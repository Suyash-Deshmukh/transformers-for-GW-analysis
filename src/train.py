import matplotlib.pyplot as plt  # type: ignore
import glob
import time
import argparse
import torch, numpy as np, os # type: ignore
from pathlib import Path
import h5py # type: ignore
from tqdm import tqdm # type: ignore
from typing import Callable, Dict, List, Tuple
from sklearn.metrics import roc_curve # type: ignore
from torch.utils.data import DataLoader, TensorDataset # type: ignore

from ml4gw import augmentations, distributions, gw, transforms, waveforms # type: ignore
from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset # type: ignore
from ml4gw.utils.slicing import sample_kernels # type: ignore
from ml4gw.distributions import PowerLaw, Sine, DeltaFunction # type: ignore

from model import WhisperModule, Wav2Vec2Module, ASTModule, WavLMModule, HubertModule, ParakeetModule, MimiModule
from utils import EarlyStopper

plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Computer Modern",
    "font.size": 16,
})

num_waveforms = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a dictionary of parameter distributions
param_dict = {
    "chirp_mass": PowerLaw(((1/4)**(3/5) * 20), ((1/4)**(3/5) * 800), -2.35),
    "mass_ratio": DeltaFunction(1),
    "chi1": DeltaFunction(0),
    "chi2": DeltaFunction(0),
    "distance": PowerLaw(100, 1000, 2),
    "phic": DeltaFunction(0),
    "inclination": Sine(),
}

params = {k: v.sample((num_waveforms,)).to(device) for k, v in param_dict.items()}

class Ml4gwDetectionModel(torch.nn.Module):
    """
    Model with methods for generating waveforms and
    performing preprocessing augmentations in 
    real-time on the GPU. Also loads training background 
    in chunks from disk, then samples batches from chunks.
    """

    def __init__(
        self,
        architecture: torch.nn.Module,
        ifos: List[str] = ["H1", "L1"],
        kernel_length: float = 1.5, # change for AST: 1
        # PSD/whitening args
        fduration: float = 2, # change for AST: 1
        psd_length: float = 16,
        sample_rate: float = 2048,
        fftlength: float = 2,
        highpass: float = 32,
        # Dataloading args
        chunk_length: float = 128,
        reads_per_chunk: int = 40,
        learning_rate: float = 0.0001,
        batch_size: int = 64, # change for AST: 16
        # Waveform generation args
        waveform_prob: float = 0.5,
        approximant: Callable = waveforms.cbc.IMRPhenomD,
        param_dict: Dict[str, torch.distributions.Distribution] = param_dict,
        waveform_duration: float = 8,
        f_min: float = 10,
        f_max: float = None,
        f_ref: float = 20,
        # Augmentation args
        inversion_prob: float = 0.5,
        reversal_prob: float = 0.5,
        min_snr: float = 12,
        max_snr: float = 100,
        # Training args
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_epochs: int = 100,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        super().__init__()
        self.nn = architecture
        self.device = device
        
        # Save hyperparameters as instance attributes
        self.ifos = ifos
        self.kernel_length = kernel_length
        self.fduration = fduration
        self.psd_length = psd_length
        self.sample_rate = sample_rate
        self.fftlength = fftlength
        self.highpass = highpass
        self.chunk_length = chunk_length
        self.reads_per_chunk = reads_per_chunk
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.waveform_prob = waveform_prob
        self.waveform_duration = waveform_duration
        self.f_min = f_min
        self.f_max = f_max
        self.f_ref = f_ref
        self.inversion_prob = inversion_prob
        self.reversal_prob = reversal_prob
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_epochs = max_epochs
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        
        # Setup directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)

        # Create augmentations
        self.inverter = augmentations.SignalInverter(prob=inversion_prob)
        self.reverser = augmentations.SignalReverser(prob=reversal_prob)

        # Real-time transformations
        self.spectral_density = transforms.SpectralDensity(
            sample_rate, fftlength, average="median", fast=False
        ).to(device)
        self.whitener = transforms.Whiten(fduration, sample_rate, highpass=highpass).to(device)

        # Get geometry information for interferometers
        detector_tensors, vertices = gw.get_ifo_geometry(*ifos)
        self.register_buffer("detector_tensors", detector_tensors.to(device))
        self.register_buffer("detector_vertices", vertices.to(device))

        # Set up frequency info for waveform generation
        nyquist = sample_rate / 2
        f_max = f_max or nyquist
        num_samples = int(waveform_duration * sample_rate)
        num_freqs = num_samples // 2 + 1
        frequencies = torch.linspace(0, nyquist, num_freqs)
        freq_mask = (frequencies >= f_min) * (frequencies < f_max)
        self.register_buffer("frequencies", frequencies.to(device))
        self.register_buffer("freq_mask", freq_mask.to(device))

        # Define parameter distributions
        self.param_dict = param_dict
        self.dec = distributions.Cosine()
        self.psi = torch.distributions.Uniform(0, torch.pi)
        self.phi = torch.distributions.Uniform(-torch.pi, torch.pi)
        self.approximant = approximant().to(device)

        # Define SNR distribution
        self.snr = distributions.PowerLaw(min_snr, max_snr, -3)

        # Define sizes in units of samples
        self.kernel_size = int(kernel_length * sample_rate)
        self.window_size = self.kernel_size + int(fduration * sample_rate)
        self.psd_size = int(psd_length * sample_rate)
        
        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(self.nn.parameters(), self.learning_rate)
        
        # Training tracking variables
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        self.train_losses = []
        self.valid_metrics = []
        self.steps = []
        
        # Custom AUROC calculator
        self.y_true_list = []
        self.y_pred_list = []

    def forward(self, X):
        return self.nn(X)
    
    # Custom AUROC implementation to replace torchmetrics
    def reset_metric(self):
        self.y_true_list = []
        self.y_pred_list = []
        
    def update_metric(self, y_pred, y_true):
        self.y_true_list.append(y_true.cpu().detach())
        self.y_pred_list.append(torch.sigmoid(y_pred).cpu().detach())
        
    def compute_auroc(self):
        """
        Compute Area Under the Receiver Operating Characteristic Curve (AUROC)
        """
        y_true = torch.cat(self.y_true_list, dim=0).numpy().flatten()
        y_pred = torch.cat(self.y_pred_list, dim=0).numpy().flatten()
        
        # Sort predictions and corresponding truth values
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true = y_true[sorted_indices]
        
        # Count positive and negative samples
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5  # If only one class present, return random guess performance
        
        # Calculate false positive rate (FPR) and true positive rate (TPR)
        tpr = np.cumsum(y_true) / n_pos
        fpr = np.cumsum(1 - y_true) / n_neg
        
        # Calculate area under curve using trapezoidal rule
        auroc = np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1])) / 2
        
        return auroc

    def compute_roc_curve(self):
        """
        Compute ROC curve data (FPR, TPR, and thresholds) using sklearn's roc_curve.
        """
        y_true = torch.cat(self.y_true_list, dim=0).numpy().flatten()
        y_pred = torch.cat(self.y_pred_list, dim=0).numpy().flatten()
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        return fpr, tpr, thresholds

    def generate_waveforms(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rvs = torch.rand(size=(batch_size,), device=self.device)
        mask = rvs < self.waveform_prob
        num_injections = mask.sum().item()
 # 1000
        params = {k: v.sample((num_injections,)).to(self.device) for k, v in self.param_dict.items()}
        hc, hp = self.approximant(
            f=self.frequencies[self.freq_mask],
            f_ref=self.f_ref,
            **params
        )
        shape = (hc.shape[0], len(self.frequencies))
        hc_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hp_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)

        # Fill the spectrum with the hc and hp values at the specified frequencies
        hc_spectrum[:, self.freq_mask] = hc
        hp_spectrum[:, self.freq_mask] = hp

        # Now, irfft and scale the waveforms by sample_rate
        hc, hp = torch.fft.irfft(hc_spectrum), torch.fft.irfft(hp_spectrum)
        hc *= self.sample_rate
        hp *= self.sample_rate

        # Roll to shift the coalescence point from the right edge
        ringdown_duration = 0.5
        ringdown_size = int(ringdown_duration * self.sample_rate)
        hc = torch.roll(hc, -ringdown_size, dims=-1)
        hp = torch.roll(hp, -ringdown_size, dims=-1)
        return hc, hp, mask

    def project_waveforms(self, hc: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
        # Sample sky parameters
        N = len(hc)
        dec = self.dec.sample((N,)).to(hc.device)
        psi = self.psi.sample((N,)).to(hc.device)
        phi = self.phi.sample((N,)).to(hc.device)

        # Project to interferometer response
        return gw.compute_observed_strain(
            dec=dec,
            psi=psi,
            phi=phi,
            detector_tensors=self.detector_tensors,
            detector_vertices=self.detector_vertices,
            sample_rate=self.sample_rate,
            cross=hc,
            plus=hp
        )

    def rescale_snrs(self, responses: torch.Tensor, psd: torch.Tensor) -> torch.Tensor:
        # Make sure everything has the same number of frequency bins
        num_freqs = int(responses.size(-1) // 2) + 1
        if psd.size(-1) != num_freqs:
            psd = torch.nn.functional.interpolate(psd, size=(num_freqs,), mode="linear")
        N = len(responses)
        target_snrs = self.snr.sample((N,)).to(responses.device)
        return gw.reweight_snrs(
            responses=responses.double(), 
            target_snrs=target_snrs,
            psd=psd,
            sample_rate=self.sample_rate,
            highpass=self.highpass,
        )

    def sample_waveforms(self, responses: torch.Tensor) -> torch.Tensor:
        # Slice off random views of each waveform to inject in arbitrary positions
        responses = responses[:, :, -self.window_size:]

        # Pad so that at least half the kernel always contains signals
        pad = [0, int(self.window_size // 2)]
        responses = torch.nn.functional.pad(responses, pad)
        return sample_kernels(responses, self.window_size, coincident=True)

    @torch.no_grad()
    def augment(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Break off "background" from target kernel and compute its PSD
        background, X = torch.split(X, [self.psd_size, self.window_size], dim=-1)
        psd = self.spectral_density(background.double())

        # Generate at most batch_size signals from our parameter distributions
        # Keep a mask that indicates which rows to inject in
        batch_size = X.size(0)
        hc, hp, mask = self.generate_waveforms(batch_size)

        # Augment with inversion and reversal
        X = self.inverter(X)
        X = self.reverser(X)

        # Sample sky parameters and project to responses, then
        # rescale the response according to a randomly sampled SNR
        responses = self.project_waveforms(hc, hp)
        responses = self.rescale_snrs(responses, psd[mask])

        # Randomly slice out a window of the waveform, add it
        # to our background, then whiten everything
        responses = self.sample_waveforms(responses)
        X[mask] += responses.float()
        X = self.whitener(X, psd)

        # Create labels, marking 1s where we injected
        y = torch.zeros((batch_size, 1), device=X.device)
        y[mask] = 1
        return X, y

    def create_train_dataloader(self):
        # Setup for batches and chunks
        samples_per_epoch = 3000 # change for AST: 1000
        batches_per_epoch = int((samples_per_epoch - 1) // self.batch_size) + 1
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1

        # Hdf5TimeSeries dataset samples batches from disk
        fnames = list(Path("/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data").iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )

        # Sample batches to pass to our NN from the chunks loaded from disk
        return ChunkedTimeSeriesDataset(
            dataset,
            kernel_size=self.window_size + self.psd_size,
            batch_size=self.batch_size,
            batches_per_chunk=batches_per_chunk,
            coincident=False
        )
    
    def create_val_dataloader(self):
        with h5py.File("/home/deshmus/transformers-for-GW-analysis/data/validation_dataset.hdf5", "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            pin_memory=True,
        )
    
    def create_eval_dataloader(self, file_name):
        with h5py.File(file_name, "r") as f:
            X = torch.Tensor(f["X"][:])
            y = torch.Tensor(f["y"][:])
        dataset = TensorDataset(X, y)
        return DataLoader(
            dataset,
            batch_size=self.batch_size * 4,
            shuffle=False,
            pin_memory=True,
        )

    def training_step(self, batch):
        X, y = self.augment(batch.to(self.device))
        self.optimizer.zero_grad()
        y_hat = self(X)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(y_hat, y)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def validation_step(self, batch):
        X, y = batch
        X = X.to(self.device)
        y = y.to(self.device)
        with torch.no_grad():
            y_hat = self(X)
            self.update_metric(y_hat, y)

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'best_metric': self.best_metric,
        }, path)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
    def log_metrics(self):
        with open(os.path.join(self.log_dir, 'metrics.csv'), 'a') as f:
            for step, loss in zip(self.steps, self.train_losses):
                f.write(f"{self.current_epoch},{step},{loss},,\n")
            for i, metric_val in enumerate(self.valid_metrics):
                f.write(f"{self.current_epoch},{self.global_step},,{metric_val}\n")
        
        # Clear lists for next epoch
        self.steps = []
        self.train_losses = []
        self.valid_metrics = []
        
    def train(self):
        # Create log file header if it doesn't exist
        if not os.path.exists(os.path.join(self.log_dir, 'metrics.csv')):
            with open(os.path.join(self.log_dir, 'metrics.csv'), 'w') as f:
                f.write("epoch,step,train_loss,step,valid_auroc\n")
        
        train_dataloader = self.create_train_dataloader()
        val_dataloader = self.create_val_dataloader()
        early_stopper = EarlyStopper(patience=20)
        
        # Setup lr scheduler
        total_steps = self.max_epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            pct_start=0.1,
            total_steps=total_steps
        )
        
        # Move model to device
        self.to(self.device)
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            
            # Training phase
            self.nn.train()
            train_loop = tqdm(train_dataloader, desc="Training")
            for batch in train_loop:
                loss = self.training_step(batch)
                
                # Update learning rate
                scheduler.step()
                
                # Logging
                self.global_step += 1
                self.steps.append(self.global_step)
                self.train_losses.append(loss)
                
                if self.global_step % 5 == 0:  # Log every 5 steps
                    train_loop.set_postfix(loss=f"{loss:.4f}")
                
            # Validation phase
            self.nn.eval()
            self.reset_metric()
            for batch in tqdm(val_dataloader, desc="Validation"):
                self.validation_step(batch)
            
            # Calculate metric
            current_metric = self.compute_auroc()
            self.valid_metrics.append(current_metric)
            print(f"Validation AUROC: {current_metric:.4f}")
            
            # Save best model
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(os.path.join(self.checkpoint_dir, f'best_model_{self.nn.model_name}.pt'))
                print(f"New best model saved with AUROC: {current_metric:.4f}")
            
                fpr, tpr, thresholds = self.compute_roc_curve()
                roc_file = os.path.join(self.log_dir, f"validation_roc_{self.nn.model_name}.hdf5")
                with h5py.File(roc_file, "w") as f:
                    f.create_dataset("fpr", data=fpr)
                    f.create_dataset("tpr", data=tpr)
                    f.create_dataset("thresholds", data=thresholds)
            
            # Log metrics
            self.log_metrics()
            
            if early_stopper.early_stop(current_metric):
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Saved ROC curve data to {roc_file}")

    def evaluate(self, hdf5_paths):
        # Load model
        self.load_checkpoint(os.path.join(self.checkpoint_dir, f'best_model_{self.nn.model_name}.pt'))
        self.to(self.device)
        self.nn.eval()
        print("Model loaded.")

        for path in hdf5_paths:
            label = "_".join(path.split("_")[3:])[:-5]
            print(f"Starting evaluation on {label}.")
            # Create dataset and dataloader
            eval_dataloader = self.create_eval_dataloader(path)

            # Reset and compute metrics
            self.reset_metric()
            for batch in tqdm(eval_dataloader, desc="Evaluation"):
                self.validation_step(batch)

            auroc = self.compute_auroc()
            print(f"{label} AUROC: {auroc:.4f}")

            # Save ROC curve
            fpr, tpr, thresholds = self.compute_roc_curve()
            roc_file = os.path.join(self.log_dir, f"{self.nn.model_name}_eval_roc_{label}.hdf5")
            with h5py.File(roc_file, "w") as f:
                f.create_dataset("fpr", data=fpr)
                f.create_dataset("tpr", data=tpr)
                f.create_dataset("thresholds", data=thresholds)
            print(f"Saved ROC curve data to {roc_file}")

def main(args):
    if args.model_name == "Whisper":
        architecture = WhisperModule(
            whisper_model_name="openai/whisper-tiny",
            model_name="whisper",
            input_sample_rate=2048,
            target_sample_rate=16000
        ).to(device)
    elif args.model_name == "Wav2Vec2":
        architecture = Wav2Vec2Module(
            wav2vec2_model_name="facebook/wav2vec2-base-960h",
            model_name="wav2vec2",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "Wav2Vec2-XLSR":
        architecture = Wav2Vec2Module(
            wav2vec2_model_name="facebook/wav2vec2-large-xlsr-53",
            model_name="wav2vec2-xlsr",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "Wav2Vec2-BERT":
        architecture = Wav2Vec2Module(
            wav2vec2_model_name="facebook/w2v-bert-2.0",
            model_name="wav2vec2-bert",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "AST":
        architecture = ASTModule(
            ast_model_name="MIT/ast-finetuned-audioset-10-10-0.4593",
            model_name="ast",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "WavLM":
        architecture = WavLMModule(
            wavlm_model_name="microsoft/wavlm-base",
            model_name="wavlm",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "Hubert":
        architecture = HubertModule(
            hubert_model_name="facebook/hubert-base-ls960",
            model_name="hubert",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "Parakeet":
        architecture = ParakeetModule(
            parakeet_model_name="nvidia/parakeet-ctc-1.1b",
            model_name="parakeet",
            input_sample_rate=2048,
            target_sample_rate=16000,
        ).to(device)
    elif args.model_name == "Mimi":
        architecture = MimiModule(
            mimi_model_name="kyutai/mimi",
            model_name="mimi",
            input_sample_rate=2048,
            target_sample_rate=24000,
        ).to(device)

    model = Ml4gwDetectionModel(architecture=architecture)

    if not args.eval_only:
        print("Starting training...")
        start_time = time.time()
        model.train()
        elapsed = time.time() - start_time
        print(f"Total training time: {elapsed:.2f} seconds")

    print("Starting evaluation...")
    start_time = time.time()
    paths = glob.glob("/home/deshmus/transformers-for-GW-analysis/data/evaluation*.hdf5")
    model.evaluate(paths)
    elapsed = time.time() - start_time
    print(f"Total evaluating time: {elapsed:.2f} seconds")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Which model to train and evalutate")
    parser.add_argument("--eval_only", action="store_true", help="Skip training and only run evaluation using best saved model")

    args = parser.parse_args()
    main(args)