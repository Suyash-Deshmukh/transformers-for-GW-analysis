import torch
import matplotlib.pyplot as plt
import glob
import time
import argparse
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "Computer Modern",
    "font.size": 16,
})

device = "cuda" if torch.cuda.is_available() else "cpu"

from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
from torch.distributions import Uniform

from sklearn.metrics import roc_curve

# Number of waveforms to sample (adjust for CPU vs GPU)
num_waveforms = 1000

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

# And then sample from each of those distributions
params = {k: v.sample((num_waveforms,)).to(device) for k, v in param_dict.items()}

import numpy as np
import os
import time
from pathlib import Path
import h5py
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import time

from ml4gw import augmentations, distributions, gw, transforms, waveforms
from ml4gw.dataloading import ChunkedTimeSeriesDataset, Hdf5TimeSeriesDataset
from ml4gw.utils.slicing import sample_kernels
from typing import Callable, Dict, List, Tuple, Any

###############################################################################
# Ml4gwDetectionModel: manages waveform generation, augmentation, training, etc.
###############################################################################
class Ml4gwDetectionModel(torch.nn.Module):
    """
    Model with methods for generating waveforms and performing preprocessing augmentations
    in real-time on the GPU. Also loads training background in chunks from disk, then samples
    batches from chunks.
    """
    def __init__(
        self,
        architecture: torch.nn.Module,
        ifos: List[str] = ["H1", "L1"],
        kernel_length: float = 1.5,
        # PSD/whitening args
        fduration: float = 2,
        psd_length: float = 16,
        sample_rate: float = 2048,
        fftlength: float = 2,
        highpass: float = 32,
        # Dataloading args
        chunk_length: float = 128,
        reads_per_chunk: int = 40,
        learning_rate: float = 0.0001,
        batch_size: int = 64,
        # Waveform generation args
        waveform_prob: float = 0.5,
        approximant: Callable = waveforms.cbc.IMRPhenomD,
        param_dict: Dict[str, torch.distributions.Distribution] = param_dict,
        waveform_duration: float = 8,
        f_min: float = 20,
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
    
    def reset_metric(self):
        self.y_true_list = []
        self.y_pred_list = []
        
    def update_metric(self, y_pred, y_true):
        self.y_true_list.append(y_true.cpu().detach())
        self.y_pred_list.append(torch.sigmoid(y_pred).cpu().detach())
        
    def compute_auroc(self):
        y_true = torch.cat(self.y_true_list, dim=0).numpy().flatten()
        y_pred = torch.cat(self.y_pred_list, dim=0).numpy().flatten()
        
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true = y_true[sorted_indices]
        
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        
        if n_pos == 0 or n_neg == 0:
            return 0.5
        
        tpr = np.cumsum(y_true) / n_pos
        fpr = np.cumsum(1 - y_true) / n_neg
        
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

        params = {k: v.sample((num_injections,)).to(self.device) for k, v in self.param_dict.items()}
        hc, hp = self.approximant(
            f=self.frequencies[self.freq_mask],
            f_ref=self.f_ref,
            **params
        )
        shape = (hc.shape[0], len(self.frequencies))
        hc_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)
        hp_spectrum = torch.zeros(shape, dtype=hc.dtype, device=self.device)

        hc_spectrum[:, self.freq_mask] = hc
        hp_spectrum[:, self.freq_mask] = hp

        hc, hp = torch.fft.irfft(hc_spectrum), torch.fft.irfft(hp_spectrum)
        hc *= self.sample_rate
        hp *= self.sample_rate

        ringdown_duration = 0.5
        ringdown_size = int(ringdown_duration * self.sample_rate)
        hc = torch.roll(hc, -ringdown_size, dims=-1)
        hp = torch.roll(hp, -ringdown_size, dims=-1)
        return hc, hp, mask

    def project_waveforms(self, hc: torch.Tensor, hp: torch.Tensor) -> torch.Tensor:
        N = len(hc)
        dec = self.dec.sample((N,)).to(hc.device)
        psi = self.psi.sample((N,)).to(hc.device)
        phi = self.phi.sample((N,)).to(hc.device)

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
        responses = responses[:, :, -self.window_size:]
        pad = [0, int(self.window_size // 2)]
        responses = torch.nn.functional.pad(responses, pad)
        return sample_kernels(responses, self.window_size, coincident=True)

    @torch.no_grad()
    def augment(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        background, X = torch.split(X, [self.psd_size, self.window_size], dim=-1)
        psd = self.spectral_density(background.double())

        batch_size = X.size(0)
        hc, hp, mask = self.generate_waveforms(batch_size)

        X = self.inverter(X)
        X = self.reverser(X)

        responses = self.project_waveforms(hc, hp)
        responses = self.rescale_snrs(responses, psd[mask])
        responses = self.sample_waveforms(responses)
        # Save a copy of the injection waveform for plotting
        # injected_waveform = responses.clone()
        X[mask] += responses.float()
        X = self.whitener(X, psd)

        # For binary classification, create labels (1 for injection, 0 for background)
        y = torch.zeros((batch_size, 1), device=X.device)
        y[mask] = 1

        # ---- Plotting a few random samples for debugging ----
        # if self.global_step == 0:
        #     num_samples_to_plot = min(3, X.size(0))
        #     indices = np.random.choice(X.size(0), size=num_samples_to_plot, replace=False)
        #     os.makedirs("sample_plots", exist_ok=True)
        #     for i in indices:
        #         fig, ax = plt.subplots()
        #         # Plot the final whitened data (noise with injection added if mask true)
        #         ax.plot(X[i, 0].cpu().numpy(), label="Whitened Data", linewidth=2)
        #         # If an injection was applied to this sample, overlay the injected waveform
        #         if y[i].item() == 1:
        #             ax.plot(injected_waveform[i, 0].cpu().numpy(), label="Injected Waveform", linestyle="--", linewidth=2)
        #         ax.set_title(f"Sample {i} (y={y[i].item()}), Epoch {self.current_epoch}, Step {self.global_step}")
        #         ax.legend()
        #         plot_path = os.path.join("sample_plots", f"sample_epoch{self.current_epoch}_step{self.global_step}_idx{i}.png")
        #         plt.savefig(plot_path)
        #         plt.close(fig)
        #     print("Saved random sample plots to 'sample_plots/'")
        
        return X, y

    def create_train_dataloader(self):
        samples_per_epoch = 3000
        batches_per_epoch = int((samples_per_epoch - 1) // self.batch_size) + 1
        batches_per_chunk = int(batches_per_epoch // 10)
        chunks_per_epoch = int(batches_per_epoch // batches_per_chunk) + 1

        fnames = list(Path("/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data").iterdir())
        dataset = Hdf5TimeSeriesDataset(
            fnames=fnames,
            channels=self.ifos,
            kernel_size=int(self.chunk_length * self.sample_rate),
            batch_size=self.reads_per_chunk,
            batches_per_epoch=chunks_per_epoch,
            coincident=False,
        )

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
        
        # print(X.shape)
        
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
        
        self.steps = []
        self.train_losses = []
        self.valid_metrics = []
        
    def train(self):
        if not os.path.exists(os.path.join(self.log_dir, 'metrics.csv')):
            with open(os.path.join(self.log_dir, 'metrics.csv'), 'w') as f:
                f.write("epoch,step,train_loss,step,valid_auroc\n")
        
        train_dataloader = self.create_train_dataloader()
        val_dataloader = self.create_val_dataloader()
        
        total_steps = self.max_epochs * len(train_dataloader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            pct_start=0.1,
            total_steps=total_steps
        )
        
        self.to(self.device)
        
        for epoch in range(self.current_epoch, self.max_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch+1}/{self.max_epochs}")
            
            self.nn.train()
            train_loop = tqdm(train_dataloader, desc="Training")
            for batch in train_loop:
                loss = self.training_step(batch)
                scheduler.step()
                self.global_step += 1
                self.steps.append(self.global_step)
                self.train_losses.append(loss)
                
                if self.global_step % 5 == 0:
                    train_loop.set_postfix(loss=f"{loss:.4f}")
                
            self.nn.eval()
            self.reset_metric()
            for batch in tqdm(val_dataloader, desc="Validation"):
                self.validation_step(batch)
            
            current_metric = self.compute_auroc()
            self.valid_metrics.append(current_metric)
            print(f"Validation AUROC: {current_metric:.4f}")
            
            if current_metric > self.best_metric:
                self.best_metric = current_metric
                self.save_checkpoint(os.path.join(self.checkpoint_dir, 'best_model.pt'))
                print(f"New best model saved with AUROC: {current_metric:.4f}")
            
            # self.save_checkpoint(os.path.join(self.checkpoint_dir, f'epoch_{epoch+1}.pt'))
            self.log_metrics()
            
        # After training, compute and save ROC curve data for the validation set using sklearn
        fpr, tpr, thresholds = self.compute_roc_curve()
        roc_file = os.path.join(self.log_dir, "validation_roc_Wav2Vec2.hdf5")
        with h5py.File(roc_file, "w") as f:
            f.create_dataset("fpr", data=fpr)
            f.create_dataset("tpr", data=tpr)
            f.create_dataset("thresholds", data=thresholds)
        print(f"Saved ROC curve data to {roc_file}")
    
    def evaluate(self, hdf5_paths):
        # Load model
        self.load_checkpoint(os.path.join(self.checkpoint_dir, f'best_model_wav2vec2.pt'))
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
            roc_file = os.path.join(self.log_dir, f"wav2vec2_eval_roc_{label}.hdf5")
            with h5py.File(roc_file, "w") as f:
                f.create_dataset("fpr", data=fpr)
                f.create_dataset("tpr", data=tpr)
                f.create_dataset("thresholds", data=thresholds)
            print(f"Saved ROC curve data to {roc_file}")

###############################################################################
# Updated Wav2Vec2DetectionNetwork: processes each channel separately.
###############################################################################
import fnmatch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from peft import LoraConfig, get_peft_model
import torchaudio.transforms as T  # Torch-native resampling


class two_channel_ligo_binary_classifier(nn.Module):
    def __init__(self, encoder, num_classes=1):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.config.hidden_size * 2, 1024), 
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, input_0, input_1):
        output_h1 = self.encoder(input_0).last_hidden_state
        output_l1 = self.encoder(input_1).last_hidden_state
        
        output_h1 = torch.mean(output_h1, dim=1)
        output_l1 = torch.mean(output_l1, dim=1)
        
        outputs = torch.cat((output_h1, output_l1), dim=1)
        logits = self.classifier(outputs)
        
        return logits
    

class Wav2Vec2DetectionNetwork(torch.nn.Module):
    """
    A detection network that uses a Wav2Vec2 encoder (augmented with DoRA via LoRA)
    and a decoder composed of fully connected layers for binary classification.
    
    The network:
      1. Accepts an input tensor of shape [batch, channels, time] (with channels=2).
      2. Processes each channel individually:
         - Resamples from the input sample rate (e.g., 2048 Hz) to the target rate (e.g., 16 kHz)
           using librosa.
         - Extracts features with the Wav2Vec2 feature extractor.
         - Passes the resampled signals through the Wav2Vec2 encoder.
      3. Pools over the time dimension for each channel.
      4. Concatenates the resulting representations and feeds them to a decoder.
    """
    def __init__(
        self,
        wav2vec2_model_name: str = "facebook/wav2vec2-base-960h",
        input_sample_rate: int = 2048,
        target_sample_rate: int = 16000,
        device: str = "cuda"
    ):
        super(Wav2Vec2DetectionNetwork, self).__init__()
        self.wav2vec2_model_name = wav2vec2_model_name
        self.input_sample_rate = input_sample_rate
        self.target_sample_rate = target_sample_rate
        self.device = device
        
        self.resampler = T.Resample(orig_freq=self.input_sample_rate, new_freq=self.target_sample_rate)

        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-960h")
        
        self.wav2vec2_model = Wav2Vec2Model.from_pretrained(self.wav2vec2_model_name)
        
        # Apply DoRA via LoRA to the attention layers
        module_names = [name for name, _ in self.wav2vec2_model.named_modules()]
        patterns = [
            "encoder.layers.*.attention.q_proj",
            "encoder.layers.*.attention.k_proj",
            "encoder.layers.*.attention.v_proj",
            "encoder.layers.*.attention.out_proj",
        ]
        matched_modules = []
        for pattern in patterns:
            matched_modules.extend(fnmatch.filter(module_names, pattern))
        lora_config = LoraConfig(use_dora=True, r=8, lora_alpha=32, target_modules=matched_modules)
        
        self.wav2vec2_model = get_peft_model(self.wav2vec2_model, lora_config)
        
        # Freeze base parameters; only train LoRA parameters.
        for name, param in self.wav2vec2_model.named_parameters():
            param.requires_grad = "lora" in name
            
        self.classifier = two_channel_ligo_binary_classifier(self.wav2vec2_model).to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Expect input x to be of shape [batch, 2, time]
        if x.dim() == 3 and x.shape[1] == 2:
            x_h1 = x[:, 0, :]  # Channel 1
            x_l1 = x[:, 1, :]  # Channel 2
        else:
            raise ValueError("Input tensor must have shape [batch, 2, time].")
        
        # Resample to target sample rate
        x_h1 = self.resampler(x_h1)
        x_l1 = self.resampler(x_l1)

        x_h1_features = self.feature_extractor(x_h1.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(x.device)
        x_h1_features = x_h1_features.input_values.squeeze(0)
        
        x_l1_features = self.feature_extractor(x_l1.cpu().numpy(), sampling_rate=16000, return_tensors="pt").to(x.device)
        x_l1_features = x_l1_features.input_values.squeeze(0)
        
        logits = self.classifier(x_h1_features, x_l1_features)
        
        return logits

###############################################################################
# Instantiate the architecture and model, then start training.
###############################################################################
# architecture = Wav2Vec2DetectionNetwork(
#     wav2vec2_model_name="facebook/wav2vec2-base-960h",
#     input_sample_rate=2048,
#     target_sample_rate=16000,
# ).to(device)

# model = Ml4gwDetectionModel(architecture=architecture)

# start_time = time.time()
# model.train()
# elapsed = time.time() - start_time
# print(f"Total training time: {elapsed:.2f} seconds")


def main(args):
    architecture = Wav2Vec2DetectionNetwork(
        wav2vec2_model_name="facebook/wav2vec2-base-960h",
        input_sample_rate=2048,
        target_sample_rate=16000,
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