import h5py
from pathlib import Path
import torch
from torch.distributions import Uniform
from ml4gw.waveforms import IMRPhenomD
from ml4gw.dataloading import Hdf5TimeSeriesDataset
from ml4gw.transforms import SpectralDensity, Whiten
from ml4gw.distributions import PowerLaw, Sine, Cosine, DeltaFunction
from ml4gw.gw import get_ifo_geometry, compute_observed_strain, reweight_snrs

def generate_eval_datasets(args):
    low_mass, high_mass, snr = args
    print(f"Starting work on evaluation_dataset_{low_mass}_{high_mass}_{snr}.hdf5")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    waveform_duration = 8
    sample_rate = 2048

    f_min = 10
    f_max = 1024
    f_ref = 20

    nyquist = sample_rate / 2
    num_samples = int(waveform_duration * sample_rate)
    num_freqs = num_samples // 2 + 1

    frequencies = torch.linspace(0, nyquist, num_freqs).to(device)
    freq_mask = (frequencies >= f_min) * (frequencies < f_max).to(device)

    num_waveforms = 1000

    # Create a dictionary of parameter distributions
    param_dict = {
        "chirp_mass": PowerLaw(((1/4)**(3/5) * low_mass), ((1/4)**(3/5) * high_mass), -2.35),
        "mass_ratio": DeltaFunction(1),
        "chi1": DeltaFunction(0),
        "chi2": DeltaFunction(0),
        "distance": PowerLaw(100, 1000, 2),
        "phic": DeltaFunction(0),
        "inclination": Sine(),
    }

    # And then sample from each of those distributions
    params = {
        k: v.sample((num_waveforms,)).to(device) for k, v in param_dict.items()
    }

    approximant = IMRPhenomD().to(device)

    hc_f, hp_f = approximant(f=frequencies[freq_mask], f_ref=f_ref, **params)

    shape = (hc_f.shape[0], num_freqs)
    hc_spectrum = torch.zeros(shape, dtype=hc_f.dtype, device=device)
    hp_spectrum = torch.zeros(shape, dtype=hc_f.dtype, device=device)

    # fill the spectrum with the
    # hc and hp values at the specified frequencies
    hc_spectrum[:, freq_mask] = hc_f
    hp_spectrum[:, freq_mask] = hp_f

    # now, irfft and scale the waveforms by sample_rate
    hc, hp = torch.fft.irfft(hc_spectrum), torch.fft.irfft(hp_spectrum)
    hc *= sample_rate
    hp *= sample_rate

    # The coalescence point is placed at the right edge, so shift it to
    # give some room for ringdown
    ringdown_duration = 0.5
    ringdown_size = int(ringdown_duration * sample_rate)
    hc = torch.roll(hc, -ringdown_size, dims=-1)
    hp = torch.roll(hp, -ringdown_size, dims=-1)

    # Define probability distributions for sky location and polarization angle
    dec = Cosine()
    psi = Uniform(0, torch.pi)
    phi = Uniform(-torch.pi, torch.pi)

    ifos = ["H1", "L1"]
    tensors, vertices = get_ifo_geometry(*ifos)

    # Pass the detector geometry, along with the polarizations and sky parameters,
    # to get the observed strain
    waveforms = compute_observed_strain(
        dec=dec.sample((num_waveforms,)).to(device),
        psi=psi.sample((num_waveforms,)).to(device),
        phi=phi.sample((num_waveforms,)).to(device),
        detector_tensors=tensors.to(device),
        detector_vertices=vertices.to(device),
        sample_rate=sample_rate,
        cross=hc,
        plus=hp,
    )

    fftlength = 2
    spectral_density = SpectralDensity(
        sample_rate=sample_rate,
        fftlength=fftlength,
        overlap=None,
        average="median",
    ).to(device)

    # This is H1 and L1 data from O3
    background_file = "/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data_test/background-1240658942-9110.hdf5"
    with h5py.File(background_file, "r") as f:
        background = [torch.Tensor(f[ifo][:]) for ifo in ifos]
        background = torch.stack(background).to(device)

    # Note cast to double
    psd = spectral_density(background.double())

    # Note need to interpolate
    if psd.shape[-1] != num_freqs:
        # Adding dummy dimensions for consistency
        while psd.ndim < 3:
            psd = psd[None]
        psd = torch.nn.functional.interpolate(
            psd, size=(num_freqs,), mode="linear"
        )

    target_snrs = DeltaFunction(snr).sample((num_waveforms,)).to(device)
    # Each waveform will be scaled by the ratio of its target SNR to its current SNR
    waveforms = reweight_snrs(
        responses=waveforms,
        target_snrs=target_snrs,
        psd=psd,
        sample_rate=sample_rate,
        highpass=f_min,
    )

    # Length of data used to estimate PSD
    psd_length = 16
    psd_size = int(psd_length * sample_rate)

    # Length of filter. A segment of length fduration / 2
    # will be cropped from either side after whitening
    fduration = 2

    # Length of window of data we'll feed to our network
    kernel_length = 1.5
    kernel_size = int(1.5 * sample_rate)

    # Total length of data to sample
    window_length = psd_length + fduration + kernel_length

    fnames = list(Path("/data/p_dsi/ligo/chattec-dgx01/chattec/LIGO/ligo_data/ml4gw_data_test").iterdir())
    dataloader = Hdf5TimeSeriesDataset(
        fnames=fnames,
        channels=ifos,
        kernel_size=int(window_length * sample_rate),
        batch_size=2
        * num_waveforms,  # Grab twice as many background samples as we have waveforms
        batches_per_epoch=1,
        coincident=False,
    )

    background_samples = [x for x in dataloader][0].to(device)

    whiten = Whiten(
        fduration=fduration, sample_rate=sample_rate, highpass=f_min
    ).to(device)

    # Create PSDs using the first psd_length seconds of each sample
    # with the SpectralDensity module we defined earlier
    psd = spectral_density(background_samples[..., :psd_size].double())

    # Take everything after the first psd_length as our input kernel
    kernel = background_samples[..., psd_size:]
    # And whiten using our PSDs
    whitened_kernel = whiten(kernel, psd)

    pad = int(fduration / 2 * sample_rate)
    injected = kernel.detach().clone()
    # Inject waveforms into every other background sample
    injected[::2, :, pad:-pad] += waveforms[..., -kernel_size:]
    # And whiten with the same PSDs as before
    whitened_injected = whiten(injected, psd)

    y = torch.zeros(len(injected))
    y[::2] = 1
    with h5py.File(f"evaluation_dataset_{low_mass}_{high_mass}_{snr}.hdf5", "w") as f:
        f.create_dataset("X", data=whitened_injected.cpu())
        f.create_dataset("y", data=y)
    print(f"Created evaluation_dataset_{low_mass}_{high_mass}_{snr}.hdf5")

parameters = [(20, 60, 12), (60, 120, 12), (120, 200, 12), (200, 400, 12), (400, 600, 12), (600, 800, 12)]
for parameter in parameters:
    generate_eval_datasets(parameter)