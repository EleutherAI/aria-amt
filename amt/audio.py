"""Contains code taken from https://github.com/openai/whisper"""

import os
import random
import torch
import torchaudio
import torch.nn.functional as F
import torchaudio.functional as AF
import numpy as np

from amt.config import load_config
from amt.tokenizer import AmtTokenizer

# hard-coded audio hyperparameters
config = load_config()["audio"]
SAMPLE_RATE = config["sample_rate"]
N_FFT = config["n_fft"]
HOP_LENGTH = config["hop_len"]
CHUNK_LENGTH = config["chunk_len"]
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input
N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(
                array, [pad for sizes in pad_widths[::-1] for pad in sizes]
            )
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


# Refactor default params are stored in config.json
class AudioTransform(torch.nn.Module):
    def __init__(
        self,
        reverb_factor: int = 1,
        min_snr: int = 20,
        max_snr: int = 50,
        max_dist_gain: int = 25,
        min_dist_gain: int = 0,
        noise_ratio: float = 0.9,
        reverb_ratio: float = 0.9,
        applause_ratio: float = 0.01,
        bandpass_ratio: float = 0.15,
        distort_ratio: float = 0.15,
        reduce_ratio: float = 0.01,
        detune_ratio: float = 0.1,
        detune_max_shift: float = 0.15,
        spec_aug_ratio: float = 0.9,
    ):
        super().__init__()
        self.tokenizer = AmtTokenizer()
        self.reverb_factor = reverb_factor
        self.min_snr = min_snr
        self.max_snr = max_snr
        self.max_dist_gain = max_dist_gain
        self.min_dist_gain = min_dist_gain

        self.config = load_config()["audio"]
        self.sample_rate = self.config["sample_rate"]
        self.chunk_len = self.config["chunk_len"]
        self.num_samples = self.sample_rate * self.chunk_len

        self.noise_ratio = noise_ratio
        self.reverb_ratio = reverb_ratio
        self.applause_ratio = applause_ratio
        self.bandpass_ratio = bandpass_ratio
        self.distort_ratio = distort_ratio
        self.reduce_ratio = reduce_ratio
        self.detune_ratio = detune_ratio
        self.detune_max_shift = detune_max_shift
        self.spec_aug_ratio = spec_aug_ratio

        self.time_mask_param = 2500
        self.freq_mask_param = 15
        self.reduction_resample_rate = 6000

        # Audio aug
        impulse_paths = self._get_paths(
            os.path.join(os.path.dirname(__file__), "assets", "impulse")
        )
        noise_paths = self._get_paths(
            os.path.join(os.path.dirname(__file__), "assets", "noise")
        )
        applause_paths = self._get_paths(
            os.path.join(os.path.dirname(__file__), "assets", "applause")
        )

        # Register impulses and noises as buffers
        self.num_impulse = 0
        for i, impulse in enumerate(self._get_impulses(impulse_paths)):
            self.register_buffer(f"impulse_{i}", impulse)
            self.num_impulse += 1

        self.num_noise = 0
        for i, noise in enumerate(self._get_noise(noise_paths)):
            self.register_buffer(f"noise_{i}", noise)
            self.num_noise += 1

        self.num_applause = 0
        for i, applause in enumerate(self._get_noise(applause_paths)):
            self.register_buffer(f"applause_{i}", applause)
            self.num_applause += 1

        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.config["n_fft"],
            hop_length=self.config["hop_len"],
        )
        self.mel_transform = torchaudio.transforms.MelScale(
            n_mels=self.config["n_mels"],
            sample_rate=self.config["sample_rate"],
            n_stft=self.config["n_fft"] // 2 + 1,
        )
        self.spec_aug = torch.nn.Sequential(
            torchaudio.transforms.TimeMasking(
                time_mask_param=self.time_mask_param,
                iid_masks=True,
            ),
            torchaudio.transforms.FrequencyMasking(
                freq_mask_param=self.freq_mask_param, iid_masks=True
            ),
        )

    def get_params(self):
        return {
            "noise_ratio": self.noise_ratio,
            "reverb_ratio": self.reverb_ratio,
            "applause_ratio": self.applause_ratio,
            "bandpass_ratio": self.bandpass_ratio,
            "distort_ratio": self.distort_ratio,
            "reduce_ratio": self.reduce_ratio,
            "detune_ratio": self.detune_ratio,
            "detune_max_shift": self.detune_max_shift,
            "spec_aug_ratio": self.spec_aug_ratio,
            "time_mask_param": self.time_mask_param,
            "freq_mask_param": self.freq_mask_param,
        }

    def _get_paths(self, dir_path):
        os.makedirs(dir_path, exist_ok=True)

        return [
            os.path.join(dir_path, f)
            for f in os.listdir(dir_path)
            if os.path.isfile(os.path.join(dir_path, f))
        ]

    def _get_impulses(self, impulse_paths: list):
        impulses = [torchaudio.load(path) for path in impulse_paths]
        impulses = [
            AF.resample(
                waveform=wav, orig_freq=sr, new_freq=config["sample_rate"]
            ).mean(0, keepdim=True)[:, : 5 * self.sample_rate]
            for wav, sr in impulses
        ]
        return [
            (wav) / (torch.linalg.vector_norm(wav, ord=2)) for wav in impulses
        ]

    def _get_noise(self, noise_paths: list):
        noises = [torchaudio.load(path) for path in noise_paths]
        noises = [
            AF.resample(
                waveform=wav, orig_freq=sr, new_freq=config["sample_rate"]
            ).mean(0, keepdim=True)[:, : self.num_samples]
            for wav, sr in noises
        ]

        for wav in noises:
            assert wav.shape[-1] == self.num_samples, "noise wav too short"

        return noises

    def apply_reverb(self, wav: torch.Tensor):
        # wav: (bz, L)
        batch_size, _ = wav.shape

        reverb_strength = (
            torch.Tensor([random.uniform(0, 1) for _ in range(batch_size)])
            .unsqueeze(-1)
            .to(wav.device)
        )
        reverb_type = random.randint(0, self.num_impulse - 1)
        impulse = getattr(self, f"impulse_{reverb_type}")

        reverb = AF.fftconvolve(wav, impulse, mode="full")[
            :, : self.num_samples
        ]
        if self.reverb_factor > 1:
            for _ in range(self.reverb_factor - 1):
                reverb = AF.fftconvolve(reverb, impulse, mode="full")[
                    : self.num_samples
                ]

        res = (reverb_strength * reverb) + ((1 - reverb_strength) * wav)

        return res

    def apply_noise(self, wav: torch.tensor):
        batch_size, _ = wav.shape

        snr_dbs = torch.tensor(
            [
                random.randint(self.min_snr, self.max_snr)
                for _ in range(batch_size)
            ]
        ).to(wav.device)
        noise_type = random.randint(0, self.num_noise - 1)
        noise = getattr(self, f"noise_{noise_type}")

        return AF.add_noise(waveform=wav, noise=noise, snr=snr_dbs)

    def apply_applause(self, wav: torch.tensor):
        batch_size, _ = wav.shape

        snr_dbs = torch.tensor(
            [random.randint(1, self.min_snr) for _ in range(batch_size)]
        ).to(wav.device)
        applause_type = random.randint(5, self.num_applause - 1)

        applause = getattr(self, f"applause_{applause_type}")

        return AF.add_noise(waveform=wav, noise=applause, snr=snr_dbs)

    def apply_bandpass(self, wav: torch.tensor):
        central_freq = random.randint(1000, 3500)
        Q = random.uniform(0.707, 1.41)

        return torchaudio.functional.bandpass_biquad(
            wav, self.sample_rate, central_freq, Q
        )

    def apply_reduction(self, wav: torch.tensor):
        """
        Limit the high-band pass filter, the low-band pass filter and the sample rate
        Designed to mimic the effect of recording on a low-quality microphone or phone.
        """
        wav = AF.highpass_biquad(wav, self.sample_rate, cutoff_freq=300)
        wav = AF.lowpass_biquad(wav, self.sample_rate, cutoff_freq=3400)
        wav_downsampled = AF.resample(
            wav,
            orig_freq=self.sample_rate,
            new_freq=self.reduction_resample_rate,
            lowpass_filter_width=3,
        )

        return AF.resample(
            wav_downsampled,
            self.reduction_resample_rate,
            self.sample_rate,
        )

    def apply_distortion(self, wav: torch.tensor):
        gain = random.randint(self.min_dist_gain, self.max_dist_gain)
        colour = random.randint(5, 95)

        return AF.overdrive(wav, gain=gain, colour=colour)

    def distortion_aug_cpu(self, wav: torch.Tensor):
        # This function should run on the cpu (i.e. in the dataloader collate
        # function) in order to not be a bottlekneck

        if random.random() < self.reduce_ratio:
            wav = self.apply_reduction(wav)
        if random.random() < self.distort_ratio:
            wav = self.apply_distortion(wav)

        return wav

    def shift_spec(self, specs: torch.Tensor, shift: int | float):
        if shift == 0:
            return specs

        freq_mult = 2 ** (shift / 12.0)
        _, num_bins, L = specs.shape
        new_num_bins = int(num_bins * freq_mult)

        # Interpolate expects extra channel dim
        specs = specs.unsqueeze(1)
        shifted_specs = torch.nn.functional.interpolate(
            specs, size=(new_num_bins, L), mode="bilinear", align_corners=False
        )
        shifted_specs = shifted_specs.squeeze(1)

        if shift > 0:
            shifted_specs = shifted_specs[:, :num_bins, :]
        else:
            padding = num_bins - shifted_specs.size(1)
            shifted_specs = torch.nn.functional.pad(
                shifted_specs, (0, 0, 0, padding), "constant", 0
            )

        return shifted_specs

    def detune_spec(self, specs: torch.Tensor):
        if random.random() < self.detune_ratio:
            detune_shift = random.uniform(
                -self.detune_max_shift, self.detune_max_shift
            )
            detuned_specs = self.shift_spec(specs, shift=detune_shift)

            return (specs + detuned_specs) / 2
        else:
            return specs

    def aug_wav(self, wav: torch.Tensor):
        # This function doesn't apply distortion. If distortion is desired it
        # should be run beforehand on the cpu with distortion_aug_cpu. Note
        # also that detuning is done to the spectrogram in log_mel, not the wav.

        # Noise
        if random.random() < self.noise_ratio:
            wav = self.apply_noise(wav)

        if random.random() < self.applause_ratio:
            wav = self.apply_applause(wav)

        # Reverb
        if random.random() < self.reverb_ratio:
            wav = self.apply_reverb(wav)

        # EQ
        if random.random() < self.bandpass_ratio:
            wav = self.apply_bandpass(wav)

        return wav

    def norm_mel(self, mel_spec: torch.Tensor):
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        max_over_mels = log_spec.max(dim=1, keepdim=True)[0]
        max_log_spec = max_over_mels.max(dim=2, keepdim=True)[0]
        log_spec = torch.maximum(log_spec, max_log_spec - 8.0)
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec

    def log_mel(
        self, wav: torch.Tensor, shift: int | None = None, detune: bool = False
    ):
        spec = self.spec_transform(wav)[..., :-1]

        if shift is not None and shift != 0:
            spec = self.shift_spec(spec, shift)
        elif detune is True:
            # Don't detune and spec shift at the same time
            spec = self.detune_spec(spec)

        mel_spec = self.mel_transform(spec)

        # Norm
        log_spec = self.norm_mel(mel_spec)

        return log_spec

    def forward(self, wav: torch.Tensor, shift: int = 0):
        # Noise, and reverb
        wav = self.aug_wav(wav)

        # Spec, detuning & pitch shift
        log_mel = self.log_mel(wav, shift, detune=True)

        # Spec aug
        if random.random() < self.spec_aug_ratio:
            log_mel = self.spec_aug(log_mel)

        return log_mel
