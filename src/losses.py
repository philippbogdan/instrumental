"""Loss functions for inverse synthesis optimization."""

import torch
import torch.nn as nn
import torchaudio.transforms as T
import auraloss.freq as AF


def _ensure_3d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape (batch, channels, samples)."""
    if x.dim() == 2:
        return x.unsqueeze(1)
    return x


class MultiResSTFTLoss:
    """Multi-resolution STFT loss using auraloss."""

    def __init__(self):
        self._loss = AF.MultiResolutionSTFTLoss(
            fft_sizes=[512, 1024, 2048],
            hop_sizes=[128, 256, 512],
            win_lengths=[512, 1024, 2048],
        )

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gen = _ensure_3d(generated)
        tgt = _ensure_3d(target)
        return self._loss(gen, tgt)


class MFCCLoss:
    """MSE loss on MFCC features. Differentiable via torchaudio."""

    def __init__(self, sample_rate: int = 44100, n_mfcc: int = 13):
        self._mfcc = T.MFCC(
            sample_rate=sample_rate,
            n_mfcc=n_mfcc,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40},
        )

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # MFCC expects (..., time) — squeeze channel dim if 3d
        gen = _ensure_3d(generated).squeeze(1)  # (batch, samples)
        tgt = _ensure_3d(target).squeeze(1)
        mfcc_gen = self._mfcc(gen)
        mfcc_tgt = self._mfcc(tgt)
        return torch.nn.functional.mse_loss(mfcc_gen, mfcc_tgt)


class WaveformL1Loss:
    """L1 loss on raw waveform samples."""

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(generated - target))


class SpectralFeatureLoss:
    """Loss on spectral centroid, spectral flatness, and RMS envelope.

    All features computed via differentiable PyTorch ops.
    """

    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        self.n_fft = n_fft
        self.hop_length = hop_length
        # Frequency bin indices for centroid computation
        self._freqs = None

    def _get_freqs(self, n_bins: int, device: torch.device) -> torch.Tensor:
        if self._freqs is None or self._freqs.shape[0] != n_bins or self._freqs.device != device:
            self._freqs = torch.arange(n_bins, dtype=torch.float32, device=device)
        return self._freqs

    def _stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude. x: (batch, samples)"""
        xr = x.reshape(-1, x.shape[-1])
        window = torch.hann_window(self.n_fft, device=xr.device)
        stft = torch.stft(
            xr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=window,
            return_complex=True,
        )  # (batch, freq_bins, time_frames)
        return stft.abs()  # differentiable magnitude

    def _spectral_centroid(self, mag: torch.Tensor) -> torch.Tensor:
        """Weighted mean frequency bin. mag: (batch, freq, time) -> (batch,)"""
        freqs = self._get_freqs(mag.shape[1], mag.device)
        power = mag ** 2  # (batch, freq, time)
        total_power = power.sum(dim=1, keepdim=True).sum(dim=2, keepdim=True).clamp(min=1e-8)
        centroid = (power * freqs.view(1, -1, 1)).sum(dim=(1, 2)) / total_power.squeeze()
        return centroid  # (batch,)

    def _spectral_flatness(self, mag: torch.Tensor) -> torch.Tensor:
        """Geometric mean / arithmetic mean of magnitude spectrum. mag: (batch, freq, time) -> (batch,)"""
        eps = 1e-8
        log_mean = torch.log(mag.clamp(min=eps)).mean(dim=(1, 2))
        geo_mean = torch.exp(log_mean)
        arith_mean = mag.mean(dim=(1, 2)).clamp(min=eps)
        return geo_mean / arith_mean  # (batch,)

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        """RMS of waveform. x: (batch, samples) -> (batch,)"""
        return torch.sqrt((x ** 2).mean(dim=-1).clamp(min=1e-8))

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gen_2d = _ensure_3d(generated).squeeze(1)  # (batch, samples)
        tgt_2d = _ensure_3d(target).squeeze(1)

        mag_gen = self._stft_mag(gen_2d)
        mag_tgt = self._stft_mag(tgt_2d)

        centroid_loss = torch.mean(torch.abs(
            self._spectral_centroid(mag_gen) - self._spectral_centroid(mag_tgt)
        ))
        flatness_loss = torch.mean(torch.abs(
            self._spectral_flatness(mag_gen) - self._spectral_flatness(mag_tgt)
        ))
        rms_loss = torch.mean(torch.abs(self._rms(gen_2d) - self._rms(tgt_2d)))

        # Normalize each component to similar scale, then weight equally
        return centroid_loss / (mag_gen.shape[1] + 1e-8) + flatness_loss + rms_loss


class HybridLoss:
    """Weighted combination: 0.7 * STFT + 0.3 * MFCC."""

    def __init__(self):
        self._stft = MultiResSTFTLoss()
        self._mfcc = MFCCLoss()

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.7 * self._stft(generated, target) + 0.3 * self._mfcc(generated, target)


def get_loss(name: str):
    """Factory function returning a loss instance by name.

    Args:
        name: One of "stft", "mfcc", "l1", "spectral", "hybrid"

    Returns:
        Loss instance with __call__(generated, target) -> scalar tensor
    """
    losses = {
        "stft": MultiResSTFTLoss,
        "mfcc": MFCCLoss,
        "l1": WaveformL1Loss,
        "spectral": SpectralFeatureLoss,
        "hybrid": HybridLoss,
    }
    if name not in losses:
        raise ValueError(f"Unknown loss '{name}'. Valid options: {list(losses.keys())}")
    return losses[name]()
