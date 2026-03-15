"""
Batched multi-resolution STFT loss — runs entirely on GPU.
Replaces the sequential MatchingLoss for CMA-ES evaluation.
"""

import torch
import math


class BatchedMultiResSTFTLoss:
    """
    Multi-resolution STFT loss computed for a batch of candidates simultaneously.
    All operations are batched tensor ops — no Python loops over candidates.

    Computes spectral convergence + log magnitude at 3 FFT sizes.
    """

    def __init__(self, fft_sizes=(512, 1024, 2048), hop_sizes=(128, 256, 512), device=None):
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.device = device or torch.device("cpu")
        self.windows = {
            n: torch.hann_window(n, device=self.device)
            for n in fft_sizes
        }

    def to(self, device):
        self.device = device
        self.windows = {n: w.to(device) for n, w in self.windows.items()}
        return self

    def __call__(self, generated_batch, target):
        """
        Args:
            generated_batch: (B, N) — batch of generated audio
            target: (N,) or (1, N) — single target to compare against

        Returns:
            (B,) — one loss value per candidate
        """
        if target.dim() == 2:
            target = target.squeeze(0)

        B, N = generated_batch.shape
        target_expanded = target.unsqueeze(0).expand(B, -1)  # (B, N)

        total_loss = torch.zeros(B, device=generated_batch.device)

        for n_fft, hop in zip(self.fft_sizes, self.hop_sizes):
            window = self.windows[n_fft]

            # Batched STFT
            stft_gen = torch.stft(generated_batch, n_fft=n_fft, hop_length=hop,
                                   window=window, return_complex=True)
            stft_tgt = torch.stft(target_expanded, n_fft=n_fft, hop_length=hop,
                                   window=window, return_complex=True)

            mag_gen = stft_gen.abs()  # (B, F, T)
            mag_tgt = stft_tgt.abs()  # (B, F, T)

            # Spectral convergence: ||mag_tgt - mag_gen|| / ||mag_tgt||
            sc = (torch.norm(mag_tgt - mag_gen, dim=(1, 2))
                  / (torch.norm(mag_tgt, dim=(1, 2)) + 1e-8))

            # Log magnitude L1
            log_gen = torch.log(mag_gen.clamp(min=1e-8))
            log_tgt = torch.log(mag_tgt.clamp(min=1e-8))
            lm = torch.mean(torch.abs(log_gen - log_tgt), dim=(1, 2))

            total_loss = total_loss + sc + lm

        return total_loss  # (B,)


class BatchedCentroidLoss:
    """Batched spectral centroid loss on GPU."""

    def __init__(self, n_fft=2048, hop_length=512, sr=44100):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr

    def __call__(self, generated_batch, target):
        if target.dim() == 2:
            target = target.squeeze(0)

        B, N = generated_batch.shape
        target_expanded = target.unsqueeze(0).expand(B, -1)
        device = generated_batch.device

        window = torch.hann_window(self.n_fft, device=device)
        freqs = torch.fft.rfftfreq(self.n_fft, d=1.0/self.sr, device=device)  # (F,)

        stft_gen = torch.stft(generated_batch, n_fft=self.n_fft, hop_length=self.hop_length,
                               window=window, return_complex=True).abs()  # (B, F, T)
        stft_tgt = torch.stft(target_expanded, n_fft=self.n_fft, hop_length=self.hop_length,
                               window=window, return_complex=True).abs()

        # Centroid: weighted mean frequency
        f = freqs.view(1, -1, 1)  # (1, F, 1)
        cent_gen = (f * stft_gen).sum(dim=1) / (stft_gen.sum(dim=1) + 1e-8)  # (B, T)
        cent_tgt = (f * stft_tgt).sum(dim=1) / (stft_tgt.sum(dim=1) + 1e-8)

        # Mean absolute centroid difference, normalized
        return torch.mean(torch.abs(cent_gen - cent_tgt), dim=1) / (self.sr / 2)  # (B,)


class BatchedMFCCLoss:
    """Batched MFCC loss on GPU."""

    def __init__(self, sr=44100, n_mfcc=13, n_fft=1024, hop_length=256, n_mels=40):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self._mfcc = None
        self._device = None

    def _get_transform(self, device):
        if self._mfcc is None or self._device != device:
            import torchaudio.transforms as T
            self._mfcc = T.MFCC(
                sample_rate=self.sr, n_mfcc=self.n_mfcc,
                melkwargs={"n_fft": self.n_fft, "hop_length": self.hop_length, "n_mels": self.n_mels}
            ).to(device)
            self._device = device
        return self._mfcc

    def __call__(self, generated_batch, target):
        if target.dim() == 2:
            target = target.squeeze(0)

        B, N = generated_batch.shape
        device = generated_batch.device
        mfcc_transform = self._get_transform(device)

        # torchaudio MFCC accepts batched input (B, N) → (B, n_mfcc, T)
        mfcc_gen = mfcc_transform(generated_batch)
        target_expanded = target.unsqueeze(0).expand(B, -1)
        mfcc_tgt = mfcc_transform(target_expanded)

        # Per-candidate MSE
        return torch.mean((mfcc_gen - mfcc_tgt) ** 2, dim=(1, 2))  # (B,)


class BatchedMatchingLoss:
    """
    Same MatchingLoss formula as src/losses.py but runs each candidate on GPU.
    Uses auraloss MelSTFT (with perceptual weighting) + centroid + MFCC.
    Loop over candidates but each eval on GPU = faster than CPU sequential.
    """

    def __init__(self, device=None):
        import auraloss.freq as AF
        import torchaudio.transforms as T

        self.device = device or torch.device("cpu")

        # Same config as the original MatchingLoss in src/losses.py
        self._mel_stft = AF.MultiResolutionSTFTLoss(
            fft_sizes=[1024, 2048, 8192],
            hop_sizes=[256, 512, 2048],
            win_lengths=[1024, 2048, 8192],
            scale="mel", n_bins=128, sample_rate=44100,
            perceptual_weighting=True,
        ).to(self.device)

        self._mfcc = T.MFCC(
            sample_rate=44100, n_mfcc=13,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40},
        ).to(self.device)

        self.centroid = BatchedCentroidLoss()

    def to(self, device):
        self.device = device
        self._mel_stft = self._mel_stft.to(device)
        self._mfcc = self._mfcc.to(device)
        return self

    def __call__(self, generated_batch, target):
        """
        Args:
            generated_batch: (B, N)
            target: (N,) or (1, N)
        Returns:
            (B,) — loss per candidate
        """
        if target.dim() == 2:
            target = target.squeeze(0)

        B, N = generated_batch.shape
        ml = min(N, target.shape[0])

        tgt_3d = target[:ml].view(1, 1, ml)  # (1, 1, ml) for auraloss
        losses = torch.zeros(B, device=self.device)

        # Loop over candidates — each on GPU
        for i in range(B):
            gen_3d = generated_batch[i, :ml].view(1, 1, ml)

            # 1.0 * mel STFT
            stft_loss = self._mel_stft(gen_3d, tgt_3d)

            # 0.05 * MFCC
            mfcc_gen = self._mfcc(gen_3d.squeeze(0))
            mfcc_tgt = self._mfcc(tgt_3d.squeeze(0))
            mfcc_loss = torch.nn.functional.mse_loss(mfcc_gen, mfcc_tgt)

            losses[i] = stft_loss + 0.05 * mfcc_loss

        # Batched centroid (already batched)
        cent_loss = self.centroid(generated_batch[:, :ml],
                                  target[:ml])  # (B,)
        losses = losses + 0.1 * cent_loss

        return losses
