"""MatchingLoss, unreduced: one loss value per candidate instead of a mean.

The server's objective is `losses.MatchingLoss` (1.0 mel-STFT + 0.1 centroid +
0.05 MFCC), evaluated one candidate at a time in a worker process. CMA-ES needs
a value per candidate, and the scalar losses average the batch away, which is
the only reason the population was ever fanned out across processes.

Keeping the batch dimension is not just passing a bigger tensor. Two of the
three terms are batch-coupled if you do that naively, and both silently change
the objective:

  * auraloss computes spectral convergence as one Frobenius ratio over the
    whole tensor, so with a batch it mixes every candidate into every
    candidate's score, whatever `reduction` says. Here it is computed per item.
  * torchaudio's AmplitudeToDB clamps at `max - top_db` where the max is taken
    over the whole tensor, so in a batch the loudest candidate sets the floor
    for the quiet ones. Here the clamp is per item.

Two things are cached rather than recomputed, both exactly equal to the naive
version. The target does not change during a search, so its spectra are
computed once per target. And the A-weighting prefilter is one filter shared by
all three resolutions, so the signal is filtered once rather than three times.
Together they roughly halve the cost of a generation.

The auraloss modules are still used, but only for the pieces that are pure
constants: the prefilter, the analysis window, the mel filterbank.

tests/test_batch_matching_loss.py asserts this agrees with the scalar loss
candidate by candidate, which is the property the optimiser depends on.
"""

import torch
import torchaudio.transforms as T
import auraloss.freq as AF

RESOLUTIONS = ((1024, 256), (2048, 512), (8192, 2048))


class BatchedMatchingLoss:
    """Per-candidate MatchingLoss. Call with (B, N) generated and (N,) target."""

    def __init__(self, device=None, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        self.device = device or 'cpu'

        # Constructed only to borrow their window, mel filterbank and
        # prefilter; the loss arithmetic below is ours.
        self._stft = [
            AF.STFTLoss(
                fft_size=n, hop_size=h, win_length=n,
                scale="mel", n_bins=128, sample_rate=sample_rate,
                perceptual_weighting=True,
            )
            for n, h in RESOLUTIONS
        ]
        self._mfcc = T.MFCC(
            sample_rate=sample_rate, n_mfcc=13,
            melkwargs={"n_fft": 1024, "hop_length": 256, "n_mels": 40},
        )
        self._n_fft_cent, self._hop_cent = 2048, 512
        self._cache_key = None
        self._cache = {}
        self.to(self.device)

    def to(self, device):
        self.device = device
        self._mfcc = self._mfcc.to(device)
        for f in self._stft:
            f.to(device)
            f.window = f.window.to(device)
            f.fb = f.fb.to(device)
            f.prefilter.to(device)
        self._cache_key, self._cache = None, {}
        return self

    # ------------------------------------------------------------ primitives

    def _aweight(self, x):
        """A-weighting. One filter, shared by every resolution."""
        f = self._stft[0].prefilter
        return torch.nn.functional.conv1d(
            x.unsqueeze(1), f.fir.weight.data, padding=f.ntaps // 2).squeeze(1)

    def _mel_mag(self, sig, f):
        s = torch.stft(sig, n_fft=f.fft_size, hop_length=f.hop_size,
                       win_length=f.win_length, window=f.window,
                       return_complex=True)
        m = torch.sqrt(torch.clamp(s.real ** 2 + s.imag ** 2, min=f.eps))
        return torch.matmul(f.fb, m)

    def _centroid(self, x):
        window = torch.hann_window(self._n_fft_cent, device=x.device)
        mag = torch.stft(x, n_fft=self._n_fft_cent, hop_length=self._hop_cent,
                         window=window, return_complex=True).abs()
        freqs = torch.arange(mag.shape[1], dtype=torch.float32, device=x.device)
        mag_sum = mag.sum(dim=2)
        total = mag_sum.sum(dim=1).clamp(min=1e-8)
        return (mag_sum * freqs.unsqueeze(0)).sum(dim=1) / total

    def _mfcc_features(self, x):
        """MFCC with the dB floor applied per item rather than per batch."""
        mel = self._mfcc.MelSpectrogram(x)                      # (B, n_mels, T)
        a2db = self._mfcc.amplitude_to_DB
        db = a2db.multiplier * torch.log10(torch.clamp(mel, min=a2db.amin))
        db = db - a2db.multiplier * a2db.db_multiplier
        if a2db.top_db is not None:
            floor = db.flatten(1).max(dim=1).values - a2db.top_db
            db = torch.maximum(db, floor.view(-1, 1, 1))
        return torch.matmul(db.transpose(-1, -2), self._mfcc.dct_mat).transpose(-1, -2)

    # ---------------------------------------------------------------- target

    def _target_terms(self, tgt):
        """Everything about the target, computed once per search."""
        key = (id(tgt), tgt.shape[-1], str(tgt.device))
        if self._cache_key == key:
            return self._cache
        t = tgt.reshape(1, -1)
        tw = self._aweight(t)
        self._cache = {
            'mel': [self._mel_mag(tw, f) for f in self._stft],
            'centroid': self._centroid(t),
            'mfcc': self._mfcc_features(t),
        }
        self._cache_key = key
        return self._cache

    # ------------------------------------------------------------------ call

    def __call__(self, generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if generated.dim() == 1:
            generated = generated.unsqueeze(0)
        n = min(generated.shape[-1], target.shape[-1])
        gen = generated[..., :n].contiguous()
        tgt = target.reshape(-1)[:n]
        B = gen.shape[0]
        cache = self._target_terms(tgt)

        gw = self._aweight(gen)
        per_res = []
        for f, ym in zip(self._stft, cache['mel']):
            xm = self._mel_mag(gw, f)
            ym_b = ym.expand(B, -1, -1)
            sc = (torch.linalg.matrix_norm(ym_b - xm, ord='fro')
                  / torch.linalg.matrix_norm(ym_b, ord='fro').clamp(min=1e-12))
            log_mag = (torch.log(xm) - torch.log(ym_b)).abs().flatten(1).mean(dim=1)
            per_res.append(f.w_sc * sc + f.w_log_mag * log_mag)
        mel = torch.stack(per_res).mean(dim=0)

        nyquist_bin = self._n_fft_cent // 2 + 1   # the scalar loss divides by this
        cent = (self._centroid(gen) - cache['centroid']).abs() / nyquist_bin

        mfcc = ((self._mfcc_features(gen) - cache['mfcc']) ** 2).flatten(1).mean(dim=1)

        return 1.0 * mel + 0.1 * cent + 0.05 * mfcc
