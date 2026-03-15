"""
Differentiable subtractive synthesizer in pure PyTorch.

Signal chain: Oscillators → Mixer → Low-pass Filter → Amplifier (ADSR envelope) → Reverb → Output

15 parameters, all in [0, 1], all differentiable.
"""

import math
from typing import List

import torch
import torch.nn.functional as F

SR = 44100  # sample rate

PARAM_DEFS = [
    ("saw_mix", 0.0, 1.0),
    ("square_mix", 0.0, 1.0),
    ("sine_mix", 0.0, 1.0),
    ("noise_mix", 0.0, 0.5),
    ("detune", -24.0, 24.0),
    ("filter_cutoff", 200.0, 16000.0),
    ("filter_resonance", 0.0, 0.95),
    ("attack", 0.001, 0.5),
    ("decay", 0.001, 1.0),
    ("sustain", 0.0, 1.0),
    ("release", 0.001, 1.0),
    ("gain", 0.0, 1.0),
    ("filter_env", -1.0, 1.0),
    ("reverb_size", 0.01, 0.99),
    ("reverb_mix", 0.0, 0.8),
]


def _denorm(params: torch.Tensor) -> dict:
    p = params.clamp(0.0, 1.0)
    result = {}
    for i, (name, lo, hi) in enumerate(PARAM_DEFS):
        result[name] = p[i] * (hi - lo) + lo
    return result


def _saw(phase: torch.Tensor) -> torch.Tensor:
    return 2.0 * (phase / (2 * math.pi)) - 1.0


def _square(phase: torch.Tensor) -> torch.Tensor:
    # Differentiable soft square using tanh (avoids hard discontinuity)
    return torch.tanh(20.0 * torch.sin(phase))


def _lowpass_filter(signal: torch.Tensor, cutoff_hz: torch.Tensor,
                    resonance: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """
    Differentiable low-pass filter via frequency-domain multiplication.

    Uses a smooth sigmoid-shaped frequency response rather than biquad coefficients.
    This gives clean gradients w.r.t. cutoff and resonance.
    """
    N = signal.shape[-1]
    freqs = torch.fft.rfftfreq(N, d=1.0 / sr, device=signal.device)

    # Smooth low-pass: sigmoid rolloff centered at cutoff
    # Steepness controlled by a fixed slope (24 dB/oct equivalent)
    # The key: this is a smooth, differentiable function of cutoff_hz
    cutoff_hz = cutoff_hz.clamp(100.0, sr / 2 - 100)

    # Normalized frequency ratio
    freq_ratio = freqs / cutoff_hz.clamp(min=1.0)

    # Sigmoid rolloff: smooth transition from 1 to 0 around cutoff
    # slope controls steepness (higher = sharper cutoff)
    slope = 12.0  # ~24 dB/oct
    magnitude = torch.sigmoid(-slope * (freq_ratio - 1.0))

    # Resonance: boost near cutoff frequency
    # Gaussian peak centered at cutoff
    if resonance > 0.01:
        res_width = 0.15  # width of resonance peak
        res_peak = resonance * 2.0 * torch.exp(-0.5 * ((freq_ratio - 1.0) / res_width) ** 2)
        magnitude = magnitude + res_peak

    # Apply in frequency domain
    X = torch.fft.rfft(signal)
    Y = X * magnitude
    return torch.fft.irfft(Y, n=N)


def _make_adsr(n_samples: int, attack: torch.Tensor, decay: torch.Tensor,
               sustain: torch.Tensor, release: torch.Tensor,
               note_duration: float = None, sr: int = SR) -> torch.Tensor:
    t = torch.linspace(0, n_samples / sr, n_samples, device=attack.device)

    attack_env = (t / attack.clamp(min=0.001)).clamp(0.0, 1.0)
    decay_start = attack
    decay_env = 1.0 - (1.0 - sustain) * ((t - decay_start) / decay.clamp(min=0.001)).clamp(0.0, 1.0)
    env = torch.where(t < attack, attack_env, decay_env)

    if note_duration is not None:
        release_start = torch.tensor(note_duration, device=attack.device)
        release_env = env * (1.0 - ((t - release_start) / release.clamp(min=0.001)).clamp(0.0, 1.0))
        env = torch.where(t > release_start, release_env, env)

    return env.clamp(0.0, 1.0)


def _simple_reverb(signal: torch.Tensor, room_size: torch.Tensor,
                    mix: torch.Tensor, sr: int = SR) -> torch.Tensor:
    mix = mix.clamp(0.0, 0.8)
    room_size = room_size.clamp(0.01, 0.99)
    N = signal.shape[-1]

    delays = [int(d * room_size.item() * 0.1 * sr) for d in [0.3, 0.5, 0.7, 1.1, 1.4, 1.9]]
    delays = [max(1, min(d, N - 1)) for d in delays]

    wet = torch.zeros_like(signal)
    for i, delay in enumerate(delays):
        decay = room_size ** (i + 1)
        padded = F.pad(signal, (delay, 0))[:N]
        wet = wet + decay * padded

    wet_peak = wet.abs().max().clamp(min=1e-8)
    wet = wet / wet_peak * signal.abs().max().clamp(min=1e-8)

    return (1.0 - mix) * signal + mix * wet


class SynthPatch:
    """
    Differentiable subtractive synthesizer.

    15 parameters in [0, 1]. Fully differentiable via PyTorch autograd.
    """

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._n_params = len(PARAM_DEFS)

    def get_param_count(self) -> int:
        return self._n_params

    def get_param_names(self) -> List[str]:
        return [name for name, _, _ in PARAM_DEFS]

    def random_params(self) -> torch.Tensor:
        return torch.rand(self._n_params)

    def render(self, params_tensor: torch.Tensor, f0_hz: float = 440.0,
              duration: float = 1.0, note_duration: float = None) -> torch.Tensor:
        p = _denorm(params_tensor)
        n_samples = int(duration * self.sr)
        t = torch.linspace(0, duration, n_samples, device=params_tensor.device)

        # Oscillators
        phase = (2.0 * math.pi * f0_hz * t) % (2.0 * math.pi)
        detune_ratio = 2.0 ** (p["detune"] / 12.0)
        phase2 = (2.0 * math.pi * f0_hz * detune_ratio * t) % (2.0 * math.pi)

        saw1 = _saw(phase)
        saw2 = _saw(phase2)
        sq1 = _square(phase)
        sq2 = _square(phase2)
        sine1 = torch.sin(phase)
        sine2 = torch.sin(phase2)
        noise = torch.randn(n_samples, device=params_tensor.device)

        signal = (
            p["saw_mix"] * (saw1 + saw2) / 2.0
            + p["square_mix"] * (sq1 + sq2) / 2.0
            + p["sine_mix"] * (sine1 + sine2) / 2.0
            + p["noise_mix"] * noise
        )
        total_mix = (p["saw_mix"] + p["square_mix"] + p["sine_mix"] + p["noise_mix"]).clamp(min=0.01)
        signal = signal / total_mix

        # ADSR envelope
        env = _make_adsr(n_samples, p["attack"], p["decay"], p["sustain"],
                         p["release"], note_duration=note_duration, sr=self.sr)

        # Filter with envelope modulation
        effective_cutoff = p["filter_cutoff"] * (1.0 + p["filter_env"] * (env.mean() - 0.5))
        signal = _lowpass_filter(signal, effective_cutoff, p["filter_resonance"], sr=self.sr)

        # Apply envelope + gain
        signal = signal * env * p["gain"]

        # Reverb
        signal = _simple_reverb(signal, p["reverb_size"], p["reverb_mix"], sr=self.sr)

        return signal.unsqueeze(0)
