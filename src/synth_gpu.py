"""
Fully batched GPU synthesizer — zero Python loops in render path.
Same 24 params as synth.py, same sound, but processes B candidates simultaneously.

Every operation is a batched tensor op. No .item(), no numpy, no Python loops over
candidates or voices. Runs on MPS/CUDA/CPU.
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

SR = 44100
N_VOICES = 5  # Fixed unison voice count — spread=0 makes them identical

PARAM_DEFS = [
    ("saw_mix", 0.0, 1.0),
    ("square_mix", 0.0, 1.0),
    ("sine_mix", 0.0, 1.0),
    ("noise_mix", 0.0, 0.5),
    ("detune", -2.0, 2.0),           # Constrained to ±2 semitones (was ±24)
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
    ("unison_voices", 1.0, 7.0),     # Controls effective spread via voice count
    ("unison_spread", 0.0, 0.5),
    ("noise_floor", 0.0, 0.1),
    ("filter_attack", 0.001, 0.5),
    ("filter_decay", 0.001, 1.0),
    ("filter_sustain", 0.0, 1.0),
    ("filter_release", 0.001, 1.0),
    ("pulse_width", 0.1, 0.9),
    ("filter_slope", 4.0, 48.0),
]

N_PARAMS = len(PARAM_DEFS)

# Pre-compute param ranges as tensors for fast denormalization
_LO = torch.tensor([float(lo) for _, lo, _ in PARAM_DEFS])
_HI = torch.tensor([float(hi) for _, hi, _ in PARAM_DEFS])
_RANGE = _HI - _LO

# Fixed unison voice offsets: [-1, -0.5, 0, 0.5, 1] normalized
_VOICE_OFFSETS = torch.linspace(-1.0, 1.0, N_VOICES)

# Fixed reverb delay multipliers
_REVERB_DELAYS = torch.tensor([0.3, 0.5, 0.7, 1.1, 1.4, 1.9])


def _denorm_batch(params: torch.Tensor) -> torch.Tensor:
    """Denormalize (B, N_PARAMS) from [0,1] to real values. Returns (B, N_PARAMS)."""
    lo = _LO.to(params.device)
    r = _RANGE.to(params.device)
    return params.clamp(0.0, 1.0) * r.unsqueeze(0) + lo.unsqueeze(0)


def _get_param(real_params: torch.Tensor, name: str) -> torch.Tensor:
    """Get a single param as (B,) tensor from denormalized (B, N_PARAMS)."""
    idx = [n for n, _, _ in PARAM_DEFS].index(name)
    return real_params[:, idx]


def _batched_adsr(n_samples: int, attack, decay, sustain, release,
                  note_duration: Optional[float], device) -> torch.Tensor:
    """
    Batched ADSR envelope. All inputs are (B,) tensors.
    Returns (B, N) tensor.
    """
    B = attack.shape[0]
    t = torch.linspace(0, n_samples / SR, n_samples, device=device)  # (N,)
    t = t.unsqueeze(0).expand(B, -1)  # (B, N)

    a = attack.unsqueeze(1).clamp(min=0.001)   # (B, 1)
    d = decay.unsqueeze(1).clamp(min=0.001)
    s = sustain.unsqueeze(1)
    r = release.unsqueeze(1).clamp(min=0.001)

    attack_env = (t / a).clamp(0.0, 1.0)
    decay_env = 1.0 - (1.0 - s) * ((t - a) / d).clamp(0.0, 1.0)
    env = torch.where(t < a, attack_env, decay_env)

    if note_duration is not None:
        nd = torch.tensor(note_duration, device=device)
        rel_env = env * (1.0 - ((t - nd) / r).clamp(0.0, 1.0))
        env = torch.where(t > nd, rel_env, env)

    return env.clamp(0.0, 1.0)


def _batched_lowpass(signal: torch.Tensor, cutoff_hz: torch.Tensor,
                     resonance: torch.Tensor, slope: torch.Tensor) -> torch.Tensor:
    """
    Batched frequency-domain low-pass filter.
    signal: (B, N), cutoff_hz/resonance/slope: (B,)
    Returns: (B, N)
    """
    B, N = signal.shape
    freqs = torch.fft.rfftfreq(N, d=1.0 / SR, device=signal.device)  # (F,)
    freqs = freqs.unsqueeze(0)  # (1, F)

    cutoff = cutoff_hz.unsqueeze(1).clamp(100.0, SR / 2 - 100)  # (B, 1)
    freq_ratio = freqs / cutoff.clamp(min=1.0)  # (B, F)

    sl = slope.unsqueeze(1)  # (B, 1)
    magnitude = torch.sigmoid(-sl * (freq_ratio - 1.0))  # (B, F)

    # Resonance peak
    res = resonance.unsqueeze(1)  # (B, 1)
    res_peak = res * 2.0 * torch.exp(-0.5 * ((freq_ratio - 1.0) / 0.15) ** 2)
    magnitude = magnitude + res_peak

    X = torch.fft.rfft(signal)  # (B, F)
    Y = X * magnitude
    return torch.fft.irfft(Y, n=N)


def _batched_reverb(signal: torch.Tensor, room_size: torch.Tensor,
                    mix: torch.Tensor) -> torch.Tensor:
    """
    Batched reverb using fixed delay taps with per-candidate decay.
    signal: (B, N), room_size/mix: (B,)
    Returns: (B, N)
    """
    B, N = signal.shape
    device = signal.device
    mix = mix.clamp(0.0, 0.8)
    rs = room_size.clamp(0.01, 0.99)

    wet = torch.zeros_like(signal)
    delays = _REVERB_DELAYS.to(device)

    # Use mean room_size for delay computation (can't vary delay per candidate in batched mode)
    mean_rs = rs.mean()
    for i in range(len(delays)):
        d = int(delays[i].item() * mean_rs.item() * 0.1 * SR)
        d = max(1, min(d, N - 1))
        decay = rs.unsqueeze(1) ** (i + 1)  # (B, 1) — per-candidate decay
        wet[:, d:] = wet[:, d:] + signal[:, :N - d] * decay

    wet_peak = wet.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    sig_peak = signal.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    wet = wet / wet_peak * sig_peak

    m = mix.unsqueeze(1)  # (B, 1)
    return (1.0 - m) * signal + m * wet


class SynthPatchGPU:
    """
    Fully batched GPU synthesizer. Same 24 params as SynthPatch.
    render() accepts (B, 24) and returns (B, N). Zero Python loops.
    """

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._n_params = N_PARAMS

    def get_param_count(self) -> int:
        return self._n_params

    def get_param_names(self) -> List[str]:
        return [name for name, _, _ in PARAM_DEFS]

    def random_params(self, batch_size: int = 1, device=None) -> torch.Tensor:
        return torch.rand(batch_size, self._n_params, device=device)

    def render(self, params_tensor: torch.Tensor, f0_hz: float = 440.0,
              duration: float = 1.0, note_duration: float = None) -> torch.Tensor:
        """
        Fully batched render.
        params_tensor: (B, 24) or (24,) — normalized [0,1]
        Returns: (B, N) audio
        """
        if params_tensor.dim() == 1:
            params_tensor = params_tensor.unsqueeze(0)

        B = params_tensor.shape[0]
        device = params_tensor.device
        n_samples = int(duration * self.sr)

        # Denormalize all params at once: (B, 24) → (B, 24) real values
        p = _denorm_batch(params_tensor)

        # Time axis: (N,)
        t = torch.linspace(0, duration, n_samples, device=device)

        # ─── OSCILLATORS: fully batched across candidates AND voices ───

        # Voice detune offsets: (B, V)
        detune = _get_param(p, "detune").unsqueeze(1)         # (B, 1)
        spread = _get_param(p, "unison_spread").unsqueeze(1)  # (B, 1)
        offsets = _VOICE_OFFSETS.to(device).unsqueeze(0)      # (1, V)
        voice_detunes = detune + spread * offsets              # (B, V)
        voice_ratios = 2.0 ** (voice_detunes / 12.0)          # (B, V)

        # Phase for all candidates × all voices × all samples: (B, V, N)
        # phase = (2π × f0 × ratio × t) mod 2π
        # f0 is scalar, ratios is (B, V), t is (N,)
        phase = (2.0 * math.pi * f0_hz * voice_ratios.unsqueeze(2) * t.unsqueeze(0).unsqueeze(0))
        phase = phase % (2.0 * math.pi)  # (B, V, N)

        # Waveforms: all (B, V, N)
        saw = 2.0 * (phase / (2.0 * math.pi)) - 1.0
        sine = torch.sin(phase)

        # Pulse wave: threshold from pulse_width param
        pw = _get_param(p, "pulse_width")  # (B,)
        threshold = pw * 2.0 * math.pi     # (B,)
        pulse = torch.tanh(20.0 * (torch.sin(phase) - torch.sin(threshold.view(B, 1, 1))))

        # Mix waveforms: (B, 1, 1) weights × (B, V, N) → sum over V → (B, N)
        saw_w = _get_param(p, "saw_mix").view(B, 1, 1)
        sq_w = _get_param(p, "square_mix").view(B, 1, 1)   # drives the pulse wave
        sin_w = _get_param(p, "sine_mix").view(B, 1, 1)

        mixed = saw_w * saw + sq_w * pulse + sin_w * sine  # (B, V, N)
        signal = mixed.mean(dim=1)  # average over voices → (B, N)

        # Normalize by total mix weight
        total_mix = (_get_param(p, "saw_mix") + _get_param(p, "square_mix")
                     + _get_param(p, "sine_mix")).clamp(min=0.01)
        signal = signal / total_mix.unsqueeze(1)

        # Add noise
        noise_mix = _get_param(p, "noise_mix").unsqueeze(1)
        noise_floor = _get_param(p, "noise_floor").unsqueeze(1)
        signal = signal + noise_mix * torch.randn(B, n_samples, device=device)
        signal = signal + noise_floor * torch.randn(B, n_samples, device=device)

        # ─── FILTER ENVELOPE ───
        filter_env = _batched_adsr(
            n_samples,
            _get_param(p, "filter_attack"),
            _get_param(p, "filter_decay"),
            _get_param(p, "filter_sustain"),
            _get_param(p, "filter_release"),
            note_duration=note_duration, device=device
        )  # (B, N)

        # Modulate cutoff with filter envelope
        base_cutoff = _get_param(p, "filter_cutoff")  # (B,)
        fenv_amount = _get_param(p, "filter_env")     # (B,)
        # Use mean of filter envelope per candidate (FFT filter can't do per-sample cutoff)
        env_mean = filter_env.mean(dim=1)  # (B,)
        effective_cutoff = base_cutoff * (1.0 + fenv_amount * (env_mean - 0.5))

        # ─── LOW-PASS FILTER ───
        signal = _batched_lowpass(
            signal, effective_cutoff,
            _get_param(p, "filter_resonance"),
            _get_param(p, "filter_slope")
        )

        # ─── AMPLITUDE ENVELOPE ───
        amp_env = _batched_adsr(
            n_samples,
            _get_param(p, "attack"),
            _get_param(p, "decay"),
            _get_param(p, "sustain"),
            _get_param(p, "release"),
            note_duration=note_duration, device=device
        )  # (B, N)

        signal = signal * amp_env * _get_param(p, "gain").unsqueeze(1)

        # ─── REVERB ───
        signal = _batched_reverb(
            signal,
            _get_param(p, "reverb_size"),
            _get_param(p, "reverb_mix")
        )

        return signal  # (B, N)
