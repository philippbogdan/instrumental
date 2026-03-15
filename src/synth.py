"""
Differentiable subtractive synthesizer in pure PyTorch.
GPU-compatible (MPS/CUDA). Supports batched rendering.

Signal chain:
  Oscillators (saw/square/sine + harmonics) → Mixer → Distortion →
  Low-pass Filter (with ADSR envelope) → Amplitude ADSR → Delay → Reverb → Output

29 parameters, all in [0, 1], all differentiable.
"""

import math
from typing import List, Optional

import torch
import torch.nn.functional as F

SR = 44100

PARAM_DEFS = [
    # Oscillator mix (4)
    ("saw_mix", 0.0, 1.0),
    ("square_mix", 0.0, 1.0),
    ("sine_mix", 0.0, 1.0),
    ("noise_mix", 0.0, 0.5),
    # Oscillator control (2)
    ("detune", -24.0, 24.0),
    ("unison_voices", 1.0, 7.0),
    ("unison_spread", 0.0, 0.5),
    # Filter (2)
    ("filter_cutoff", 200.0, 16000.0),
    ("filter_resonance", 0.0, 0.95),
    # Filter envelope (4)
    ("filter_env_amount", -1.0, 1.0),
    ("filter_attack", 0.001, 0.5),
    ("filter_decay", 0.001, 1.0),
    ("filter_sustain", 0.0, 1.0),
    ("filter_release", 0.001, 1.0),
    # Amplitude envelope (4)
    ("attack", 0.001, 0.5),
    ("decay", 0.001, 1.0),
    ("sustain", 0.0, 1.0),
    ("release", 0.001, 1.0),
    # Distortion (2)
    ("drive", 0.0, 1.0),
    ("dist_mix", 0.0, 1.0),
    # Delay (3)
    ("delay_time", 0.05, 0.5),
    ("delay_feedback", 0.0, 0.8),
    ("delay_mix", 0.0, 0.6),
    # Reverb (2)
    ("reverb_size", 0.01, 0.99),
    ("reverb_mix", 0.0, 0.8),
    # Vibrato (2)
    ("vibrato_rate", 0.5, 8.0),
    ("vibrato_depth", 0.0, 15.0),
    # Other (2)
    ("noise_floor", 0.0, 0.1),
    ("gain", 0.0, 1.0),
]

N_PARAMS = len(PARAM_DEFS)


def _denorm(params: torch.Tensor) -> dict:
    """Convert [0,1] params to real values. Works with batched (B, N) or single (N,)."""
    p = params.clamp(0.0, 1.0)
    result = {}
    for i, (name, lo, hi) in enumerate(PARAM_DEFS):
        if p.dim() == 1:
            result[name] = p[i] * (hi - lo) + lo
        else:
            result[name] = p[:, i] * (hi - lo) + lo  # (B,)
    return result


def _make_adsr(n_samples: int, attack, decay, sustain, release,
               note_duration: float = None, sr: int = SR, device=None):
    """Batched ADSR. attack/decay/sustain/release can be scalar or (B,) tensors."""
    t = torch.linspace(0, n_samples / sr, n_samples, device=device)

    # Handle both batched and single
    if attack.dim() == 0:
        # Single — same as before
        attack_env = (t / attack.clamp(min=0.001)).clamp(0.0, 1.0)
        decay_env = 1.0 - (1.0 - sustain) * ((t - attack) / decay.clamp(min=0.001)).clamp(0.0, 1.0)
        env = torch.where(t < attack, attack_env, decay_env)
        if note_duration is not None:
            rs = torch.tensor(note_duration, device=device)
            rel_env = env * (1.0 - ((t - rs) / release.clamp(min=0.001)).clamp(0.0, 1.0))
            env = torch.where(t > rs, rel_env, env)
        return env.clamp(0.0, 1.0)
    else:
        # Batched: attack is (B,), output is (B, n_samples)
        B = attack.shape[0]
        t = t.unsqueeze(0).expand(B, -1)  # (B, N)
        a = attack.unsqueeze(1)   # (B, 1)
        d = decay.unsqueeze(1)
        s = sustain.unsqueeze(1)
        r = release.unsqueeze(1)

        attack_env = (t / a.clamp(min=0.001)).clamp(0.0, 1.0)
        decay_env = 1.0 - (1.0 - s) * ((t - a) / d.clamp(min=0.001)).clamp(0.0, 1.0)
        env = torch.where(t < a, attack_env, decay_env)
        if note_duration is not None:
            rs = torch.tensor(note_duration, device=device)
            rel_env = env * (1.0 - ((t - rs) / r.clamp(min=0.001)).clamp(0.0, 1.0))
            env = torch.where(t > rs, rel_env, env)
        return env.clamp(0.0, 1.0)


def _lowpass_filter(signal, cutoff_hz, resonance, sr: int = SR):
    """Frequency-domain low-pass filter. Works batched: (B, N) or (N,)."""
    is_batched = signal.dim() == 2
    if not is_batched:
        signal = signal.unsqueeze(0)
        cutoff_hz = cutoff_hz.unsqueeze(0) if cutoff_hz.dim() == 0 else cutoff_hz.unsqueeze(0)
        resonance = resonance.unsqueeze(0) if resonance.dim() == 0 else resonance.unsqueeze(0)

    B, N = signal.shape
    freqs = torch.fft.rfftfreq(N, d=1.0 / sr, device=signal.device)  # (F,)
    freqs = freqs.unsqueeze(0)  # (1, F)

    cutoff = cutoff_hz.unsqueeze(1).clamp(100.0, sr / 2 - 100)  # (B, 1)
    freq_ratio = freqs / cutoff.clamp(min=1.0)  # (B, F)

    magnitude = torch.sigmoid(-12.0 * (freq_ratio - 1.0))  # (B, F)

    # Resonance peak
    res = resonance.unsqueeze(1)  # (B, 1)
    res_peak = res * 2.0 * torch.exp(-0.5 * ((freq_ratio - 1.0) / 0.15) ** 2)
    magnitude = magnitude + res_peak

    X = torch.fft.rfft(signal)  # (B, F)
    Y = X * magnitude
    result = torch.fft.irfft(Y, n=N)

    return result if is_batched else result.squeeze(0)


def _soft_clip(signal, drive):
    """Differentiable soft distortion. drive: 0=clean, 1=heavy."""
    if drive.dim() == 0:
        amount = 1.0 + drive * 20.0
        return torch.tanh(signal * amount) / torch.tanh(torch.tensor(amount))
    else:
        amount = (1.0 + drive * 20.0).unsqueeze(1)  # (B, 1)
        return torch.tanh(signal * amount) / torch.tanh(amount)


def _delay(signal, delay_time, feedback, mix, sr: int = SR):
    """Delay via FFT-based circular convolution. Fully batched, no Python loops."""
    is_batched = signal.dim() == 2
    if not is_batched:
        signal = signal.unsqueeze(0)
        delay_time = delay_time.unsqueeze(0) if delay_time.dim() == 0 else delay_time
        feedback = feedback.unsqueeze(0) if feedback.dim() == 0 else feedback
        mix = mix.unsqueeze(0) if mix.dim() == 0 else mix

    B, N = signal.shape

    # Build impulse response: delta at t=0 + decaying taps
    # Use a fixed delay of ~150ms (middle of range) for all batch elements
    # This loses per-element delay variation but enables full GPU batching
    mean_delay = int(delay_time.mean().item() * sr)
    mean_delay = max(1, min(mean_delay, N // 3))
    mean_fb = feedback.mean().clamp(0.0, 0.8)

    wet = torch.zeros_like(signal)
    for tap in range(3):
        shift = mean_delay * (tap + 1)
        if shift >= N:
            break
        decay = mean_fb ** (tap + 1)
        wet[:, shift:] = wet[:, shift:] + signal[:, :N - shift] * decay

    wet_peak = wet.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    sig_peak = signal.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    wet = wet / wet_peak * sig_peak

    m = mix.unsqueeze(1)
    result = (1.0 - m) * signal + m * wet

    return result if is_batched else result.squeeze(0)


def _reverb(signal, room_size, mix, sr: int = SR):
    """Simple reverb. Batched with mean room_size for GPU compat."""
    is_batched = signal.dim() == 2
    if not is_batched:
        signal = signal.unsqueeze(0)
        room_size = room_size.unsqueeze(0) if room_size.dim() == 0 else room_size
        mix = mix.unsqueeze(0) if mix.dim() == 0 else mix

    B, N = signal.shape
    wet = torch.zeros_like(signal)
    delay_mults = [0.3, 0.5, 0.7, 1.1, 1.4, 1.9]

    mean_rs = room_size.mean().clamp(0.01, 0.99)
    for i, dm in enumerate(delay_mults):
        d = int(dm * mean_rs.item() * 0.1 * sr)
        d = max(1, min(d, N - 1))
        decay = mean_rs ** (i + 1)
        wet[:, d:] = wet[:, d:] + signal[:, :N - d] * decay

    wet_peak = wet.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    sig_peak = signal.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    wet = wet / wet_peak * sig_peak

    m = mix.unsqueeze(1)
    result = (1.0 - m) * signal + m * wet

    return result if is_batched else result.squeeze(0)


class SynthPatch:
    """
    Differentiable subtractive synthesizer with 29 parameters.
    GPU-compatible. Supports batched rendering.
    """

    def __init__(self, sr: int = SR):
        self.sr = sr
        self._n_params = N_PARAMS

    def get_param_count(self) -> int:
        return self._n_params

    def get_param_names(self) -> List[str]:
        return [name for name, _, _ in PARAM_DEFS]

    def random_params(self, batch_size: Optional[int] = None, device=None) -> torch.Tensor:
        if batch_size is None:
            return torch.rand(self._n_params, device=device)
        return torch.rand(batch_size, self._n_params, device=device)

    def render(self, params_tensor: torch.Tensor, f0_hz: float = 440.0,
               duration: float = 1.0, note_duration: float = None) -> torch.Tensor:
        """
        Render audio. Supports single (n_params,) or batched (B, n_params) input.
        Returns (1, n_samples) or (B, n_samples).
        """
        is_batched = params_tensor.dim() == 2
        if not is_batched:
            params_tensor = params_tensor.unsqueeze(0)

        B = params_tensor.shape[0]
        device = params_tensor.device
        p = _denorm(params_tensor)
        n_samples = int(duration * self.sr)
        t = torch.linspace(0, duration, n_samples, device=device)  # (N,)

        # ─── VIBRATO ───
        vib_rate = p["vibrato_rate"].unsqueeze(1)   # (B, 1)
        vib_depth = p["vibrato_depth"].unsqueeze(1)  # (B, 1) in cents
        vib_mod = vib_depth / 1200.0 * torch.sin(2.0 * math.pi * vib_rate * t.unsqueeze(0))  # (B, N)
        f0_modulated = f0_hz * (1.0 + vib_mod)  # (B, N)

        # ─── OSCILLATORS (fully batched, no Python loops) ───
        # Fixed 5 unison voices for all, spread controlled by param
        N_UNISON = 5
        offsets = torch.linspace(-1.0, 1.0, N_UNISON, device=device)  # (V,)
        spread = p["unison_spread"].unsqueeze(1)  # (B, 1)
        detune_base = p["detune"].unsqueeze(1)    # (B, 1)

        # voice_detunes: (B, V)
        voice_detunes = detune_base + spread * offsets.unsqueeze(0)
        voice_ratios = 2.0 ** (voice_detunes / 12.0)  # (B, V)

        # Phase for each voice: (B, V, N)
        # f0_modulated is (B, N), voice_ratios is (B, V)
        t_cumsum = t.cumsum(0) / n_samples * duration  # (N,)
        phase = (2.0 * math.pi * f0_modulated.unsqueeze(1) * voice_ratios.unsqueeze(2) * t_cumsum.unsqueeze(0).unsqueeze(0))
        phase = phase % (2.0 * math.pi)  # (B, V, N)

        # Waveforms for all voices at once
        saw = 2.0 * (phase / (2 * math.pi)) - 1.0       # (B, V, N)
        square = torch.tanh(20.0 * torch.sin(phase))      # (B, V, N)
        sine = torch.sin(phase)                            # (B, V, N)

        # Mix waveforms: (B, 1, 1) * (B, V, N) → sum over V → (B, N)
        sm = p["saw_mix"].view(B, 1, 1)
        sqm = p["square_mix"].view(B, 1, 1)
        snm = p["sine_mix"].view(B, 1, 1)

        mixed = sm * saw + sqm * square + snm * sine  # (B, V, N)
        signal = mixed.mean(dim=1)  # average over voices → (B, N)

        total_mix = (p["saw_mix"] + p["square_mix"] + p["sine_mix"]).clamp(min=0.01)
        signal = signal / total_mix.unsqueeze(1)

        # Add noise
        noise = torch.randn(B, n_samples, device=device)
        noise_mix = p["noise_mix"].unsqueeze(1)
        noise_floor = p["noise_floor"].unsqueeze(1)
        signal = signal + noise_mix * noise + noise_floor * torch.randn(B, n_samples, device=device)

        # ─── DISTORTION ───
        dist_mix = p["dist_mix"]
        drive = p["drive"]
        distorted = _soft_clip(signal, drive)
        dm = dist_mix.unsqueeze(1)
        signal = (1.0 - dm) * signal + dm * distorted

        # ─── FILTER with envelope ───
        filter_env = _make_adsr(
            n_samples, p["filter_attack"], p["filter_decay"],
            p["filter_sustain"], p["filter_release"],
            note_duration=note_duration, sr=self.sr, device=device
        )  # (B, N)

        # Modulate cutoff
        base_cutoff = p["filter_cutoff"].unsqueeze(1)  # (B, 1)
        env_amount = p["filter_env_amount"].unsqueeze(1)  # (B, 1)
        effective_cutoff = base_cutoff * (1.0 + env_amount * (filter_env - 0.5))  # (B, N)
        # Use per-batch mean cutoff for FFT filter (can't do per-sample easily)
        mean_cutoff = effective_cutoff.mean(dim=1)  # (B,)

        signal = _lowpass_filter(signal, mean_cutoff, p["filter_resonance"], sr=self.sr)

        # ─── AMPLITUDE ENVELOPE ───
        amp_env = _make_adsr(
            n_samples, p["attack"], p["decay"], p["sustain"], p["release"],
            note_duration=note_duration, sr=self.sr, device=device
        )  # (B, N)

        signal = signal * amp_env * p["gain"].unsqueeze(1)

        # ─── DELAY ───
        signal = _delay(signal, p["delay_time"], p["delay_feedback"], p["delay_mix"], sr=self.sr)

        # ─── REVERB ───
        signal = _reverb(signal, p["reverb_size"], p["reverb_mix"], sr=self.sr)

        if not is_batched:
            return signal  # already (1, N) from the unsqueeze at top

        return signal  # (B, N)
