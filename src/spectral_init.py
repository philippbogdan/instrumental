"""
Spectral analysis-based parameter initialization for the synthesizer.

Analyzes a target audio signal and returns a 15-dimensional parameter vector
(normalized [0,1]) that approximates the target's spectral characteristics.
"""

import numpy as np
import torch
import librosa


def spectral_init(target_audio_np: np.ndarray, sr: int = 44100) -> torch.Tensor:
    """
    Analyze target audio and return initial synth parameters.

    Parameters
    ----------
    target_audio_np : np.ndarray
        1D mono audio signal.
    sr : int
        Sample rate (default 44100).

    Returns
    -------
    torch.Tensor
        Shape (15,) with values in [0, 1].
    """
    y = target_audio_np.astype(np.float64)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # --- 1. Spectral rolloff -> filter_cutoff ---
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    rolloff_hz = float(np.median(rolloff))
    norm_cutoff = np.clip((rolloff_hz - 200.0) / (16000.0 - 200.0), 0.0, 1.0)

    # --- 2. Harmonic analysis -> waveform type ---
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'), sr=sr
    )
    # Use median of voiced frames as fundamental
    voiced_f0 = f0[voiced_flag]
    if len(voiced_f0) > 0:
        f0_est = float(np.median(voiced_f0))
    else:
        f0_est = 440.0

    # Measure even vs odd harmonic energy (exclude fundamental H1)
    S = np.abs(librosa.stft(y))
    freqs = librosa.fft_frequencies(sr=sr)

    harmonic_energies = {}
    for h in range(1, 13):  # harmonics 1-12
        target_freq = f0_est * h
        idx = np.argmin(np.abs(freqs - target_freq))
        harmonic_energies[h] = float(np.mean(S[idx, :] ** 2))

    total_all = sum(harmonic_energies.values())
    fund_energy = harmonic_energies.get(1, 0.0)
    fundamental_dominance = fund_energy / total_all if total_all > 0 else 0.0

    # For even/odd ratio, exclude fundamental (H1) to avoid skewing
    even_energy = sum(harmonic_energies[h] for h in range(2, 13) if h % 2 == 0)
    odd_energy = sum(harmonic_energies[h] for h in range(3, 13) if h % 2 == 1)
    overtone_total = even_energy + odd_energy

    if overtone_total > 0:
        even_ratio = even_energy / overtone_total
    else:
        even_ratio = 0.5

    # Saw: all harmonics present (even_ratio ~0.5 among overtones)
    # Square: only odd harmonics (even_ratio ~0)
    # Sine: fundamental dominates, few overtones

    if fundamental_dominance > 0.85:
        saw_mix = 0.1
        square_mix = 0.1
        sine_mix = 0.9
    elif even_ratio > 0.15:
        # Even harmonics present -> saw-like
        saw_mix = 0.7 + 0.2 * min(1.0, even_ratio / 0.5)
        square_mix = 0.1
        sine_mix = 0.1
    else:
        # Mostly odd harmonics -> square
        saw_mix = 0.1
        square_mix = 0.7 + 0.2 * min(1.0, (0.15 - even_ratio) / 0.15)
        sine_mix = 0.1

    # --- 3. Spectral flatness -> noise_mix ---
    flatness = librosa.feature.spectral_flatness(y=y)
    flatness_val = float(np.median(flatness))
    # noise_mix param range is [0, 0.5], normalized to [0, 1]
    noise_mix = np.clip(flatness_val * 2.0, 0.0, 1.0)

    # --- 4. RMS -> gain ---
    rms = librosa.feature.rms(y=y)
    rms_val = float(np.mean(rms))
    gain = np.clip(rms_val / 0.1, 0.0, 1.0)

    # --- 5. Onset steepness -> attack ---
    rms_frames = rms[0]
    peak_idx = np.argmax(rms_frames)
    if peak_idx > 0:
        # Time to reach peak RMS
        attack_time = float(peak_idx) * 512.0 / sr  # hop_length default = 512
        # Normalize: attack param range [0.001, 0.5]
        norm_attack = np.clip((attack_time - 0.001) / (0.5 - 0.001), 0.0, 1.0)
    else:
        norm_attack = 0.05  # very fast attack

    # --- 6. Defaults for remaining params ---
    detune = 0.5           # center = 0 semitones
    resonance = 0.1
    decay = 0.3
    sustain = 0.5
    release = 0.3
    filter_env = 0.5       # center = 0
    reverb_size = 0.3
    reverb_mix = 0.2

    # Build the 15-dim vector in PARAM_DEFS order
    params = torch.tensor([
        saw_mix,            # 0: saw_mix
        square_mix,         # 1: square_mix
        sine_mix,           # 2: sine_mix
        noise_mix,          # 3: noise_mix
        detune,             # 4: detune
        norm_cutoff,        # 5: filter_cutoff
        resonance,          # 6: filter_resonance
        norm_attack,        # 7: attack
        decay,              # 8: decay
        sustain,            # 9: sustain
        release,            # 10: release
        gain,               # 11: gain
        filter_env,         # 12: filter_env
        reverb_size,        # 13: reverb_size
        reverb_mix,         # 14: reverb_mix
    ], dtype=torch.float32)

    return params.clamp(0.0, 1.0)
