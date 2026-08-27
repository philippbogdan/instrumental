"""
Two-stage sound matching:
  Stage 1: Extract exact harmonics from target (no optimization)
  Stage 2: CMA-ES the envelope/filter/reverb/gain to match (8-core)
"""

import torch, numpy as np, soundfile as sf, subprocess, time, librosa, cma, math
from multiprocessing import Pool
from src.synth import _make_adsr, _simple_reverb, _lowpass_filter

SR = 44100
N_HARMONICS = 12

# ─── STAGE 1: Extract harmonics from each target note ───
print("=== Stage 1: Extracting harmonics ===", flush=True)

TARGET_NOTES = []
HARMONIC_PROFILES = {}

for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, _ = sf.read(f)
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1/SR)

    amps = []
    for h in range(1, N_HARMONICS + 1):
        idx = np.argmin(np.abs(freqs - freq * h))
        w = slice(max(0, idx-3), idx+4)
        amps.append(float(np.max(fft[w])))

    h1 = amps[0]
    amps_norm = [a / h1 for a in amps]
    HARMONIC_PROFILES[round(freq)] = amps_norm
    TARGET_NOTES.append((audio, freq, len(audio)/SR))

    print(f"  {freq:.0f} Hz: " + " ".join(f"H{i+1}={a:.2f}" for i, a in enumerate(amps_norm[:6])))

# ─── STAGE 2: Optimize envelope/filter/reverb/gain ───
# 10 free params: attack, decay, sustain, release, filter_cutoff, filter_reso,
#                 reverb_size, reverb_mix, gain, noise_floor

PARAM_NAMES = ['attack', 'decay', 'sustain', 'release', 'filter_cutoff',
               'filter_reso', 'reverb_size', 'reverb_mix', 'gain', 'noise_floor']
PARAM_RANGES = [
    (0.001, 0.5),   # attack
    (0.001, 1.0),   # decay
    (0.0, 1.0),     # sustain
    (0.001, 1.0),   # release
    (200, 16000),    # filter_cutoff
    (0.0, 0.95),    # filter_reso
    (0.01, 0.99),   # reverb_size
    (0.0, 0.8),     # reverb_mix
    (0.0, 1.0),     # gain
    (0.0, 0.1),     # noise_floor
]


def render_note(free_params_real, freq, dur, note_dur):
    """Render using extracted harmonics + free envelope/filter params."""
    closest = min(HARMONIC_PROFILES.keys(), key=lambda p: abs(p - freq))
    harms = HARMONIC_PROFILES[closest]

    n_samples = int(dur * SR)
    t = torch.linspace(0, dur, n_samples)

    # Additive synthesis with target's exact harmonics
    # Use 3 unison voices with slight spread for character
    n_voices = 5
    spread = 0.01  # very subtle, just enough for life
    offsets = [spread * (2.0 * i / (n_voices - 1) - 1.0) for i in range(n_voices)]

    signal = torch.zeros(n_samples)
    for vo in offsets:
        ratio = 2.0 ** (vo / 12.0)
        voice = torch.zeros(n_samples)
        for h_idx, amp in enumerate(harms):
            h = h_idx + 1
            phase = (2.0 * math.pi * freq * h * ratio * t) % (2.0 * math.pi)
            # Use saw-like phase shape weighted by target amplitude
            saw_component = 2.0 * (phase / (2 * math.pi)) - 1.0
            sine_component = torch.sin(phase)
            # Blend: more saw for low harmonics (character), more sine for high (clean)
            blend = min(1.0, h / 6.0)  # 0=all saw, 1=all sine
            wave = (1.0 - blend) * saw_component + blend * sine_component
            voice = voice + amp * wave
        signal = signal + voice
    signal = signal / n_voices

    # Noise
    signal = signal + free_params_real[9] * torch.randn(n_samples)

    # Envelope
    env = _make_adsr(n_samples,
                     torch.tensor(free_params_real[0]),  # attack
                     torch.tensor(free_params_real[1]),  # decay
                     torch.tensor(free_params_real[2]),  # sustain
                     torch.tensor(free_params_real[3]),  # release
                     note_duration=note_dur, sr=SR)

    # Filter
    signal = _lowpass_filter(signal,
                             torch.tensor(free_params_real[4]),  # cutoff
                             torch.tensor(free_params_real[5]),  # reso
                             sr=SR)

    # Apply envelope + gain
    signal = signal * env * free_params_real[8]

    # Reverb
    signal = _simple_reverb(signal,
                            torch.tensor(free_params_real[6]),  # size
                            torch.tensor(free_params_real[7]),  # mix
                            sr=SR)
    return signal


def evaluate_single(free_01):
    """Evaluate one candidate. free_01 is [0,1]^10."""
    from src.losses import get_loss
    loss_fn = get_loss("matching")

    # Denormalize
    real = []
    for val, (lo, hi) in zip(free_01, PARAM_RANGES):
        real.append(float(np.clip(val, 0, 1)) * (hi - lo) + lo)

    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = render_note(real, freq, dur + 0.15, dur)
        gen = gen.unsqueeze(0)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


if __name__ == "__main__":
    print(f"\n=== Stage 2: CMA-ES on {len(PARAM_NAMES)} params, 8-core ===", flush=True)

    # Smart init: attack fast, sustain high, cutoff high, gain moderate
    x0 = [0.01, 0.2, 0.8, 0.5, 0.8, 0.1, 0.4, 0.05, 0.3, 0.3]

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, 0.3, {
        'bounds': [[0]*10, [1]*10],
        'maxfevals': 10000, 'popsize': 24, 'verbose': -9,
    })
    best = float('inf')
    evals = 0
    with Pool(8) as pool:
        while not es.stop():
            sols = es.ask()
            fits = pool.map(evaluate_single, sols)
            es.tell(sols, fits)
            evals += len(sols)
            if min(fits) < best: best = min(fits)
            if evals % 2000 < 24:
                print(f"  evals {evals}: best={best:.4f}", flush=True)

    elapsed = time.time() - t0
    best_01 = es.result.xbest
    best_real = [float(np.clip(v, 0, 1)) * (hi - lo) + lo for v, (lo, hi) in zip(best_01, PARAM_RANGES)]

    print(f"\nDone: {best:.4f} in {elapsed:.0f}s", flush=True)
    for n, v in zip(PARAM_NAMES, best_real):
        print(f"  {n:>15}: {v:.4f}")

    # Render full sequence
    note_data = [
        (0.023,0.148,294.5),(0.171,0.154,278.0),(0.325,0.145,220.6),
        (0.470,0.151,294.5),(0.621,0.145,278.0),(0.778,0.136,220.6),
        (0.914,0.154,292.8),(1.068,0.148,278.0),(1.216,0.148,292.8),
        (1.364,0.145,278.0),(1.509,0.145,292.8),(1.654,0.148,278.0),
        (1.802,0.157,220.6),(1.959,0.145,294.5),(2.104,0.148,278.0),
        (2.252,0.151,220.6),(2.403,0.145,292.8),(2.548,0.151,278.0),
        (2.699,0.145,220.6),(2.844,0.151,294.5),(2.995,0.470,292.8),
    ]
    orig_full, _ = librosa.load('target.wav', sr=SR, mono=True)
    s, e = int(1.312*SR), int(4.5*SR)
    rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))
    full_audio = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = render_note(best_real, freq, dur+0.3, dur)
        audio_np = audio.detach().numpy()
        pos = int((1.312+onset)*SR)
        end = pos+len(audio_np)
        if end <= len(full_audio): full_audio[pos:end] += audio_np
        else:
            trim = len(full_audio)-pos
            if trim > 0: full_audio[pos:pos+trim] += audio_np[:trim]
    rms_m = np.sqrt(np.mean(full_audio[s:e]**2))
    full_audio *= rms_orig/(rms_m+1e-8)
    if np.max(np.abs(full_audio)) > 0.99: full_audio = full_audio / np.max(np.abs(full_audio)) * 0.99
    sf.write('output/matched_v21_two_stage.wav', full_audio.astype(np.float32), SR)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V18 (previous best) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v18_filter_adsr.wav"])
    print("--- V21 (extracted harmonics + optimized envelope) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v21_two_stage.wav"])
