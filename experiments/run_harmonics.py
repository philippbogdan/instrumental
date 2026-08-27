"""Freeze v11, optimize 8 harmonic multipliers on top. 8-core."""

import torch, numpy as np, soundfile as sf, subprocess, time, librosa, cma, math
from multiprocessing import Pool
from src.synth import _make_adsr, _simple_reverb, _lowpass_filter

SR = 44100

# v11 envelope/filter/reverb params (denormalized)
V11 = {
    'saw_mix': 0.7329, 'square_mix': 0.9156, 'sine_mix': 0.0029,
    'noise_mix': 0.0127 * 0.5, 'noise_floor': 0.4182 * 0.1,
    'detune': (0.4997 - 0.5) * 48, 'unison_voices': int(round(0.7223 * 6 + 1)),
    'unison_spread': 0.0012 * 0.5,
    'filter_cutoff': 0.9937 * 15800 + 200, 'filter_resonance': 0.2039 * 0.95,
    'attack': 0.0002 * 0.499 + 0.001, 'decay': 0.1137 * 0.999 + 0.001,
    'sustain': 0.8624, 'release': 0.9048 * 0.999 + 0.001,
    'gain': 0.1422, 'filter_env': 0.2972 * 2 - 1,
    'reverb_size': 0.5472 * 0.98 + 0.01, 'reverb_mix': 0.0247 * 0.8,
}

TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((audio, freq, len(audio)/SR))


def render_with_harmonic_mults(h_mults, freq, dur, note_dur):
    """Render using v11 base + harmonic multipliers."""
    v = V11
    n_samples = int(dur * SR)
    t = torch.linspace(0, dur, n_samples)

    # Base waveforms with unison
    n_voices = v['unison_voices']
    spread = v['unison_spread']
    if n_voices == 1:
        offsets = [0.0]
    else:
        offsets = [spread * (2.0 * i / (n_voices - 1) - 1.0) for i in range(n_voices)]

    signal = torch.zeros(n_samples)
    for vo in offsets:
        total_detune = v['detune'] + vo
        ratio = 2.0 ** (total_detune / 12.0)
        f_voice = freq * ratio

        # Instead of fixed saw/square, build from harmonics with multipliers
        voice = torch.zeros(n_samples)
        for h in range(1, 9):  # harmonics 1-8
            phase = (2.0 * math.pi * f_voice * h * t) % (2.0 * math.pi)
            # Base amplitude: saw gives 1/h, square gives 1/h for odd, 0 for even
            saw_amp = 1.0 / h
            sq_amp = (1.0 / h) if h % 2 == 1 else 0.0
            base_amp = v['saw_mix'] * saw_amp + v['square_mix'] * sq_amp + v['sine_mix'] * (1.0 if h == 1 else 0.0)
            # Apply the learned multiplier
            voice = voice + base_amp * h_mults[h-1] * torch.sin(phase)

        signal = signal + voice
    signal = signal / n_voices

    # Normalize
    total_mix = max(v['saw_mix'] + v['square_mix'] + v['sine_mix'], 0.01)
    signal = signal / total_mix

    # Noise
    signal = signal + v['noise_mix'] * torch.randn(n_samples)
    signal = signal + v['noise_floor'] * torch.randn(n_samples)

    # Envelope
    env = _make_adsr(n_samples, torch.tensor(v['attack']), torch.tensor(v['decay']),
                     torch.tensor(v['sustain']), torch.tensor(v['release']),
                     note_duration=note_dur, sr=SR)

    # Filter
    effective_cutoff = torch.tensor(v['filter_cutoff'] * (1.0 + v['filter_env'] * (env.mean().item() - 0.5)))
    signal = _lowpass_filter(signal, effective_cutoff, torch.tensor(v['filter_resonance']), sr=SR)

    # Apply envelope + gain
    signal = signal * env * v['gain']

    # Reverb
    signal = _simple_reverb(signal, torch.tensor(v['reverb_size']), torch.tensor(v['reverb_mix']), sr=SR)

    return signal


def evaluate_single(h_mults_np):
    from src.losses import get_loss
    loss_fn = get_loss("matching")
    # h_mults in [0, 1] from CMA-ES → scale to [0.0, 3.0] (can boost up to 3x)
    h_mults = [m * 3.0 for m in h_mults_np]
    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = render_with_harmonic_mults(h_mults, freq, dur + 0.15, dur)
        gen = gen.unsqueeze(0)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


if __name__ == "__main__":
    print("Optimizing 8 harmonic multipliers on frozen v11 base, 8-core", flush=True)

    # Start with multipliers at 1/3 (=1.0x after scaling) — neutral
    x0 = [1.0/3.0] * 8

    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0, 0.2, {
        'bounds': [[0]*8, [1]*8],
        'maxfevals': 5000, 'popsize': 16, 'verbose': -9,
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
            if evals % 1000 < 16:
                print(f"  evals {evals}: best={best:.4f}", flush=True)

    elapsed = time.time() - t0
    best_mults = [m * 3.0 for m in es.result.xbest]
    print(f"\nDone: {best:.4f} in {elapsed:.0f}s", flush=True)
    print("Harmonic multipliers (1.0 = unchanged from saw/square base):")
    for h, m in enumerate(best_mults, 1):
        print(f"  H{h}: {m:.3f}x")

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
        audio = render_with_harmonic_mults(best_mults, freq, dur+0.3, dur)
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
    sf.write('output/matched_v20_harmonics.wav', full_audio.astype(np.float32), SR)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V18 (current best) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v18_filter_adsr.wav"])
    print("--- V20 (v11 + optimized harmonic multipliers) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v20_harmonics.wav"])
