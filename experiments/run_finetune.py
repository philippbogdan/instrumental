"""Freeze v11's 18 params, optimize only the 11 new params with tight ranges. 8-core."""

import torch
import numpy as np
import soundfile as sf
import subprocess
import time
import librosa
import cma
from multiprocessing import Pool
from src.synth import SynthPatch, PARAM_DEFS, N_PARAMS
from src.losses import get_loss

# v11 base mapped to 29-param synth
V11_BASE = [
    0.7329,  # 0: saw_mix — FROZEN
    0.9156,  # 1: square_mix — FROZEN
    0.0029,  # 2: sine_mix — FROZEN
    0.0127,  # 3: noise_mix — FROZEN
    0.5,     # 4: detune — FROZEN at 0
    0.5,     # 5: unison_voices — FROZEN at ~4
    0.0012,  # 6: unison_spread — FROZEN
    0.9937,  # 7: filter_cutoff — FROZEN
    0.2039,  # 8: filter_resonance — FROZEN
    0.5,     # 9: filter_env_amount — OPTIMIZE
    0.5,     # 10: filter_attack — OPTIMIZE
    0.5,     # 11: filter_decay — OPTIMIZE
    0.5,     # 12: filter_sustain — OPTIMIZE
    0.5,     # 13: filter_release — OPTIMIZE
    0.0002,  # 14: attack — FROZEN
    0.1137,  # 15: decay — FROZEN
    0.8624,  # 16: sustain — FROZEN
    0.9048,  # 17: release — FROZEN
    0.0,     # 18: drive — OPTIMIZE (range 0-0.3, subtle only)
    0.0,     # 19: dist_mix — OPTIMIZE (range 0-0.3)
    0.3,     # 20: delay_time — OPTIMIZE
    0.0,     # 21: delay_feedback — OPTIMIZE (range 0-0.4)
    0.0,     # 22: delay_mix — OPTIMIZE (range 0-0.3)
    0.5472,  # 23: reverb_size — FROZEN
    0.0247,  # 24: reverb_mix — FROZEN
    0.5,     # 25: vibrato_rate — OPTIMIZE
    0.0,     # 26: vibrato_depth — OPTIMIZE (range 0-0.2, subtle)
    0.4182,  # 27: noise_floor — FROZEN
    0.1422,  # 28: gain — FROZEN
]

# Indices to optimize and their constrained ranges [lo, hi]
FREE_PARAMS = [
    (9,  0.0, 1.0),   # filter_env_amount
    (10, 0.0, 1.0),   # filter_attack
    (11, 0.0, 1.0),   # filter_decay
    (12, 0.0, 1.0),   # filter_sustain
    (13, 0.0, 1.0),   # filter_release
    (18, 0.0, 0.3),   # drive (subtle)
    (19, 0.0, 0.3),   # dist_mix (subtle)
    (20, 0.1, 0.6),   # delay_time
    (21, 0.0, 0.4),   # delay_feedback (no runaway)
    (22, 0.0, 0.3),   # delay_mix (subtle)
    (25, 0.2, 0.8),   # vibrato_rate
    (26, 0.0, 0.15),  # vibrato_depth (very subtle)
]

FREE_INDICES = [f[0] for f in FREE_PARAMS]
FREE_LO = [f[1] for f in FREE_PARAMS]
FREE_HI = [f[2] for f in FREE_PARAMS]
N_FREE = len(FREE_PARAMS)

TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((audio, freq, len(audio) / 44100))


def evaluate_single(free_np):
    """Evaluate one candidate."""
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = list(V11_BASE)
    for i, idx in enumerate(FREE_INDICES):
        # Map [0,1] CMA-ES space to constrained range
        val = free_np[i] * (FREE_HI[i] - FREE_LO[i]) + FREE_LO[i]
        params[idx] = float(np.clip(val, FREE_LO[i], FREE_HI[i]))
    params_t = torch.tensor(params, dtype=torch.float32)

    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params_t, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


if __name__ == "__main__":
    print(f"Total params: {N_PARAMS}, Frozen: {N_PARAMS - N_FREE}, Optimizing: {N_FREE}")
    print(f"Free params: {[PARAM_DEFS[i][0] for i in FREE_INDICES]}")

    BUDGET = 10000
    POPSIZE = 24

    print(f"\n=== CMA-ES: {BUDGET} evals, {N_FREE} free params, 8-core ===")
    t0 = time.time()

    es = cma.CMAEvolutionStrategy([0.5]*N_FREE, 0.3, {
        'bounds': [[0]*N_FREE, [1]*N_FREE],
        'maxfevals': BUDGET,
        'popsize': POPSIZE,
        'verbose': -9,
    })

    best_loss = float('inf')
    evals = 0

    with Pool(8) as pool:
        while not es.stop():
            sols = es.ask()
            fits = pool.map(evaluate_single, sols)
            es.tell(sols, fits)
            evals += len(sols)
            if min(fits) < best_loss:
                best_loss = min(fits)
            if evals % 2000 < POPSIZE:
                elapsed = time.time() - t0
                print(f"  evals {evals:>6}: best={best_loss:.4f}  ({evals/elapsed:.0f}/s)", flush=True)

    total_time = time.time() - t0
    print(f"\nDone: loss={best_loss:.4f}, {evals} evals in {total_time:.1f}s", flush=True)

    # Build final params
    best_free = es.result.xbest
    final = list(V11_BASE)
    print(f"\nOptimized params:")
    for i, idx in enumerate(FREE_INDICES):
        val = best_free[i] * (FREE_HI[i] - FREE_LO[i]) + FREE_LO[i]
        final[idx] = float(np.clip(val, FREE_LO[i], FREE_HI[i]))
        name = PARAM_DEFS[idx][0]
        print(f"  {name:>20}: {final[idx]:.4f}")

    final_t = torch.tensor(final, dtype=torch.float32)

    # Render
    synth = SynthPatch()
    note_data = [
        (0.023, 0.148, 294.5), (0.171, 0.154, 278.0), (0.325, 0.145, 220.6),
        (0.470, 0.151, 294.5), (0.621, 0.145, 278.0), (0.778, 0.136, 220.6),
        (0.914, 0.154, 292.8), (1.068, 0.148, 278.0), (1.216, 0.148, 292.8),
        (1.364, 0.145, 278.0), (1.509, 0.145, 292.8), (1.654, 0.148, 278.0),
        (1.802, 0.157, 220.6), (1.959, 0.145, 294.5), (2.104, 0.148, 278.0),
        (2.252, 0.151, 220.6), (2.403, 0.145, 292.8), (2.548, 0.151, 278.0),
        (2.699, 0.145, 220.6), (2.844, 0.151, 294.5), (2.995, 0.470, 292.8),
    ]
    orig_full, _ = librosa.load('target.wav', sr=44100, mono=True)
    start_time = 1.312
    tail = 0.3
    s, e = int(1.312*44100), int(4.5*44100)
    rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))

    full_audio = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = synth.render(final_t, f0_hz=freq, duration=dur+tail, note_duration=dur)
        audio_np = audio.detach().numpy().squeeze()
        pos = int((start_time + onset) * 44100)
        end = pos + len(audio_np)
        if end <= len(full_audio):
            full_audio[pos:end] += audio_np
        else:
            trim = len(full_audio) - pos
            if trim > 0:
                full_audio[pos:pos+trim] += audio_np[:trim]

    rms_m = np.sqrt(np.mean(full_audio[s:e]**2))
    full_audio *= rms_orig / (rms_m + 1e-8)
    if np.max(np.abs(full_audio)) > 0.99:
        full_audio = full_audio / np.max(np.abs(full_audio)) * 0.99

    sf.write('output/matched_v17_finetune.wav', full_audio.astype(np.float32), 44100)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V11 (baseline) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v11_50k.wav"])
    print("--- V17 (v11 + 12 new params, constrained) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v17_finetune.wav"])
