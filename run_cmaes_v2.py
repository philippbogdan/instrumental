"""CMA-ES with 29-param synth, 8-core parallel, correct MatchingLoss, 100K evals."""

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
from src.spectral_init import spectral_init

# Preload targets (pickle-friendly for multiprocessing)
TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((audio, freq, len(audio) / 44100))


def evaluate_single(params_np):
    """One eval in a worker process. Uses the original MatchingLoss."""
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


if __name__ == "__main__":
    print(f"Params: {N_PARAMS}")

    # Spectral init → pad to 29
    first_note = TARGET_NOTES[2][0]
    init_15 = spectral_init(first_note, 44100)

    init_29 = torch.full((N_PARAMS,), 0.5)
    init_29[0] = init_15[0]   # saw_mix
    init_29[1] = init_15[1]   # square_mix
    init_29[2] = init_15[2]   # sine_mix
    init_29[3] = init_15[3]   # noise_mix
    init_29[4] = init_15[4]   # detune
    init_29[7] = init_15[5]   # filter_cutoff
    init_29[8] = init_15[6]   # filter_resonance
    init_29[14] = init_15[7]  # attack
    init_29[15] = init_15[8]  # decay
    init_29[16] = init_15[9]  # sustain
    init_29[17] = init_15[10] # release
    init_29[28] = init_15[11] # gain
    init_29[5] = 0.5    # unison_voices ~4
    init_29[6] = 0.15   # unison_spread
    init_29[9] = 0.5    # filter_env_amount
    init_29[10] = 0.01  # filter_attack
    init_29[11] = 0.3   # filter_decay
    init_29[12] = 0.5   # filter_sustain
    init_29[13] = 0.3   # filter_release
    init_29[18] = 0.05  # drive
    init_29[19] = 0.05  # dist_mix
    init_29[20] = 0.3   # delay_time
    init_29[21] = 0.2   # delay_feedback
    init_29[22] = 0.1   # delay_mix
    init_29[23] = 0.4   # reverb_size
    init_29[24] = 0.15  # reverb_mix
    init_29[25] = 0.3   # vibrato_rate
    init_29[26] = 0.05  # vibrato_depth
    init_29[27] = 0.1   # noise_floor

    BUDGET = 100000
    POPSIZE = 48  # multiple of 8 cores

    print(f"\n=== CMA-ES: {BUDGET} evals, {N_PARAMS} params, pop={POPSIZE}, 8-core CPU ===")
    t0 = time.time()

    es = cma.CMAEvolutionStrategy(init_29.numpy(), 0.4, {
        'bounds': [[0]*N_PARAMS, [1]*N_PARAMS],
        'maxfevals': BUDGET,
        'popsize': POPSIZE,
        'verbose': -9,
        'seed': 42,
        'tolx': 1e-8,
        'tolfun': 1e-10,
        'tolstagnation': 20000,
    })

    best_loss = float('inf')
    evals = 0

    with Pool(8) as pool:
        while not es.stop():
            solutions = es.ask()
            fitnesses = pool.map(evaluate_single, solutions)
            es.tell(solutions, fitnesses)
            evals += len(solutions)
            if min(fitnesses) < best_loss:
                best_loss = min(fitnesses)
            if evals % 5000 < POPSIZE:
                elapsed = time.time() - t0
                rate = evals / elapsed
                eta = (BUDGET - evals) / rate if rate > 0 else 0
                print(f"  evals {evals:>7}: best={best_loss:.4f}  ({rate:.0f}/s, ETA {eta:.0f}s)", flush=True)

    total_time = time.time() - t0
    best_params = torch.tensor(es.result.xbest, dtype=torch.float32)
    print(f"\nDone: loss={best_loss:.4f}, {evals} evals in {total_time:.1f}s ({evals/total_time:.0f}/s)", flush=True)

    print("\nParameters:", flush=True)
    for name, val in zip(SynthPatch().get_param_names(), best_params):
        idx = SynthPatch().get_param_names().index(name)
        lo, hi = PARAM_DEFS[idx][1], PARAM_DEFS[idx][2]
        real_val = val.item() * (hi - lo) + lo
        print(f"  {name:>20}: {val.item():.4f}  (={real_val:.3f})")

    np.save('output/best_params_29p.npy', best_params.numpy())

    # Render full sequence
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
    s, e = int(1.312 * 44100), int(4.5 * 44100)
    rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))

    full_audio = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = synth.render(best_params, f0_hz=freq, duration=dur+tail, note_duration=dur)
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

    sf.write('output/matched_v16_29params.wav', full_audio.astype(np.float32), 44100)

    cent_o = np.mean(librosa.feature.spectral_centroid(y=orig_full[s:e], sr=44100))
    cent_m = np.mean(librosa.feature.spectral_centroid(y=full_audio[s:e], sr=44100))
    print(f"\nCentroid: orig={cent_o:.0f} match={cent_m:.0f} diff={abs(cent_o-cent_m):.0f} Hz", flush=True)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V16 (29 params, 100K evals, 8-core) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v16_29params.wav"])
