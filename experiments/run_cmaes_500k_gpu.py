"""Run CMA-ES with 500K evaluations on MPS GPU."""

import torch
import numpy as np
import soundfile as sf
import subprocess
import time
import librosa
import cma
from src.synth import SynthPatch, PARAM_DEFS
from src.losses import get_loss
from src.spectral_init import spectral_init

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

# Preload targets onto GPU
TARGET_DATA = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_DATA.append((torch.tensor(audio, dtype=torch.float32, device=DEVICE), freq, len(audio) / 44100))

# Loss function on GPU
loss_fn = get_loss("matching")
# Move internal modules to MPS
if hasattr(loss_fn, '_stft') and hasattr(loss_fn._stft, '_loss'):
    loss_fn._stft._loss = loss_fn._stft._loss.to(DEVICE)
if hasattr(loss_fn, '_mfcc') and hasattr(loss_fn._mfcc, '_mfcc'):
    loss_fn._mfcc._mfcc = loss_fn._mfcc._mfcc.to(DEVICE)

synth = SynthPatch()


def evaluate_batch_gpu(solutions):
    """Evaluate all solutions sequentially on GPU (faster than CPU multiprocessing for small renders)."""
    results = []
    for params_np in solutions:
        params = torch.tensor(params_np, dtype=torch.float32)
        total = 0.0
        for target_gpu, freq, dur in TARGET_DATA:
            gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
            gen_gpu = gen.to(DEVICE)
            target = target_gpu.unsqueeze(0)
            ml = min(target.shape[1], gen_gpu.shape[1])
            gen_3d = gen_gpu[:, :ml].unsqueeze(0)
            tgt_3d = target[:, :ml].unsqueeze(0)
            loss = loss_fn(gen_3d, tgt_3d)
            total += loss.item()
        results.append(total / len(TARGET_DATA))
    return results


if __name__ == "__main__":
    # Spectral init
    init_15 = spectral_init(TARGET_DATA[2][0].cpu().numpy(), 44100)
    init_18 = torch.cat([init_15, torch.tensor([0.5, 0.3, 0.3])]).numpy()

    BUDGET = 500000
    POPSIZE = 80

    print(f"=== CMA-ES: {BUDGET} evals, 18 params, MPS GPU ===")

    t0 = time.time()

    es = cma.CMAEvolutionStrategy(init_18, 0.5, {
        'bounds': [[0]*18, [1]*18],
        'maxfevals': BUDGET,
        'popsize': POPSIZE,
        'verbose': -9,
        'seed': 42,
        'tolx': 1e-8,
        'tolfun': 1e-10,
        'tolstagnation': 50000,
    })

    best_loss = float('inf')
    evals = 0

    while not es.stop():
        solutions = es.ask()
        fitnesses = evaluate_batch_gpu(solutions)
        es.tell(solutions, fitnesses)
        evals += len(solutions)
        if min(fitnesses) < best_loss:
            best_loss = min(fitnesses)
        if evals % 10000 < POPSIZE:
            elapsed = time.time() - t0
            rate = evals / elapsed
            eta = (BUDGET - evals) / rate
            print(f"  evals {evals:>7}: best={best_loss:.4f}  ({rate:.0f} evals/s, ETA {eta:.0f}s)")

    total_time = time.time() - t0
    best_params = torch.tensor(es.result.xbest, dtype=torch.float32)
    print(f"\nDone: loss={best_loss:.4f}, {evals} evals in {total_time:.1f}s ({evals/total_time:.0f} evals/s)")

    print("\nRecovered parameters:")
    for name, val in zip(synth.get_param_names(), best_params):
        idx = synth.get_param_names().index(name)
        lo, hi = PARAM_DEFS[idx][1], PARAM_DEFS[idx][2]
        real_val = val.item() * (hi - lo) + lo
        print(f"  {name:>18}: {val.item():.4f}  (={real_val:.3f})")

    np.save('output/best_params_500k.npy', best_params.numpy())

    # Render full sequence
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

    sf.write('output/matched_v14_500k.wav', full_audio.astype(np.float32), 44100)

    cent_o = np.mean(librosa.feature.spectral_centroid(y=orig_full[s:e], sr=44100))
    cent_m = np.mean(librosa.feature.spectral_centroid(y=full_audio[s:e], sr=44100))
    print(f"\nCentroid: orig={cent_o:.0f} match={cent_m:.0f} diff={abs(cent_o-cent_m):.0f} Hz")

    print("\n--- ORIGINAL ---")
    subprocess.run(["afplay", "target.wav"])
    print("--- V14 (500K evals, GPU) ---")
    subprocess.run(["afplay", "output/matched_v14_500k.wav"])
