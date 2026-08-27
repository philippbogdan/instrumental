"""Freeze v11's 18 params, optimize only 4 new filter ADSR params. 8-core."""

import torch, numpy as np, soundfile as sf, subprocess, time, librosa, cma
from multiprocessing import Pool
from src.synth import SynthPatch, PARAM_DEFS
from src.losses import get_loss

# v11 best (first 18 params) + 4 new filter ADSR at 0.5 (neutral)
V11_BASE = [
    0.7329, 0.9156, 0.0029, 0.0127, 0.4997,
    0.9937, 0.2039, 0.0002, 0.1137, 0.8624,
    0.9048, 0.1422, 0.2972, 0.5472, 0.0247,
    0.7223, 0.0012, 0.4182,
    0.5, 0.5, 0.5, 0.5  # filter ADSR — optimize these
]

FREE_INDICES = [18, 19, 20, 21]  # filter_attack, filter_decay, filter_sustain, filter_release
N_FREE = 4

TARGET_NOTES = []
for f, freq in [('notes/note_03_A3_221Hz.wav', 221.0), ('notes/note_02_Cs4_278Hz.wav', 278.0), ('notes/note_01_D4_295Hz.wav', 295.0)]:
    audio, sr = sf.read(f)
    TARGET_NOTES.append((audio, freq, len(audio)/44100))


def evaluate_single(free_np):
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = list(V11_BASE)
    for i, idx in enumerate(FREE_INDICES):
        params[idx] = float(np.clip(free_np[i], 0, 1))
    params_t = torch.tensor(params, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params_t, f0_hz=freq, duration=dur, note_duration=dur*0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(TARGET_NOTES)


if __name__ == "__main__":
    print(f"22-param synth. Frozen: 18 (v11). Optimizing: 4 (filter ADSR)", flush=True)
    t0 = time.time()
    es = cma.CMAEvolutionStrategy([0.5]*N_FREE, 0.3, {
        'bounds': [[0]*N_FREE, [1]*N_FREE],
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
    print(f"\nDone: {best:.4f} in {elapsed:.0f}s", flush=True)
    best_free = es.result.xbest
    for n, v in zip(['filter_attack','filter_decay','filter_sustain','filter_release'], best_free):
        print(f"  {n}: {v:.4f}")

    final = list(V11_BASE)
    for i, idx in enumerate(FREE_INDICES): final[idx] = float(best_free[i])
    final_t = torch.tensor(final, dtype=torch.float32)

    synth = SynthPatch()
    note_data = [
        (0.023,0.148,294.5),(0.171,0.154,278.0),(0.325,0.145,220.6),
        (0.470,0.151,294.5),(0.621,0.145,278.0),(0.778,0.136,220.6),
        (0.914,0.154,292.8),(1.068,0.148,278.0),(1.216,0.148,292.8),
        (1.364,0.145,278.0),(1.509,0.145,292.8),(1.654,0.148,278.0),
        (1.802,0.157,220.6),(1.959,0.145,294.5),(2.104,0.148,278.0),
        (2.252,0.151,220.6),(2.403,0.145,292.8),(2.548,0.151,278.0),
        (2.699,0.145,220.6),(2.844,0.151,294.5),(2.995,0.470,292.8),
    ]
    orig_full, _ = librosa.load('target.wav', sr=44100, mono=True)
    s, e = int(1.312*44100), int(4.5*44100)
    rms_orig = np.sqrt(np.mean(orig_full[s:e]**2))
    full_audio = np.zeros(len(orig_full))
    for onset, dur, freq in note_data:
        audio = synth.render(final_t, f0_hz=freq, duration=dur+0.3, note_duration=dur)
        audio_np = audio.detach().numpy().squeeze()
        pos = int((1.312+onset)*44100)
        end = pos+len(audio_np)
        if end <= len(full_audio): full_audio[pos:end] += audio_np
        else:
            trim = len(full_audio)-pos
            if trim > 0: full_audio[pos:pos+trim] += audio_np[:trim]
    rms_m = np.sqrt(np.mean(full_audio[s:e]**2))
    full_audio *= rms_orig/(rms_m+1e-8)
    if np.max(np.abs(full_audio)) > 0.99: full_audio = full_audio / np.max(np.abs(full_audio)) * 0.99
    sf.write('output/matched_v18_filter_adsr.wav', full_audio.astype(np.float32), 44100)

    print("\n--- ORIGINAL ---", flush=True)
    subprocess.run(["afplay", "target.wav"])
    print("--- V11 (baseline) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v11_50k.wav"])
    print("--- V18 (v11 + filter ADSR) ---", flush=True)
    subprocess.run(["afplay", "output/matched_v18_filter_adsr.wav"])
