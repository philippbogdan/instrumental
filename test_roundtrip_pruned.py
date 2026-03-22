"""Round-trip test with pruned 13-param search space."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np
import soundfile as sf
from synth import SynthPatch, PARAM_DEFS, _denorm
from losses import MatchingLoss
from spectral_init import spectral_init
import cma
import time

# Same pruning config as server.py
OPTIMIZED_INDICES = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22]
N_OPT = len(OPTIMIZED_INDICES)
FROZEN = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
FROZEN[1] = 0.0; FROZEN[13] = 0.3; FROZEN[14] = 0.05
FROZEN[15] = 0.0; FROZEN[16] = 0.0; FROZEN[17] = 0.0
FROZEN[18] = 0.05; FROZEN[19] = 0.3; FROZEN[20] = 0.5; FROZEN[21] = 0.3
FROZEN[23] = 0.25; FROZEN[24] = 0.5; FROZEN[25] = 0.5; FROZEN[26] = 0.5; FROZEN[27] = 0.5

def expand(reduced):
    full = FROZEN.copy()
    for i, idx in enumerate(OPTIMIZED_INDICES):
        full[idx] = reduced[i]
    return full

def reduce(full):
    return np.array([full[idx] for idx in OPTIMIZED_INDICES], dtype=np.float32)

# Ground truth (same bright saw+square)
GT_FULL = np.array([
    0.80, 0.60, 0.05, 0.00, 0.50, 0.70, 0.10, 0.01,
    0.30, 0.70, 0.40, 0.80, 0.55, 0.30, 0.10, 0.30,
    0.10, 0.00, 0.10, 0.30, 0.60, 0.30, 0.50, 0.25,
    0.50, 0.50, 0.50, 0.50,
], dtype=np.float32)
GT_REDUCED = reduce(GT_FULL)

# Generate target
synth = SynthPatch()
gt_params = torch.tensor(GT_FULL, dtype=torch.float32)
freqs = [220.0, 277.2, 293.7]
target_notes = []
for freq in freqs:
    audio = synth.render(gt_params, f0_hz=freq, duration=0.5, note_duration=0.45)
    target_notes.append((audio.detach().numpy().squeeze(), freq, 0.5))

# Spectral init
init_15 = spectral_init(target_notes[0][0], 44100)
x0_full = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
for i in range(15): x0_full[i] = float(init_15[i])
x0_reduced = reduce(x0_full)

loss_fn = MatchingLoss()

def evaluate(reduced_np):
    full_np = expand(np.clip(reduced_np, 0, 1))
    params = torch.tensor(full_np, dtype=torch.float32)
    s = SynthPatch()
    total = 0.0
    for audio_np, freq, dur in target_notes:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = s.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(target_notes)

# Run 5 times with different inits
print(f"=== 5 CMA-ES Runs — PRUNED {N_OPT} params (was 28) ===")
print(f"  Budget per run: 2000 evals\n")

results = []
t0 = time.time()
for run_id in range(5):
    if run_id == 0:
        x0 = x0_reduced.copy()
    else:
        x0 = x0_reduced + np.random.randn(N_OPT) * 0.2
        x0 = np.clip(x0, 0, 1)

    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.15, {
        "bounds": [[0]*N_OPT, [1]*N_OPT],
        "maxfevals": 2000, "popsize": 20, "verbose": -9, "seed": run_id*100+1,
    })
    best = float("inf")
    while not es.stop():
        sols = es.ask()
        fits = [evaluate(s) for s in sols]
        es.tell(sols, fits)
        if min(fits) < best: best = min(fits)

    best_full = expand(es.result.xbest)
    results.append((run_id, best, best_full, es.result.xbest))
    print(f"  Run {run_id}: loss={best:.4f}")

total_time = time.time() - t0
results.sort(key=lambda r: r[1])

# Summary table
opt_names = [PARAM_DEFS[i][0] for i in OPTIMIZED_INDICES]
print(f"\n  Total time: {total_time:.1f}s\n")
print(f"  {'Run':>4}  {'Loss':>8}  {'saw':>5}  {'sine':>5}  {'filt':>6}  {'gain':>5}  {'f_env':>6}  {'pw':>5}")
for run_id, loss, full, red in results:
    print(f"  {run_id:>4}  {loss:>8.4f}  {red[0]:>5.2f}  {red[1]:>5.2f}  {red[4]:>6.2f}  {red[10]:>5.2f}  {red[11]:>6.2f}  {red[12]:>5.2f}")

# Best run comparison
_, best_loss, best_full, best_red = results[0]
print(f"\n=== Best Run (loss {best_loss:.4f}) — Optimized Params Only ===")
print(f"  {'Param':>20}  {'GT':>6}  {'Recov':>6}  {'Delta':>6}  {'Status'}")
close = 0
for i, idx in enumerate(OPTIMIZED_INDICES):
    name = PARAM_DEFS[idx][0]
    gt = GT_REDUCED[i]
    rec = float(best_red[i])
    delta = abs(gt - rec)
    status = "OK" if delta < 0.1 else "CLOSE" if delta < 0.2 else "MISS"
    if delta < 0.1: close += 1
    print(f"  {name:>20}  {gt:>6.2f}  {rec:>6.2f}  {delta:>6.2f}  {status}")
print(f"\n  Optimized params within 0.1: {close}/{N_OPT}")

# Audio quality
from scipy import signal as sig
def centroid_hz(audio, sr):
    freqs, psd = sig.welch(audio.astype(np.float32), sr, nperseg=2048)
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

gen = synth.render(torch.tensor(best_full, dtype=torch.float32), f0_hz=220.0, duration=0.5, note_duration=0.45)
gen_np = gen.detach().numpy().squeeze()
tgt_np = target_notes[0][0]
ml = min(len(tgt_np), len(gen_np))
corr = float(np.corrcoef(tgt_np[:ml], gen_np[:ml])[0, 1])
tc = centroid_hz(tgt_np, 44100)
gc = centroid_hz(gen_np, 44100)

print(f"\n  Waveform correlation: {corr:.4f}")
print(f"  Spectral centroid — Target: {tc:.0f}Hz, Recovered: {gc:.0f}Hz (delta: {abs(tc-gc):.0f}Hz)")
print(f"\n=== vs Previous Results ===")
print(f"  28-param continuous: 10/28 within 0.1, corr=-0.03, centroid delta=329Hz, loss=2.78")
print(f"  28-param discrete:  13/28 within 0.15, corr=-0.09, centroid delta=175Hz, loss=3.78")
print(f"  13-param pruned:    {close}/{N_OPT} within 0.1, corr={corr:.2f}, centroid delta={abs(tc-gc):.0f}Hz, loss={best_loss:.2f}")

sf.write("/tmp/roundtrip_pruned_best.wav", gen_np, 44100)
print(f"\n  Saved: /tmp/roundtrip_pruned_best.wav")
