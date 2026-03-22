"""Round-trip test with discretized params + 5 parallel runs."""
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
import multiprocessing

# Ground truth (same as before)
GROUND_TRUTH = [
    0.80, 0.60, 0.05, 0.00, 0.50, 0.70, 0.10, 0.01,
    0.30, 0.70, 0.40, 0.80, 0.55, 0.30, 0.10, 0.30,
    0.10, 0.00, 0.10, 0.30, 0.60, 0.30, 0.50, 0.25,
    0.50, 0.50, 0.50, 0.50,
]

STEP = 0.1  # discretization step

def discretize(params):
    """Round each param to nearest STEP."""
    return np.round(np.clip(params, 0, 1) / STEP) * STEP

# Generate target notes
synth = SynthPatch()
gt_params = torch.tensor(GROUND_TRUTH, dtype=torch.float32)
freqs = [220.0, 277.2, 293.7]
target_notes = []
for freq in freqs:
    audio = synth.render(gt_params, f0_hz=freq, duration=0.5, note_duration=0.45)
    target_notes.append((audio.detach().numpy().squeeze(), freq, 0.5))

# Spectral init
init_15 = spectral_init(target_notes[0][0], 44100)
x0_base = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
for i in range(15):
    x0_base[i] = float(init_15[i])

N = len(PARAM_DEFS)
loss_fn = MatchingLoss()

def evaluate_discrete(params_np):
    """Evaluate with discretized params."""
    params_d = discretize(params_np)
    params = torch.tensor(params_d, dtype=torch.float32)
    s = SynthPatch()
    total = 0.0
    for audio_np, freq, dur in target_notes:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = s.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(target_notes)

def run_single(args):
    """Run one CMA-ES instance."""
    run_id, x0, budget = args
    np.random.seed(run_id * 42 + 7)

    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.15, {
        "bounds": [[0] * N, [1] * N],
        "maxfevals": budget,
        "popsize": 20,
        "verbose": -9,
        "seed": run_id * 100 + 1,
    })

    best_loss = float("inf")
    while not es.stop():
        sols = es.ask()
        fits = [evaluate_discrete(s) for s in sols]
        es.tell(sols, fits)
        if min(fits) < best_loss:
            best_loss = min(fits)

    best_params = discretize(es.result.xbest)
    return run_id, best_loss, best_params

# Create 5 different initializations
print("=== 5 Parallel CMA-ES Runs (discretized, 2k evals each) ===")
inits = []
for i in range(5):
    if i == 0:
        x0 = x0_base.copy()  # spectral init
    else:
        # Perturbed spectral init
        x0 = x0_base.copy() + np.random.randn(N) * 0.2
        x0 = np.clip(x0, 0, 1)
    inits.append((i, x0, 2000))

print(f"  Budget per run: 2000 evals")
print(f"  Total budget: 10000 evals")
print(f"  Discretization step: {STEP}")
print(f"  Param values: {int(1/STEP)+1} per dimension")
print()

t0 = time.time()

# Run sequentially (spawn mode on macOS doesn't support top-level Pool)
results = []
for init in inits:
    run_id, loss, params = run_single(init)
    print(f"  Run {run_id}: loss={loss:.4f}")
    results.append((run_id, loss, params))

total_time = time.time() - t0

# Sort by loss
results.sort(key=lambda r: r[1])

print(f"  Total time: {total_time:.1f}s\n")
print(f"  {'Run':>4}  {'Loss':>10}  {'filter_cutoff':>14}  {'saw':>6}  {'square':>8}  {'gain':>6}  {'eq1_g':>7}  {'eq2_g':>7}")
for run_id, loss, params in results:
    d = _denorm(torch.tensor(params, dtype=torch.float32))
    print(f"  {run_id:>4}  {loss:>10.4f}  {d['filter_cutoff'].item():>14.0f}  {params[0]:>6.1f}  {params[1]:>8.1f}  {params[11]:>6.1f}  {params[25]:>7.1f}  {params[27]:>7.1f}")

# Best result comparison
best_run, best_loss, best_params = results[0]
print(f"\n=== Best Run: #{best_run} (loss {best_loss:.4f}) ===")
print(f"  {'Param':>20}  {'GT':>6}  {'Recov':>6}  {'Delta':>6}  {'Status'}")
close = 0
for i, (name, lo, hi) in enumerate(PARAM_DEFS):
    gt = GROUND_TRUTH[i]
    rec = float(best_params[i])
    delta = abs(gt - rec)
    status = "OK" if delta < 0.15 else "CLOSE" if delta < 0.25 else "MISS"
    if delta < 0.15: close += 1
    print(f"  {name:>20}  {gt:>6.1f}  {rec:>6.1f}  {delta:>6.2f}  {status}")

print(f"\n  Params within 0.15: {close}/{N}")
print(f"  Mean absolute error: {np.mean(np.abs(np.array(GROUND_TRUTH) - best_params)):.4f}")

# Audio quality
from scipy import signal as sig
def centroid_hz(audio, sr):
    freqs, psd = sig.welch(audio.astype(np.float32), sr, nperseg=2048)
    return float(np.sum(freqs * psd) / (np.sum(psd) + 1e-8))

s = SynthPatch()
gen = s.render(torch.tensor(best_params, dtype=torch.float32), f0_hz=220.0, duration=0.5, note_duration=0.45)
gen_np = gen.detach().numpy().squeeze()
target_np = target_notes[0][0]
ml = min(len(target_np), len(gen_np))
corr = float(np.corrcoef(target_np[:ml], gen_np[:ml])[0, 1])
tc = centroid_hz(target_np, 44100)
gc = centroid_hz(gen_np, 44100)
print(f"\n  Waveform correlation: {corr:.4f}")
print(f"  Spectral centroid — Target: {tc:.0f}Hz, Recovered: {gc:.0f}Hz (delta: {abs(tc-gc):.0f}Hz)")

# Compare with continuous baseline
print(f"\n=== vs Continuous Baseline (from previous test) ===")
print(f"  Continuous: 10/28 within 0.1, corr=-0.03, centroid delta=329Hz")
print(f"  Discrete:   {close}/{N} within 0.15, corr={corr:.2f}, centroid delta={abs(tc-gc):.0f}Hz")

sf.write("/tmp/roundtrip_discrete_best.wav", gen_np, 44100)
print(f"\n  Saved: /tmp/roundtrip_discrete_best.wav")
