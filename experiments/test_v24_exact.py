"""Test with v24's exact config: 24 params, original loss, same targets."""
import sys, os, torch, numpy as np, soundfile as sf, time, cma
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from synth import SynthPatch, PARAM_DEFS
from losses import MatchingLoss
from spectral_init import spectral_init

if __name__ == "__main__":
    TARGET_NOTES = []
    for f, freq in [("notes/note_03_A3_221Hz.wav", 221.0), ("notes/note_02_Cs4_278Hz.wav", 278.0), ("notes/note_01_D4_295Hz.wav", 295.0)]:
        audio, sr = sf.read(f)
        TARGET_NOTES.append((audio.astype(np.float32), freq, len(audio)/44100))
        print(f"  {f}: {len(audio)/44100:.3f}s")

    # Spectral init (same as v24)
    init_15 = spectral_init(TARGET_NOTES[0][0], 44100)
    x0 = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
    for i in range(15): x0[i] = float(init_15[i])

    loss_fn = MatchingLoss()

    # Optimize ALL 28 params (like v24 did with 24)
    def evaluate(params_np):
        s = SynthPatch()
        p = torch.tensor(np.clip(params_np, 0, 1).astype(np.float32))
        total = 0.0
        for audio_np, freq, dur in TARGET_NOTES:
            t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            g = s.render(p, f0_hz=freq, duration=dur, note_duration=dur*0.9)
            ml = min(t.shape[1], g.shape[1])
            total += loss_fn(g[:,:ml].unsqueeze(0), t[:,:ml].unsqueeze(0)).item()
        return total / len(TARGET_NOTES)

    # v24 used sigma=0.15, popsize=20, 10k evals
    print("\nOptimizing ALL 28 params (v24 style, 10k evals)...")
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0.tolist(), 0.15, {
        "bounds": [[0]*28, [1]*28], "maxfevals": 10000, "popsize": 20, "verbose": -9
    })
    best = float("inf")
    while not es.stop():
        sols = es.ask()
        fits = [evaluate(s) for s in sols]
        es.tell(sols, fits)
        if min(fits) < best: best = min(fits)
        if es.result.evaluations % 2000 < 25:
            print(f"  {es.result.evaluations} evals, loss={best:.4f}")

    bp = es.result.xbest
    print(f"\nFinal loss: {best:.4f} (v24 was 2.09)")
    print(f"Time: {time.time()-t0:.0f}s")
    print(f"\nKey params: saw={bp[0]:.2f} square={bp[1]:.2f} sine={bp[2]:.2f} "
          f"cutoff={bp[5]:.2f} gain={bp[11]:.2f} eq1g={bp[25]:.2f} eq2g={bp[27]:.2f}")
