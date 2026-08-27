"""Test current optimizer on v24's original clean targets."""
import sys, os, torch, numpy as np, soundfile as sf, time, cma
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from synth import SynthPatch, PARAM_DEFS
from losses import MatchingLoss
from spectral_init import spectral_init

if __name__ == "__main__":
    # v24's original clean targets
    TARGET_NOTES = []
    for f, freq in [("notes/note_03_A3_221Hz.wav", 221.0), ("notes/note_02_Cs4_278Hz.wav", 278.0), ("notes/note_01_D4_295Hz.wav", 295.0)]:
        audio, sr = sf.read(f)
        TARGET_NOTES.append((audio.astype(np.float32), freq, len(audio)/44100))
        print(f"  {f}: {len(audio)/44100:.3f}s")

    # Spectral init
    init_15 = spectral_init(TARGET_NOTES[0][0], 44100)
    x0 = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
    for i in range(15): x0[i] = float(init_15[i])

    # Pruned 13-param search
    OPT_IDX = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 22]
    FROZEN = np.full(len(PARAM_DEFS), 0.5, dtype=np.float32)
    FROZEN[1]=0; FROZEN[13]=0.3; FROZEN[14]=0.05; FROZEN[15]=0; FROZEN[16]=0; FROZEN[17]=0
    FROZEN[18]=0.05; FROZEN[19]=0.3; FROZEN[20]=0.5; FROZEN[21]=0.3; FROZEN[23]=0.25
    for i in range(24, 28): FROZEN[i] = 0.5

    def expand(r):
        f = FROZEN.copy()
        for i, idx in enumerate(OPT_IDX): f[idx] = r[i]
        return f

    x0r = np.array([x0[i] for i in OPT_IDX])
    loss_fn = MatchingLoss()

    def evaluate(rp):
        fp = expand(np.clip(rp, 0, 1))
        s = SynthPatch()
        p = torch.tensor(fp, dtype=torch.float32)
        total = 0.0
        for audio_np, freq, dur in TARGET_NOTES:
            t = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
            g = s.render(p, f0_hz=freq, duration=dur, note_duration=dur*0.9)
            ml = min(t.shape[1], g.shape[1])
            total += loss_fn(g[:,:ml].unsqueeze(0), t[:,:ml].unsqueeze(0)).item()
        return total / len(TARGET_NOTES)

    print("\nOptimizing on v24 clean targets (13 params, 10k evals)...")
    t0 = time.time()
    es = cma.CMAEvolutionStrategy(x0r.tolist(), 0.15, {
        "bounds": [[0]*13, [1]*13], "maxfevals": 10000, "popsize": 20, "verbose": -9
    })
    best = float("inf")
    while not es.stop():
        sols = es.ask()
        fits = [evaluate(s) for s in sols]
        es.tell(sols, fits)
        if min(fits) < best: best = min(fits)
        if es.result.evaluations % 2000 < 25:
            print(f"  {es.result.evaluations} evals, loss={best:.4f}")

    print(f"\nFinal loss: {best:.4f} (v24 was 2.09)")
    print(f"Time: {time.time()-t0:.0f}s")

    bp = expand(es.result.xbest)
    s = SynthPatch()
    p = torch.tensor(bp, dtype=torch.float32)
    g = s.render(p, f0_hz=295.0, duration=0.5, note_duration=0.45)
    sf.write("/tmp/v24_clean_result.wav", g.detach().numpy().squeeze(), 44100)
    print("Saved: /tmp/v24_clean_result.wav")
