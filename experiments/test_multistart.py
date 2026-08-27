"""Test multi-start CMA-ES on v24's clean targets."""
import sys, os, torch, numpy as np, soundfile as sf, time, cma, multiprocessing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from synth import SynthPatch, PARAM_DEFS
from losses import MatchingLoss, get_loss
from spectral_init import spectral_init

N_PARAMS = len(PARAM_DEFS)
_WORKER_TARGET_NOTES = []

def _evaluate_single(params_np):
    synth = SynthPatch()
    loss_fn = get_loss("matching")
    params = torch.tensor(params_np, dtype=torch.float32)
    total = 0.0
    for audio_np, freq, dur in _WORKER_TARGET_NOTES:
        target = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0)
        gen = synth.render(params, f0_hz=freq, duration=dur, note_duration=dur * 0.9)
        ml = min(target.shape[1], gen.shape[1])
        loss = loss_fn(gen[:, :ml].unsqueeze(0), target[:, :ml].unsqueeze(0))
        total += loss.item()
    return total / len(_WORKER_TARGET_NOTES)

def _init_worker(target_notes):
    global _WORKER_TARGET_NOTES
    _WORKER_TARGET_NOTES = target_notes

if __name__ == "__main__":
    multiprocessing.set_start_method("fork", force=True)

    TARGET_NOTES = []
    for f, freq in [("notes/note_03_A3_221Hz.wav", 221.0), ("notes/note_02_Cs4_278Hz.wav", 278.0), ("notes/note_01_D4_295Hz.wav", 295.0)]:
        audio, sr = sf.read(f)
        TARGET_NOTES.append((audio.astype(np.float32), freq, len(audio)/44100))
        print(f"  {f}: {len(audio)/44100:.3f}s")

    # Spectral init
    init_15 = spectral_init(TARGET_NOTES[0][0], 44100)
    x0 = np.full(N_PARAMS, 0.5, dtype=np.float32)
    for i in range(15): x0[i] = float(init_15[i])

    # Create 3 diverse starts
    starts = [x0.copy()]

    s1 = x0.copy()
    s1[0] = 0.9; s1[1] = 0.1; s1[2] = 0.05; s1[5] = 0.9
    starts.append(s1)

    s2 = x0.copy()
    s2[0] = 0.1; s2[1] = 0.9; s2[2] = 0.05; s2[5] = 0.9
    starts.append(s2)

    n_cores = max(1, multiprocessing.cpu_count() - 1)
    evals_per = 3333

    print(f"\n=== Multi-Start CMA-ES: 3 starts x {evals_per} evals ===")
    t0 = time.time()
    results = []

    with multiprocessing.Pool(n_cores, initializer=_init_worker, initargs=(TARGET_NOTES,)) as pool:
        for i, start_x0 in enumerate(starts):
            labels = ["spectral-init", "saw-heavy", "square-heavy"]
            es = cma.CMAEvolutionStrategy(start_x0.tolist(), 0.15, {
                "bounds": [[0]*N_PARAMS, [1]*N_PARAMS],
                "maxfevals": evals_per, "popsize": 20, "verbose": -9,
            })
            best = float("inf")
            while not es.stop():
                sols = es.ask()
                fits = pool.map(_evaluate_single, sols)
                es.tell(sols, fits)
                if min(fits) < best: best = min(fits)

            bp = es.result.xbest
            results.append((i, best, bp))
            print(f"  Start {i} ({labels[i]}): loss={best:.4f}  "
                  f"saw={bp[0]:.2f} sq={bp[1]:.2f} sine={bp[2]:.2f} "
                  f"cutoff={bp[5]:.2f} gain={bp[11]:.2f}")

    results.sort(key=lambda r: r[1])
    best_idx, best_loss, best_params = results[0]
    labels = ["spectral-init", "saw-heavy", "square-heavy"]
    print(f"\n  WINNER: Start {best_idx} ({labels[best_idx]}), loss={best_loss:.4f}")
    print(f"  Time: {time.time()-t0:.0f}s")
    print(f"  Target: v24 = 2.09, v24 re-eval = 2.40")
