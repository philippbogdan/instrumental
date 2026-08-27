"""Real-world-shaped load against the matching pipeline, measured on this Mac.

Not a synthetic microbenchmark: arrivals are Poisson with a launch-day burst,
each visitor runs the work a real /api/match-sequence does, and the heavy
stages are serialised by the same two locks the server uses. What comes out is
what a visitor would feel (queue wait, total latency) and what the machine
would feel (system memory over baseline, sampled throughout).

    python3 loadsim.py --users 12 --window 300 --evals 10000 --backend mps

Backends:
    pool  the shipped design, multiprocessing over the scalar synth
    cpu   batched synth, one process
    mps   batched synth on the M4 GPU

--separate runs real Demucs on the clip; without it the run measures the
optimiser alone.
"""

import argparse
import os
import queue
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 44100
DUR = 0.5
F0 = 261.63
N_PARAMS = 28
CLIP = '/tmp/clip30.wav'   # 30s stereo, the length a Deezer preview gives

# One at a time, exactly as server.py does it.
demucs_lock = threading.Lock()
cmaes_lock = threading.Lock()


# ----------------------------------------------------------------- sampling

def used_mb():
    """Memory actually committed on this machine, in MB.

    Sums active, wired and compressed pages from vm_stat rather than adding up
    per-process RSS, because forked workers share pages copy-on-write and
    summing RSS double counts them badly.
    """
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    page = 16384
    vals = {}
    for line in out.splitlines():
        if ':' not in line:
            continue
        k, v = line.split(':', 1)
        v = v.strip().rstrip('.')
        if v.isdigit():
            vals[k.strip()] = int(v)
    pages = (vals.get('Pages active', 0) + vals.get('Pages wired down', 0)
             + vals.get('Pages occupied by compressor', 0))
    return pages * page / 1e6


class Sampler(threading.Thread):
    def __init__(self, period=0.25):
        super().__init__(daemon=True)
        self.period, self.stop_flag = period, threading.Event()
        self.baseline = used_mb()
        self.peak = self.baseline
        self.trace = []

    def run(self):
        while not self.stop_flag.is_set():
            m = used_mb()
            self.peak = max(self.peak, m)
            self.trace.append(m)
            time.sleep(self.period)

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=2)
        return self.peak - self.baseline


# ------------------------------------------------------------------- stages

def separate(path):
    """Stem separation, as the server runs it: a subprocess that then exits."""
    with demucs_lock:
        out = tempfile.mkdtemp(prefix='demucs-')
        t0 = time.time()
        subprocess.run(
            [sys.executable, '-m', 'demucs', '-d', 'mps', '--two-stems', 'other',
             '-n', 'htdemucs', '--segment', '7', '-o', out, path],
            capture_output=True)
        return time.time() - t0


def match_batched(target, n_evals, device):
    from src.synth_gpu import SynthPatchGPU
    from src.batch_loss import BatchedMultiResSTFTLoss
    import cma
    synth = SynthPatchGPU()
    loss = BatchedMultiResSTFTLoss(device=device)
    tgt = target.to(device)
    popsize = max(16, os.cpu_count() * 2)

    with cmaes_lock:
        t0 = time.time()
        es = cma.CMAEvolutionStrategy(
            [0.5] * N_PARAMS, 0.25,
            {'bounds': [0, 1], 'popsize': popsize, 'maxfevals': n_evals,
             'verbose': -9})
        while not es.stop():
            sols = es.ask()
            pop = torch.tensor(np.array(sols), dtype=torch.float32, device=device)
            gen = synth.render(pop, f0_hz=F0, duration=DUR)
            n = min(gen.shape[-1], tgt.shape[-1])
            fit = loss(gen[..., :n], tgt[..., :n])
            if device == 'mps':
                torch.mps.synchronize()
            es.tell(sols, fit.cpu().tolist())
        return time.time() - t0, float(es.result.fbest)


_POOL_TARGET = None


def _pool_init(t):
    global _POOL_TARGET
    _POOL_TARGET = t


def _pool_eval(p_np):
    from src.synth import SynthPatch
    from src.losses import get_loss
    gen = SynthPatch().render(torch.tensor(p_np, dtype=torch.float32),
                              f0_hz=F0, duration=DUR).reshape(1, -1)
    tgt = _POOL_TARGET.reshape(1, -1)
    n = min(gen.shape[-1], tgt.shape[-1])
    return float(get_loss('matching')(gen[..., :n], tgt[..., :n]))


def match_pool(target, n_evals):
    import multiprocessing as mp
    import cma
    popsize = max(16, os.cpu_count() * 2)
    with cmaes_lock:
        t0 = time.time()
        with mp.Pool(popsize, initializer=_pool_init, initargs=(target,)) as pool:
            es = cma.CMAEvolutionStrategy(
                [0.5] * N_PARAMS, 0.25,
                {'bounds': [0, 1], 'popsize': popsize, 'maxfevals': n_evals,
                 'verbose': -9})
            while not es.stop():
                sols = es.ask()
                es.tell(sols, pool.map(_pool_eval, [np.asarray(s) for s in sols]))
        return time.time() - t0, float(es.result.fbest)


# ------------------------------------------------------------------ visitor

def visitor(idx, target, args, results, sampler):
    arrival = time.time()
    stages = {}
    if args.separate:
        waiting = time.time()
        stages['separate'] = separate(CLIP)
        stages['sep_wait'] = time.time() - waiting - stages['separate']

    waiting = time.time()
    if args.backend == 'pool':
        dt, best = match_pool(target, args.evals)
    else:
        dt, best = match_batched(target, args.evals, args.backend)
    stages['match'] = dt
    stages['match_wait'] = time.time() - waiting - dt
    stages['total'] = time.time() - arrival
    stages['best'] = best
    results.put((idx, stages))
    print(f'  visitor {idx:2d}  total {stages["total"]:6.1f}s  '
          f'queued {stages["match_wait"] + stages.get("sep_wait", 0):5.1f}s  '
          f'loss {best:.3f}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--users', type=int, default=12)
    ap.add_argument('--window', type=float, default=300, help='arrival window, seconds')
    ap.add_argument('--evals', type=int, default=10000)
    ap.add_argument('--backend', choices=['pool', 'cpu', 'mps'], default='mps')
    ap.add_argument('--separate', action='store_true', help='run real Demucs per visitor')
    args = ap.parse_args()

    from src.synth_gpu import SynthPatchGPU
    g = torch.Generator().manual_seed(7)
    target = SynthPatchGPU().render(torch.rand(1, N_PARAMS, generator=g),
                                    f0_hz=F0, duration=DUR)[0]

    # Launch-day shape: half the visitors land in the first fifth of the window,
    # the rest trail off behind them.
    rng = random.Random(5)
    burst = args.users // 2
    arrivals = sorted([rng.uniform(0, args.window * 0.2) for _ in range(burst)]
                      + [rng.uniform(args.window * 0.2, args.window)
                         for _ in range(args.users - burst)])

    print(f'{args.users} visitors over {args.window:.0f}s, backend={args.backend}, '
          f'evals={args.evals}, demucs={"on" if args.separate else "off"}')
    print(f'arrivals at: {", ".join(f"{a:.0f}s" for a in arrivals)}')

    sampler = Sampler()
    sampler.start()
    print(f'baseline memory in use: {sampler.baseline / 1000:.2f} GB')

    results, threads = queue.Queue(), []
    t0 = time.time()
    for i, at in enumerate(arrivals):
        delay = at - (time.time() - t0)
        if delay > 0:
            time.sleep(delay)
        t = threading.Thread(target=visitor,
                             args=(i, target, args, results, sampler))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    wall = time.time() - t0
    over_baseline = sampler.stop()

    rows = [r[1] for r in sorted(results.queue)]
    tot = sorted(r['total'] for r in rows)
    wait = sorted(r['match_wait'] + r.get('sep_wait', 0) for r in rows)
    print(f'\nwall {wall:.0f}s for {len(rows)} visitors')
    print(f'total latency   median {statistics.median(tot):6.1f}s   '
          f'worst {tot[-1]:6.1f}s')
    print(f'queue wait      median {statistics.median(wait):6.1f}s   '
          f'worst {wait[-1]:6.1f}s')
    print(f'match compute   median '
          f'{statistics.median([r["match"] for r in rows]):6.1f}s')
    if args.separate:
        print(f'demucs          median '
              f'{statistics.median([r["separate"] for r in rows]):6.1f}s')
    print(f'memory over baseline, peak: {over_baseline:.0f} MB')


if __name__ == '__main__':
    main()
