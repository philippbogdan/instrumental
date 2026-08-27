"""Measure the three evaluation paths so the hosting decision rests on numbers.

  pool  the shipped design: multiprocessing.Pool over the scalar CPU synth
  cpu   the orphaned batched synth, whole population in one tensor op
  mps   the same, on the M4 GPU

Run one path per process so peak RSS is attributable:

    python3 bench_batched.py pool|cpu|mps [n_evals]

Correctness first: `python3 bench_batched.py check` renders the same params
through both synths and prints the difference.
"""

import os
import sys
import time
import resource

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SR = 44100
DUR = 0.5          # a note, not a track: what the matcher actually evaluates
F0 = 261.63
N_PARAMS = 28


def peak_rss_mb():
    """Peak RSS of this process and any children it waited on, in MB."""
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return (me + kids) / 1e6      # macOS reports bytes


def target_audio():
    """A fixed synthetic target, so every path chases the same thing."""
    from src.synth_gpu import SynthPatchGPU
    g = torch.Generator().manual_seed(7)
    p = torch.rand(1, N_PARAMS, generator=g)
    return SynthPatchGPU().render(p, f0_hz=F0, duration=DUR)[0], p[0]


def _logmel(x):
    """What the matcher actually compares: magnitude spectra, not waveforms.

    Both synths fill the noise oscillator from an unseeded randn, so two renders
    of one patch never line up sample for sample. Only the spectrum is
    meaningful.
    """
    if x.dim() == 1:
        x = x.unsqueeze(0)
    w = torch.hann_window(1024, device=x.device)
    m = torch.stft(x, n_fft=1024, hop_length=256, window=w,
                   return_complex=True).abs()
    return torch.log(m.clamp(min=1e-6))


def _spec_dist(a, b):
    """Relative spectral distance, 0 is identical."""
    A, B = _logmel(a), _logmel(b)
    n = min(A.shape[-1], B.shape[-1])
    A, B = A[..., :n], B[..., :n]
    return float((A - B).norm() / (B.norm() + 1e-9))


def check():
    from src.synth import SynthPatch
    from src.synth_gpu import SynthPatchGPU
    names = SynthPatchGPU().get_param_names()
    i_noise = [names.index('noise_mix'), names.index('noise_floor')]

    g = torch.Generator().manual_seed(11)
    params = torch.rand(4, N_PARAMS, generator=g)
    silent = params.clone()
    silent[:, i_noise] = 0.0        # isolate the deterministic signal path

    cpu, gpu = SynthPatch(), SynthPatchGPU()
    for label, ps in (('with noise', params), ('noise off', silent)):
        batch = gpu.render(ps, f0_hz=F0, duration=DUR)
        rows = []
        for i in range(ps.shape[0]):
            one = cpu.render(ps[i], f0_hz=F0, duration=DUR)
            rows.append(f'{_spec_dist(batch[i], one):.3f}')
        print(f'  {label:10s} spectral distance batched vs scalar: ' + '  '.join(rows))

    if torch.backends.mps.is_available():
        for label, ps in (('with noise', params), ('noise off', silent)):
            c = gpu.render(ps, f0_hz=F0, duration=DUR)
            m = gpu.render(ps.to('mps'), f0_hz=F0, duration=DUR).cpu()
            print(f'  {label:10s} spectral distance mps vs cpu:      '
                  f'{_spec_dist(m, c):.3f}')


# ---------------------------------------------------------------- pool path

_TARGET = None


def _init(t):
    global _TARGET
    _TARGET = t


def _eval_one(p_np):
    from src.synth import SynthPatch
    from src.losses import get_loss
    synth = SynthPatch()
    loss = get_loss('matching')
    gen = synth.render(torch.tensor(p_np, dtype=torch.float32),
                       f0_hz=F0, duration=DUR)
    gen, tgt = gen.reshape(1, -1), _TARGET.reshape(1, -1)
    n = min(gen.shape[-1], tgt.shape[-1])
    return float(loss(gen[..., :n], tgt[..., :n]))


def run_pool(n_evals):
    import multiprocessing as mp
    mp.set_start_method('fork', force=True)
    tgt, _ = target_audio()
    popsize = max(16, os.cpu_count() * 2)
    gens = max(1, n_evals // popsize)
    rng = np.random.default_rng(3)

    t0 = time.time()
    with mp.Pool(popsize, initializer=_init, initargs=(tgt,)) as pool:
        for _ in range(gens):
            pop = [rng.random(N_PARAMS) for _ in range(popsize)]
            pool.map(_eval_one, pop)
    return time.time() - t0, popsize * gens, popsize


# ------------------------------------------------------------- batched path

def run_batched(n_evals, device):
    from src.synth_gpu import SynthPatchGPU
    from src.batch_loss import BatchedMultiResSTFTLoss
    tgt, _ = target_audio()
    tgt = tgt.to(device)
    synth = SynthPatchGPU()
    loss = BatchedMultiResSTFTLoss(device=device)
    popsize = max(16, os.cpu_count() * 2)
    gens = max(1, n_evals // popsize)
    g = torch.Generator().manual_seed(3)

    t0 = time.time()
    for _ in range(gens):
        pop = torch.rand(popsize, N_PARAMS, generator=g).to(device)
        gen = synth.render(pop, f0_hz=F0, duration=DUR)
        n = min(gen.shape[-1], tgt.shape[-1])
        # BatchedMultiResSTFTLoss expands the target itself, so pass it 1-D.
        losses = loss(gen[..., :n], tgt[..., :n])
        if device == 'mps':
            torch.mps.synchronize()
        float(losses.sum())
    return time.time() - t0, popsize * gens, popsize


if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'check'
    n_evals = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

    if mode == 'check':
        check()
        sys.exit(0)

    if mode == 'pool':
        dt, evals, popsize = run_pool(n_evals)
    else:
        dt, evals, popsize = run_batched(n_evals, 'mps' if mode == 'mps' else 'cpu')

    print(f'{mode:5s} evals={evals} popsize={popsize} '
          f'wall={dt:.1f}s  {evals / dt:.0f} evals/s  peak_rss={peak_rss_mb():.0f}MB')
