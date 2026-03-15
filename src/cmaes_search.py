"""
CMA-ES global search over synthesizer parameters.

Uses the Covariance Matrix Adaptation Evolution Strategy to find synth params
that best match target audio. This is a gradient-free optimizer well suited for
the multimodal, non-convex landscape of audio synthesis matching.
"""

import time

import cma
import numpy as np
import torch

from src.synth import SynthPatch


def cmaes_search(
    target_notes,
    synth: SynthPatch,
    loss_fn,
    n_evals: int = 5000,
    init_params=None,
    sigma0: float = 0.3,
    popsize: int = 16,
) -> dict:
    """
    CMA-ES global search over synth params to match target notes.

    Args:
        target_notes: list of dicts with keys 'audio' (torch.Tensor 1D), 'freq' (float Hz)
        synth: SynthPatch instance (from src.synth)
        loss_fn: callable(generated_3d, target_3d) -> scalar
        n_evals: max function evaluations
        init_params: optional numpy array (15,) for starting point. If None, starts at 0.5.
        sigma0: initial step size
        popsize: population size

    Returns:
        dict with 'params' (torch.Tensor), 'loss' (float), 'n_evals' (int), 'time_seconds' (float)
    """
    n_params = synth.get_param_count()
    x0 = init_params if init_params is not None else np.full(n_params, 0.5)
    x0 = np.asarray(x0, dtype=np.float64)

    bounds = [np.zeros(n_params).tolist(), np.ones(n_params).tolist()]

    # Pre-process targets: ensure 3D tensors for loss_fn compatibility
    prepared_targets = []
    for note in target_notes:
        audio = note["audio"]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # (1, 1, T)
        prepared_targets.append({"audio": audio, "freq": note["freq"]})

    eval_count = 0
    t0 = time.time()

    def objective(x):
        nonlocal eval_count
        params = torch.tensor(np.clip(x, 0.0, 1.0), dtype=torch.float32)
        total_loss = 0.0

        with torch.no_grad():
            for target in prepared_targets:
                duration = target["audio"].shape[-1] / synth.sr
                gen = synth.render(params, f0_hz=target["freq"], duration=duration)
                # Ensure 3D: (batch, channels, time)
                if gen.dim() == 1:
                    gen = gen.unsqueeze(0).unsqueeze(0)
                elif gen.dim() == 2:
                    gen = gen.unsqueeze(0)
                total_loss += loss_fn(gen, target["audio"]).item()

        eval_count += 1
        if eval_count % 500 == 0:
            print(f"  CMA-ES eval {eval_count}: loss={total_loss:.6f}")

        return total_loss

    opts = cma.CMAOptions()
    opts.set("bounds", bounds)
    opts.set("maxfevals", n_evals)
    opts.set("popsize", popsize)
    opts.set("verbose", -9)  # suppress CMA-ES internal output

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    while not es.stop():
        solutions = es.ask()
        fitnesses = [objective(s) for s in solutions]
        es.tell(solutions, fitnesses)

    best_params = np.clip(es.result.xbest, 0.0, 1.0)
    best_loss = es.result.fbest
    elapsed = time.time() - t0

    print(f"  CMA-ES finished: loss={best_loss:.6f}, evals={eval_count}, time={elapsed:.1f}s")

    return {
        "params": torch.tensor(best_params, dtype=torch.float32),
        "loss": float(best_loss),
        "n_evals": eval_count,
        "time_seconds": elapsed,
    }
