"""
Optimization loop for inverse synthesis: find synth parameters that minimize
the perceptual loss between rendered audio and a target audio clip.
"""

import time
from typing import List, Optional

import numpy as np
import torch

from src.synth import SynthPatch
from src.losses import MultiResSTFTLoss, get_loss
from src.spectral_init import spectral_init
from src.cmaes_search import cmaes_search


def match_sound(
    target_audio: torch.Tensor,
    synth: SynthPatch,
    loss_fn,
    f0_hz: float = 440.0,
    n_steps: int = 500,
    lr: float = 0.01,
    ground_truth_params: Optional[torch.Tensor] = None,
) -> dict:
    """
    Optimize synth parameters to match target_audio using gradient descent.

    Args:
        target_audio: (1, n_samples) tensor — the reference sound to match
        synth: SynthPatch instance
        loss_fn: callable(generated, target) -> scalar loss tensor
        f0_hz: fundamental frequency to render at (Hz)
        n_steps: maximum gradient descent steps
        lr: Adam learning rate
        ground_truth_params: optional (n_params,) tensor; if provided, tracks param L2 error

    Returns:
        dict with keys:
          params       - best found params tensor (n_params,)
          loss         - final loss value (float)
          param_l2     - L2 distance to ground truth, or None
          history      - list of {"step", "loss", "param_l2"} logged every 50 steps
          time_seconds - wall-clock time (float)
    """
    target = target_audio.detach()

    # Initialize params in [0, 1]
    params = synth.random_params().requires_grad_(True)
    optimizer = torch.optim.Adam([params], lr=lr)

    best_params = params.detach().clone()
    best_loss = float("inf")
    history = []
    t0 = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        generated = synth.render(params, f0_hz=f0_hz)
        loss = loss_fn(generated, target)

        if torch.isnan(loss):
            # Skip NaN losses (can happen on degenerate param configs)
            with torch.no_grad():
                params.copy_(best_params)
            continue

        loss.backward()

        # Zero out NaN gradients and clip to prevent explosions
        if params.grad is not None:
            params.grad.nan_to_num_(nan=0.0)
            torch.nn.utils.clip_grad_norm_([params], max_norm=10.0)

        optimizer.step()

        # Clamp to valid [0, 1] range after each update
        with torch.no_grad():
            params.clamp_(0.0, 1.0)

        loss_val = loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params.detach().clone()

        # Log every 50 steps
        if step % 50 == 0:
            param_l2 = None
            if ground_truth_params is not None:
                param_l2 = torch.norm(params.detach() - ground_truth_params.detach()).item()
            history.append({"step": step, "loss": loss_val, "param_l2": param_l2})
            print(
                f"  step {step:4d}: loss={loss_val:.6f}"
                + (f"  param_l2={param_l2:.6f}" if param_l2 is not None else "")
            )

        # Early stopping
        if loss_val < 0.001:
            print(f"  Early stop at step {step}: loss={loss_val:.6f}")
            break

    elapsed = time.time() - t0

    # Final param_l2
    final_param_l2 = None
    if ground_truth_params is not None:
        final_param_l2 = torch.norm(best_params - ground_truth_params.detach()).item()

    return {
        "params": best_params,
        "loss": best_loss,
        "param_l2": final_param_l2,
        "history": history,
        "time_seconds": elapsed,
    }


def run_level0_experiment(n_runs: int = 5) -> dict:
    """
    Level 0 experiment: verify that the optimization pipeline works end-to-end.

    Uses a fixed ground-truth parameter set (deterministic seed), renders a target
    audio clip, then runs match_sound n_runs times from different random initializations.

    Returns:
        dict with keys:
          all_losses     - list of final losses (one per run)
          all_param_l2s  - list of final param L2 errors
          all_times      - list of wall-clock times
          all_converged  - list of booleans (True if loss < 0.01)
    """
    synth = SynthPatch()
    loss_fn = MultiResSTFTLoss()

    # Deterministic ground truth
    torch.manual_seed(42)
    gt_params = synth.random_params()
    target_audio = synth.render(gt_params).detach()

    print(f"Level 0 experiment: {n_runs} runs, {synth.get_param_count()} params")
    print(f"Target audio shape: {target_audio.shape}")

    all_losses = []
    all_param_l2s = []
    all_times = []
    all_converged = []

    for run_idx in range(n_runs):
        print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
        result = match_sound(
            target_audio,
            synth,
            loss_fn,
            f0_hz=440.0,
            n_steps=500,
            lr=0.01,
            ground_truth_params=gt_params,
        )
        all_losses.append(result["loss"])
        all_param_l2s.append(result["param_l2"])
        all_times.append(result["time_seconds"])
        all_converged.append(result["loss"] < 0.01)
        print(
            f"  Run {run_idx + 1} done: loss={result['loss']:.6f} "
            f"param_l2={result['param_l2']:.6f} "
            f"time={result['time_seconds']:.1f}s "
            f"converged={result['loss'] < 0.01}"
        )

    print(f"\n=== Level 0 Summary ===")
    print(f"  Converged: {sum(all_converged)}/{n_runs}")
    print(f"  Mean loss: {sum(all_losses)/len(all_losses):.6f}")
    print(f"  Mean param L2: {sum(all_param_l2s)/len(all_param_l2s):.6f}")
    print(f"  Mean time: {sum(all_times)/len(all_times):.1f}s")

    return {
        "all_losses": all_losses,
        "all_param_l2s": all_param_l2s,
        "all_times": all_times,
        "all_converged": all_converged,
    }


def match_sound_v2(
    target_notes: List[dict],
    synth: SynthPatch,
    loss_fn,
    init_params: Optional[torch.Tensor] = None,
    n_steps: int = 500,
    lr: float = 0.02,
) -> dict:
    """
    Improved optimization with cosine-annealing LR, gradient noise, and multi-pitch loss.

    Args:
        target_notes: list of dicts {'audio': torch.Tensor, 'freq': float}
        synth: SynthPatch instance
        loss_fn: callable(generated, target) -> scalar loss tensor
        init_params: optional starting params tensor (n_params,); random if None
        n_steps: gradient descent steps
        lr: initial Adam learning rate

    Returns:
        dict with keys: params, loss, history, time_seconds
    """
    if init_params is not None:
        params = init_params.clone().detach().float().requires_grad_(True)
    else:
        params = synth.random_params().requires_grad_(True)

    optimizer = torch.optim.Adam([params], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=0.001
    )

    # Pre-process targets to 3D for loss_fn
    prepared_targets = []
    for note in target_notes:
        audio = note["audio"]
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # (1, T)
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)  # (1, 1, T)
        prepared_targets.append({"audio": audio.detach(), "freq": note["freq"]})

    best_params = params.detach().clone()
    best_loss = float("inf")
    history = []
    t0 = time.time()

    for step in range(n_steps):
        optimizer.zero_grad()

        # Multi-pitch loss: sum across all target notes
        total_loss = torch.tensor(0.0)
        for target in prepared_targets:
            duration = target["audio"].shape[-1] / synth.sr
            generated = synth.render(params, f0_hz=target["freq"], duration=duration)
            # Ensure 3D
            if generated.dim() == 1:
                generated = generated.unsqueeze(0).unsqueeze(0)
            elif generated.dim() == 2:
                generated = generated.unsqueeze(0)
            total_loss = total_loss + loss_fn(generated, target["audio"])

        if torch.isnan(total_loss):
            with torch.no_grad():
                params.copy_(best_params)
            continue

        total_loss.backward()

        # Gradient noise for exploration
        if params.grad is not None:
            params.grad.nan_to_num_(nan=0.0)
            noise = torch.randn_like(params.grad) * 0.01 / (1 + step) ** 0.55
            params.grad.add_(noise)
            torch.nn.utils.clip_grad_norm_([params], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            params.clamp_(0.0, 1.0)

        loss_val = total_loss.item()

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params.detach().clone()

        if step % 50 == 0:
            history.append({"step": step, "loss": loss_val})
            print(f"  step {step:4d}: loss={loss_val:.6f}")

        if loss_val < 0.001:
            print(f"  Early stop at step {step}: loss={loss_val:.6f}")
            break

    elapsed = time.time() - t0

    return {
        "params": best_params,
        "loss": best_loss,
        "history": history,
        "time_seconds": elapsed,
    }


def full_pipeline(
    target_notes: List[dict],
    synth: SynthPatch,
    n_cmaes: int = 3000,
    n_adam: int = 500,
) -> dict:
    """
    Full CMA-ES -> Adam hybrid optimization pipeline.

    Step 1: Spectral init on the first note's audio -> init_params
    Step 2: CMA-ES global search with init_params -> cmaes_params
    Step 3: Adam refinement with match_sound_v2 -> final result

    Args:
        target_notes: list of dicts {'audio': torch.Tensor, 'freq': float}
        synth: SynthPatch instance
        n_cmaes: CMA-ES function evaluations
        n_adam: Adam gradient descent steps

    Returns:
        dict with keys: params, loss, history, time_seconds,
                        init_params, cmaes_params
    """
    loss_fn = get_loss("matching")
    pipeline_t0 = time.time()

    # --- Step 1: Spectral init ---
    t0 = time.time()
    first_audio = target_notes[0]["audio"]
    if first_audio.dim() > 1:
        first_audio_np = first_audio.squeeze().numpy()
    else:
        first_audio_np = first_audio.numpy()
    init_params = spectral_init(first_audio_np, sr=synth.sr)
    t1 = time.time()
    print(f"[Pipeline] Step 1 - Spectral init: {t1 - t0:.1f}s")

    # --- Step 2: CMA-ES global search ---
    t0 = time.time()
    cmaes_result = cmaes_search(
        target_notes,
        synth,
        loss_fn,
        n_evals=n_cmaes,
        init_params=init_params.numpy(),
    )
    cmaes_params = cmaes_result["params"]
    t1 = time.time()
    print(f"[Pipeline] Step 2 - CMA-ES: {t1 - t0:.1f}s (loss={cmaes_result['loss']:.6f})")

    # --- Step 3: Adam refinement ---
    t0 = time.time()
    result = match_sound_v2(
        target_notes,
        synth,
        loss_fn,
        init_params=cmaes_params,
        n_steps=n_adam,
    )
    t1 = time.time()
    print(f"[Pipeline] Step 3 - Adam: {t1 - t0:.1f}s (loss={result['loss']:.6f})")

    total_time = time.time() - pipeline_t0
    print(f"[Pipeline] Total: {total_time:.1f}s")

    result["init_params"] = init_params
    result["cmaes_params"] = cmaes_params
    result["time_seconds"] = total_time

    return result
