"""
Optimization loop for inverse synthesis: find synth parameters that minimize
the perceptual loss between rendered audio and a target audio clip.
"""

import time
from typing import Optional

import torch

from src.synth import SynthPatch
from src.losses import MultiResSTFTLoss


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
