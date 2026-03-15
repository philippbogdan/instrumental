"""
Level 0 experiment runner.

Runs the inverse synthesis pipeline and evaluates against Level 0 criteria:
  - STFT loss < 0.01 on all runs
  - Param L2 < 0.05 on all runs
  - Time < 60s on all runs
  - All runs converged
"""

import os

import soundfile as sf
import torch

from src.optimize import run_level0_experiment
from src.synth import SynthPatch
from src.losses import MultiResSTFTLoss


def main():
    os.makedirs("output", exist_ok=True)

    # Run the experiment
    results = run_level0_experiment(n_runs=5)

    all_losses = results["all_losses"]
    all_param_l2s = results["all_param_l2s"]
    all_times = results["all_times"]
    all_converged = results["all_converged"]

    # Save audio samples from the last run for reference
    synth = SynthPatch()
    torch.manual_seed(42)
    gt_params = synth.random_params()
    target_audio = synth.render(gt_params).detach()

    loss_fn = MultiResSTFTLoss()
    from src.optimize import match_sound
    result = match_sound(
        target_audio,
        synth,
        loss_fn,
        f0_hz=440.0,
        n_steps=500,
        lr=0.01,
        ground_truth_params=gt_params,
    )
    matched_audio = synth.render(result["params"]).detach()

    # Save as WAV via soundfile
    target_np = target_audio.squeeze(0).numpy()
    matched_np = matched_audio.squeeze(0).numpy()
    sf.write("output/target.wav", target_np, 44100)
    sf.write("output/matched.wav", matched_np, 44100)
    print("\nSaved output/target.wav and output/matched.wav")

    # Evaluate Level 0 criteria
    print("\n=== Level 0 Criteria Evaluation ===")

    n_runs = len(all_losses)

    criterion_loss = all(l < 0.01 for l in all_losses)
    criterion_param_l2 = all(
        (p < 0.05 if p is not None else False) for p in all_param_l2s
    )
    criterion_time = all(t < 60.0 for t in all_times)
    criterion_converged = sum(all_converged) == n_runs

    print(f"  STFT loss < 0.01 on all {n_runs} runs:   {'PASS' if criterion_loss else 'FAIL'}")
    for i, l in enumerate(all_losses):
        print(f"    Run {i+1}: {l:.6f}")

    print(f"  Param L2 < 0.05 on all {n_runs} runs:    {'PASS' if criterion_param_l2 else 'FAIL'}")
    for i, p in enumerate(all_param_l2s):
        val = f"{p:.6f}" if p is not None else "N/A"
        print(f"    Run {i+1}: {val}")

    print(f"  Time < 60s on all {n_runs} runs:          {'PASS' if criterion_time else 'FAIL'}")
    for i, t in enumerate(all_times):
        print(f"    Run {i+1}: {t:.1f}s")

    print(f"  {n_runs}/{n_runs} converged:                     {'PASS' if criterion_converged else 'FAIL'}")
    for i, c in enumerate(all_converged):
        print(f"    Run {i+1}: {'converged' if c else 'NOT converged'}")

    overall = all([criterion_loss, criterion_param_l2, criterion_time, criterion_converged])
    print(f"\n=== Overall Level 0: {'PASS' if overall else 'FAIL'} ===")


if __name__ == "__main__":
    main()
