"""The batched objective must rank candidates the way the scalar one does.

If it does not, the optimiser is climbing a different hill from the one the
server reports, and every match would be tuned to an artefact of the rewrite.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.batch_matching_loss import BatchedMatchingLoss
from src.losses import get_loss
from src.synth_gpu import SynthPatchGPU

F0, DUR, N_PARAMS, B = 261.63, 0.5, 28, 12


def _population(device='cpu'):
    g = torch.Generator().manual_seed(23)
    params = torch.rand(B, N_PARAMS, generator=g)
    synth = SynthPatchGPU()
    target = synth.render(torch.rand(1, N_PARAMS, generator=g),
                          f0_hz=F0, duration=DUR)[0]
    gen = synth.render(params, f0_hz=F0, duration=DUR)
    return gen.to(device), target.to(device)


def _scalar_losses(gen, target):
    loss_fn = get_loss('matching')
    out = []
    for i in range(gen.shape[0]):
        g1 = gen[i].reshape(1, -1)
        t1 = target.reshape(1, -1)
        n = min(g1.shape[-1], t1.shape[-1])
        out.append(float(loss_fn(g1[..., :n], t1[..., :n])))
    return torch.tensor(out)


def test_matches_scalar_per_candidate():
    gen, target = _population()
    batched = BatchedMatchingLoss()(gen, target)
    scalar = _scalar_losses(gen, target)
    assert batched.shape == (B,)
    torch.testing.assert_close(batched, scalar, rtol=2e-3, atol=2e-3)


def test_preserves_ranking():
    """Weaker property, but the one CMA-ES actually depends on."""
    gen, target = _population()
    batched = BatchedMatchingLoss()(gen, target)
    scalar = _scalar_losses(gen, target)
    assert torch.equal(batched.argsort(), scalar.argsort())


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason='no MPS')
def test_mps_agrees_with_cpu():
    gen, target = _population()
    cpu = BatchedMatchingLoss(device='cpu')(gen, target)
    mps = BatchedMatchingLoss(device='mps')(gen.to('mps'), target.to('mps')).cpu()
    torch.testing.assert_close(mps, cpu, rtol=5e-2, atol=5e-2)
    assert torch.equal(mps.argsort(), cpu.argsort())
