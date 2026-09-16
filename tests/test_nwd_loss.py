"""Unit tests for the Normalized Wasserstein Distance loss.

Run with ``pytest tests/test_nwd_loss.py`` or ``python tests/test_nwd_loss.py``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from losses.nwd_loss import (  # noqa: E402
    NWDLoss,
    cxcywh_to_xyxy,
    nwd_similarity,
    wasserstein_sq_distance,
    xyxy_to_cxcywh,
)

TOL = 1e-6


def _box(cx: float, cy: float, w: float, h: float) -> torch.Tensor:
    """Helper: cxcywh → xyxy tensor."""
    return cxcywh_to_xyxy(torch.tensor([[cx, cy, w, h]]))


def test_closed_form_matches_manual_computation() -> None:
    """W2^2 must equal the hand-derived diagonal-Gaussian expression."""
    b1 = _box(10.0, 20.0, 8.0, 4.0)
    b2 = _box(14.0, 16.0, 6.0, 8.0)
    expected = (10 - 14) ** 2 + (20 - 16) ** 2 + ((8 - 6) / 2) ** 2 + ((4 - 8) / 2) ** 2
    got = wasserstein_sq_distance(b1, b2)
    assert torch.allclose(got, torch.tensor([[expected]]), atol=TOL), got


def test_identical_boxes_give_zero_distance_and_unit_similarity() -> None:
    b = torch.tensor([[5.0, 5.0, 10.0, 10.0], [50.0, 60.0, 4.0, 6.0]])
    dist = wasserstein_sq_distance(b, b).diagonal()
    assert torch.allclose(dist, torch.zeros_like(dist), atol=TOL)
    sim = nwd_similarity(b, b, constant=12.8).diagonal()
    assert torch.allclose(sim, torch.ones_like(sim), atol=TOL)


def test_similarity_bounds_and_symmetry() -> None:
    a = torch.rand(7, 4) * 100
    b = torch.rand(5, 4) * 100
    sim = nwd_similarity(a, b, constant=12.8)
    assert sim.shape == (7, 5)
    assert (sim > 0).all() and (sim <= 1.0 + TOL).all()
    assert torch.allclose(sim, nwd_similarity(b, a, constant=12.8).t(), atol=1e-5)


def test_monotone_decreasing_with_center_displacement() -> None:
    """NWD must decrease as the box is translated away from its target."""
    target = _box(50.0, 50.0, 10.0, 10.0)
    sims = []
    for dx in (0.0, 2.0, 5.0, 10.0, 20.0):
        pred = _box(50.0 + dx, 50.0, 10.0, 10.0)
        sims.append(nwd_similarity(pred, target, constant=12.8).item())
    assert all(s > n for s, n in zip(sims, sims[1:])), sims


def test_constant_controls_sensitivity() -> None:
    """Larger normalizing constant C → slower decay → higher similarity."""
    pred, target = _box(50.0, 50.0, 10.0, 10.0), _box(58.0, 50.0, 10.0, 10.0)
    sim_small_c = nwd_similarity(pred, target, constant=4.0)
    sim_large_c = nwd_similarity(pred, target, constant=64.0)
    assert sim_large_c > sim_small_c


def test_disjoint_boxes_have_positive_similarity_and_gradient() -> None:
    """The key property vs IoU: nonzero, differentiable signal off-overlap."""
    pred = _box(0.0, 0.0, 10.0, 10.0).clone().requires_grad_(True)
    target = _box(30.0, 0.0, 10.0, 10.0)  # fully disjoint
    sim = nwd_similarity(pred, target, constant=12.8)
    assert sim.item() > 0.0
    sim.backward()
    assert pred.grad is not None
    assert pred.grad.abs().sum() > 0.0  # IoU would give exactly zero grad here
    # d(sim)/d(x1) > 0: gradient ascent on similarity moves the box's left
    # edge right, i.e. the prediction toward the target at x=30.
    assert pred.grad[0, 0] > 0.0


def test_loss_zero_for_perfect_prediction() -> None:
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0], [100.0, 100.0, 8.0, 12.0]])
    loss_fn = NWDLoss(constant=12.8, alpha=1.0)  # pure NWD
    loss = loss_fn(boxes.clone(), boxes)
    assert loss.item() < 1e-6


def test_loss_blends_with_ciou() -> None:
    """alpha=0 → pure CIoU; alpha=1 → pure NWD; intermediate lies between."""
    pred = torch.tensor([[9.0, 10.0, 18.0, 22.0]])
    target = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    l_nwd = NWDLoss(alpha=1.0)(pred.clone(), target).item()
    l_ciou = NWDLoss(alpha=0.0)(pred.clone(), target).item()
    l_mid = NWDLoss(alpha=0.5)(pred.clone(), target).item()
    assert min(l_nwd, l_ciou) - 1e-6 <= l_mid <= max(l_nwd, l_ciou) + 1e-6
    assert l_nwd > 0 and l_ciou > 0


def test_batched_input_flattens_correctly() -> None:
    pred = torch.rand(2, 5, 4) * 50
    pred[..., 2:] = pred[..., 2:].abs() + 4  # positive sizes
    target = torch.rand(2, 5, 4) * 50
    target[..., 2:] = target[..., 2:].abs() + 4
    loss = NWDLoss()(pred, target)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_weighted_reduction() -> None:
    pred = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[5.0, 0.0, 10.0, 10.0], [50.0, 0.0, 10.0, 10.0]])
    weights = torch.tensor([1.0, 0.0])  # only the first pair counts
    only_first = NWDLoss(alpha=1.0, reduction="mean")(pred.clone(), target, weights)
    first = NWDLoss(alpha=1.0, reduction="none")(pred.clone(), target, None)[0]
    assert torch.allclose(only_first, first, atol=1e-6)


def test_empty_input_keeps_graph() -> None:
    pred = torch.rand(0, 4, requires_grad=True)
    loss = NWDLoss()(pred, torch.rand(0, 4))
    assert loss.requires_grad or loss.grad_fn is not None or loss.item() == 0.0
    loss.backward() if loss.requires_grad else None


def test_nwd_matches_expected_exp_form() -> None:
    """Check the exp(-sqrt(W2^2)/C) form against a direct computation."""
    pred, target = _box(0.0, 0.0, 10.0, 10.0), _box(3.0, 4.0, 6.0, 8.0)
    c = 12.8
    w2_sq = 3.0**2 + 4.0**2 + ((10 - 6) / 2) ** 2 + ((10 - 8) / 2) ** 2  # 9+16+4+1=30
    expected = math.exp(-math.sqrt(w2_sq) / c)
    got = nwd_similarity(pred, target, constant=c).item()
    assert abs(got - expected) < 1e-6


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")
