"""The WSD plateau must not move when the decay is rescheduled.

That invariance is the entire reason to use this schedule instead of cosine.
A stable phase of unknown length only works if raising decay_start and
resuming reproduces every learning rate the checkpoint was already trained
with; if the plateau shifted, a resumed run would silently replay a different
schedule than the one that produced its weights, and the loss curve either
side of the switch would not be comparable.

The last test here is the negative control: it pins down what switching to
`cosine` actually does at the same point, because that is the mistake this
schedule exists to prevent and a number is more convincing than a warning.
"""
import numpy as np
import pytest
import torch

from utils.lr_scheduler import get_scheduler_with_warmup

BASE_LR = 5e-5
WARMUP = 1000
TOTAL = 200000


def rates(schedule_type, steps, **kwargs):
    """The learning rate this schedule produces at each step in `steps`.

    Stepping the scheduler one step at a time would take 200,000 iterations
    per case; LambdaLR's lambda is a pure function of the step counter, so
    setting last_epoch and reading the rate back is the same computation.
    """
    out = []
    for s in steps:
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([p], lr=BASE_LR)
        # LambdaLR multiplies by initial_lr, which it takes from the optimizer
        # on construction, so a fresh optimizer per point keeps them independent.
        opt.param_groups[0]["initial_lr"] = BASE_LR
        sch = get_scheduler_with_warmup(
            opt, warmup_steps=WARMUP, training_steps=TOTAL,
            schedule_type=schedule_type, last_epoch=s - 1, **kwargs)
        out.append(sch.get_last_lr()[0])
    return np.array(out)


def test_plateau_is_independent_of_decay_start():
    """Two different decay points give identical rates before either of them."""
    steps = [0, 1, 500, WARMUP, 5000, 50000, 100000, 149999]
    early = rates("wsd", steps, decay_start=150000)
    late = rates("wsd", steps, decay_start=190000)
    never = rates("wsd", steps, decay_start=-1)
    np.testing.assert_allclose(early, late, rtol=0, atol=0)
    np.testing.assert_allclose(early, never, rtol=0, atol=0)


def test_plateau_matches_constant_warmup_exactly():
    """Before the decay, wsd IS constant_warmup - including through warmup.

    A run started under constant_warmup and continued under wsd must not see
    a discontinuity, since that is how the stable phase will actually be
    extended in practice.
    """
    steps = [0, 1, 499, WARMUP - 1, WARMUP, WARMUP + 1, 20000, 120000]
    np.testing.assert_allclose(
        rates("wsd", steps, decay_start=150000),
        rates("constant_warmup", steps),
        rtol=0, atol=0)


def test_decay_is_continuous_and_reaches_zero():
    """Full rate at the switch, no cliff; zero at the end of the budget."""
    start = 150000
    at_switch = rates("wsd", [start], decay_start=start)[0]
    assert at_switch == pytest.approx(BASE_LR, rel=1e-12), (
        "the first step of the decay must still be at the plateau value")

    curve = rates("wsd", list(range(start, TOTAL + 1, 2500)), decay_start=start)
    assert np.all(np.diff(curve) <= 0), "the decay must be monotonic"
    assert curve[-1] == pytest.approx(0.0, abs=1e-12)
    # Halfway through the decay a half-cosine is at half amplitude.
    mid = rates("wsd", [(start + TOTAL) // 2], decay_start=start)[0]
    assert mid == pytest.approx(BASE_LR * 0.5, rel=1e-3)


def test_beyond_the_budget_clamps_at_zero():
    """estimated_stepping_batches can end up below a resumed step counter."""
    over = rates("wsd", [TOTAL + 1, TOTAL + 50000], decay_start=150000)
    assert np.all(over >= 0.0)
    assert np.all(over <= 1e-12)


def test_switching_to_cosine_instead_is_a_cliff():
    """The negative control: what `cosine` does at the same switch point.

    cosine_decay places the restored step counter on a curve that began at
    warmup_steps, so resuming a 150,000-step run under cosine does not start
    a decay - it jumps to wherever that curve already is, about a seventh of
    the base rate. This test exists so that number is written down.
    """
    cliff = rates("cosine", [150000])[0]
    assert cliff == pytest.approx(BASE_LR * 0.146, rel=0.02)
    assert cliff < rates("wsd", [150000], decay_start=150000)[0] / 6
