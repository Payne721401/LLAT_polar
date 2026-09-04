"""tools/chain_state.py, which decides whether a chained GB200 run continues.

The chain in job_scripts/train_gb200.sh submits its successor before it trains,
so the two numbers it reads - the step reached and the step aimed at - are what
stand between "the run continues" and two failures that make no noise. Report
the step too high and the chain stops early, leaving a truncated run that looks
finished. Report it too low, or the target as zero, and the chain either never
stops or stops immediately; the second is what a cumulative max_time already
caused once, twenty-seven seconds into every hop.

Each test here has a negative control: the checkpoint name is broken, the
overlay key removed, the mtimes reversed. A test that passes whether or not the
mechanism is there proves nothing.
"""
import importlib.util
import os
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(ROOT, "tools", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load("chain_state")

# The exact format train.py's ModelCheckpoint produces. If that filename ever
# changes, this constant and the parser have to move together, and this test is
# where the mismatch surfaces.
REAL_NAME = "00d-01.9h-vl0.2722397-e51-s23465.ckpt"


# --------------------------------------------------------------------------
# The step, out of the checkpoint filename
# --------------------------------------------------------------------------

def test_step_read_from_a_real_checkpoint_name(tmp_path):
    p = tmp_path / REAL_NAME
    p.write_bytes(b"")
    assert cs.checkpoint_step(str(p)) == 23465


def test_step_follows_the_last_ckpt_symlink(tmp_path):
    """save_last='link' means last.ckpt is never the file holding the step."""
    real = tmp_path / REAL_NAME
    real.write_bytes(b"")
    link = tmp_path / "last.ckpt"
    try:
        link.symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    assert cs.checkpoint_step(str(link)) == 23465


def test_step_is_none_when_the_name_carries_no_step(tmp_path):
    """Negative control: strip the -s suffix and the parser must not invent one.

    None and 0 are different answers - a fresh run is at zero, an unreadable
    name is unknown - and collapsing them is how a chain would restart a
    finished run or stop a live one.
    """
    p = tmp_path / "00d-01.9h-vl0.2722397-e51.ckpt"
    p.write_bytes(b"")
    assert cs.checkpoint_step(str(p)) is None


def test_step_is_none_for_a_missing_path(tmp_path):
    assert cs.checkpoint_step(str(tmp_path / "nothing.ckpt")) is None
    assert cs.checkpoint_step(None) is None


def test_step_is_not_confused_by_digits_elsewhere_in_the_name(tmp_path):
    """-s must anchor at the end, or 'vl0.27' style fields can be misread."""
    p = tmp_path / "00d-01.9h-s999-vl0.2722397-e51-s23465.ckpt"
    p.write_bytes(b"")
    assert cs.checkpoint_step(str(p)) == 23465


# --------------------------------------------------------------------------
# Which checkpoint, when a resumed run has several version_* directories
# --------------------------------------------------------------------------

def _make_version(rundir, version, name, mtime):
    d = rundir / "lightning_logs" / f"version_{version}" / "checkpoints"
    d.mkdir(parents=True)
    p = d / name
    p.write_bytes(b"")
    link = d / "last.ckpt"
    link.symlink_to(p)
    os.utime(p, (mtime, mtime))
    os.utime(link, (mtime, mtime), follow_symlinks=False)
    return link


def test_newest_checkpoint_is_by_mtime_not_by_version_number(tmp_path):
    """Version directories are named after job ids, which do not sort by age.

    Negative control is built in: the older directory has the HIGHER number, so
    a parser sorting on the name returns the wrong one and this fails.
    """
    try:
        old = _make_version(tmp_path, "999999", "a-e1-s100.ckpt", time.time() - 3600)
        new = _make_version(tmp_path, "111111", "b-e9-s900.ckpt", time.time())
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable on this platform")
    picked = cs.newest_checkpoint(str(tmp_path))
    assert picked == str(new)
    assert cs.checkpoint_step(picked) == 900
    assert cs.checkpoint_step(str(old)) == 100


def test_newest_checkpoint_is_none_on_an_empty_rundir(tmp_path):
    assert cs.newest_checkpoint(str(tmp_path)) is None


# --------------------------------------------------------------------------
# The target, out of config plus overlay
# --------------------------------------------------------------------------

def _yaml(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return str(p)


def test_overlay_max_steps_wins_over_config(tmp_path):
    base = _yaml(tmp_path, "config.yaml", "trainer:\n  max_steps: 105000\n")
    over = _yaml(tmp_path, "over.yaml", "trainer:\n  max_steps: 300000\n")
    assert cs.target_steps([base, over]) == 300000
    # Negative control: reverse the order and the other one wins, so the test
    # is measuring precedence rather than just finding a number.
    assert cs.target_steps([over, base]) == 105000


def test_config_max_steps_survives_an_overlay_that_does_not_set_it(tmp_path):
    base = _yaml(tmp_path, "config.yaml", "trainer:\n  max_steps: 105000\n")
    over = _yaml(tmp_path, "over.yaml", "data:\n  batch_size: 2\n")
    assert cs.target_steps([base, over]) == 105000


def test_target_is_none_when_nothing_sets_max_steps(tmp_path):
    """None, not 0. The job script refuses to chain toward an unknown target."""
    base = _yaml(tmp_path, "config.yaml", "trainer:\n  max_epochs: 3\n")
    assert cs.target_steps([base]) is None


def test_target_handles_an_empty_and_a_trainerless_document(tmp_path):
    empty = _yaml(tmp_path, "empty.yaml", "")
    nomodel = _yaml(tmp_path, "d.yaml", "data:\n  batch_size: 2\n")
    base = _yaml(tmp_path, "config.yaml", "trainer:\n  max_steps: 7\n")
    assert cs.target_steps([empty, nomodel, base]) == 7


# --------------------------------------------------------------------------
# The decision the job script makes from those two numbers
# --------------------------------------------------------------------------

def _decide(step, target, hops):
    """The condition ladder in train_gb200.sh, kept in one place to compare."""
    if target < 0:
        return "stop: no target"
    if hops <= 0:
        return "stop: out of hops"
    if step >= target:
        return "stop: target reached"
    return "chain"


@pytest.mark.parametrize("step,target,hops,expected", [
    (0,      300000, 24, "chain"),             # first segment
    (23465,  300000, 23, "chain"),             # mid run, the real case
    (300000, 300000, 5,  "stop: target reached"),
    (300001, 300000, 5,  "stop: target reached"),
    (-1,     300000, 24, "chain"),             # step unknown: hops bound it
    (-1,     300000, 0,  "stop: out of hops"),
    (0,      -1,     24, "stop: no target"),   # nothing to aim at
])
def test_chain_decision(step, target, hops, expected):
    assert _decide(step, target, hops) == expected


def test_cli_prints_two_integers(tmp_path, capsys):
    """The shell reads this with `read STEP TARGET`, so it must stay two words.

    -1 rather than an empty field for a missing value: an empty word would
    shift the target into the step and the chain would read a step of 300000.
    """
    import subprocess
    import sys
    base = _yaml(tmp_path, "config.yaml", "trainer:\n  max_steps: 300000\n")
    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "chain_state.py"),
         "--rundir", str(tmp_path), "--config", base],
        capture_output=True, text=True, check=True).stdout.split()
    assert len(out) == 2
    assert out == ["-1", "300000"]          # no checkpoint yet, target set

    out = subprocess.run(
        [sys.executable, os.path.join(ROOT, "tools", "chain_state.py"),
         "--rundir", str(tmp_path), "--config", base, "--fresh"],
        capture_output=True, text=True, check=True).stdout.split()
    assert out == ["0", "300000"]           # FRESH=1 means step zero, not unknown
