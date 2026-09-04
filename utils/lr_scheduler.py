import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def get_scheduler_with_warmup(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    training_steps: int,
    cycles: float = 0.5,
    last_epoch: int = -1,
    schedule_type: str = "cosine",
    decay_start: int = -1,
):
    def cosine_decay(current_step):
        # Warmup
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        # decadence
        progress = (current_step - warmup_steps) / max(1, training_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * float(cycles) * 2.0 * progress)))

    def constant(current_step):
        return 1.0

    def constant_warmup(current_step):
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        return 1.0

    def wsd(current_step):
        """Warmup, then a constant plateau, then a cosine decay to zero.

        The plateau's value does not depend on where the decay will start, so
        the total budget can be chosen after the run is already going: raise
        decay_start, resume, and every learning rate the run has already used
        is reproduced exactly and the curve is continuous across the switch.
        That is the only reason to prefer this over `cosine`.

        Switching a constant_warmup run to `cosine` instead does NOT do this.
        cosine_decay reads the step counter Lightning restores from the
        checkpoint and places it on a curve that began at warmup_steps, so a
        switch at step 150,000 with training_steps 200,000 lands at 0.146 of
        the base rate on the first step after the switch. That is a cliff, and
        the decay phase this schedule exists to provide never happens.

        decay_start below zero means "no decay yet" and the schedule is then
        identical to constant_warmup, which is what a stable phase of unknown
        length needs.
        """
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        start = training_steps if decay_start < 0 else decay_start
        if current_step < start:
            return 1.0
        progress = (current_step - start) / max(1, training_steps - start)
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * min(1.0, progress))))

    match schedule_type:
        case "cosine":
            return LambdaLR(optimizer, cosine_decay, last_epoch)
        case "constant":
            return LambdaLR(optimizer, constant, last_epoch)
        case "constant_warmup":
            return LambdaLR(optimizer, constant_warmup, last_epoch)
        case "wsd":
            return LambdaLR(optimizer, wsd, last_epoch)
        case _:
            raise ValueError(f"Unsupported schedule type: {schedule_type}")
