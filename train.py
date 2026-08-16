# from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, DeviceStatsMonitor
import os
import time
import torch
import lightning.__version__ as lightning_version
import lightning.pytorch as pl
from lightning.pytorch.cli import LightningCLI
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from models.lightning_modules import PanguLightningModule
from utils.data_modules import ERA5TCDataModule
from typing_extensions import override

import torch

print(f"pytorch {torch.__version__}")
print(f"lightning {lightning_version}")

# torch.multiprocessing.set_sharing_strategy('file_system')
torch.set_float32_matmul_precision("high")


class ElapsedTimeCallback(Callback):
    @override
    def on_train_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Record the start time at the beginning of training
        self.start_time = time.time()

    @override
    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        # Calculate elapsed time in seconds since training start
        elapsed = time.time() - self.start_time
        # Log the elapsed time (in seconds) as a metric;
        # this will make it available for checkpoint filename formatting
        pl_module.log("elapsed_time_totals", elapsed, prog_bar=False, logger=False)
        # (Optional) If you want to log minutes and seconds separately:
        days, seconds = divmod(int(elapsed), 86400)
        hours = seconds / 3600
        pl_module.log("elapsed_time_days", days, logger=False)
        pl_module.log("elapsed_time_hours", hours, logger=False)


def allow_numpy_scalars_in_checkpoints():
    """Let a resume actually load the checkpoint it is resuming from.

    LightningCLI reads the checkpoint before training starts, to resolve the
    config, and does it with `torch.load(..., weights_only=True)`. Since PyTorch
    2.4 that refuses anything not on an allowlist, and a Lightning checkpoint
    contains numpy scalars - the logged metrics, `val_loss` among them - so the
    load raises `UnpicklingError: Unsupported global: numpy._core.multiarray.
    scalar` and the job dies in twenty-six seconds.

    This only ever bites on resume. A fresh run writes checkpoints happily and
    never reads one back, which is why the path went unexercised until a job had
    to be continued.

    The module moved from `numpy.core` to `numpy._core` in numpy 2, so both are
    tried. `dtype` and the float dtypes usually come up next once the scalar is
    allowed, so they are added in the same pass rather than one failed job at a
    time. Allowlisting is narrower than TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1, which
    would disable the check for every load in the process.
    """
    import numpy as np

    wanted = []
    for path in ("_core.multiarray.scalar", "core.multiarray.scalar"):
        obj = np
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
        except AttributeError:
            continue
        wanted.append(obj)
    for name in ("dtype",):
        wanted.append(getattr(np, name))
    for name in ("Float64DType", "Float32DType", "Int64DType"):
        dt = getattr(getattr(np, "dtypes", None), name, None)
        if dt is not None:
            wanted.append(dt)

    add = getattr(torch.serialization, "add_safe_globals", None)
    if add is not None and wanted:
        add(list(dict.fromkeys(wanted)))


def main():
    allow_numpy_scalars_in_checkpoints()

    # A checkpoint here is 305 MB - 24.1 M parameters plus two Adam moments - and
    # one is written every epoch, which at 455 steps is under two minutes. Keeping
    # thirty of them is 9.2 GB per run, and a run that fills the quota dies
    # mid-write: the file is truncated, the exception is a disk error rather than
    # anything about the model, and forty-seven minutes of healthy training is
    # discarded with nothing in the log to say why. That is how job 257976 ended.
    #
    # Settable from the environment so a job script can lower it without editing
    # code, and defaulting to the old value so no existing run changes silently.
    save_top_k = int(os.environ.get("SAVE_TOP_K", "30"))
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=save_top_k,
        save_last='link',
        filename="{elapsed_time_days:02.0f}d-{elapsed_time_hours:04.1f}h-vl{val_loss:9.7f}-e{epoch}-s{step}",
        auto_insert_metric_name=False,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    # device_stats_monitor = DeviceStatsMonitor(cpu_stats=True)

    cli = LightningCLI(
        model_class=PanguLightningModule,
        datamodule_class=ERA5TCDataModule,
        trainer_defaults={
            "profiler": "pytorch",
            "callbacks": [ElapsedTimeCallback(), checkpoint_callback, lr_monitor],
            # "callbacks": [checkpoint_callback, lr_monitor, device_stats_monitor]
        },
    )


if __name__ == "__main__":
    main()
