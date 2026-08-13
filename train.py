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


def main():
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
