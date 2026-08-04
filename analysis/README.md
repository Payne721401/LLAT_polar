# analysis/

Post-run analysis artefacts. One report per training run, named
`YYYY-MM-DD_job<JobID>_report.md`.

| File | Contents |
|---|---|
| `2026-08-04_job233052_report.md` | Run report: throughput, LR annealing, gradient-explosion diagnosis, presentation outline, inference defect list |
| `training_analysis.ipynb` | Four diagnostic figures plus a summary table; re-runnable |
| `run_mine_curves.csv` | Main curves for that run |
| `run_mine_per_var.csv` | Per-variable and per-layer metrics for that run |
| `run_baseline_curves.csv` | Y.-Y. Cheng's baseline run, for comparison |

Raw TensorBoard event files (~80 MB each) are **not** version-controlled; see
`.gitignore`. Only the extracted CSVs, the notebook and the reports are tracked.

---

## Extracting the CSVs

Event files are append-only, so this works **while training is still running** — there
is no need to wait for a job to finish:

```bash
# cwd = repository root
python tools/events_to_csv.py runs/<experiment>/lightning_logs/version_0 --all
```

This writes `curves.csv` (main curves, including `gradient_2norm`) and `per_var.csv`
(per-variable `val_RMSE` / `val_norm_L1`, plus per-layer `grad_2.0_norm/*`) next to the
event file. Both are a few hundred KB, so they `scp` in seconds where the event file
does not.

`tools/events_to_csv.py` parses the protobuf records directly and **does not require
tensorboard to be installed**.

---

## Running the notebook

Use the `ty-dev` environment (local CPU). The notebook reads its CSVs by relative path,
so the working directory must be this one:

```bash
cd analysis
jupyter lab training_analysis.ipynb
```

To analyse a new run, copy its `curves.csv` / `per_var.csv` here under a new name and
edit the filename constants — they are collected in the first cell.
