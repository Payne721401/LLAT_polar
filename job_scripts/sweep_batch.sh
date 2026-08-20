#!/bin/bash
# One allocation, every batch size, one node - so node-to-node variation cannot
# get into the comparison.
#
#   sbatch job_scripts/sweep_batch.sh
#   cat runs/sweep_batch/results.txt
#
# The earlier sweep used one job per batch size, which put each on a different
# node: b6 came out at 205 s where the trend says 185, and there was no way to
# tell a real effect from a busy neighbour. Running them in sequence inside a
# single allocation removes that entirely - same GPUs, same interconnect, same
# filesystem contention, back to back.
#
# It also costs less. Seven configurations at 600 steps is about 26 minutes of
# compute against seven separate queue waits and seven start-ups.
#
# Repeats: --repeat 2 or 3 and take the median if the spread still matters. Two
# passes over seven sizes is under an hour.

#SBATCH --account=MST115002
#SBATCH --job-name=sweep_batch
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --time=04:00:00
#SBATCH --output=job_logs/job-%j.out
#SBATCH --error=job_logs/job-%j.err
set -u

BATCHES="${BATCHES:-1 2 4 6 8 12 16 24 32}"
STEPS="${STEPS:-600}"
REPEAT="${REPEAT:-1}"
OUT="runs/sweep_batch"
mkdir -p "$OUT"
RESULTS="$OUT/results.txt"

echo "node $(hostname)   steps $STEPS   repeats $REPEAT" | tee "$RESULTS"
printf '%6s %8s %10s %11s %12s\n' batch pass seconds "per step" "samples/s" | tee -a "$RESULTS"

for pass in $(seq 1 "$REPEAT"); do
  for B in $BATCHES; do
    Y="$OUT/_b${B}.yaml"
    sed -e "s/^  batch_size: .*/  batch_size: $B/" \
        -e "s/^  max_steps: .*/  max_steps: $STEPS/" \
        experiments/bench_batch.yaml > "$Y"

    # A fresh RUNDIR each time: resuming into a previous pass would restore an
    # optimiser state and a step count and measure the wrong thing.
    RD="$OUT/run_b${B}_p${pass}"
    rm -rf "$RD"

    T0=$(date +%s)
    SAVE_TOP_K=1 FRESH=1 srun -n 8 python train.py fit \
        --config config.yaml --config "$Y" \
        --trainer.default_root_dir "$RD" > "$OUT/log_b${B}_p${pass}.txt" 2>&1
    RC=$?
    T1=$(date +%s)
    S=$((T1 - T0))

    if [ $RC -ne 0 ]; then
      # Out of memory is the expected way this ends at large batch, and it is a
      # result rather than a failure - it is where the sweep stops.
      REASON=$(grep -m1 -o 'CUDA out of memory\|OutOfMemoryError' "$OUT/log_b${B}_p${pass}.txt")
      printf '%6s %8s %10s %11s %12s\n' "$B" "$pass" "$S" "-" "${REASON:-FAILED rc=$RC}" | tee -a "$RESULTS"
      continue
    fi

    PS=$(python -c "print(f'{$S/$STEPS:.3f}')")
    SPS=$(python -c "print(f'{$STEPS*$B*8/$S:.0f}')")
    printf '%6s %8s %10s %11s %12s\n' "$B" "$pass" "$S" "$PS" "$SPS" | tee -a "$RESULTS"
    rm -rf "$RD"
  done
done

echo "" | tee -a "$RESULTS"
echo "samples/s is throughput; per step is what decides how many optimiser" | tee -a "$RESULTS"
echo "updates an hour of wall clock buys. They point opposite ways, and which" | tee -a "$RESULTS"
echo "one matters is a question about the model, not about the machine." | tee -a "$RESULTS"
