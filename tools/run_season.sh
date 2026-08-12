#!/usr/bin/env bash
# Run every storm of a season, restartably.
#
# A whole season is the smallest experiment that can tell whether a change to the
# model helped. One case cannot: Kong-rey's 296 km at +24 h might be the model or
# might be Kong-rey, and after a retrain there is no way to tell which moved.
#
# This exists as a file rather than a command to paste because the loop is
# multi-line, and a shell prompt copied in with it produces a page of syntax
# errors around a run that half-succeeded.
#
# Restartable: a storm whose output directory already exists is skipped, so an
# interrupted sweep resumes where it stopped. Delete the directory to redo one.
#
# Usage
# -----
#   tools/run_season.sh --data-root /wk2/yungyun/FCNV2_TC \
#                       --track-csv /wk2/yungyun/ERA5_2024_for_TC/TC_list_JMA_v2 \
#                       --out ~/LLAT_polar_runs \
#                       --fcnv2-weight /wk2/yungyun/code_space/FCNV2_test/weight \
#                       --mode one-way --hours 120 --max-starts 3
#
# Everything after the recognised options is passed through to
# run_coupled_forecast.py unchanged, so --frame-speed-scale and friends work.
#
# On parallelism: leave it at 1 for one-way. A single process peaked at 9.2 GB of
# a 10 GB card, so a second would not fit, and a CUDA out-of-memory kills the run
# after the minutes already spent. Twenty-nine storms at three initial times take
# about an hour serially, which is not worth risking. --jobs is there for
# standalone mode, which touches no GPU at all.
set -u

JOBS=1
MODE=one-way
HOURS=120
OUT=""
DATA_ROOT=""
PASS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --jobs)       JOBS="$2";      shift 2 ;;
        --mode)       MODE="$2";      PASS+=(--mode "$2");      shift 2 ;;
        --hours)      HOURS="$2";     PASS+=(--hours "$2");     shift 2 ;;
        --out)        OUT="$2";       PASS+=(--out "$2");       shift 2 ;;
        --data-root)  DATA_ROOT="$2"; PASS+=(--data-root "$2"); shift 2 ;;
        *)            PASS+=("$1");   shift ;;
    esac
done

if [ -z "$DATA_ROOT" ] || [ -z "$OUT" ]; then
    echo "need --data-root and --out; see the header of this file" >&2
    exit 2
fi

HERE="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${OUT/#\~/$HOME}"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
LOGS="$OUT/logs"
mkdir -p "$LOGS"

STORMS=$(ls -d "$DATA_ROOT"/*W 2>/dev/null | xargs -n1 basename)
if [ -z "$STORMS" ]; then
    echo "no {TC_ID}W directories under $DATA_ROOT" >&2
    exit 2
fi
echo "$(echo "$STORMS" | wc -l) storms, mode $MODE, ${HOURS} h, $JOBS at a time"
echo "logs in $LOGS"

run_one() {
    tc="$1"
    # Any start_from_* under this storm means the sweep already covered it.
    if compgen -G "$OUT/$tc/"*"/start_from_"* > /dev/null 2>&1; then
        echo "  $tc  already done, skipping"
        return 0
    fi
    if python "$HERE/run_coupled_forecast.py" --tc-id "$tc" "${PASS[@]}" \
            > "$LOGS/$tc.log" 2>&1; then
        echo "  $tc  ok"
    else
        # Keep going. One storm with a missing initial condition should not end
        # a sweep that has already spent half an hour.
        echo "  $tc  FAILED — see $LOGS/$tc.log"
    fi
}
export -f run_one
export OUT HERE HOURS
export PASS_STR="${PASS[*]}"

start=$(date +%s)
RESULTS=$(mktemp)
trap 'rm -f "$RESULTS"' EXIT
if [ "$JOBS" -le 1 ]; then
    for tc in $STORMS; do run_one "$tc"; done | tee "$RESULTS"
else
    # xargs cannot see the array, so the parallel path rebuilds the command.
    echo "$STORMS" | xargs -P "$JOBS" -I{} sh -c '
        if ls -d "$0/{}"/*/start_from_* >/dev/null 2>&1; then
            echo "  {} already done, skipping"
        elif python "$1/run_coupled_forecast.py" --tc-id {} $2 \
                > "$0/logs/{}.log" 2>&1; then
            echo "  {} ok"
        else
            echo "  {} FAILED"
        fi' "$OUT" "$HERE" "$PASS_STR" | tee "$RESULTS"
fi

ok=$(grep -c ' ok$' "$RESULTS" || true)
skipped=$(grep -c 'already done' "$RESULTS" || true)
failed=$(grep -c 'FAILED' "$RESULTS" || true)

echo
echo "finished in $(( ($(date +%s) - start) / 60 )) min: $ok run, $skipped skipped, $failed failed"
if [ "$failed" -gt 0 ]; then
    echo "the failures, and the last line of each log:"
    grep 'FAILED' "$RESULTS" | awk '{print $1}' | while read -r tc; do
        echo "  $tc: $(tail -1 "$LOGS/$tc.log")"
    done
fi
