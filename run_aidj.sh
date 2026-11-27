#!/bin/zsh
# AIDJ launcher: ./run_aidj.sh [extra mac_realtime.py options]
DIR="$HOME/projects/StreamDiffusion"
PY="$DIR/.venv/bin/python"
LOG="$DIR/aidj_last.log"

cd "$DIR" || { echo "AIDJ folder not found: $DIR"; exit 1; }
[ -x "$PY" ] || { echo "venv python not found: $PY"; exit 1; }

# 前回の AIDJ が残っていると MIDI ポートが掴まれたままになるので止める
OLD=$(pgrep -f "mac_realtime.py")
if [ -n "$OLD" ]; then
  echo "stopping old AIDJ ($OLD)"
  kill $OLD 2>/dev/null; sleep 2; kill -9 $OLD 2>/dev/null
fi

# conda 等の環境が混ざらないようにする
unset PYTHONPATH PYTHONHOME CONDA_PREFIX
export PYTORCH_ENABLE_MPS_FALLBACK=1

echo "starting AIDJ (log: $LOG)"
"$PY" mac_realtime.py "$@" 2>&1 | tee "$LOG"
echo "AIDJ exited. log: $LOG"
