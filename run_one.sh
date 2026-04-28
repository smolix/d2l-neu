#!/bin/bash
GPU=$1
VENV=$2
NB=$3
LABEL=$4
cd /home/smola/d2l/d2l-neu
ABS_VENV="/home/smola/d2l/d2l-neu/$VENV"
export LD_LIBRARY_PATH="$(find $ABS_VENV/lib -path "*nvidia/*/lib" -type d 2>/dev/null | paste -sd: -):$LD_LIBRARY_PATH"
START=$(date +%s)
CUDA_VISIBLE_DEVICES=$GPU $ABS_VENV/bin/jupyter nbconvert --to notebook --execute --inplace "$NB" --ExecutePreprocessor.timeout=2400 > /tmp/nbrun_$LABEL.log 2>&1
RC=$?
END=$(date +%s)
ELAPSED=$((END - START))
echo "$LABEL rc=$RC elapsed=${ELAPSED}s" | tee -a /tmp/nbrun_summary.log
