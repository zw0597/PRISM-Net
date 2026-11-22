#!/bin/bash
# R8 (reviewer_protocol.md): PRISM-Net with orbit consistency (oc), canonical acquisition (canon) and both; 5 folds, 1000 steps, R4 stack.
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv_mps/bin/python
mkdir -p logs_gpu
jobs=()
for f in 0 1 2 3 4; do for v in oc canon occanon; do jobs+=("$v $f"); done; done
run() { set -- $1; v=$1; f=$2; out=results_r8_$v; t=prismr+paug_group_f${f}_s0
  case $v in oc) opt="--oc 1.0";; canon) opt="--canon";; occanon) opt="--oc 1.0 --canon";; esac
  [ -f $out/$t.json ] && return
  echo "$(date '+%H:%M:%S') start $v $t"
  $PY -W ignore -u train_r8.py --model prismr+paug --protocol group --fold $f --res 80 --lr 2e-3 --bs 64 --iters 1000 --threads 4 \
      --seed 0 --device auto --out $out $opt >> logs_gpu/r8_${v}_f$f.log 2>&1 || echo "FAILED $v $t"
  echo "$(date '+%H:%M:%S') done $v $t $(grep 'TEST ACC' logs_gpu/r8_${v}_f$f.log | tail -1)"; }
export -f run; export PY
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I{} bash -c 'run "{}"'
echo "== R8 DONE =="
