#!/bin/bash
# R4 on the Apple M5 Max (MPS): PRISM-Net, ResNet-18, VGG-11 x five folds x {1000, 3000} steps; two workers.
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv_mps/bin/python
mkdir -p logs_gpu
jobs=()
for f in 0 1 2 3 4; do for it in 3000 1000; do for m in prismr+paug resnet18 vgg11_bn; do jobs+=("$m $f $it"); done; done; done
run() { set -- $1; m=$1; f=$2; it=$3; out=results_gpu_$it; t=${m}_group_f${f}_s0
  [ -f $out/$t.json ] && return
  echo "$(date '+%H:%M:%S') start $t $it"
  $PY -u train.py --model $m --protocol group --fold $f --res 80 --lr 2e-3 --bs 64 --iters $it --threads 4 --seed 0 \
      --device auto --out $out > logs_gpu/${t}_$it.log 2>&1 || echo "FAILED $t $it"
  echo "$(date '+%H:%M:%S') done $t $it $(grep 'TEST ACC' logs_gpu/${t}_$it.log | tail -1)"; }
export -f run; export PY
printf '%s\n' "${jobs[@]}" | xargs -P 2 -I{} bash -c 'run "{}"'
echo "== R4 DONE =="
