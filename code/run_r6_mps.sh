#!/bin/bash
# R6 (reviewer_protocol.md): the 12 models not in R4 x five folds x 3000 steps on the M5 Max (MPS), same command as R4; three workers.
cd "$(dirname "$0")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv_mps/bin/python
mkdir -p logs_gpu
jobs=()
for f in 0 1 2 3 4; do for m in convnextv2_atto repvit_m0_9 efficientnet_b0 mobilenetv4_conv_small prismx+paug vit gfnet gfnet+paug \
    mobilenet_v3_small mobilenet_v3_small+paug shufflenet_v2_x1_0 resnet18+paug; do jobs+=("$m $f 3000"); done; done
run() { set -- $1; m=$1; f=$2; it=$3; out=results_gpu_$it; t=${m}_group_f${f}_s0
  [ -f $out/$t.json ] && return
  echo "$(date '+%H:%M:%S') start $t $it"
  $PY -u train.py --model $m --protocol group --fold $f --res 80 --lr 2e-3 --bs 64 --iters $it --threads 4 --seed 0 \
      --device auto --out $out >> logs_gpu/${t}_$it.log 2>&1 || echo "FAILED $t $it"
  echo "$(date '+%H:%M:%S') done $t $it $(grep 'TEST ACC' logs_gpu/${t}_$it.log | tail -1)"; }
export -f run; export PY
printf '%s\n' "${jobs[@]}" | xargs -P 3 -I{} bash -c 'run "{}"'
echo "== R6 DONE =="
