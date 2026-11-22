#!/bin/bash
# VGG-11 (BN) baseline (reviewer_protocol.md R2); one worker, two threads; claims a new job only in the first 25 min
cd .; start=$(date +%s)
for j in group:0 group:1 group:2 group:3 group:4 official:0; do
  IFS=: read p f <<< "$j"; t=vgg11_bn_${p}_f${f}_s0
  [ -f results/$t.json ] && continue
  [ $(( $(date +%s) - start )) -gt 3000 ] && exit 0
  mkdir results/lock_$t 2>/dev/null || continue
  python3 train.py --model vgg11_bn --protocol $p --fold $f --res 80 --lr 2e-3 --bs 64 --iters 1000 --threads 2 --seed 0 --save_model --out ./results > results/$t.log 2>&1
  rmdir results/lock_$t
done
