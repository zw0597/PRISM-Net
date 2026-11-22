#!/bin/bash
# R3: VGG-11 seeds 1 and 2 on five folds; one worker, two threads; claims a new job only in the first 70 min
cd .; start=$(date +%s); rmdir results/lock_vgg11_bn_group_f*_s[12] 2>/dev/null
for s in 1 2; do for f in 0 1 2 3 4; do
  t=vgg11_bn_group_f${f}_s${s}
  [ -f results/$t.json ] && continue
  [ $(( $(date +%s) - start )) -gt 4200 ] && exit 0
  mkdir results/lock_$t 2>/dev/null || continue
  python3 train.py --model vgg11_bn --protocol group --fold $f --res 80 --lr 2e-3 --bs 64 --iters 1000 --threads 2 --seed $s --out ./results > results/$t.log 2>&1
  rmdir results/lock_$t
done; done
