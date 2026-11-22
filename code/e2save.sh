#!/bin/bash
# re-run E2 with --save_model (deterministic; predictions checked against the stored runs); stops claiming after 65 min
cd .; start=$(date +%s); mkdir -p rerun_e2
for f in 0 1 2 3 4; do for m in mobilenetv4_conv_small convnextv2_atto repvit_m0_9; do
  t=${m}_group_f${f}_s0
  [ -f rerun_e2/$t.pt ] && continue
  mkdir rerun_e2/lock_$t 2>/dev/null || continue
  [ $(( $(date +%s) - start )) -gt 3900 ] && { rmdir rerun_e2/lock_$t; exit 0; }
  python3 train.py --model $m --protocol group --fold $f --bs 64 --iters 1000 --threads 1 --seed 0 --res 80 --lr 2e-3 --out ./rerun_e2 --save_model > rerun_e2/$t.log 2>&1
  rmdir rerun_e2/lock_$t
done; done
