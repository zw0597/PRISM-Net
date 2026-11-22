#!/bin/bash
# worker: runs pending fine-tune jobs; safe to start several; stops after 100 min
cd .; start=$(date +%s)
for b in 0.05 0.2; do for m in prismr+paug resnet18 resnet18+paug; do for f in 0 1 2 3 4; do
  t=${m}_f${f}_b$(python3 -c "print(int($b*100))")
  [ -f swrd/ft/$t.npy ] && continue
  mkdir swrd/ft_lock_$t 2>/dev/null || continue
  [ $(( $(date +%s) - start )) -gt 6000 ] && { rmdir swrd/ft_lock_$t; exit 0; }
  python3 swrd_finetune.py --model $m --fold $f --budget $b >> swrd/ftlog.txt 2>&1
  rmdir swrd/ft_lock_$t
done; done; done
