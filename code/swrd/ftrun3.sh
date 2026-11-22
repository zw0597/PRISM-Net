#!/bin/bash
cd .; start=$(date +%s)
for job in $(python3 -c "
import json;s=json.load(open('swrd/round2_selection.json'))
for b in ('5','20'):
  for f in range(5):
    for k,r in s.items():
      m,bb=k.split('|')
      if bb==b: print(f'{m}:{f}:{bb}:{r}')"); do
  IFS=: read m f b r <<< "$job"
  t=${m}_f${f}_b${b}_${r}_pgaR; [ $r = C0 ] && t=${m}_f${f}_b${b}_pgaR
  [ -f swrd/ft/$t.npy ] && continue
  mkdir swrd/ft_lock_$t 2>/dev/null || continue
  [ $(( $(date +%s) - start )) -gt 5400 ] && { rmdir swrd/ft_lock_$t; exit 0; }
  python3 swrd_finetune2.py --model $m --fold $f --budget $(python3 -c "print($b/100)") --recipe $r --pga R >> swrd/ftlog3.txt 2>&1
  rmdir swrd/ft_lock_$t
done
