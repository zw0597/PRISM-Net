#!/bin/bash
# usage: ftrun2.sh "model:fold:budget:recipe ..." ; claims jobs via lock dirs, skips finished, stops after 100 min
cd .; start=$(date +%s)
for job in $1; do
  IFS=: read m f b r <<< "$job"; bb=$(python3 -c "print(int($b*100))")
  t=${m}_f${f}_b${bb}_${r}
  [ -f swrd/ft/$t.npy ] && continue
  mkdir swrd/ft_lock_$t 2>/dev/null || continue
  [ $(( $(date +%s) - start )) -gt 6000 ] && { rmdir swrd/ft_lock_$t; exit 0; }
  python3 swrd_finetune2.py --model $m --fold $f --budget $b --recipe $r >> swrd/ftlog2.txt 2>&1
  rmdir swrd/ft_lock_$t
done
