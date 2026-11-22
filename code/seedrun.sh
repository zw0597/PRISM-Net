#!/bin/bash
# usage: seedrun.sh "model:seed model:seed ..."  -- runs 5 folds each, skips finished, stops claiming after 95 min
cd .
T0=$(date +%s)
for ms in $1; do m=${ms%%:*}; s=${ms##*:}
  for f in 0 1 2 3 4; do
    t=${m}_group_f${f}_s${s}
    [ -f results/$t.json ] && continue
    [ $(( $(date +%s) - T0 )) -gt 5700 ] && exit 0
    mkdir results/$t.lockdir 2>/dev/null || continue
    python3 train.py --model $m --protocol group --fold $f --res 80 --bs 64 --iters 1000 --threads 1 --seed $s --out results > results/$t.log 2>&1
    rmdir results/$t.lockdir
  done
done
