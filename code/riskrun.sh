#!/bin/bash
# worker for E1/E2; claims jobs via lock dirs; stops claiming after 100 min
cd .; start=$(date +%s)
JOBS="prismrpt+paug:0 resnet18pt:0 resnet18pt+paug:0 convnextv2_atto:0 repvit_m0_9:0 mobilenetv4_conv_small:0"
for f in 1 2 3 4; do JOBS="$JOBS prismrpt+paug:$f resnet18pt:$f resnet18pt+paug:$f convnextv2_atto:$f repvit_m0_9:$f mobilenetv4_conv_small:$f"; done
for j in $JOBS; do
  IFS=: read m f <<< "$j"
  case $m in *pt*) out=results_pt; extra="--res 112 --lr 5e-4 --no_corrupt";; *) out=results; extra="--res 80 --lr 2e-3";; esac
  t=${m}_group_f${f}_s0
  [ -f $out/$t.json ] && continue
  mkdir $out/lock_$t 2>/dev/null || continue
  [ $(( $(date +%s) - start )) -gt 3900 ] && { rmdir $out/lock_$t; exit 0; }
  python3 train.py --model $m --protocol group --fold $f --bs 64 --iters 1000 --threads 1 --seed 0 --out ./$out $extra > $out/$t.log 2>&1
  rmdir $out/lock_$t
done
