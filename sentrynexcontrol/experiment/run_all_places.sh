#!/bin/bash

PLACES=(06 00 08)

for PLACE in "${PLACES[@]}"; do
  echo "=============================="
  echo "[CALIB] place=${PLACE}"
  echo "=============================="

  python ex.py \
    --place ${PLACE} \
    --mode calib \
    --calib_method robust \
    --cc_k 2.5 \
    --final_k 2.5 \
    --calib_max_imgs 100 \
    --calib_n_ref 3 \
    --radius 1 \
    --seed 0 \
    --dino_model dinov2_vits14 \
    --dino_top_m 3

  echo "=============================="
  echo "[INFER] place=${PLACE}"
  echo "=============================="

  python ex.py \
    --place ${PLACE} \
    --mode infer \
    --n_ref_candidates 3 \
    --radius 1 \
    --seed 0 \
    --dino_model dinov2_vits14 \
    --dino_top_m 3

done

echo "✅ ALL DONE"