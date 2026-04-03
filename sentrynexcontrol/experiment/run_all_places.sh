#!/bin/bash
06

PLACES=(06 00 08)

for PLACE in "${PLACES[@]}"; do
  echo "=============================="
  echo "[CALIB] place=${PLACE}"
  echo "=============================="

  python ex.py \
    --place ${PLACE} \
    --mode calib \
    --calib_method robust \
    --calib_k 3.0 \
    --calib_max_imgs 30 \
    --calib_n_ref 3 \
    --proposal_top_k 3

  echo "=============================="
  echo "[INFER] place=${PLACE}"
  echo "=============================="

  python ex.py \
    --place ${PLACE} \
    --mode infer \
    --n_ref_candidates 3 \
    --proposal_top_k 3

done

echo "✅ ALL DONE"