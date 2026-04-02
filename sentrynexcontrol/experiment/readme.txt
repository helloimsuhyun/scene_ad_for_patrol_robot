python3 ex.py \
  --place 08 \
  --mode calib \
  --radius 1 \
  --calib_method robust \
  --calib_k 3.0 \
  --calib_max_imgs 100 \
  --calib_n_ref 3 \
  --dino_model dinov2_vits14 \
  --dino_top_m 3



python3 ex.py \
  --place 06 \
  --mode infer \
  --radius 1 \
  --n_ref_candidates 3 \
  --dino_model dinov2_vits14 \
  --dino_top_m 3