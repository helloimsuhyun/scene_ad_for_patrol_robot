python3 ex.py \
  --place 00 \
  --mode calib \
  --radius 1 \
  --calib_method robust \
  --calib_k 3.0 \
  --calib_max_imgs 30 \
  --calib_n_ref 5 \
  --dino_model dinov2_vits14 \
  --dino_top_m 5



python3 ex.py \
  --place 00 \
  --mode infer \
  --radius 1 \
  --n_ref_candidates 5 \
  --dino_model dinov2_vits14 \
  --dino_top_m 5