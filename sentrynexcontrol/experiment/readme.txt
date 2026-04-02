python3 ex.py \
  --place 00 \
  --mode calib \
  --radius 1 \
  --calib_max_imgs 100 \
  --calib_n_ref 3 \
  --dino_model dinov2_vits14 \
  --dino_top_m 3


python3 ex.py \
  --place 00 \
  --mode infer \
  --radius 1 \
  --n_ref_candidates 3 \
  --dino_model dinov2_vits14 \
  --dino_top_m 3



python ex.py \
  --place 06 \
  --mode calib \
  --calib_method robust \
  --calib_k 2.5 \
  --calib_max_imgs 30 \
  --calib_n_ref 5 \
  --dino_model dinov2_vits14 \
  --dino_top_m 5 \
  --n_ref_candidates 5


python ex.py \
  --place 06 \
  --mode infer \
  --dino_model dinov2_vits14 \
  --dino_top_m 5 \
  --n_ref_candidates 5

