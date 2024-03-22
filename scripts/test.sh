#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python test.py --dataset dtu --batch_size 1 \
  --testpath ./MVS_data/DTU/dtu_test \
  --testlist ./lists/dtu/test.txt \
  --resume ./saved/models/DINOv2/mvsformer++/model_best.pth \
  --outdir ./DTU_outputs/debug \
  --interval_scale 1.06 --num_view 5 \
  --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma \
  --disp_threshold 0.1 --num_consistent 2 --prob_threshold 0.5
