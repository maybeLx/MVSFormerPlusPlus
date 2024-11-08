echo Auditorium > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=2  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3

echo Ballroom > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=2  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR} \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3

 echo Courtroom > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=2  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3

 echo Museum > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=2  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3

  echo Palace > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=3  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3

   echo Temple > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=3  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/advanced --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.3
