

echo Family > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=0  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method gipuma \
  --disp_threshold 0.4 --num_consistent 5  --prob_threshold 0.5 --conf 0.5 --conf_choose mean --use_short_range


echo Francis > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=0  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose stage4 --conf 0.6 --use_short_range


 echo Horse > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=0  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose mean --conf 0.6 --use_short_range

  echo Lighthouse > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=0  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose stage4 --conf 0.6 --use_short_range


  echo M60 > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=0  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose stage4 --conf 0.6 --use_short_range


  echo Train > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=1  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --fusion_view 15 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose stage4 --conf 0.6 --use_short_range

   echo Panther > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=1  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920 --filter_method dpcd \
 --disp_threshold 0.1   --conf_choose stage4 --conf 0.6 --use_short_range


   echo Playground > ./lists/tanksandtemples/demo.txt && CUDA_VISIBLE_DEVICES=1  python test_tt.py --dataset tt --batch_size 1 \
 --testpath ./MVS_data/tankandtemples/intermediate --testlist ./lists/tanksandtemples/demo.txt \
 --resume ${MODEL_WEIGHT_PATH}    --outdir ${OUTPUT_DIR}  \
 --interval_scale 1.06 --num_view 20 --numdepth 192 --max_h 1088 --max_w 1920  --filter_method gipuma \
 --disp_threshold 0.3 --num_consistent 5 --conf 0.5 --conf_choose stage4 --use_short_range
