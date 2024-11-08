## MVSFormer++: Revealing the Devil in Transformer’s Details for Multi-View Stereo (ICLR2024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvsformer-revealing-the-devil-in-transformer/3d-reconstruction-on-dtu)](https://paperswithcode.com/sota/3d-reconstruction-on-dtu?p=mvsformer-revealing-the-devil-in-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvsformer-revealing-the-devil-in-transformer/point-clouds-on-tanks-and-temples)](https://paperswithcode.com/sota/point-clouds-on-tanks-and-temples?p=mvsformer-revealing-the-devil-in-transformer)

[[arXiv]](https://arxiv.org/abs/2401.11673) [[openreview]](https://openreview.net/forum?id=wXWfvSpYHh)

- [x] Releasing codes of training and testing

- [x] Releasing pre-trained models trained on DTU

- [x] Releasing pre-trained models fine-tuned on BlendedMVS and test code.

- [x] **NEWS**: Updating a script to process mvs data with depth range free cameras (such as nerf transforms.json)

## Installation

​	Please first see [FlashAttention2](https://github.com/Dao-AILab/flash-attention) for original requirements and compilation instructions, or you should make sure torch>=2.1 to use ```F.scaled_dot_product_attention``` with custom scales.

```
git clone https://github.com/maybeLx/MVSFormerPlusPlus.git
cd MVSFormerPlusPlus
pip install -r requirements.txt
```

​	We also highly recommend to install fusibile from (https://github.com/YoYo000/fusibile) for the depth fusion.

```
git clone https://github.com/YoYo000/fusibile.git
cd fusibile
cmake .
make
```

**Tips:** You should revise CUDA_NVCC_FLAGS in CMakeLists.txt according the gpu device you used. We set `-gencode arch=compute_70,code=sm_70` instead of `-gencode arch=compute_60,code=sm_60` with V100 GPUs. For other GPU types, you can follow

```
# 1080Ti
-gencode arch=compute_60,code=sm_60

# 2080Ti
-gencode arch=compute_75,code=sm_75

# 3090Ti
-gencode arch=compute_86,code=sm_86

# V100
-gencode arch=compute_70,code=sm_70
```

More compile relations could be found in [here](https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/)

## Datasets

Please refer to [MVSFormer](https://github.com/ewrfcas/MVSFormer)

##  Training

## Pretrained weights

[DINOv2-B](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)

Training MVSFormer++ on DTU with 4 48GB A6000 GPUs costs around 1 day. We set the max epoch=15 in DTU.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config config/mvsformer++.json \
                                             --exp_name MVSFormer++ \
                                             --DDP
```

We should finetune our model based on BlendedMVS before the testing on Tanks&Temples.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config ./saved/models/DINOv2/mvsformer++/mvsformer++_ft.json  \
                                             --exp_name MVSFormer++_blendedmvs_dtu_mixed_M0 \
                                             --dataloader_type "BlendedLoader" \
                                             --data_path ${YOUR_BLENDEMVS_PATH}
                                             --dtu_model_path ./saved/models/DINOv2/mvsformer++/model_best.pth \
                                             --finetune \
                                             --balanced_training \
                                             --DDP
```

## Test

Pretrained models and additional pair.txt : [OneDrive](https://1drv.ms/f/s!AnZvbwfkzTydkk0TEvkA7M8cRY92?e=ZjncP5)

For testing on DTU:

```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset dtu --batch_size 1  \
                                      --testpath ${dtu_test_path}   --testlist ./lists/dtu/test.txt   \
                                      --resume ${MODEL_WEIGHT_PATH}   \
                                      --outdir ${OUTPUT_DIR}   --interval_scale 1.06 --num_view 5   \
                                      --numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma   \
                                      --disp_threshold 0.1 --num_consistent 2 --prob_threshold 0.5
```
For testing on Tanks&Temple:

You can run `test_tt_adv.sh` and `test_tt_inter.sh` in the scripts. Please specify the `${MODEL_WEIGHT_PATH}` and `${OUTPUT_DIR}`. We recommend using the checkpoint (`tnt_ckpt` available on [OneDrive](https://1drv.ms/f/s!AnZvbwfkzTydkk0TEvkA7M8cRY92?e=ZjncP5)) for testing on this dataset.

## Test on your own data
Our MVSFormer++ requires camera parameters and view selection file. If you do not have them, you can use `Colmap` to estimate cameras and convert them to MVSNet format by `colmap2mvsnet.py`. Please arrange your files as follows.
```
- <dense_folder>
    - images_col  # input images of Colmap
    - sparse_col  # SfM output from colmap in .txt format
    - cams        # output MVSNet cameras, to be generated
    - images      # output MVSNet input images, to be generated
    - pair.txt    # output view selection file, to be generated
    - undistorted  # undistorted images folder, contains cams, images, pair.txt, sparse
```
An example of running `Colmap`
```
colmap feature_extractor \
    --database_path <dense_folder>/database.db \
    --image_path <dense_folder>/images_col

colmap exhaustive_matcher \
    --database_path <dense_folder>/database.db

colmap mapper \
    --database_path <dense_folder>/database.db \
    --image_path <dense_folder>/images_col \
    --output_path <dense_folder>/sparse_col

colmap model_converter \
    --input_path <dense_folder>/sparse_col/0 \
    --output_path <dense_folder>/sparse_col \
    --output_type TXT

colmap image_undistorter --image_path <dense_folder>/images_col \
        --input_path <dense_folder>/sparse_col/0 \
        --output_path <dense_folder>/undistorted  \
        --output_type COLMAP
cd <dense_folder>/undistorted && mv images images_col

```
Run `colmap2mvsnet.py` by
```
python colmap2mvsnet.py --dense_folder <dense_folder>/undistorted --max_d 256 --convert_format
```
For poses without sparse points (nerf poses), which is necessary for achieving depth range, we recommend use ```nerf2mvsnet.py``` instead.
Note that you should activate ```nerf2opencv``` to to process nerf poses with different coordinates.
```
python nerf2mvsnet.py --dense_folder <dense_folder> --max_d 256 --save_ply --nerf2opencv
```

Please note that: the resolution of input images must be divisible by 64. we can change the parameter of `max_h` and `max_w`. For test on your own dataset:
```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset mydataset --batch_size 1  \
                                      --testpath ${scene_test_path}   --testlist ${scene_test_list}   \
                                      --resume ${MODEL_WEIGHT_PATH}   \
                                      --outdir ${OUTPUT_DIR}   --interval_scale 1.06 --num_view 5   \
                                      --numdepth 192 --max_h ${max_h} --max_w ${max_w} --filter_method dpcd   \
                                      --conf 0.5
```


## Cite
If you found our program helpful, please consider citing:

```
@inproceedings{cao2024mvsformer++,
      title={MVSFormer++: Revealing the Devil in Transformer's Details for Multi-View Stereo}, 
      author={Chenjie Cao, Xinlin Ren and Yanwei Fu},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024}
}
```

## Acknowledgments
We borrow the code from [VisMVSNet](https://github.com/jzhangbs/Vis-MVSNet), [MVSFormer](https://github.com/ewrfcas/MVSFormer). We express gratitude for these works' contributions!
