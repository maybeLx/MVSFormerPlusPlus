#### MVSFormer++: Revealing the Devil in Transformer’s Details for Multi-View Stereo (ICLR2024)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvsformer-revealing-the-devil-in-transformer/3d-reconstruction-on-dtu)](https://paperswithcode.com/sota/3d-reconstruction-on-dtu?p=mvsformer-revealing-the-devil-in-transformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mvsformer-revealing-the-devil-in-transformer/point-clouds-on-tanks-and-temples)](https://paperswithcode.com/sota/point-clouds-on-tanks-and-temples?p=mvsformer-revealing-the-devil-in-transformer)

[[arXiv]](https://arxiv.org/abs/2401.11673) [[openreview]](https://openreview.net/forum?id=wXWfvSpYHh)

- [x] Releasing codes of training and testing

- [x] Releasing pre-trained models trained on DTU

- [ ] Releasing pre-trained models fine-tuned on Tanks&Temples and test code.

- [ ] Releasing point cloud of DTU

  ####  Installation

​	Please first see [FlashAttention2](https://github.com/Dao-AILab/flash-attention) for original requirements and compilation instructions.

```
git clone https://github.com/ewrfcas/MVSFormer.git
cd MVSFormer
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

#### Datasets

Please refer to [MVSFormer](https://github.com/ewrfcas/MVSFormer)

####  Training

##### Pretrained weights

[DINOv2-B](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth)

Training MVSFormer++ on DTU with 4 48GB A6000 GPUs cost 2 days. We set the max epoch=15 in DTU.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config configs/mvsformer++.json \
                                         --exp_name MVSFormer++ \
                                         --DDP
```

We should finetune our model based on BlendedMVS before the testing on Tanks&Temples.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py --config ./saved/models/DINOv2/mvsformer++/config_ft.json  \
--exp_name MVSFormer++_blendedmvs_dtu_mixed_M0 \
--dataloader_type "BlendedLoader" \
--data_path ${YOUR_BLENDEMVS_PATH}
--dtu_model_path ./saved/models/DINOv2/mvsformer++/model_best.pth \
--finetune \
--balanced_training \
--DDP
```

#### Test

Pretrained models and additional pair.txt : [OneDrive]()

For testing on DTU:

```
CUDA_VISIBLE_DEVICES=0 python test.py --dataset dtu --batch_size 1  \
--testpath ${dtu_test_path}   --testlist ./lists/dtu/test.txt   \
--resume ${MODEL_WEIGHT_PATH}   \
--outdir ${OUTPUT_DIR}   --interval_scale 1.06 --num_view 5   \
--numdepth 192 --max_h 1152 --max_w 1536 --filter_method gipuma   \
--disp_threshold 0.1 --num_consistent 2 --prob_threshold 0.5
```



#### Cite

If you found our program helpful, please consider citing:

```
@inproceedings{cao2024mvsformer++,
      title={MVSFormer++: Revealing the Devil in Transformer's Details for Multi-View Stereo}, 
      author={Chenjie Cao, Xinlin Ren and Yanwei Fu},
      booktitle={International Conference on Learning Representations (ICLR)},
      year={2024}
}
```
