import os
import pdb
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import collections

from datasets.data_io import read_pfm
from .color_jittor import ColorJitter

np.random.seed(123)
random.seed(123)


class RandomGamma():
    def __init__(self, min_gamma=0.7, max_gamma=1.5, clip_image=False):
        self._min_gamma = min_gamma
        self._max_gamma = max_gamma
        self._clip_image = clip_image

    @staticmethod
    def get_params(min_gamma, max_gamma):
        return np.random.uniform(min_gamma, max_gamma)

    @staticmethod
    def adjust_gamma(image, gamma, clip_image):
        adjusted = torch.pow(image, gamma)
        if clip_image:
            adjusted.clamp_(0.0, 1.0)
        return adjusted

    def __call__(self, img, gamma):
        # gamma = self.get_params(self._min_gamma, self._max_gamma)
        return self.adjust_gamma(img, gamma, self._clip_image)


# the DTU dataset preprocessed by Yao Yao (only for training)
class DTUMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, random_crop=False, augment=False,
                 aug_args=None, height=512, width=640, **kwargs):
        super(DTUMVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.height = height
        self.width = width
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        if mode != 'train':
            self.random_crop = False
            self.augment = False
        else:
            self.random_crop = random_crop
            self.augment = augment
        self.kwargs = kwargs
        self.multi_scale = kwargs.get('multi_scale', False)
        self.multi_scale_args = kwargs['multi_scale_args']
        self.resize_scale = kwargs.get('resize_scale', 0.5)

        self.scales = self.multi_scale_args['scales']#[::-1]
        self.resize_range = self.multi_scale_args['resize_range']
        self.batch_size = kwargs.get('batch_size', 4)
        self.world_size = kwargs.get('world_size', 1)
        self.img_size_map = []

        # print("mvsdataset kwargs", self.kwargs)

        if self.augment and mode == 'train':
            self.color_jittor = ColorJitter(brightness=aug_args['brightness'], contrast=aug_args['contrast'],
                                            saturation=aug_args['saturation'], hue=aug_args['hue'])
            self.to_tensor = transforms.ToTensor()
            self.random_gamma = RandomGamma(min_gamma=aug_args['min_gamma'], max_gamma=aug_args['max_gamma'], clip_image=True)
            self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    lights = np.arange(7)
                    for light_idx in lights:
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def reset_dataset(self, shuffled_idx):
        self.idx_map = {}
        barrel_idx = 0
        count = 0
        for sid in shuffled_idx:
            self.idx_map[sid] = barrel_idx
            count += 1
            if count == self.batch_size:
                count = 0
                barrel_idx += 1

        # random img size:256~512
        if self.mode == 'train':
            barrel_num = int(len(self.metas) / (self.batch_size * self.world_size))
            barrel_num += 2
            self.img_size_map = np.arange(0, len(self.scales))

    def __len__(self):
        # return len(self.generate_img_index)
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename).convert('RGB')
        return img

    def read_depth(self, filename):
        # read pfm depth file
        return np.array(read_pfm(filename)[0], dtype=np.float32)

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        # np_img = self.prepare_img(np_img)
        return np_img

    def read_depth_hr(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        # return self.prepare_img(depth)
        return depth

    def generate_stage_depth(self, depth):
        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth
        }
        return depth_ms

    def center_crop_img(self, img, new_h=None, new_w=None):
        h, w = img.shape[:2]

        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            img = img[start_h:finish_h, start_w:finish_w]
        return img

    def center_crop_cam(self, intrinsics, h, w, new_h=None, new_w=None):
        if new_h != h or new_w != w:
            start_h = (h - new_h) // 2
            start_w = (w - new_w) // 2
            new_intrinsics = intrinsics.copy()
            new_intrinsics[0][2] = new_intrinsics[0][2] - start_w
            new_intrinsics[1][2] = new_intrinsics[1][2] - start_h
            return new_intrinsics
        else:
            return intrinsics

    def pre_resize(self, img, depth, intrinsic, mask, resize_scale):
        ori_h, ori_w, _ = img.shape
        img = cv2.resize(img, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_AREA)
        h, w, _ = img.shape

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, :] *= resize_scale
        output_intrinsics[1, :] *= resize_scale

        if depth is not None:
            depth = cv2.resize(depth, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        if mask is not None:
            mask = cv2.resize(mask, (int(ori_w * resize_scale), int(ori_h * resize_scale)), interpolation=cv2.INTER_NEAREST)

        return img, depth, output_intrinsics, mask

    def final_crop(self, img, depth, intrinsic, mask, crop_h, crop_w, offset_y=None, offset_x=None):
        h, w, _ = img.shape
        if offset_x is None or offset_y is None:
            if self.random_crop:
                offset_y = random.randint(0, h - crop_h)
                offset_x = random.randint(0, w - crop_w)
            else:
                offset_y = (h - crop_h) // 2
                offset_x = (w - crop_w) // 2
        cropped_image = img[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w, :]

        output_intrinsics = intrinsic.copy()
        output_intrinsics[0, 2] -= offset_x
        output_intrinsics[1, 2] -= offset_y

        if depth is not None:
            cropped_depth = depth[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_depth = None

        if mask is not None:
            cropped_mask = mask[offset_y:offset_y + crop_h, offset_x:offset_x + crop_w]
        else:
            cropped_mask = None

        return cropped_image, cropped_depth, output_intrinsics, cropped_mask

    def __getitem__(self, idx):
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(0)
        
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        if self.mode == 'train':
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        # view_ids = [ref_view] + src_views

        imgs = []
        depth_values = None
        depth_ms = None
        mask = None
        proj_matrices = []

        if self.augment:
            fn_idx = torch.randperm(4)
            brightness_factor = torch.tensor(1.0).uniform_(self.color_jittor.brightness[0], self.color_jittor.brightness[1]).item()
            contrast_factor = torch.tensor(1.0).uniform_(self.color_jittor.contrast[0], self.color_jittor.contrast[1]).item()
            saturation_factor = torch.tensor(1.0).uniform_(self.color_jittor.saturation[0], self.color_jittor.saturation[1]).item()
            hue_factor = torch.tensor(1.0).uniform_(self.color_jittor.hue[0], self.color_jittor.hue[1]).item()
            gamma_factor = self.random_gamma.get_params(self.random_gamma._min_gamma, self.random_gamma._max_gamma)
        else:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor, gamma_factor = None, None, None, None, None, None

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            # NOTE that DTU_origin/Rectified saves the images with the original size (1200x1600)
            img_filename = os.path.join(self.datapath, 'Rectified_raw/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            # these poses are based on original resolution
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/{:0>8}_cam.txt').format(vid)
            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            if i == 0:
                depth_hr = self.read_depth_hr(depth_filename_hr)
                depth_mask_hr = self.read_mask_hr(mask_filename_hr)
            else:
                depth_hr = None
                depth_mask_hr = None

            # resize according to crop size
            if self.mode == 'train':
                [crop_h, crop_w] = self.scales[self.idx_map[idx] % len(self.scales)]
                enlarge_scale = self.resize_range[0] + random.random() * (self.resize_range[1] - self.resize_range[0])
                resize_scale_h = np.clip((crop_h * enlarge_scale) / 1200, 0.45, 1.0)
                resize_scale_w = np.clip((crop_w * enlarge_scale) / 1600, 0.45, 1.0)
                resize_scale = max(resize_scale_h, resize_scale_w)
            else:
                crop_h, crop_w = self.height, self.width
                resize_scale = self.resize_scale

            img = np.asarray(img)
            if resize_scale != 1.0:
                img, depth_hr, intrinsics, depth_mask_hr = self.pre_resize(img, depth_hr, intrinsics, depth_mask_hr, resize_scale)

            if i == 0:  # reference view
                while True:  # get resonable offset
                    # finally random crop
                    img_, depth_hr_, intrinsics_, depth_mask_hr_ = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                                       crop_h=crop_h, crop_w=crop_w)
                    mask_read_ms_ = self.generate_stage_depth(depth_mask_hr_)
                    if self.mode != 'train' or np.any(mask_read_ms_['stage1'] > 0.0) or not self.random_crop:
                        break

                depth_ms = self.generate_stage_depth(depth_hr_)
                mask = mask_read_ms_
                img = img_
                intrinsics = intrinsics_
                # get depth values
                depth_max = depth_interval * self.ndepths + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)
            else:
                img, depth_hr, intrinsics, depth_mask_hr = self.final_crop(img, depth_hr, intrinsics, depth_mask_hr,
                                                                                     crop_h=crop_h, crop_w=crop_w)
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            img = Image.fromarray(img)
            if not self.augment:
                imgs.append(self.transforms(img))
            else:
                img_aug = self.color_jittor(img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
                img_aug = self.to_tensor(img_aug)
                img_aug = self.random_gamma(img_aug, gamma_factor)
                img_aug = self.normalize(img_aug)
                imgs.append(img_aug)

        # all
        imgs = torch.stack(imgs)
        # ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125

        proj_matrices_ms = {
            "stage1": stage0_pjmats,  # 1/8
            "stage2": stage1_pjmats,  # 1/4
            "stage3": stage2_pjmats,  # 1/2
            "stage4": proj_matrices  # 1/1
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "mask": mask}
