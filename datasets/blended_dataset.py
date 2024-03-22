import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.data_io import *
from .color_jittor import ColorJitter
import random
import cv2

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


class BlendedMVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, crop=False, augment=False,
                 aug_args=None, height=256, width=320, **kwargs):
        super(BlendedMVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale

        if mode != 'train':
            self.crop = False
            self.augment = False
            self.random_resize = False
            self.random_mask = False
        else:
            self.crop = crop
            self.augment = augment
            self.random_resize = kwargs.get('random_resize', False)
        self.kwargs = kwargs
        self.resize_scale = kwargs.get('resize_scale', 0.5)
        self.min_scale = kwargs.get('min_scale', 1.0)
        self.max_scale = kwargs.get('max_scale', 1.0)
        self.pre_crop = kwargs.get('pre_crop', False)
        self.pre_height = kwargs.get('pre_height', 1200)
        self.pre_width = kwargs.get('pre_width', 1600)
        self.consist_crop = kwargs.get('consist_crop', False)

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
        # scans = self.listfile

        interval_scale_dict = {}
        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print("{}< num_views:{}".format(len(src_views), self.nviews))
                            src_views += [src_views[0]] * (self.nviews - len(src_views))
                        # src_views = src_views[:(self.nviews-1)]
                        metas.append((scan, ref_view, src_views, scan))

        # self.interval_scale = interval_scale_dict
        print("dataset ", self.mode, "metas: ", len(metas), "interval_scale: {}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # intrinsics[0, 2] -= 64.0
        # intrinsics[1, 2] -= 32.0
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= self.interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def prepare_img(self, img):
        h, w = img.shape[:2]
        target_h, target_w = 576, 768  # 512, 640
        start_h, start_w = (h - target_h) // 2, (w - target_w) // 2
        img_crop = img[start_h: start_h + target_h, start_w: start_w + target_w]
        return img_crop

    def read_img(self, filename):
        np_img = np.asarray(Image.open(filename).convert('RGB'))
        np_img = self.prepare_img(np_img)

        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth = self.prepare_img(depth)
        h, w = depth.shape
        depth_ms = {
            "stage1": cv2.resize(depth, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(depth, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage4": depth,
        }

        return depth_ms

    def read_mask(self, filename):
        depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        np_img = (depth > 0).astype(np.float32)
        np_img = self.prepare_img(np_img)
        # np_img = cv2.resize(np_img, (768, 576), interpolation=cv2.INTER_NEAREST)
        h, w = np_img.shape
        np_img_ms = {
            "stage1": cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage3": cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage4": np_img,
        }
        return np_img_ms

    def __getitem__(self, idx):
        # global s_h, s_w
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # use only the reference view and first nviews-1 source views
        if self.mode == 'train':
            src_views = src_views[:7]
            np.random.shuffle(src_views)
        view_ids = [ref_view] + src_views[:(self.nviews - 1)]
        # view_ids = [ref_view] + src_views #[:self.nviews - 1]

        imgs = []
        depth_values = None
        proj_matrices = []
        mask, depth_ms = None, None
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
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            mask_filename = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))

            img = self.read_img(img_filename)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)  # 读进来是1/4的camera
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                mask_read_ms = self.read_mask(mask_filename)
                depth_ms = self.read_depth(depth_filename)

                # get depth values
                depth_max = depth_interval * (self.ndepths - 0.5) + depth_min
                depth_values = np.arange(depth_min, depth_max, depth_interval, dtype=np.float32)

                mask = mask_read_ms

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
        imgs = torch.stack(imgs)  # .transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5

        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": proj_matrices,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_values": depth_values,
                "mask": mask,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
