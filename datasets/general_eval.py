import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

from datasets.data_io import *

s_h, s_w = 0, 0


class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.max_h, self.max_w = kwargs["max_h"], kwargs["max_w"]
        self.fix_res = kwargs.get("fix_res", False)  # whether to fix the resolution of input image.
        self.fix_wh = False
        self.kwargs = kwargs
        self.stage3 = kwargs.get('stage3', False)

        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        assert self.mode == "test"
        self.metas = self.build_list()
        self.list_begin = []

    def build_list(self):
        metas = []  # {}
        if type(self.listfile) is list:
            scans = self.listfile
        else:
            with open(self.listfile, 'r') as f:
                scans = f.readlines()
                scans = [s.strip() for s in scans]

        interval_scale_dict = {}
        # scans
        for scan in scans:
            # determine the interval scale of each scene. default is 1.06
            if isinstance(self.interval_scale, float):
                interval_scale_dict[scan] = self.interval_scale
            else:
                interval_scale_dict[scan] = self.interval_scale[scan]

            pair_file = "{}/pair.txt".format(scan)
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
                        src_views = src_views[:(self.nviews - 1)]
                        metas.append((scan, ref_view, src_views, scan))

        self.interval_scale = interval_scale_dict
        print("dataset", self.mode, "metas:", len(metas), "interval_scale:{}".format(self.interval_scale))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename, interval_scale):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        if self.kwargs["dataset"] == "tt":
            intrinsics[1, 2] += 4
        intrinsics[:2, :] /= 4.0
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])

        if 'cams_1' in filename:  # only used in DTU
            depth_interval = 2.5
        else:
            depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 3:
            num_depth = lines[11].split()[2]
            depth_max = depth_min + int(float(num_depth)) * depth_interval
            depth_interval = (depth_max - depth_min) / self.ndepths

        if self.kwargs['dataset'] == 'eth3d':
            depth_max = float(lines[11].split()[1])
            depth_interval = (depth_max - depth_min) / self.ndepths

        depth_interval *= interval_scale

        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename).convert('RGB')  # 0~255
        np_img = np.asarray(img)
        if self.kwargs["dataset"] == "tt":
            np_img = np.pad(np_img, ((4, 4), (0, 0), (0, 0)), 'edge')
        img = Image.fromarray(np_img)
        return img

    def scale_mvs_input(self, img, intrinsics, max_w, max_h, base=64):
        h, w = img.shape[:2]
        new_h, new_w = max_h, max_w

        scale_w = 1.0 * new_w / w
        scale_h = 1.0 * new_h / h
        intrinsics[0, :] *= scale_w
        intrinsics[1, :] *= scale_h

        img = cv2.resize(img, (int(new_w), int(new_h)))

        return img, intrinsics

    def read_mask_hr(self, filename):
        img = Image.open(filename)
        np_img = np.array(img, dtype=np.float32)
        np_img = (np_img > 10).astype(np.float32)
        np_img = cv2.resize(np_img, (self.max_w, self.max_h), interpolation=cv2.INTER_NEAREST)

        np_img_ms = {
            "stage4": np_img
        }
        return np_img_ms

    def read_depth_hr(self, filename):
        # read pfm depth file
        depth_hr = np.array(read_pfm(filename)[0], dtype=np.float32)
        depth_lr = cv2.resize(depth_hr, (self.max_w, self.max_h), interpolation=cv2.INTER_NEAREST)

        depth_lr_ms = {
            "stage4": depth_lr
        }
        return depth_lr_ms

    def __getitem__(self, idx):
        global s_h, s_w
        # key, real_idx = self.generate_img_index[idx]
        meta = self.metas[idx]
        scan, ref_view, src_views, scene_name = meta
        # scan = scene_name = key
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views  # [:self.nviews - 1]

        imgs = []
        depth_values = None
        depth_ms = None
        mask = None
        proj_matrices = []
        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            if self.kwargs["dataset"] == "tt":
                if self.kwargs['use_short_range']:
                    proj_mat_filename = os.path.join(self.datapath, 'short_range_cameras/cams_{}/{:0>8}_cam.txt'.format(scan.lower(), vid))
                else:
                    proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))
            else:
                proj_mat_filename = os.path.join(self.datapath, '{}/cams_1/{:0>8}_cam.txt'.format(scan, vid))
                if not os.path.exists(proj_mat_filename):
                    proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            if self.kwargs["dataset"] == 'dtu':  # only used for metric
                mask_filename_hr = os.path.join("/".join(self.datapath.split('/')[:-1]), 'mvs_training/Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
                depth_filename_hr = os.path.join("/".join(self.datapath.split('/')[:-1]), 'mvs_training/Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            img = self.read_img(img_filename)
            img = np.array(img)
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename, interval_scale=self.interval_scale[scene_name])
            # scale input
            img, intrinsics = self.scale_mvs_input(img, intrinsics, self.max_w, self.max_h)

            if self.fix_res:
                # using the same standard height or width in entire scene.
                s_h, s_w = img.shape[:2]
                self.fix_res = False
                self.fix_wh = True

            if i == 0:
                if not self.fix_wh:
                    # using the same standard height or width in each nviews.
                    s_h, s_w = img.shape[:2]

            # resize to standard height or width
            c_h, c_w = img.shape[:2]
            if (c_h != s_h) or (c_w != s_w):
                scale_h = 1.0 * s_h / c_h
                scale_w = 1.0 * s_w / c_w
                img = cv2.resize(img, (s_w, s_h))
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

            img = Image.fromarray(img)
            imgs.append(self.transforms(img))
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                if self.kwargs['dataset'] == 'dtu':
                    mask_read_ms = self.read_mask_hr(mask_filename_hr)
                    depth_ms = self.read_depth_hr(depth_filename_hr)
                    mask = mask_read_ms
                depth_values = np.arange(depth_min, depth_interval * (self.ndepths - 0.5) + depth_min, depth_interval, dtype=np.float32)

        # all
        imgs = torch.stack(imgs)  # [V,3,H,W]
        proj_matrices = np.stack(proj_matrices)

        stage0_pjmats = proj_matrices.copy()
        stage0_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5
        stage1_pjmats = proj_matrices.copy()
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4

        proj_matrices_ms = {
            "stage1": stage0_pjmats,
            "stage2": stage1_pjmats,
            "stage3": stage2_pjmats,
            "stage4": stage3_pjmats
        }

        if getattr(self, 'stage3', False):
            proj_matrices_ms = {
                "stage1": proj_matrices_ms['stage2'],  # 1/4
                "stage2": proj_matrices_ms['stage3'],  # 1/2
                "stage3": proj_matrices_ms['stage4'],  # 1/1
            }

        if self.kwargs['dataset'] == 'dtu':
            return {"imgs": imgs,
                    "proj_matrices": proj_matrices_ms,
                    "depth_values": depth_values,
                    "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                    "depth": depth_ms,
                    "mask": mask}
        else:
            return {"imgs": imgs,
                    "proj_matrices": proj_matrices_ms,
                    "depth_values": depth_values,
                    "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
