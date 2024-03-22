import torch.nn as nn
import torch
import math
from einops import rearrange, repeat

class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(384, 384)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 384 corresponds to 3072 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(torch.arange(0, d_model // 2, 2).float() * (-math.log(10000.0) / (d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


class PositionEncodingSineNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(128, 128)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()
        self.d_model = d_model
        self.max_shape = max_shape
        self.pe_dict = dict()

    def reset_pe(self, new_shape, device):
        (H, W) = new_shape
        pe = torch.zeros((self.d_model, H, W))
        y_position = torch.ones((H, W)).cumsum(0).float().unsqueeze(0) * self.max_shape[0] / H
        x_position = torch.ones((H, W)).cumsum(1).float().unsqueeze(0) * self.max_shape[1] / W

        div_term = torch.exp(torch.arange(0, self.d_model // 2, 2).float() * (-math.log(10000.0) / (self.d_model // 2)))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        return pe.unsqueeze(0).to(device)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        # if in testing, and test_shape!=train_shape, reset PE
        _, _, H, W = x.shape
        if f"{H}-{W}" in self.pe_dict:  # if the cache has this PE weights, use it
            pe = self.pe_dict[f"{H}-{W}"]
        else:  # or re-generate new PE weights for H-W
            pe = self.reset_pe((H, W), x.device)
            self.pe_dict[f"{H}-{W}"] = pe  # save new PE

        return x + pe  # the shape must be the same


class EpipoleEncodingSineNorm(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, train_max_size=384):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()
        self.d_model = d_model
        self.train_max_size = train_max_size

    def forward(self, feat, Fmat):
        """
        Args:
            x: [N, C, H, W]
            Fmat: [N, 3, 3]
        """
        batch, _, height, width = feat.shape
        max_size = max(height, width)
        # calculate distance from each pixel to epipole lines
        y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=feat.device),
                               torch.arange(0, width, dtype=torch.float32, device=feat.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(height * width), x.view(height * width)
        xyz = torch.stack((x, y, torch.ones_like(x)), dim=0).unsqueeze(0).repeat(batch, 1, 1)  # [N,3,HW]
        xy = xyz[:, :2]  # [N,2,HW]
        epi_lines = Fmat @ xyz  # [N,3,HW] Ax+By+C=0
        A, B, C = epi_lines[:, 0], epi_lines[:, 1], epi_lines[:, 2]
        epi_dis = torch.abs(A * xy[:, 0] + B * xy[:, 1] + C)
        epi_dis = epi_dis / (torch.sqrt(A ** 2 + B ** 2) + 1e-7)
        epi_dis = torch.clamp(epi_dis, 0, max_size)  # [N,HW]
        epi_dis = epi_dis * self.train_max_size / max_size  # aligned with training size
        epi_dis = epi_dis.unsqueeze(1)  # [N,1,HW]

        pe = torch.zeros((batch, self.d_model, height, width), dtype=torch.float32, device=feat.device)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        div_term = div_term[None, :, None]  # [1,C//4,1]
        pe[:, 0::2, :, :] = torch.sin(epi_dis * div_term).reshape(batch, -1, height, width)
        pe[:, 1::2, :, :] = torch.cos(epi_dis * div_term).reshape(batch, -1, height, width)  # [N,C//2,H,W]

        return x + pe


def get_position_3d(B, H, W, K, depth_values, depth_min, depth_max, height_min, height_max, width_min, width_max, normalize=True):
    num_depth = depth_values.shape[1]
    with torch.no_grad():
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=K.device),
                               torch.arange(0, W, dtype=torch.float32, device=K.device)], indexing='ij')
        y, x = y.contiguous(), x.contiguous()
        y, x = y.view(H * W), x.view(H * W)
        xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
        xyz = torch.unsqueeze(xyz, 0).repeat(B, 1, 1)  # [B, 3, H*W]
        xyz = torch.matmul(torch.inverse(K), xyz)  # [B, 3, H*W]
        # point position in 3d space
        position3d = xyz.unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_values.view(B, 1, num_depth, -1)  # [B, 3, 1, H*W]->[B,3,D,H*W]
        if normalize:  # minmax normalization
            if height_min is None or height_max is None or width_min is None or width_max is None:
                width_min, width_max = position3d[:, 0].min(), position3d[:, 0].max()
                height_min, height_max = position3d[:, 1].min(), position3d[:, 1].max()
            position3d[:, 0] = (position3d[:, 0] - width_min) / (width_max - width_min + 1e-5)
            position3d[:, 1] = (position3d[:, 1] - height_min) / (height_max - height_min + 1e-5)
            # normalizing depth (z) into 0~1 based on depth_min and depth_max
            position3d[:, 2] = (torch.clamp(position3d[:, 2], depth_min, depth_max) - depth_min) / (depth_max - depth_min + 1e-5)

        position3d = position3d.reshape(B, 3, num_depth, H, W)

    return position3d, height_min, height_max, width_min, width_max


def PositionEncoding3D(position3d, C, rescale=4.0):
    # position3d must be in 0~1, actually [0~rescale]
    # position3d:[B,3,D,H,W]
    # rescale: 这个参数是trade-off远程衰减和震荡程度的系数，目测下来对0~1和维度8的PE 4.0比较好
    B, _, D, H, W = position3d.shape
    div_term = torch.exp(torch.arange(0, C, 2).float() * (-math.log(10000.0) / C)).to(position3d.device)
    div_term = div_term[None, :, None]  # [1,C//2,1]

    pe_x = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_x = position3d[:, 0].reshape(B, 1, D * H * W)
    pe_x[:, 0::2, :] = torch.sin(pos_x * rescale * div_term).reshape(B, -1, D * H * W)
    pe_x[:, 1::2, :] = torch.cos(pos_x * rescale * div_term).reshape(B, -1, D * H * W)  # [N,C,DHW]

    pe_y = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_y = position3d[:, 1].reshape(B, 1, D * H * W)
    pe_y[:, 0::2, :] = torch.sin(pos_y * rescale * div_term).reshape(B, -1, D * H * W)
    pe_y[:, 1::2, :] = torch.cos(pos_y * rescale * div_term).reshape(B, -1, D * H * W)  # [N,C,DHW]

    pe_z = torch.zeros((B, C, D * H * W), dtype=torch.float32, device=position3d.device)
    pos_z = position3d[:, 2].reshape(B, 1, D * H * W)
    pe_z[:, 0::2, :] = torch.sin(pos_z * rescale * div_term).reshape(B, -1, D * H * W)
    pe_z[:, 1::2, :] = torch.cos(pos_z * rescale * div_term).reshape(B, -1, D * H * W)  # [N,C,DHW]

    pe = torch.cat([pe_x, pe_y, pe_z], dim=1).reshape(B, C * 3, D, H, W)

    return pe

