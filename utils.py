import json
import math
import pdb
import random
import warnings
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import numpy as np
import torch
import torchvision.utils as vutils
import yaml
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import LambdaLR
import pdb
import itertools


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if vars is None:
        return vars
    elif isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        if vars.dtype == torch.bfloat16:
            vars = vars.to(torch.float32)
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.to(torch.device("cuda"))
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step, fname=None):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            if fname is not None:
                name = '{}/{}/{}'.format(mode, fname, key)
            else:
                name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                if fname is not None:
                    name = '{}/{}/{}_{}'.format(mode, fname, key, idx)
                else:
                    name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input, n=1.0):
        self.count += n
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                if k not in self.data:
                    self.data[k] = v
                else:
                    self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}

    def reset(self):
        self.data = {}
        self.count = 0


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *kwargs):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *kwargs)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, thres=None):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    error = (depth_est - depth_gt).abs()
    if thres is not None:
        error = error[(error >= float(thres[0])) & (error <= float(thres[1]))]
        if error.shape[0] == 0:
            return torch.tensor(0, device=error.device, dtype=error.dtype)
    return torch.mean(error)


import torch.distributed as dist


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def reduce_scalar_outputs(scalar_outputs):
    world_size = get_world_size()
    if world_size < 2:
        return scalar_outputs
    with torch.no_grad():
        names = []
        scalars = []
        for k in sorted(scalar_outputs.keys()):
            names.append(k)
            scalars.append(scalar_outputs[k])
        scalars = torch.stack(scalars, dim=0)
        dist.reduce(scalars, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            scalars /= world_size
        reduced_scalars = {k: v for k, v in zip(names, scalars)}

    return reduced_scalars


from bisect import bisect_right


# FIXME ideally this would be achieved with a CombinedLRScheduler,
# separating MultiStepLR with WarmupLR
# but the current LRScheduler design doesn't allow it
class WarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            milestones,
            gamma=0.1,
            warmup_factor=1.0 / 3,
            warmup_iters=500,
            warmup_method="linear",
            last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of" " increasing integers. Got {}",
                milestones,
            )

        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = float(self.last_epoch) / self.warmup_iters
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        # print("base_lr {}, warmup_factor {}, self.gamma {}, self.milesotnes {}, self.last_epoch{}".format(
        #    self.base_lrs[0], warmup_factor, self.gamma, self.milestones, self.last_epoch))
        return [
            base_lr
            * warmup_factor
            * self.gamma ** bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def local_pcd(depth, intr):
    nx = depth.shape[1]  # w
    ny = depth.shape[0]  # h
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    x = x.reshape(nx * ny)
    y = y.reshape(nx * ny)
    p2d = np.array([x, y, np.ones_like(y)])
    p3d = np.matmul(np.linalg.inv(intr), p2d)
    depth = depth.reshape(1, nx * ny)
    p3d *= depth
    p3d = np.transpose(p3d, (1, 0))
    p3d = p3d.reshape(ny, nx, 3).astype(np.float32)
    return p3d


def generate_pointcloud(rgb, depth, ply_file, intr, scale=1.0):
    """
    Generate a colored point cloud in PLY format from a color and a depth image.

    Input:
    rgb_file -- filename of color image
    depth_file -- filename of depth image
    ply_file -- filename of ply file

    """
    fx, fy, cx, cy = intr[0, 0], intr[1, 1], intr[0, 2], intr[1, 2]
    points = []
    for v in range(rgb.shape[0]):
        for u in range(rgb.shape[1]):
            color = rgb[v, u]  # rgb.getpixel((u, v))
            Z = depth[v, u] / scale
            if Z == 0: continue
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append("%f %f %f %d %d %d 0\n" % (X, Y, Z, color[0], color[1], color[2]))
    file = open(ply_file, "w")
    file.write('''ply
            format ascii 1.0
            element vertex %d
            property float x
            property float y
            property float z
            property uchar red
            property uchar green
            property uchar blue
            property uchar alpha
            end_header
            %s
            ''' % (len(points), "".join(points)))
    file.close()
    print("save ply, fx:{}, fy:{}, cx:{}, cy:{}".format(fx, fy, cx, cy))


def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def torch_init_model(model, total_dict, key, rank=0):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    if rank == 0:
        print("missing keys:{}".format(missing_keys))
        print('unexpected keys:{}'.format(unexpected_keys))
        print('error msgs:{}'.format(error_msgs))


def load_config(config_path, display=False):
    config = OmegaConf.load(config_path)
    if display:
        print(yaml.dump(OmegaConf.to_container(config)))
    return config


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def get_lr_schedule_with_warmup(optimizer, num_warmup_steps, total_steps, min_lr, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        else:
            lr_weight = min_lr + (1. - min_lr) * 0.5 * (1. + math.cos(math.pi * (current_step - num_warmup_steps) / (total_steps - num_warmup_steps)))
        return lr_weight

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_parameter_groups(opt_args, model, freeze_vit=None):
    param_groups = []
    # normal params
    param_groups.append({"params": [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("vit.") and not n.startswith('encoder.')],
                         'lr': opt_args['lr'], 'weight_decay': 0.0}  )

    # if opt_args.get('vgg_lr', -1) > -1:
    vgg_lr = opt_args.get('vgg_lr', 1e-3)
    param_groups.append(
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("encoder.")],
         'lr': vgg_lr, 'weight_decay': 0.0})

    # vit params
    if freeze_vit is False:
        lora_cfg = getattr(model, "lora_cfg", None)
        if lora_cfg is not None and lora_cfg['rank'] > 0:  # use lora
            param_groups.append({"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.") and 'lora_' in n],
                                 'lr': opt_args['vit_lr'], 'weight_decay': opt_args['weight_decay']})
        else:
            param_groups.append({"params": [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("vit.")],
                                 'lr': opt_args['vit_lr'], 'weight_decay': opt_args['weight_decay']})

    return param_groups


def init_model(config):
    if 'DINOv2' in config['arch']['args']['model_type']:
        from models.networks.DINOv2_mvsformer_model import DINOv2MVSNet
        model = DINOv2MVSNet(config['arch']['args'])
    elif config['arch']['args']['model_type'] == 'casmvs':
        from models.networks.casmvs_model import CasMVSNet
        model = CasMVSNet(config['arch']['args'])
    else:
        raise NotImplementedError(f"Unknown model type {config['arch']['args']['model_type']}...")
    return model
