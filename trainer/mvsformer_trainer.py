import collections
import os
import pdb
import time
import torch
import cv2
import torch.distributed as dist
from tqdm import tqdm
import torch.utils.checkpoint as cp
from base import BaseTrainer
from models.losses import *
from utils import *


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, optimizer, config, data_loader, valid_data_loader=None, lr_scheduler=None, writer=None,
                 rank=0, ddp=False,
                 train_sampler=None, debug=False):
        super().__init__(model, optimizer, config, writer=writer, rank=rank, ddp=ddp)
        self.config = config
        self.ddp = ddp
        self.debug = debug
        self.data_loader = data_loader
        self.train_sampler = train_sampler
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler

        self.log_step = config['trainer']['logging_every']
        self.depth_type = self.config['arch']['args']['depth_type']
        self.ndepths = self.config['arch']['args']['ndepths']
        self.multi_scale = self.config['data_loader'][0]['args']['multi_scale']
        self.scale_batch_map = self.config['data_loader'][0]['args']['multi_scale_args']['scale_batch_map']
        # self.multi_ratio_config = self.config['data_loader'][0]['args']['multi_scale_args'].get('multi_ratio', None)
        # if self.multi_ratio_config is not None:
        #     self.multi_ratio_scale_batch_map = self.multi_ratio_config['scale_batch_map']
        # else:
        #     self.multi_ratio_scale_batch_map = None

        self.inverse_depth = config['arch']['args']['inverse_depth']

        self.loss_arg = config['arch']['loss']
        self.grad_norm = config['trainer'].get('grad_norm', None)
        self.dataset_name = config['arch']['dataset_name']
        self.train_metrics = DictAverageMeter()
        self.valid_metrics = DictAverageMeter()
        self.loss_downscale = config['optimizer']['args'].get('loss_downscale', 1.0)
        self.scale_dir = True
        if config['fp16'] is True:
            self.fp16 = True
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.fp16 = False
        self.bf16 = config["arch"].get("bf16", False)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """

        # if self.ddp:
        self.train_sampler.set_epoch(epoch)  # Shuffle each epoch
        if self.multi_scale:
            for i in range(len(self.data_loader)):
                self.data_loader[i].dataset.reset_dataset(self.train_sampler)


        self.model.train()
        dist_group = torch.distributed.group.WORLD

        global_step = 0
        scaled_grads = collections.defaultdict(list)
        pre_scale = int(self.scaler.get_scale()) if self.fp16 else None

        # training
        for dl in self.data_loader:
            if self.rank == 0:
                t_loader = tqdm(dl, desc=f"Epoch: {epoch}/{self.epochs}. Train.",
                                bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}', total=len(dl))
            else:
                t_loader = dl
            for batch_idx, sample in enumerate(t_loader):

                num_stage = 3 if self.config['data_loader'][0]['args'].get('stage3', False) else 4
                sample_cuda = tocuda(sample)
                depth_gt_ms = sample_cuda["depth"]
                mask_ms = sample_cuda["mask"]
                imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]
                depth_values = sample_cuda["depth_values"]
                depth_interval = depth_values[:, 1] - depth_values[:, 0]

                self.optimizer.zero_grad()

                # gradient accumulate
                if self.multi_scale:
                    # if self.multi_ratio_scale_batch_map is not None:
                    #     ratio_idx = dl.dataset.scale2idx[tuple(imgs.shape[3:5])]
                    #     bs = dl.dataset.scale_batch_map[str(ratio_idx)]
                    # else:
                    bs = self.scale_batch_map[str(imgs.shape[3])]
                else:
                    bs = imgs.shape[0]
                iters_to_accumulate = imgs.shape[0] // bs
                total_loss = torch.tensor(0.0, device="cuda")
                total_loss_dict = collections.defaultdict(float)

                for bi in range(iters_to_accumulate):
                    b_start = bi * bs
                    b_end = (bi + 1) * bs
                    cam_params_tmp = {}
                    depth_gt_ms_tmp = {}
                    mask_ms_tmp = {}
                    imgs_tmp = imgs[b_start:b_end]
                    for k in cam_params:
                        cam_params_tmp[k] = cam_params[k][b_start:b_end]
                        depth_gt_ms_tmp[k] = depth_gt_ms[k][b_start:b_end]
                        mask_ms_tmp[k] = mask_ms[k][b_start:b_end]

                    if self.fp16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.bf16 else torch.float16):

                            outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end])
                    else:

                        outputs = self.model.forward(imgs_tmp, cam_params_tmp, depth_values[b_start:b_end])

                    if type(self.depth_type) == list:
                        loss_dict = get_multi_stage_losses(self.loss_arg, self.depth_type, outputs, depth_gt_ms_tmp, mask_ms_tmp,
                                                           depth_interval[b_start:b_end], self.inverse_depth)
                    else:
                        loss_dict = get_loss(self.loss_arg, self.depth_type, outputs, depth_gt_ms_tmp,
                                             mask_ms_tmp, depth_interval[b_start:b_end], self.inverse_depth)

                    loss = torch.tensor(0.0, device="cuda")
                    for key in loss_dict:
                        loss = loss + loss_dict[key] / iters_to_accumulate
                        total_loss_dict[key] = total_loss_dict[key] + loss_dict[key] / iters_to_accumulate
                    total_loss += loss

                    if self.fp16:
                        self.scaler.scale(loss * self.loss_downscale).backward()
                    else:
                        (loss * self.loss_downscale).backward()

                if self.debug:
                    # DEBUG:scaled grad
                    with torch.no_grad():
                        for group in self.optimizer.param_groups:
                            for param in group["params"]:
                                if param.grad is None:
                                    continue
                                if param.grad.is_sparse:
                                    if param.grad.dtype is torch.float16:
                                        param.grad = param.grad.coalesce()
                                    to_unscale = param.grad._values()
                                else:
                                    to_unscale = param.grad
                                v = to_unscale.clone().abs().max()
                                if torch.isinf(v) or torch.isnan(v):
                                    print('Rank', str(self.rank) + ':', 'INF in', group['layer_name'], 'of step',
                                          global_step, '!!!')
                                scaled_grads[group['layer_name']].append(v.item() / self.scaler.get_scale())

                if self.grad_norm is not None:
                    if self.fp16:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm, error_if_nonfinite=False)

                if self.fp16:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    new_scale = int(self.scaler.get_scale())
                    if new_scale == pre_scale:  # 只有scale不变表示优化进行了
                        self.lr_scheduler.step()
                    pre_scale = new_scale
                else:
                    self.optimizer.step()
                    self.lr_scheduler.step()
                global_step = (epoch - 1) * len(dl) + batch_idx

                # forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1000.0 ** 2)
                # print(f"imgs shape:{imgs.shape},, iters_to_accumulate:{iters_to_accumulate}, max_mem: {forward_max_memory_allocated}")


                # no_grad_param = []
                # for name,param in self.model.named_parameters():
                #     if param.grad is None:
                #         no_grad_param.append(name)
                # xx = [n for n, p in self.model.named_parameters() if n.startswith("vit.")]
                # not_in_no_grad_param = [k for k in xx if k not in no_grad_param]
                # not_in_vit = [k for k in no_grad_param if k not in xx]
                # pdb.set_trace()

                if self.rank == 0:
                    desc = f"Epoch: {epoch}/{self.epochs}. " \
                           f"Train. " \
                           f"Scale:({imgs.shape[-2]}x{imgs.shape[-1]}), " \
                           f"Loss: {'%.2f' % total_loss.item()}"
                    # FIXME:Temp codes
                    if "stage4_uncertainty" in loss_dict:
                        desc += f", VarLoss: {'%.2f' % loss_dict['stage4_uncertainty'].item()}"

                    if self.fp16:
                        desc += ', scale={:d}'.format(int(self.scaler.get_scale()))
                    t_loader.set_description(desc)
                    t_loader.refresh()

                if self.debug and batch_idx % 50 == 0 and self.rank == 0:
                    scaled_grads_dict = {}
                    for k in scaled_grads:
                        scaled_grads_dict[k] = np.max(scaled_grads[k])
                    save_scalars(self.writer, 'grads', scaled_grads_dict, global_step)
                    scaled_grads = collections.defaultdict(list)

                if batch_idx % self.log_step == 0 and self.rank == 0:
                    scalar_outputs = {"loss": total_loss.item()}
                    for key in total_loss_dict:
                        scalar_outputs['loss_' + key] = loss_dict[key].item()
                    image_outputs = {"pred_depth": outputs['refined_depth'] * mask_ms_tmp[f'stage{num_stage}'],
                                     "pred_depth_nomask": outputs['refined_depth'],
                                     "conf": outputs['photometric_confidence'],
                                     "gt_depth": depth_gt_ms_tmp[f'stage{num_stage}'], "ref_img": imgs_tmp[:, 0]}
                    if self.fp16:
                        scalar_outputs['loss_scale'] = float(self.scaler.get_scale())
                    for i, lr_value in enumerate(self.lr_scheduler.get_last_lr()):
                        scalar_outputs[f'lr_{i}'] = lr_value
                    save_scalars(self.writer, 'train', scalar_outputs, global_step)
                    save_images(self.writer, 'train', image_outputs, global_step)
                    del scalar_outputs, image_outputs

        val_metrics = self._valid_epoch(epoch)

        if self.ddp:
            dist.barrier(group=dist_group)
            for k in val_metrics:
                dist.all_reduce(val_metrics[k], group=dist_group, async_op=False)
                val_metrics[k] /= dist.get_world_size(dist_group)
                val_metrics[k] = val_metrics[k].item()

        if self.rank == 0:
            save_scalars(self.writer, 'test', val_metrics, epoch)
            print("Global Test avg_test_scalars:", val_metrics)
        else:
            val_metrics = {'useless_for_other_ranks': -1}
        if self.ddp:
            dist.barrier(group=dist_group)

        return val_metrics

    def _valid_epoch(self, epoch, temp=False):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        temp: just for test
        :return: A log that contains information about validation
        """
        self.model.eval()
        seen_scans = set()
        global_val_metrics = {}
        with torch.no_grad():
            for val_data_idx, dl in enumerate(self.valid_data_loader):
                self.valid_metrics.reset()
                for batch_idx, sample in enumerate(tqdm(dl)):
                    sample_cuda = tocuda(sample)
                    depth_gt_ms = sample_cuda["depth"]
                    mask_ms = sample_cuda["mask"]
                    num_stage = 3 if self.config['data_loader'][0]['args'].get('stage3', False) else 4
                    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
                    mask = mask_ms["stage{}".format(num_stage)]

                    imgs, cam_params = sample_cuda["imgs"], sample_cuda["proj_matrices"]

                    depth_values = sample_cuda["depth_values"]
                    depth_interval = depth_values[:, 1] - depth_values[:, 0]
                    if self.fp16:
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if self.bf16 else torch.float16):
                            outputs = self.model.forward(imgs, cam_params, depth_values)
                    else:
                        outputs = self.model.forward(imgs, cam_params, depth_values)

                    depth_est = outputs["refined_depth"].detach()
                    if self.config['data_loader'][val_data_idx]['type'] == 'BlendedLoader':
                        scalar_outputs = collections.defaultdict(float)
                        for j in range(depth_interval.shape[0]):
                            di = depth_interval[j].item()
                            scalar_outputs_ = {
                                "abs_depth_thres0-2mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 2]),
                                "abs_depth_thres0-4mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 4]),
                                "abs_depth_thres0-8mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 8]),
                                "abs_depth_thres0-14mm_error": AbsDepthError_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, [0, di * 14]),
                                "thres2mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 2),
                                "thres4mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 4),
                                "thres8mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 8),
                                "thres14mm_error": Thres_metrics(depth_est[j:j + 1], depth_gt[j:j + 1], mask[j:j + 1] > 0.5, di * 14)}
                            for k in scalar_outputs_:
                                scalar_outputs[k] += scalar_outputs_[k]
                        for k in scalar_outputs:
                            scalar_outputs[k] /= depth_interval.shape[0]
                    else:
                        di = depth_interval[0].item() / 2.65
                        scalar_outputs = {"abs_depth_thres0-2mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 2]),
                                          "abs_depth_thres0-4mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 4]),
                                          "abs_depth_thres0-8mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 8]),
                                          "abs_depth_thres0-14mm_error": AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, [0, di * 14]),
                                          "thres2mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 2),
                                          "thres4mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 4),
                                          "thres8mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 8),
                                          "thres14mm_error": Thres_metrics(depth_est, depth_gt, mask > 0.5, di * 14)}

                    scalar_outputs = tensor2float(scalar_outputs)
                    image_outputs = {"pred_depth": outputs['refined_depth'][0:1] * mask[0:1], "gt_depth": depth_gt[0:1], "ref_img": imgs[0:1, 0]}

                    self.valid_metrics.update(scalar_outputs)

                    if self.rank == 0 and val_data_idx == 0:
                        # 每个scan存一张
                        filenames = sample['filename']
                        for filename in filenames:
                            scan = filename.split('/')[0]
                            if scan not in seen_scans:
                                seen_scans.add(scan)
                                save_images(self.writer, 'test', image_outputs, epoch, fname=scan)

                    if temp and batch_idx > 5:
                        break

                val_metrics = self.valid_metrics.mean()
                val_metrics['mean_error'] = val_metrics['thres2mm_error'] + val_metrics['thres4mm_error'] + \
                                            val_metrics['thres8mm_error'] + val_metrics['thres14mm_error']
                val_metrics['mean_error'] = val_metrics['mean_error'] / 4.0

                if len(self.valid_data_loader) > 1 and self.rank == 0:
                    print(f'Eval complete for dataset{val_data_idx}...')
                if val_data_idx > 0:
                    val_metrics_copy = {}
                    for k in val_metrics:
                        val_metrics_copy[f'ex_valset_{val_data_idx}_' + k] = val_metrics[k]
                    val_metrics = val_metrics_copy

                global_val_metrics.update(val_metrics)
                # save_scalars(self.writer, 'test', val_metrics, epoch)
                # print(f"Rank{self.rank}, avg_test_scalars:", val_metrics)

        for k in global_val_metrics:
            global_val_metrics[k] = torch.tensor(global_val_metrics[k], device=self.rank, dtype=torch.float32)
        self.model.train()

        return global_val_metrics
