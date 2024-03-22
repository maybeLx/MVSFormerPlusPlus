import math
from models.warping import homo_warping_3D_with_mask
from models.module import *
# from models.swin.block import SwinTransformerCostReg
import torch.utils.checkpoint as cp

class identity_with(object):
    def __init__(self, enabled=True):
        self._enabled = enabled

    def __enter__(self):
        pass

    def __exit__(self, *args):
        pass


autocast = torch.cuda.amp.autocast if torch.__version__ >= '1.6.0' else identity_with


class StageNet(nn.Module):
    def __init__(self, args, ndepth, stage_idx):
        super(StageNet, self).__init__()
        self.args = args
        self.fusion_type = args.get('fusion_type', 'cnn')
        self.ndepth = ndepth
        self.stage_idx = stage_idx
        self.cost_reg_type = args.get("cost_reg_type", ["Normal", "Normal", "Normal", "Normal"])[stage_idx]
        self.depth_type = args["depth_type"]
        if type(self.depth_type) == list:
            self.depth_type = self.depth_type[stage_idx]

        in_channels = args['base_ch']
        if type(in_channels) == list:
            in_channels = in_channels[stage_idx]
        if self.fusion_type == 'cnn':
            self.vis = nn.Sequential(ConvBnReLU(1, 16), ConvBnReLU(16, 16), ConvBnReLU(16, 8), nn.Conv2d(8, 1, 1), nn.Sigmoid())
        else:
            raise NotImplementedError(f"Not implemented fusion type: {self.fusion_type}.")

        if self.cost_reg_type == "PureTransformerCostReg":
            args['transformer_config'][stage_idx]['base_channel'] = in_channels
            self.cost_reg = PureTransformerCostReg(in_channels, **args['transformer_config'][stage_idx])
        else:
            model_th = args.get('model_th', 8)
            if ndepth <= model_th:  # do not downsample in depth range
                self.cost_reg = CostRegNet3D(in_channels, in_channels)
            else:
                self.cost_reg = CostRegNet(in_channels, in_channels)

    def forward(self, features, proj_matrices, depth_values, tmp, position3d=None):
        ref_feat = features[:, 0]
        src_feats = features[:, 1:]
        src_feats = torch.unbind(src_feats, dim=1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(src_feats) == len(proj_matrices) - 1, "Different number of images and projection matrices"

        # step 1. feature extraction
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        # step 2. differentiable homograph, build cost volume
        volume_sum = 0.0
        vis_sum = 0.0
        similarities = []
        with autocast(enabled=False):
            for src_feat, src_proj in zip(src_feats, src_projs):
                # warpped features
                src_feat = src_feat.to(torch.float32)
                src_proj_new = src_proj[:, 0].clone()
                src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
                ref_proj_new = ref_proj[:, 0].clone()
                ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
                warped_volume, proj_mask = homo_warping_3D_with_mask(src_feat, src_proj_new, ref_proj_new, depth_values)

                B, C, D, H, W = warped_volume.shape
                G = self.args['base_ch']
                if type(G) == list:
                    G = G[self.stage_idx]

                if G < C:
                    warped_volume = warped_volume.view(B, G, C // G, D, H, W)
                    ref_volume = ref_feat.view(B, G, C // G, 1, H, W).repeat(1, 1, 1, D, 1, 1).to(torch.float32)
                    in_prod_vol = (ref_volume * warped_volume).mean(dim=2)  # [B,G,D,H,W]
                elif G == C:
                    ref_volume = ref_feat.view(B, G, 1, H, W).to(torch.float32)
                    in_prod_vol = ref_volume * warped_volume  # [B,C(G),D,H,W]
                else:
                    raise AssertionError("G must <= C!")

                if self.fusion_type == 'cnn':
                    sim_vol = in_prod_vol.sum(dim=1)  # [B,D,H,W]
                    sim_vol_norm = F.softmax(sim_vol.detach(), dim=1)
                    entropy = (- sim_vol_norm * torch.log(sim_vol_norm + 1e-7)).sum(dim=1, keepdim=True)
                    vis_weight = self.vis(entropy)
                else:
                    raise NotImplementedError

                volume_sum = volume_sum + in_prod_vol * vis_weight.unsqueeze(1)
                vis_sum = vis_sum + vis_weight

            # aggregate multiple feature volumes by variance
            volume_mean = volume_sum / (vis_sum.unsqueeze(1) + 1e-6)  # volume_sum / (num_views - 1)

        cost_reg = self.cost_reg(volume_mean, position3d)

        prob_volume_pre = cost_reg.squeeze(1)
        prob_volume = F.softmax(prob_volume_pre, dim=1)

        if self.depth_type == 'ce':
            if self.training:
                _, idx = torch.max(prob_volume, dim=1)
                # vanilla argmax
                depth = torch.gather(depth_values, dim=1, index=idx.unsqueeze(1)).squeeze(1)
            else:
                # regression (t)
                depth = depth_regression(F.softmax(prob_volume_pre * tmp, dim=1), depth_values=depth_values)
            # conf
            photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]

        else:
            depth = depth_regression(prob_volume, depth_values=depth_values)
            if self.ndepth >= 32:
                photometric_confidence = conf_regression(prob_volume, n=4)
            elif self.ndepth == 16:
                photometric_confidence = conf_regression(prob_volume, n=3)
            elif self.ndepth == 8:
                photometric_confidence = conf_regression(prob_volume, n=2)
            else:  # D == 4
                photometric_confidence = prob_volume.max(1)[0]  # [B,H,W]

        outputs = {'depth': depth, 'prob_volume': prob_volume, "photometric_confidence": photometric_confidence.detach(),
                   'depth_values': depth_values, 'prob_volume_pre': prob_volume_pre}

        return outputs
