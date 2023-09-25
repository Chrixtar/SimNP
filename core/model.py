import os
import random
from typing import Dict, Optional, List
import warnings

from easydict import EasyDict as edict
import numpy as np
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch.nn import ModuleDict, Parameter
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm

from torch_knnquery import VoxelGrid

from . import detectors, pose_estimators, extractors, global_extractors, fields, renderers

from utils.camera import get_keypoint_projections, to_cam, img_to_world, get_cam_loc_in_world, mirror_world_to_cam
from utils.chamfer import chamfer_distance
from utils.data import trim_tensor_dict, recursive_to_cpu, join_tensor_dicts
from utils.log import log
from utils import loss
from utils.util import sparse_batched_mask_sampling, get_nested
from utils.vis import tb_image, tb_point_cloud, tb_point_cloud2, tb_attn, tb_attn2, dump_images, weights_to_colors, dump_emb_weights_as_alpha, dump_emb_weights_as_color, dump_emb_weights_as_heat# , dump_kp_pos_img


class Model(LightningModule):
    def __init__(
        self, 
        opt, 
        n_obj: int, 
        view_idx: Dict[str, Optional[List[int]]],
        n_repeats: int = 1
    ):
        super(Model, self).__init__()
        self.opt = opt
        os.makedirs(opt.output_path, exist_ok=True)
        
        use_detector_cache = getattr(self.opt, "use_detector_cache", False)
        self.detector = getattr(detectors, opt.model.detector.network)(3, opt.rendering.cube_scale, n_obj, opt.model.kp.num, predict_lrf=getattr(opt.model.kp, "lrf", False), use_cache=use_detector_cache, **opt.model.detector.kwargs)
        if getattr(opt.model, "extractor", None) is not None:
            if getattr(opt.model, "pose_estimator", None) is not None:
                self.pose_estimator = getattr(pose_estimators, opt.model.pose_estimator.network)(n_obj, opt.data.cam_dist, n_repeats, **opt.model.pose_estimator.kwargs)
            else:
                self.pose_estimator = None
            if getattr(opt.model, "voxel_grid", None) is not None:
                self.voxel_grid = VoxelGrid(**opt.model.voxel_grid)
            else:
                self.voxel_grid = None
            self.kp_ids = Parameter(torch.randn(opt.model.kp.num, opt.model.kp.id_dim))
            self.extractor = getattr(extractors, opt.model.extractor.network)(opt.model.kp.num, opt.model.kp.id_dim, opt.model.kp.feat_dim, n_obj, **opt.model.extractor.kwargs)
            kp_feat_dim = opt.model.kp.feat_dim
            if opt.model.kp.use_id_as_feat:
                kp_feat_dim += opt.model.kp.id_dim
            if getattr(opt.model, "global_extractor", None) is not None:
                self.global_extractor = getattr(global_extractors, opt.model.global_extractor.network)(n_obj, **opt.model.global_extractor.kwargs)
                global_dim = self.global_extractor.out_dim
            else:
                self.global_extractor = None
                global_dim = None
            self.field = getattr(fields, opt.model.field.network)(kp_feat_dim, global_dim, self.voxel_grid, opt.model.field.aggregator, **opt.model.field.kwargs, nerf=opt.model.field.nerf)
            self.renderer = getattr(renderers, opt.model.renderer.network)(self.field, white_back=opt.data.white_back, **(opt.model.renderer.kwargs | opt.rendering))
        else:
            self.extractor = None

        # Set train_metrics, val_metrics, test_metrics
        
        self.split_eval_per_view = {}
        for split in ("train", "val", "test"):
            if split in view_idx:
                split_metrics_options = getattr(opt.metrics, split, None)
                if split_metrics_options is not None:
                    self.split_eval_per_view[split] = split_metrics_options.pop("eval_per_view", False)
                    setattr(self, f"{split}_metrics", self.import_metrics(
                        split_metrics_options, 
                        split, 
                        view_idx[split] if self.split_eval_per_view[split] else None)
                    )
                    break
            setattr(self, f"{split}_metrics", None)

        self.same_train_val_split = getattr(self.opt.data.train, "split", "train") == getattr(self.opt.data.val, "split", "val")
        self.requires_detector_prior = hasattr(self.opt.loss_weight, "kp_prior") or hasattr(self.opt.loss_weight, "prior_chamfer_3d") or hasattr(self.opt.loss_weight, "kp_prior_latent")
        self.requires_detector_latent = hasattr(self.opt.loss_weight, "kp_prior_latent")

    @staticmethod
    def import_metrics(metric_options, split: str, view_idx=None):
        metric_list = []
        for name, kwargs in metric_options.items():
            for module in [torchmetrics, torchmetrics.image.lpip]:
                if hasattr(module, name):
                    metric_list.append(getattr(module, name)(**kwargs))
                    break
            else:
                raise ValueError(f"Cannot find desired metric {name}")
        metrics = torchmetrics.MetricCollection(metric_list)
        split_metrics = ModuleDict({"all": metrics.clone(prefix=f"{split}/metric/all/")})
        if view_idx is not None:
            for i in view_idx:
                split_metrics.add_module(f"view_{i}", metrics.clone(prefix=f"{split}/metric/view_{i}/"))
        return split_metrics


    def state_dict(self, *args, **kwargs):
        # Remove metrics from state_dict
        sd: Dict[str, Tensor] = super(Model, self).state_dict(*args, **kwargs)
        return {k: v for k, v in sd.items() if "metrics" not in k}

    def configure_optimizers(self):
        log.info("setting up optimizers...")
        self.optimizer = getattr(torch.optim,self.opt.optim.algo)
        optim_list = [dict(params=filter(lambda p: p.requires_grad, self.parameters()),lr=self.opt.optim.lr),]
        res = {"optimizer": self.optimizer(optim_list)}
        # self.optim = self.optimizer(optim_list)
        if self.opt.optim.sched: # self.setup_optimizer_scheduler(opt)
            scheduler = getattr(torch.optim.lr_scheduler, self.opt.optim.sched.type)
            lr_scheduler_config = {k: v for k, v in self.opt.optim.sched.items() if k not in ("type", "kwargs")}
            lr_scheduler_config["scheduler"] = scheduler(res["optimizer"], **self.opt.optim.sched.kwargs)
            res["lr_scheduler"] = lr_scheduler_config
        return res

    def get_field_input_feat(self, kp_feat: Tensor) -> Tensor:
        if self.opt.model.kp.use_id_as_feat:
            kp_feat = torch.cat((self.kp_ids.expand(*kp_feat.shape[:-1], -1), kp_feat), dim=-1)
        return kp_feat
    
    def detect_keypoints(self, var, use_emb: bool = True):
        if getattr(self.opt.data, "use_gt_scale", True) and get_nested(var, "dpc", "scale") is not None:
            apply_scale = True
            scale = var.dpc.scale
        else:
            apply_scale = use_emb
            scale = None
        if getattr(self.opt.data, "use_gt_offset", True) and get_nested(var, "dpc", "offset") is not None:
            apply_offset = True
            offset = var.dpc.offset
        else:
            apply_offset = use_emb
            offset = None
        var.pred = edict(self.detector(
            var.obj_idx, 
            get_nested(var, "views", "img"),
            mask=get_nested(var, "views", "mask"),
            pix=get_nested(var, "views", "pix"),
            extr=get_nested(var, "views", "extr"),
            intr=get_nested(var, "views", "intr"),
            scale=scale,
            offset=offset,
            apply_scale=apply_scale,
            apply_offset=apply_offset,
            return_kp_pos=use_emb,
            return_prior=self.requires_detector_prior,
            return_latent=self.requires_detector_latent
        ))

    def setup_voxelgrid(self, kp_pos: Tensor) -> None:
        if self.voxel_grid is not None:
            self.voxel_grid.set_pointset(
                kp_pos.detach(), 
                torch.full(
                    (kp_pos.shape[0],), 
                    fill_value=self.opt.model.kp.num, 
                    device=kp_pos.device, 
                    dtype=torch.int
                )
            )
    
    def extract_feat(self, var):
        # [B, n_kp, feat_dim]
        var.pred.extracted_feat, var.pred.extractor_inter_res = self.extractor(
            idx=var.obj_idx,
            kp_pos=var.pred.canonical_kp_pos,  
            kp_ids=self.kp_ids, 
        )
        var.pred.global_feat = self.global_extractor(var.obj_idx) if self.global_extractor is not None else None

    def render(
        self, 
        var, 
        kp_pos: Tensor, 
        sample: bool, 
        kp_weights: bool = False, 
        pose_estimation: bool = False
    ):
        if getattr(self.opt.data, "use_gt_scale", False) and get_nested(var, "dpc", "scale") is not None:
            scale = var.dpc.scale
        else:
            scale = None
        if getattr(self.opt.data, "use_gt_offset", False) and get_nested(var, "dpc", "offset") is not None:
            offset = var.dpc.offset
        else:
            offset = None
        kp_feat = self.get_field_input_feat(var.pred.extracted_feat)
        if self.pose_estimator is not None and pose_estimation:
            assert var.views.idx.shape[1] == 1, "Pose estimation only compatible with num_views == 1"
            extr = var.pred.extr = self.pose_estimator(var.obj_idx).unsqueeze(1)
        else:
            extr = var.views.extr
        if self.should_be_detached("kp_feat"):
            kp_feat = kp_feat.detach()
        if self.should_be_detached("extr"):
            extr = extr.detach()
        var.pred.views = self.renderer(
            kp_pos,
            getattr(var.pred, "kp_rot", None),
            getattr(var.pred, "kp_mir", None),
            kp_feat,
            var.pred.global_feat,
            offset,
            scale,
            getattr(var.views, "pix", None),
            extr,
            var.views.intr,
            resolution=self.opt.sizes.render,
            sample=sample,
            global_step=self.global_step,
            return_channels=True,
            return_kp_weights=kp_weights
        )

    def forward(
        self, 
        var: edict, 
        sample: bool, 
        kp_weights: bool = False,
        use_emb: bool = True, 
        apply_kp_noise: bool = False,
        pose_estimation: bool = False
    ) -> edict:
        self.detect_keypoints(var, use_emb)
        if self.extractor is not None:
            kp_pos = var.pred.kp_pos
            if apply_kp_noise and hasattr(self.opt, "kp_noise"):
                if hasattr(self.opt.kp_noise, "std"):
                    kp_pos = kp_pos + torch.randn_like(kp_pos) * self.opt.kp_noise.std
                if hasattr(self.opt.kp_noise, "offset_std"):
                    kp_pos = kp_pos + torch.randn(3, device=kp_pos.device) * self.opt.kp_noise.offset_std
                if hasattr(self.opt.kp_noise, "scale_std"):
                    kp_pos = kp_pos * (1 + torch.randn(1, device=kp_pos.device) * self.opt.kp_noise.scale_std)
            
            self.setup_voxelgrid(kp_pos)
            self.extract_feat(var)
            self.render(var, kp_pos, sample, kp_weights, pose_estimation=pose_estimation)
        return var

    @staticmethod
    def subsample_gt(gt_map, ray_idx):
        samples = gt_map.flatten(-2, -1).transpose(-1, -2)
        if ray_idx is not None:
            samples = samples.expand(*ray_idx.shape[:-2], *samples.shape[-2:])
            samples = samples.gather(
                dim=-2, 
                index=ray_idx.expand(*ray_idx.shape[:-1], samples.shape[-1])
            )
        return samples

    @staticmethod
    def unflatten_pred(pred):
        res = pred.transpose(-1, -2)
        m = res.shape[-1]
        side = round(m ** (0.5))
        return res.reshape(*res.shape[:-1], side, side)

    def should_be_detached(self, t: str):
        if hasattr(self.opt, "detach_schedule") and hasattr(self.opt.detach_schedule, t):
            schedule = getattr(self.opt.detach_schedule, t)
            return max(self.current_epoch-schedule[2], 0) % schedule[0] < schedule[1]
        return False

    def is_active_loss(self, loss_name: str, split: str):
        loss_weight = getattr(self.opt.loss_weight, split, self.opt.loss_weight)
        if not hasattr(loss_weight, loss_name):
            return False
        if hasattr(self.opt, "loss_schedule"):
            if hasattr(self.opt.loss_schedule, loss_name):
                loss_schedule = getattr(self.opt.loss_schedule, loss_name)
                left =  loss_schedule[0] is None or loss_schedule[0] <= self.current_epoch
                right = loss_schedule[1] is None or self.current_epoch < loss_schedule[1]
                return left and right
        return True

    def compute_loss(self, var, split: str):
        res = edict()
        pred = var.pred
        loss_weight = getattr(self.opt.loss_weight, split, self.opt.loss_weight)
        if hasattr(pred, "views"):
            pred = pred.views
            gt = var.views
            weight = self.subsample_gt(gt.weight, pred.get("ray_idx", None)) \
                if hasattr(gt, "weight") else None
            if self.is_active_loss("img", split):
                img = self.subsample_gt(gt.img, pred.get("ray_idx", None))
                res.img = edict()
                for l in loss_weight.img.keys():
                    res.img[l] = getattr(loss, l)(pred.channels, img, weight=weight)
            if self.is_active_loss("mask", split):
                mask = self.subsample_gt(gt.mask, pred.get("ray_idx", None))
                res.mask = loss.bce(pred.mask, mask, weight=weight)
            if self.is_active_loss("eikonal", split):
                gradients = pred.grad[pred.valid_pts_mask.expand(*pred.valid_pts_mask.shape[:-1], 3)]
                res.eikonal = loss.mse(torch.linalg.norm(gradients, ord=2, dim=-1), 1.0)

        # KP position losses
        if self.is_active_loss("kp_chamfer_3d", split):
            res.kp_chamfer_3d = self.kp_chamfer_3d_loss(var)
        if self.is_active_loss("kp_prior", split):
            res.kp_prior = self.kp_prior_loss(var)
        if self.is_active_loss("kp_chamfer_2d", split):
            res.kp_chamfer_2d = self.kp_chamfer_2d_loss(var)
        if self.is_active_loss("kp_negative_depth", split):
            res.kp_negative_depth = self.kp_negative_depth_loss(var)
        if self.is_active_loss("kp_depth", split):
            res.kp_depth, var.views.depth_pc = self.kp_depth_loss(var)
        if self.is_active_loss("kp_surface", split):
            res.kp_surface = self.kp_surface_loss(var)

        # KP Pos Prior losses
        if self.is_active_loss("prior_chamfer_3d", split):
            res.prior_chamfer_3d = self.prior_chamfer_3d_loss(var)
        if self.is_active_loss("kp_prior_latent", split):
            res.kp_prior_latent = self.kp_prior_latent_loss(var)
        
        # KP Rot losses
        if self.is_active_loss("rot_prior", split):
            res.rot_prior = self.rot_prior_loss(var)
        if self.is_active_loss("mir_prior", split):
            res.mir_prior = self.mir_prior_loss(var)

        if self.is_active_loss("pose", split):
            res.pose = self.pose_loss(var)

        return res
    
    def compute_metrics(self, var, split="train"):
        if hasattr(self, f"{split}_metrics"):
            metrics = getattr(self, f"{split}_metrics")
            pred = var.pred
            if hasattr(pred, "views"):
                pred = pred.views
                gt = var.views
                if hasattr(pred, "ray_idx"):
                    g = self.subsample_gt(gt.img, pred.ray_idx)
                    p = pred.channels
                else:
                    g = gt.img
                    p = self.unflatten_pred(pred.channels)
                # Flatten batch and view dimensions
                g = g.flatten(0, 1)
                p = p.flatten(0, 1).detach()
                view_idx = gt.name.flatten()
                metrics["all"](p, g)
                if self.split_eval_per_view[split]:
                    unique_view_idx = torch.unique(view_idx)
                    for idx_t in unique_view_idx:
                        i = idx_t.item()
                        mask = view_idx == i
                        metrics[f"view_{i}"](p[mask], g[mask])

    @staticmethod
    def get_cam_loc_mask(cam_masking, extr):
        cam_locs = get_cam_loc_in_world(extr)
        masks = []
        for idx, name in enumerate(("x", "y", "z")):
            if hasattr(cam_masking, name):
                interval = getattr(cam_masking, name)
                if interval[0] is not None:
                    masks.append(interval[0] <= cam_locs[..., idx])
                if interval[1] is not None:
                    masks.append(cam_locs[..., idx] <= interval[1])
        masks = torch.stack(masks)
        mask = masks.any(dim=0)
        return mask

    @staticmethod
    def align_kp_scale(gt, pred):
        """
        Arguments:
            gt: [..., num_kp, 3]
            pred: [..., num_kp, 3]
        """
        gt_scale = torch.linalg.norm(gt.abs().max(dim=-2).values, dim=-1)       # [...]
        pred_scale = torch.linalg.norm(pred.abs().max(dim=-2).values, dim=-1)   # [...]
        scale_factor = gt_scale / pred_scale
        return scale_factor[..., None, None] * pred

    def kp_chamfer_3d_loss(self, var):
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "kp_chamfer_3d") and hasattr(self.opt.loss.kp_chamfer_3d, "canonical"):
            gt = var.dpc.points
            pred = var.pred.canonical_kp_pos
        else:
            gt = var.dpc.points
            if hasattr(var.dpc, "scale"):
                gt = gt * var.dpc.scale[:, None]
            if hasattr(var.dpc, "offset"):
                gt = gt + var.dpc.offset[:, None]
            pred = var.pred.kp_pos
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "kp_chamfer_3d") and hasattr(self.opt.loss.kp_chamfer_3d, "scale_invariant"):
            pred = self.align_kp_scale(gt, pred)
        cd = chamfer_distance(pred, gt)
        res = get_nested(self.opt, "loss", "kp_chamfer_3d", "pred_gt", default=1) * cd["cham_xy"] \
            + get_nested(self.opt, "loss", "kp_chamfer_3d", "gt_pred", default=1) * cd["cham_yx"]
        return res
    
    def prior_chamfer_3d_loss(self, var):
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "prior_chamfer_3d") and hasattr(self.opt.loss.prior_chamfer_3d, "canonical"):
            gt = var.dpc.points
            pred = var.pred.canonical_prior_kp_pos
        else:
            gt = var.dpc.points
            if hasattr(var.dpc, "scale"):
                gt = gt * var.dpc.scale[:, None]
            if hasattr(var.dpc, "offset"):
                gt = gt + var.dpc.offset[:, None]
            pred = var.pred.prior_kp_pos
        gt = gt.unsqueeze(1)    # Add view dimension
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "prior_chamfer_3d") and hasattr(self.opt.loss.prior_chamfer_3d, "scale_invariant"):
            pred = self.align_kp_scale(gt, pred)
        num_views, num_kp = pred.shape[1:3]
        gt = gt.expand(-1, num_views, -1, -1)
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "prior_chamfer_3d") and hasattr(self.opt.loss.prior_chamfer_3d, "cam_masking"):
            mask = self.get_cam_loc_mask(self.opt.loss.prior_chamfer_3d.cam_masking, var.views.extr)[..., None, None]
            pred = torch.masked_select(pred, mask)
            gt = torch.masked_select(gt, mask)
        pred = pred.view(-1, num_kp, 3)
        gt = gt.reshape(pred.shape[0], -1, 3)
        cd = chamfer_distance(pred, gt)
        res = cd["cham_xy"] + cd["cham_yx"]
        return res
    
    def kp_prior_loss(self, var):
        canon_kp = var.pred.canonical_kp_pos.unsqueeze(1)   # add view dimension
        prior_kp = var.pred.canonical_prior_kp_pos  # [B, num_views, num_kp, 3]
        canon_kp = canon_kp.expand(*prior_kp.shape)
        if hasattr(self.opt, "loss") and hasattr(self.opt.loss, "kp_prior"):
            if hasattr(self.opt.loss.kp_prior, "cam_masking"):
                num_kp = canon_kp.shape[2]
                mask = self.get_cam_loc_mask(self.opt.loss.kp_prior.cam_masking, var.views.extr)[..., None, None].expand(*prior_kp.shape)
                canon_kp = canon_kp[mask].view(-1, num_kp, 3)
                prior_kp = prior_kp[mask].view(-1, num_kp, 3)
            if getattr(self.opt.loss.kp_prior, "detach_canon", False):
                canon_kp = canon_kp.detach()
            if getattr(self.opt.loss.kp_prior, "detach_prior", False):
                prior_kp = prior_kp.detach()
        return loss.mse(canon_kp, prior_kp)
    
    def kp_prior_latent_loss(self, var):
        return loss.mse(var.pred.kp_latent.unsqueeze(1), var.pred.prior_kp_latent)

    def rot_prior_loss(self, var):
        return loss.mse(var.pred.kp_rot.unsqueeze(1), var.pred.prior_kp_rot)
    
    def mir_prior_loss(self, var):
        return loss.mse(var.pred.kp_mir.unsqueeze(1), var.pred.prior_kp_mir)
    
    def kp_negative_depth_loss(self, var):
        kp_pos = to_cam(var.pred.kp_pos.unsqueeze(1), var.views.extr)      # [B, num_views, num_kp, 3]
        neg_depth = torch.clamp(kp_pos[..., 2], max=0)
        res = torch.mean(neg_depth ** 2)
        return res

    @staticmethod
    def chamfer_2d_loss(
        kp_pos: Tensor, 
        mask: Tensor, 
        extr: Tensor, 
        intr: Tensor, 
        pix_scale: Optional[Tensor] = None, 
        pix_offset: Optional[Tensor] = None,
    ):
        """
        kp_pos: [B, n_kp, 3]
        mask: [B, num_views, 1, H, W]
        extr: [B, num_views, 4, 4]
        intr: [B, num_views, 3, 3]
        pix_scale: [B, num_views, 1]
        pix_offset: [B, num_views, 2]
        """
        device = mask.device
        # width, height
        img_size = torch.tensor(mask.shape[-2:], device=mask.device).flip(-1)
        # [B, num_views, num_kp, 2]
        kp_pix = get_keypoint_projections(
            kp_pos.unsqueeze(1), 
            extr, 
            intr,
            pix_scale,
            pix_offset
        )[0]
        if torch.any(torch.isinf(kp_pix)):
            return torch.tensor(0., device=device) 
        # Flatten batch and view dimensions
        mask = mask.flatten(0, 2) >= 0.5    # [B * num_views, H, W]
        kp_pix = kp_pix.flatten(0, 1)
        # Sample equal number of nnz elements from mask
        mask_pix = sparse_batched_mask_sampling(mask)   # [B, min_count, 2]
        if mask_pix is None:
            return torch.tensor(0., device=device)
        # Flip to obtain x, y coordinates
        mask_pix = mask_pix.flip(-1) + 0.5
        # Normalize coordinates
        kp_pix = kp_pix / img_size - 0.5
        mask_pix = mask_pix / img_size - 0.5
        cd = chamfer_distance(kp_pix, mask_pix)
        res = cd["cham_xy"] + cd["cham_yx"]
        return res

    def kp_chamfer_2d_loss(self, var):
        return self.chamfer_2d_loss(
            var.pred.kp_pos, 
            var.views.mask, 
            var.views.extr, 
            var.views.intr,
            getattr(var.views, "pix_scale", None),
            getattr(var.views, "pix_offset", None)
        )
    
    def pose_loss(self, var):
        return self.chamfer_2d_loss(
            var.pred.kp_pos.detach(), 
            var.views.mask, 
            var.pred.extr, 
            var.views.intr,
            getattr(var.views, "pix_scale", None),
            getattr(var.views, "pix_offset", None)
        )

    @staticmethod
    def get_gt_depth_samples(depth, mask: Optional[Tensor] = None):
        depth = depth.flatten(0, 2).unsqueeze(-1)
        depth_mask = depth[..., 0] >= 0
        if mask is not None:
            mask = mask.flatten(0, 2) >= 0.5
            depth_mask = torch.logical_and(depth_mask, mask)
        samples = sparse_batched_mask_sampling(depth_mask, depth) # [B*num_views, min_count, 3]
        return samples

    @staticmethod
    def depth_to_world_samples(samples, extr, intr):
        B, num_views = extr.shape[:2]
        samples = samples[..., -3:].view(B, num_views, -1, 3)
        # Flip to obtain x, y coordinates
        sample_pix = samples[..., :2].flip(-1) + 0.5
        sample_depth = samples[..., -1]
        sample_world = img_to_world(sample_pix, intr, extr, sample_depth)   # [B, num_views, min_count, 3]
        return sample_world

    @staticmethod
    def kp_depth_loss_helper(kp_pos, samples, extr, intr):
        B, num_views = extr.shape[:2]
        device = extr.device
        if samples is None:
            return torch.tensor(0., device=device), None
        sample_world = Model.depth_to_world_samples(samples, extr, intr)
        sample_world = sample_world.flatten(0, 1)
        kp_pos = kp_pos.unsqueeze(1).expand(-1, num_views, -1, -1).flatten(0, 1)
        res = chamfer_distance(sample_world, kp_pos, return_yx=False)["cham_xy"]
        return res, sample_world.view(B, num_views, -1, 3)

    def kp_depth_loss(self, var):
        if not hasattr(self.opt, "loss") or not hasattr(self.opt.loss, "kp_depth") or getattr(self.opt.loss.kp_depth, "gt", True):
            samples = self.get_gt_depth_samples(var.views.depth, var.views.mask)  # [B*num_views, min_count, 3]
        else:
            ray_idx = var.pred.views.get("ray_idx", None)
            mask = var.pred.views.mask > 0.5
            depth = var.pred.views.depth.detach()
            if ray_idx is None:
                # mask: [B, num_tar, self.opt.sizes.render**2, 1]
                mask = mask.view(-1, self.opt.sizes.render, self.opt.sizes.render)
                depth = depth.view(-1, self.opt.sizes.render, self.opt.sizes.render, 1)
                samples = sparse_batched_mask_sampling(mask, depth) # [B*num_views, min_count, 3]
            else:
                mask = mask.flatten(0, 1)[..., 0]   # [B*num_views, num_sample_rays]
                depth = depth.flatten(0, 1)
                idx = ray_idx.flatten(0, 1)
                data = torch.cat((torch.div(idx, self.opt.sizes.render, rounding_mode="trunc"), idx % self.opt.sizes.render, depth), dim=-1)
                samples = sparse_batched_mask_sampling(mask, data) # [B*num_views, min_count, 4]
        return self.kp_depth_loss_helper(var.pred.kp_pos, samples, var.views.extr, var.views.intr)

    def kp_surface_loss(self, var):
        x = var.pred.kp_pos.view(-1, 1, self.opt.model.kp.num, 1, 3)    # [B, 1, n_kp, 1, 3] 
        kp_feat = self.get_field_input_feat(var.pred.extracted_feat)    # [B, n_kp, feat_dim]
        decoded = self.field(
            x,
            None,
            var.pred.kp_pos.detach(),
            kp_feat.detach(),
            var.pred.global_feat.detach(),
            sample=False,
            return_channels=False,
            return_mask=True,
            return_pts=False,
            return_grad=False,
            return_kp_weights=False
        )
        kp_sdf = decoded["shape"][decoded["mask"]].view(-1, self.opt.model.kp.num)
        res = loss.mse(kp_sdf, 0)
        return res

    def training_step(self, batch, batch_idx):
        var = edict(batch)
        batch_size = var.views.name.numel()
        vis = self.global_step % self.opt.freq.vis == 0
        # vis = self.trainer.is_last_batch
        if vis:
            if self.opt.vis_batch_size < self.opt.train_batch_size:
                batch_size = min(batch_size, self.opt.vis_batch_size)
                var = trim_tensor_dict(var, self.opt.vis_batch_size, dim=0)
            if self.opt.vis_view_size < self.opt.train_view_size:
                var.views = trim_tensor_dict(var.views, self.opt.vis_view_size, dim=1)
        with torch.set_grad_enabled(not vis):   # disable grad for visualization because of too much memory consumption
            var = self.forward(var, sample=not vis, apply_kp_noise=True, pose_estimation=True)
            loss = self.compute_loss(var, split="train")
            loss = self.summarize_loss(loss, split="train")
            self.compute_metrics(var, split="train")
            self.log_scalars(loss, batch_size=batch_size, split="train")
        if vis or not (isinstance(loss.all, Tensor) and loss.all.requires_grad):
            loss.all = torch.zeros(1, device=var.obj_idx.device, requires_grad=True)
        return {"loss": loss.all, "var": recursive_to_cpu(var) if vis else None, "step": self.global_step}

    def training_epoch_end(self, outputs):
        outputs = filter(lambda out: out["var"] is not None, outputs)
        for out in outputs:
            self.visualize(var=out["var"], step=out["step"], split="train")

    def _eval(self, batch, split: str, kp_weights: bool = False, use_emb: bool = True):
        var = edict(batch)
        batch_size = var.views.name.numel()
        var = self.forward(var, sample=False, kp_weights=kp_weights, use_emb=use_emb)
        loss = self.compute_loss(var, split=split)
        loss = self.summarize_loss(loss, split=split)
        self.compute_metrics(var, split=split)
        self.log_scalars(loss, batch_size=batch_size, split=split, epoch=True)
        return var
        # return var, loss

    def on_validation_epoch_start(self) -> None:
        self.val_vis_batches = set(random.sample(
            range(self.trainer.num_val_batches[0]), 
            min(self.opt.vis.num_batches, self.trainer.num_val_batches[0]))
        )

    def validation_step(self, batch, batch_idx):
        var = self._eval(batch, "val", use_emb=self.same_train_val_split)
        return recursive_to_cpu(var) if batch_idx in self.val_vis_batches else None
    
    def validation_epoch_end(self, val_step_outputs):
        vars = list(filter(lambda var: var is not None, val_step_outputs))
        if vars:
            var = join_tensor_dicts(vars, dim=0)
            self.visualize(var, step=self.global_step, split="val")

    def test_step(self, batch, batch_idx):
        var = self._eval(batch, "test", kp_weights=True)
        if getattr(self.opt, "dump_vis", False) and (not getattr(self.opt, "skip_existing", False) or not self.dump_exists(var)):
            self.dump_results(var)

    def dump_exists(self, var):
        dump_dir = getattr(self.opt, "dump_dir", "dump")
        for i, o in enumerate(var.obj_name):
            # if os.path.isfile(os.path.join(self.opt.output_path, dump_dir, o, "image", "251.png")):
            #     continue
            for v_t in var.views.name[i]:
                v = v_t.item()
                if not os.path.isfile(os.path.join(self.opt.output_path, dump_dir, o, "image", str(v) + ".png")):
                    return False
        return True

    def predict_step(self, batch, batch_idx):
        var = edict(batch)
        # TODO remove this again
        if getattr(self.opt, "skip_existing", False) and self.dump_exists(var):
            return None
        var = self.forward(var, sample=False, kp_weights=True)
        if getattr(self.opt, "dump_vis", False):
            self.dump_results(var)
        return var

    @staticmethod
    def summarize_loss_helper(loss, weights):
        loss_all = 0
        # weigh losses
        for key in loss:
            assert(key in weights), f"Key {key} not in loss_weight"
            if isinstance(loss[key], edict):
                weighted = Model.summarize_loss_helper(loss[key], weights[key])
            else:
                assert(loss[key].shape==())
                assert not torch.isinf(loss[key]),"loss {} is Inf".format(key)
                assert not torch.isnan(loss[key]),"loss {} is NaN".format(key)
                weighted = float(weights[key])*loss[key]
            loss_all = loss_all + weighted
        return loss_all

    def summarize_loss(self, loss, split):
        weights = getattr(self.opt.loss_weight, split, self.opt.loss_weight)
        loss_all = Model.summarize_loss_helper(loss, weights)
        loss.update(all=loss_all)
        return loss

    @staticmethod
    def flatten_scalars(scalars: edict, prefix: str = "") -> dict:
        res = {}
        for key, value in scalars.items():
            flattened_key = f"{prefix}/{key}"
            if isinstance(value, edict):
                res.update(Model.flatten_scalars(value, prefix=flattened_key))
            else:
                res[flattened_key] = value
        return res

    def log_scalars(self, loss, batch_size, log_metrics=True, split="train", epoch=False):
        loss = self.flatten_scalars(loss, prefix=f"{split}/loss")
        self.log_dict(loss, on_step=not epoch, on_epoch=epoch, batch_size=batch_size)
        metrics = getattr(self, f"{split}_metrics", None)
        if log_metrics and metrics is not None:
            for view_metrics in metrics.values():
                self.log_dict(view_metrics, on_step=not epoch, on_epoch=epoch, batch_size=batch_size)
    
    @staticmethod
    def depth_to_vis(depth):
        min = torch.amin(depth, dim=(-2, -1), keepdim=True)
        max = torch.amax(depth, dim=(-2, -1), keepdim=True)
        return 1 - (depth - min) / (max - min)

    @torch.no_grad()
    def visualize(self, var, step=0, split="train"):
        pred = var.pred
        gt = var.views
        exist_tb_split_opt = hasattr(self.opt.tb, split)
        tb_split_opt = getattr(self.opt.tb, split, None)
        if hasattr(pred, "kp_pos"):
            if not exist_tb_split_opt or getattr(tb_split_opt, "kp_pos_img", True):
                tb_image(self.opt, self.logger.experiment, step, split, "kp_pos_img", gt.img, kp=pred.kp_pos, extr=gt.extr, intr=gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), from_range=(0, 1))
            if not exist_tb_split_opt or getattr(tb_split_opt, "kp_pos_pc_img", True):
                tb_point_cloud(self.opt, self.logger.experiment, step, split, "kp_pos_pc_img", pred.canonical_kp_pos, gt.extr, gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), gt=var.dpc.points)
            if not exist_tb_split_opt or getattr(tb_split_opt, "kp_pos_pc", True):
                tb_point_cloud2(self.opt, self.logger.experiment, step, split, pred.canonical_kp_pos, var.dpc.points)
            if hasattr(pred, "extr") and (not exist_tb_split_opt or getattr(tb_split_opt, "pose_img", True)):
                tb_image(self.opt, self.logger.experiment, step, split, "pose_img", gt.img, kp=pred.kp_pos, extr=pred.extr, intr=gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), from_range=(0, 1))
        if hasattr(pred, "prior_kp_pos"):
            if not exist_tb_split_opt or getattr(tb_split_opt, "kp_prior_img", True):
                tb_image(self.opt, self.logger.experiment, step, split, "kp_prior_img", gt.img, kp=pred.prior_kp_pos, extr=gt.extr, intr=gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), from_range=(0, 1))
            if not exist_tb_split_opt or getattr(tb_split_opt, "kp_prior_pc_img", True):
                tb_point_cloud(self.opt, self.logger.experiment, step, split, "kp_prior_pc_img", pred.canonical_prior_kp_pos, gt.extr, gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), gt=var.dpc.points)
        if hasattr(gt, "depth_pc") and gt.depth_pc is not None:
            if not exist_tb_split_opt or getattr(tb_split_opt, "depth_pc_img", True):
                tb_image(self.opt, self.logger.experiment, step, split, "depth_pc_img", gt.img, kp=gt.depth_pc, extr=gt.extr, intr=gt.intr, from_range=(0, 1))
            if not exist_tb_split_opt or getattr(tb_split_opt, "depth_pc_pc_img", True):
                tb_point_cloud(self.opt, self.logger.experiment, step, split, "depth_pc_pc_img", gt.depth_pc, gt.extr, gt.intr, gt=var.dpc.points)
        
        if hasattr(pred, "extractor_inter_res") and "attn_map" in pred.extractor_inter_res:
            if not exist_tb_split_opt or getattr(tb_split_opt, "attn_pc_color", True):
                tb_attn(self.opt, self.logger.experiment, step, split, pred.canonical_kp_pos, pred.extractor_inter_res["attn_map"])
            if not exist_tb_split_opt or getattr(tb_split_opt, "attn_pc_gray", True):
                tb_attn2(self.opt, self.logger.experiment, step, split, pred.canonical_kp_pos, pred.extractor_inter_res["attn_map"])

        if not exist_tb_split_opt or getattr(tb_split_opt, "image_gt", True):
            tb_image(self.opt, self.logger.experiment, step, split, "image_gt", gt.img, extr=gt.extr, from_range=(0, 1))
        if hasattr(gt, "mask"):
            if not exist_tb_split_opt or getattr(tb_split_opt, "mask_gt", True):
                tb_image(self.opt, self.logger.experiment, step, split, "mask_gt", gt.mask)
        if hasattr(pred, "views"):
            p = pred.views
            if not exist_tb_split_opt or getattr(tb_split_opt, "image_pred", True):
                tb_image(self.opt, self.logger.experiment, step, split, "image_pred", self.unflatten_pred(p.channels), from_range=(0, 1))
            if hasattr(p, "kp_weights") and (not exist_tb_split_opt or getattr(tb_split_opt, "kp_weights", True)):
                color_map = self.unflatten_pred(weights_to_colors(p.kp_weights, self.opt.vis.kp.cmap))
                tb_image(self.opt, self.logger.experiment, step, split, "kp_weights", color_map)
            if hasattr(p, "depth") and (not exist_tb_split_opt or getattr(tb_split_opt, "depth_pred", True)):
                tb_image(self.opt, self.logger.experiment, step, split, "depth_pred", self.depth_to_vis(self.unflatten_pred(p.depth)))
            if hasattr(p, "mask") and (not exist_tb_split_opt or getattr(tb_split_opt, "mask_pred", True)):
                tb_image(self.opt, self.logger.experiment, step, split, "mask_pred", self.unflatten_pred(p.mask))

    @torch.no_grad()
    def dump_results(self, var):
        pred = var.pred
        gt = var.views
        exist_dump_opt = hasattr(self.opt, "dump")
        dump_opt = getattr(self.opt, "dump", None)
        
        if not exist_dump_opt or getattr(dump_opt, "kp_pos", True):
            # Save kp_pos
            for i, o in enumerate(var.obj_name):
                d = os.path.join(self.opt.output_path, "dump", o)
                os.makedirs(d, exist_ok=True)
                f = os.path.join(d, "kp_pos.npy")
                np.save(f, pred.kp_pos[i].cpu().numpy())
            
        if exist_dump_opt and getattr(dump_opt, "prior_kp_pos", False):
            # Save kp_prior
            for i, o in enumerate(var.obj_name):
                for j, view in enumerate(var.views.name[i]):
                    d = os.path.join(self.opt.output_path, "dump", o, "prior_kp_pos")
                    os.makedirs(d, exist_ok=True)
                    f = os.path.join(d, f"{view}.npy")
                    np.save(f, pred.prior_kp_pos[i, j].cpu().numpy())

        if hasattr(pred, "views"):
            p = pred.views
            img = self.unflatten_pred(p.channels)
            if not exist_dump_opt or getattr(dump_opt, "image", True):
                dump_images(self.opt, var.obj_name, "image", gt.name, img, from_range=(0, 1))
            if exist_dump_opt and getattr(dump_opt, "kp_pos_img", False):
                dump_images(self.opt, var.obj_name, "kp_pos_img", gt.name, gt.img, kp=pred.kp_pos, extr=gt.extr, intr=gt.intr, pix_scale=getattr(gt, "pix_scale", None), pix_offset=getattr(gt, "pix_offset", None), from_range=(0, 1))
            if hasattr(p, "kp_weights"):
                # kp_weights: [B, num_views, num_rays, num_kp]
                if not exist_dump_opt or getattr(dump_opt, "kp_weights", True):
                    kp_weight_img = self.unflatten_pred(weights_to_colors(p.kp_weights, self.opt.vis.kp.cmap))
                    dump_images(self.opt, var.obj_name, "kp_weights", gt.name, kp_weight_img, from_range=(0, 1))
                if hasattr(pred, "extractor_inter_res") and "attn_map" in pred.extractor_inter_res:
                    # attn_map: [B, n_kp, n_emb]
                    emb_weights = p.kp_weights @ pred.extractor_inter_res["attn_map"].unsqueeze(1)
                    emb_img = self.unflatten_pred(emb_weights)
                    if not exist_dump_opt or getattr(dump_opt, "emb_weights_alpha", True):
                        dump_emb_weights_as_alpha(self.opt, var.obj_name, gt.name, img, emb_img)
                    if not exist_dump_opt or getattr(dump_opt, "emb_weights_color", True):
                        # dump_emb_weights_as_color(self.opt, var.obj_name, gt.view_idx, emb_img, pred.extractor_inter_res["attn_map"])
                        emb_color_img = self.unflatten_pred(weights_to_colors(emb_weights, self.opt.vis.kp.cmap))
                        dump_images(self.opt, var.obj_name, "emb_weights_color", gt.name, emb_color_img, from_range=(0, 1))
                    if not exist_dump_opt or getattr(dump_opt, "emb_weights_heat", True):
                        dump_emb_weights_as_heat(self.opt, var.obj_name, gt.name, emb_img)
            if hasattr(p, "depth") and (not exist_dump_opt or getattr(dump_opt, "depth", True)):
                dump_images(self.opt, var.obj_name, "depth", gt.name, self.depth_to_vis(self.unflatten_pred(p.depth)))
            if hasattr(p, "mask") and (not exist_dump_opt or getattr(dump_opt, "mask", True)):
                dump_images(self.opt, var.obj_name, "mask", gt.name, self.unflatten_pred(p.mask))

    def interpolate(
        self, 
        idx: Tensor, 
        num_steps: int, 
        detector: bool, 
        extractor: bool, 
        res_extractor: bool, 
        global_extractor: bool, 
        module_kwargs
    ) -> None:
        """
        Arguments
            idx: [n, 2]
        """
        for module_name, flag in (("detector", detector), ("extractor", extractor), ("res_extractor", res_extractor), ("global_extractor", global_extractor)):
            module = getattr(self, module_name, None)
            if module is not None:
                if flag:
                    module.interpolate(idx, num_steps, **module_kwargs.get(module_name, {}))
                else:
                    module.copy_selection(idx[:, 0], num_steps)
        
    def randomize(
        self,
        idx: Tensor,
        num_random: int,
        std: Dict[str,float],
        detector: bool,
        extractor: bool,
        res_extractor: bool,
        global_extractor: bool,
        module_kwargs
    ) -> None:
        """
        Arguments:
            idx: [n]
        """
        for module_name, flag in (("detector", detector), ("extractor", extractor), ("res_extractor", res_extractor), ("global_extractor", global_extractor)):
            module = getattr(self, module_name, None)
            if module is not None:
                if flag:
                    module.randomize(idx, num_random, std[module_name], **module_kwargs.get(module_name, {}))
                else:
                    module.copy_selection(idx, num_random)