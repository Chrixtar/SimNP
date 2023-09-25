"""
Compute metrics on rendered images (after eval.py).
First computes per-object metric then reduces them. If --multicat is used
then also summarized per-categority metrics. Use --reduce_only to skip the
per-object computation step.

Note eval.py already outputs PSNR/SSIM.
This also computes LPIPS and is useful for double-checking metric is correct.
"""

import os
from collections import defaultdict
import os.path as osp
import argparse
import skimage.measure
from tqdm import tqdm
import warnings
import lpips
import numpy as np
import torch
import imageio
import json
from random import shuffle
from skimage.filters import gaussian
# from scipy.ndimage import gaussian_filter

parser = argparse.ArgumentParser(description="Calculate PSNR for rendered images.")
parser.add_argument("--category", "-C", type=str, default="cars", help="Category name")
parser.add_argument("--datadir", "-D", type=str, default="/BS/wewer/work/datasets/rendered", help="Path to dataset root dir")
parser.add_argument("--dataset", type=str, default="SRN", help="Name of the dataset")
parser.add_argument("--num_views", type=int, default=251, help="Number of views per object")
parser.add_argument("--experiment", "-E", type=str, required=True, help="Path to the experiment")
parser.add_argument("--gpu_id", type=int, default=0, help="GPU id. Only single GPU supported for this script.")
parser.add_argument("--overwrite", action="store_true", help="overwriting existing metrics.txt")
parser.add_argument("--lpips_batch_size", type=int, default=32, help="Batch size for LPIPS")
parser.add_argument("--reduce_only", "-R", action="store_true", help="skip the map (per-obj metric computation)")
parser.add_argument("--skip_views", type=str, default="8525") # "64")
parser.add_argument("--sigma", type=float, default=0)
args = parser.parse_args()


skip_views = set(map(lambda x: int(x.strip()), args.skip_views.split(",")))
data_root = os.path.join(args.datadir, args.dataset, args.category)
render_root = os.path.join(args.experiment, "dump")


def run_map():
    objs = [o for o in os.listdir(render_root) if os.path.isdir(os.path.join(render_root, o))]
    
    # Split by _ because of possible repetitions
    objs_data = [osp.join(data_root, x.split("_")[0]) for x in objs]
    objs_rend = [osp.join(render_root, x) for x in objs]

    objs = list(zip(objs_data, objs_rend))
    shuffle(objs)

    cuda = "cuda:" + str(args.gpu_id)
    lpips_vgg = lpips.LPIPS(net="vgg").to(device=cuda)

    def get_metrics(rgb, gt):
        ssim = skimage.measure.compare_ssim(rgb, gt, multichannel=True, data_range=1)
        psnr = skimage.measure.compare_psnr(rgb, gt, data_range=1)
        return psnr, ssim

    def is_valid_view(path):
        name, ext = osp.splitext(path)
        return name.isnumeric() and (ext == ".jpg" or ext == ".png")

    def process_obj(data_path, rend_path):
        gt_img_path = osp.join(data_path, "rgb")
        rend_img_path = osp.join(rend_path, "image")
        out_path = osp.join(rend_path, f"metrics{f'_{args.sigma}' if args.sigma > 0 else ''}.json")
        if osp.exists(out_path) and not args.overwrite:
            return
        ims = [x for x in sorted(os.listdir(gt_img_path)) if is_valid_view(x)]
        if len(ims) < args.num_views:
            return
        psnr_all = []
        ssim_all = []
        view_all = []
        psnr_avg = 0.0
        ssim_avg = 0.0
        gts = []
        preds = []
        num_ims = 0

        for im_name in ims:
            im_path = osp.join(gt_img_path, im_name)
            im_name_id = int(osp.splitext(im_name)[0])
            if im_name_id not in skip_views:
                im_name_out = f"{im_name_id}.png"
                im_rend_path = osp.join(rend_img_path, im_name_out)
                if osp.exists(im_rend_path):
                    gt = imageio.imread(im_path).astype(np.float32)[..., :3] / 255.0
                    pred = imageio.imread(im_rend_path).astype(np.float32) / 255.0
                    if args.sigma > 0:
                        pred = gaussian(pred, args.sigma, multichannel=True)
                    psnr, ssim = get_metrics(pred, gt)
                    psnr_all.append(psnr)
                    ssim_all.append(ssim)
                    view_all.append(im_name_id)
                    psnr_avg += psnr
                    ssim_avg += ssim
                    gts.append(torch.from_numpy(gt).permute(2, 0, 1) * 2.0 - 1.0)
                    preds.append(torch.from_numpy(pred).permute(2, 0, 1) * 2.0 - 1.0)
                    num_ims += 1
        psnr_avg /= num_ims
        ssim_avg /= num_ims

        gts = torch.stack(gts)
        preds = torch.stack(preds)

        lpips_all = []
        preds_spl = torch.split(preds, args.lpips_batch_size, dim=0)
        gts_spl = torch.split(gts, args.lpips_batch_size, dim=0)
        with torch.no_grad():
            for predi, gti in zip(preds_spl, gts_spl):
                lpips_i = lpips_vgg(predi.to(device=cuda), gti.to(device=cuda))
                lpips_all.append(lpips_i)
            lpips = torch.cat(lpips_all).flatten()
        lpips_avg = lpips.mean().item()
        lpips_all = lpips.tolist()
        psnr = dict(zip(view_all, psnr_all))
        ssim = dict(zip(view_all, ssim_all))
        lpips = dict(zip(view_all, lpips_all))
        psnr["all"] = psnr_avg
        ssim["all"] = ssim_avg
        lpips["all"] = lpips_avg
        out = dict(psnr=psnr, ssim=ssim, lpips=lpips)
        with open(out_path, "w") as fp:
            json.dump(out, fp)

    for obj_path, obj_rend_path in tqdm(objs):
        process_obj(obj_path, obj_rend_path)


def run_reduce():
    objs = [o for o in os.listdir(render_root) if os.path.isdir(os.path.join(render_root, o))]

    print(">>> PROCESSING", len(objs), "OBJECTS")

    METRIC_NAMES = ["psnr", "ssim", "lpips"]

    out_metrics_path = osp.join(render_root, f"metrics{f'_{args.sigma}' if args.sigma > 0 else ''}.json")

    metric_sum = {}
    metric_all = {}
    for name in METRIC_NAMES:
        metric_sum[name] = defaultdict(float)
        metric_all[name] = {}

    for obj in tqdm(objs):
        obj_root = osp.join(render_root, obj)
        metrics_path = osp.join(obj_root, f"metrics{f'_{args.sigma}' if args.sigma > 0 else ''}.json")
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        for metric, scores in metrics.items():
            metric_all[metric][obj] = scores["all"]
            for view, score in scores.items():
                metric_sum[metric][view] += float(score)

    for name in METRIC_NAMES:
        metric_all[name] = {k: v for k, v in sorted(metric_all[name].items(), key=lambda item: item[1])}
        print(f"Lowest {name}: {list(metric_all[name].items())[:5]}")
        print(f"Highest {name}: {list(metric_all[name].items())[-5:]}")
        for view in metric_sum[name].keys():
            metric_sum[name][view] /= len(objs)
            print(name, view, metric_sum[name][view])

    with open(out_metrics_path, "w") as fp:
        json.dump(metric_sum, fp)


if __name__ == "__main__":
    if not args.reduce_only:
        print(">>> Compute")
        run_map()
    print(">>> Reduce")
    run_reduce()
