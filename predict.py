import argparse
import os
import json

from easydict import EasyDict as edict
import pytorch_lightning as pl
import torch

import dataloaders

from core.model import Model
from utils import options
from utils.log import log


METRIC_ABBREVIATIONS = {
    "PeakSignalNoiseRatio": "psnr",
    "LearnedPerceptualImagePatchSimilarity": "lpips",
    "StructuralSimilarityIndexMeasure": "ssim"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Testing")
    parser.add_argument("-a", "--accelerator", help="Accelerator type", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-d", "--devices", help="Number of devices", type=int, default=1)
    parser.add_argument("-e", "--experiment", help="Path to experiment", type=str, required=True)
    parser.add_argument("-da", "--date", help="Date and time of experiment", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", help="Name of checkpoint", type=str, default="last")
    parser.add_argument("-ld", "--load_depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-rm", "--reload_metrics", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-m", "--metrics", help="Path to options file containing metrics, if None use metrics from loaded experiment options", type=str, default="options/srn/base.yaml")
    parser.add_argument("-epv", "--eval_per_view", help="Evaluate per view", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-pre", "--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-dw", "--dataloader_workers", help="Number of dataloader workers", default=4)
    parser.add_argument("-pw", "--preload_workers", help="Number of preload workers", default=8)
    parser.add_argument("-bs", "--batch_size", help="Batch size during testing", type=int, default=4)
    parser.add_argument("-vs", "--view_size", help="View size during testing (has to divide number of views per instance)", type=int, default=1)
    parser.add_argument("-dv", "--dump_vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-r", "--resolution", type=int, default=128)
    parser.add_argument("-se", "--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-dkp", "--dump_kp_pos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-di", "--dump_image", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-dkpi", "--dump_kp_pos_img", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dkpw", "--dump_kp_weights", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dewa", "--dump_emb_weights_alpha", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dewc", "--dump_emb_weights_color", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dewh", "--dump_emb_weights_heat", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dd", "--dump_depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dm", "--dump_mask", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args()

    experiment_path = os.path.join(args.experiment, args.date)
    opt = options.load_options(os.path.join(experiment_path, "options.yaml"))
    opt.rendering.randomize_depth_samples = False   # Deactivate randomness in rendering
    if args.reload_metrics:
        metric_opt = options.load_options(args.metrics)
        opt.metrics = metric_opt.metrics
    opt.metrics.test.eval_per_view = args.eval_per_view

    opt.data.preload = args.preload
    opt.data.preload_workers = args.preload_workers
    opt.data.dataloader_workers = args.dataloader_workers
    opt.data.load_depth = args.load_depth
    opt.vis_batch_size = args.batch_size
    opt.vis_view_size = 1
    # opt.data.use_gt_pc_alignment = False
    opt.data.test = edict(split="test", blacklist=False)
    opt.dump_vis = args.dump_vis
    opt.sizes.dump = args.resolution
    opt.skip_existing = args.skip_existing
    opt.dump = edict(
        kp_pos=args.dump_kp_pos,
        kp_pos_img=args.dump_kp_pos_img,
        image=args.dump_image,
        kp_weights=args.dump_kp_weights,
        emb_weights_alpha=args.dump_emb_weights_alpha,
        emb_weights_color=args.dump_emb_weights_color,
        emb_weights_heat=args.dump_emb_weights_heat,
        depth=args.dump_depth,
        mask=args.dump_mask
    )
    dataloader_class = getattr(dataloaders, opt.data.dataset)
    log.info("loading test data...")
    test_data = dataloader_class(opt, setting="test")
    n_obj = test_data.n_obj
    view_idx = {"test": test_data.view_idx}
    test_loader = test_data.setup_loader(opt.vis_batch_size, opt.data.dataloader_workers, shuffle=False)

    trainer = pl.Trainer(
        accelerator=args.accelerator, 
        auto_select_gpus=True, 
        devices=args.devices
    )

    model_path = os.path.join(args.experiment, args.date, "checkpoints", args.checkpoint + ".ckpt")
    m = Model.load_from_checkpoint(checkpoint_path=model_path, opt=opt, n_obj=n_obj, view_idx=view_idx, strict=False)
    res = trainer.test(m, dataloaders=test_loader)[0]
    with open(os.path.join(experiment_path, "test_result.json"), "w") as fp:
        json.dump(res, fp)
    
    metrics = {
        "psnr": {},
        "lpips": {},
        "ssim": {}
    }
    for k, v in res.items():
        key_split = k.split("/")
        if key_split[1] == "metric":
            view, metric = key_split[2:4]
            metric_abbreviation = METRIC_ABBREVIATIONS[metric]
            if view == "all":
                metrics[metric_abbreviation][view] = v
            else:
                view_int = int(view.split("_")[1])
                metrics[metric_abbreviation][view_int] = v
    with open(os.path.join(experiment_path, "metrics.json"), "w") as fp:
        json.dump(metrics, fp)