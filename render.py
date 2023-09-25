import argparse
import os

from easydict import EasyDict as edict
import pytorch_lightning as pl
import torch

import dataloaders

from core.model import Model
from utils import options
from utils.log import log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("-a", "--accelerator", help="Accelerator type", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-d", "--devices", help="Number of devices", type=int, default=1)
    parser.add_argument("-e", "--experiment", help="Path to experiment", type=str, required=True)
    parser.add_argument("-da", "--date", help="Date and time of experiment", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", help="Name of checkpoint", type=str, default="last")
    parser.add_argument("-o", "--object", help="Names of objects", type=str, default=None)
    parser.add_argument("-ds", "--data_setting", help="Setting of the dataset", type=str, default="train")
    parser.add_argument("-vst", "--view_start", type=int, default=0)
    parser.add_argument("-ve", "--view_end", type=int, default=251)
    parser.add_argument("-vss", "--view_step_size", type=int, default=4)
    parser.add_argument("-pre", "--preload", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dw", "--dataloader_workers", help="Number of dataloader workers", default=4)
    parser.add_argument("-pw", "--preload_workers", help="Number of preload workers", default=8)
    parser.add_argument("-ld", "--load_depth", help="Whether to load depth in the dataloader or not", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-bs", "--batch_size", help="Batch size during testing", type=int, default=1)
    parser.add_argument("-vs", "--view_size", help="View size during testing (has to divide number of views per instance)", type=int, default=1)
    parser.add_argument("-r", "--resolution", help="Resolution of output images", type=int, default=128)
    parser.add_argument("-se", "--skip_existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("-dkp", "--dump_kp_pos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dkpi", "--dump_kp_pos_img", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dpkp", "--dump_prior_kp_pos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-di", "--dump_image", action=argparse.BooleanOptionalAction, default=True)
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

    # Add object names as post whitelist such that object indices are not affected!
    if args.object is not None:
        opt.data.train.whitelist = list(map(str.strip, args.object.split(",")))
        opt.data.train.post_whitelist = True
    opt.data.preload = args.preload
    opt.data.preload_workers = args.preload_workers
    opt.data.dataloader_workers = args.dataloader_workers
    opt.data.load_depth = args.load_depth
    if args.data_setting == "train":
        opt.train_batch_size = args.batch_size
        opt.train_view_size = args.view_size
    else:
        opt.vis_batch_size = args.batch_size
        opt.vis_view_size = args.view_size
        setattr(opt.data, args.data_setting, opt.data.train)
    settings_opt = getattr(opt.data, args.data_setting)
    setattr(settings_opt, "view_slices", [[args.view_start, args.view_end, args.view_step_size]])
    # if "view_slices" in opt.data.train:
    #     opt.data.train.pop("view_slices")   # Remove view_slices in order to take all views in split
    opt.loss_weight = edict()
    opt.sizes.render = args.resolution
    opt.sizes.dump = args.resolution
    opt.dump_vis = True
    opt.dump_dir = "render"
    opt.skip_existing = args.skip_existing
    opt.dump = edict(
        kp_pos=args.dump_kp_pos,
        kp_pos_img=args.dump_kp_pos_img,
        prior_kp_pos=args.dump_prior_kp_pos,
        image=args.dump_image,
        kp_weights=args.dump_kp_weights,
        emb_weights_alpha=args.dump_emb_weights_alpha,
        emb_weights_color=args.dump_emb_weights_color,
        emb_weights_heat=args.dump_emb_weights_heat,
        depth=args.dump_depth,
        mask=args.dump_mask
    )
    if "repetitions_per_epoch" in opt.data.train:
        opt.data.train.pop("repetitions_per_epoch")
    dataloader_class = getattr(dataloaders, opt.data.dataset)
    log.info("loading data...")
    dataset = dataloader_class(opt, setting=args.data_setting)  # Setting train, because we want to evaluate split, on which it was trained on (test-time optimization)
    n_obj = dataset.n_obj

    data_loader = dataset.setup_loader(args.batch_size, opt.data.dataloader_workers, shuffle=True )

    trainer = pl.Trainer(
        accelerator=args.accelerator, 
        auto_select_gpus=True, 
        devices=args.devices
    )

    model_path = os.path.join(args.experiment, args.date, "checkpoints", args.checkpoint + ".ckpt")
    m = Model.load_from_checkpoint(checkpoint_path=model_path, opt=opt, n_obj=n_obj, view_idx={}, strict=False)

    trainer.predict(m, dataloaders=data_loader, return_predictions=False)
