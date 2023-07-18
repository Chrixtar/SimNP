import argparse
import os

from easydict import EasyDict as edict
import pytorch_lightning as pl
import torch

from core.model import Model
from utils import options

from dataloaders.srn_random import SRNRandom


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization")
    parser.add_argument("-a", "--accelerator", help="Accelerator type", type=str, default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-d", "--devices", help="Number of devices", type=int, default=1)

    parser.add_argument("-e", "--experiment", help="Path to experiment", type=str, required=True)
    parser.add_argument("-da", "--date", help="Date and time of experiment", type=str, required=True)
    parser.add_argument("-c", "--checkpoint", help="Name of checkpoint", type=str, default="last")

    parser.add_argument("-o", "--objects", help="Names of objects", type=str, required=True)
    parser.add_argument("-nr", "--num_random", help="Number of random inits", type=int, required=True)
    parser.add_argument("-de_std", "--de_std", help="Standard deviation of random detector initialization", type=float, default=1)
    parser.add_argument("-ex_std", "--ex_std", help="Standard deviation of random extractor initialization", type=float, default=1)
    parser.add_argument("-res_std", "--res_std", help="Standard deviation of random residual extractor initialization", type=float, default=1)
    parser.add_argument("-glob_std", "--glob_std", help="Standard deviation of random global extractor initialization", type=float, default=1)

    parser.add_argument("-de", "--detector", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-ex", "--extractor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-res", "--res_extractor", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-glob", "--global_extractor", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("-emb", "--embeddings", help="Comma separated list of embedding indices for extractor sym embedding", type=str, default=None)

    parser.add_argument("-vss", "--view_step_size", type=int, default=8)
    parser.add_argument("-bs", "--batch_size", help="Batch size during testing", type=int, default=1)
    parser.add_argument("-vs", "--view_size", help="View size during testing (has to divide number of views per instance)", type=int, default=1)
    parser.add_argument("-r", "--resolution", help="Resolution of output images", type=int, default=256)
    parser.add_argument("-se", "--skip_existing", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("-dkp", "--dump_kp_pos", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("-dkpi", "--dump_kp_pos_img", action=argparse.BooleanOptionalAction, default=False)
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

    if args.embeddings is not None:
        embeddings = sorted([int(s.strip()) for s in args.embeddings.split(",")])
    else:
        embeddings = None

    opt.data.view_step_size = args.view_step_size
    opt.train_batch_size = args.batch_size
    opt.train_view_size = args.view_size
    opt.sizes.render = args.resolution
    opt.sizes.dump = args.resolution
    opt.dump_vis = True
    opt.dump_dir = f"template"
    if args.detector:
        opt.dump_dir += "_de"
    if args.extractor:
        opt.dump_dir += "_ex"
        if embeddings is not None:
            opt.dump_dir += "_".join(map(str, embeddings))
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
        mask=args.dump_mask,
    )

    obj_names = list(map(str.strip, args.objects.split(",")))

    dataset = SRNRandom(opt, "train", obj_names, args.num_random)  # Setting train, because we want to evaluate split, on which it was trained on (test-time optimization)
    data_loader = dataset.setup_loader(opt.train_batch_size, opt.data.dataloader_workers, shuffle=False)

    obj_idx = torch.tensor([dataset.obj_name_to_idx[obj_name] for obj_name in obj_names])

    model_path = os.path.join(args.experiment, args.date, "checkpoints", args.checkpoint + ".ckpt")
    m = Model.load_from_checkpoint(checkpoint_path=model_path, opt=opt, n_obj=dataset.n_obj, view_idx={}, strict=False)
    m.randomize(
        obj_idx, 
        args.num_random,
        {
            "detector": args.de_std,
            "extractor": args.ex_std,
            "res_extractor": args.res_std,
            "global_extractor": args.glob_std
        },
        detector=args.detector,
        extractor=args.extractor,
        res_extractor=args.res_extractor,
        global_extractor=args.global_extractor,
        module_kwargs={
            "extractor": {
                "embedding_subset": torch.tensor(embeddings) if embeddings is not None else None
            }
        }
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator, 
        auto_select_gpus=True, 
        devices=args.devices
    )

    trainer.predict(m, dataloaders=data_loader, return_predictions=False)
