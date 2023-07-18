import os
import sys

import pytorch_lightning as pl

import dataloaders

from core.model import Model
from core.freezer import Freezer
from utils import options
from utils.log import log
from utils.model import clean_up_checkpoints


if __name__ == "__main__":
    log.process(os.getpid())
    log.title("[{}] (training)".format(sys.argv[0]))

    opt_cmd = options.parse_arguments(sys.argv[1:])
    opt = options.set(opt_cmd=opt_cmd)
    options.save_options_file(opt)

    # Load train and validation data
    dataloader_class = getattr(dataloaders, opt.data.dataset)
    log.info("loading training data...")
    train_data = dataloader_class(opt, setting="train")
    n_obj = train_data.n_obj
    view_idx = {"train": train_data.view_idx}
    train_loader = train_data.setup_loader(opt.train_batch_size, getattr(opt.data.train, "dataloader_workers", opt.data.dataloader_workers), shuffle=True)
    log.info("loading validation data...")
    if opt.data.val is not None:
        val_data = dataloader_class(opt, setting="val")
        view_idx["val"] = val_data.view_idx
        val_loader = val_data.setup_loader(opt.vis_batch_size, getattr(opt.data.val, "dataloader_workers", opt.data.dataloader_workers), shuffle=False)
    else:
        val_loader = None

    # Define callbacks
    callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=f"{opt.output_path}/checkpoints",
            monitor=getattr(opt.optim, "monitor", None),
            filename="{epoch}-{step}", 
            save_last=True, 
            every_n_epochs=opt.freq.ckpt_ep if "ckpt_ep" in opt.freq else None,
            every_n_train_steps=opt.freq.ckpt_it if "ckpt_it" in opt.freq else None,
            verbose=True,
            save_on_train_epoch_end=True
        )
    ]
    if hasattr(opt.optim, "early_stopping"):
        callbacks.append(pl.callbacks.EarlyStopping(**opt.optim.early_stopping))
    if hasattr(opt, "freeze"):
        callbacks.append(Freezer(**opt.freeze))

    trainer = pl.Trainer(
        accelerator=opt.accelerator, 
        strategy="ddp" if opt.devices > 1 else None,
        auto_select_gpus=True, 
        callbacks=callbacks, 
        check_val_every_n_epoch=opt.freq.eval_ep,
        val_check_interval=opt.freq.eval_interval,
        devices=opt.devices, 
        logger=pl.loggers.TensorBoardLogger(save_dir=opt.output_path, name="", version=""),
        log_every_n_steps=opt.freq.scalar,
        max_epochs=opt.max_epoch,
        profiler=pl.profiler.AdvancedProfiler(dirpath=opt.output_path, filename="profile") if opt.profile else None,
        **(opt.trainer if hasattr(opt, "trainer") else {}),
        # detect_anomaly=True
    )

    if opt.load is not None:
        ckpt = os.path.join(opt.load, "checkpoints", "last.ckpt")
        m = Model.load_from_checkpoint(checkpoint_path=ckpt, strict=opt.resume, opt=opt, n_obj=n_obj, view_idx=view_idx)
    else:
        ckpt = None
        m = Model(opt, n_obj, view_idx)
    
    trainer.fit(m, train_loader, val_loader, ckpt_path=ckpt if getattr(opt, "resume", False) else None)
    if opt.clean_up_checkpoints:
        clean_up_checkpoints(opt.output_path)
