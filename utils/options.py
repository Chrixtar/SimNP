import numpy as np
import os,sys,time
import torch
import random
import string
import time
import yaml
from easydict import EasyDict as edict

from utils.util import to_dict
from utils.log import log

# torch.backends.cudnn.enabled = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def parse_arguments(args):
    # parse from command line (syntax: --key1.key2.key3=value)
    opt_cmd = {}
    for arg in args:
        assert(arg.startswith("--"))
        if "=" not in arg[2:]: # --key means key=True, --key! means key=False
            key_str,value = (arg[2:-1],"false") if arg[-1]=="!" else (arg[2:],"true")
        else:
            key_str,value = arg[2:].split("=")
        keys_sub = key_str.split(".")
        opt_sub = opt_cmd
        for k in keys_sub[:-1]:
            if k not in opt_sub: opt_sub[k] = {}
            opt_sub = opt_sub[k]
        assert keys_sub[-1] not in opt_sub,keys_sub[-1]
        opt_sub[keys_sub[-1]] = yaml.safe_load(value)
    opt_cmd = edict(opt_cmd)
    return opt_cmd

def set(opt_cmd={}):
    log.info("setting configurations...")
    assert("yaml" in opt_cmd)
    opt_base = load_options(opt_cmd.yaml)
    # Loading training config results into too many bugs!
    # if opt_base.load is not None:
    #     opt_load = load_options(os.path.join(opt_base.load, "options.yaml"))
    #     opt_base = overwrite_options(opt_load, opt_base)
    opt_base.fname =  os.path.splitext(os.path.basename(opt_cmd.yaml))[0]
    # overwrite with command line arguments
    opt = overwrite_options(opt_base, opt_cmd, safe_check=opt_cmd.safe_check if "safe_check" in opt_cmd else True)
    process_options(opt)
    if opt.load is not None and opt.resume:
        opt.output_path = opt.load
    log.options(opt)
    return opt

def load_options(fname):
    with open(fname) as file:
        opt = edict(yaml.load(file, yaml.FullLoader))   # edict(yaml.safe_load(file))
    if "_parent_" in opt:
        # load parent yaml file(s) as base options
        parent_fnames = opt.pop("_parent_")
        if type(parent_fnames) is str:
            parent_fnames = [parent_fnames]
        for parent_fname in parent_fnames:
            opt_parent = load_options(parent_fname)
            opt_parent = overwrite_options(opt_parent,opt,key_stack=[])
            opt = opt_parent
    if "_import_" in opt:
        import_fnames = opt.pop("_import_")
        for key, fname in import_fnames.items():
            opt_import = load_options(fname)
            opt_import = overwrite_options(opt_import, getattr(opt, key, {}), key_stack=[])
            opt[key] = opt_import
    print("loading {}...".format(fname))
    return opt

def overwrite_options(opt, opt_over, key_stack=None, safe_check=False):
    if key_stack is None:
        key_stack = []
    for key, value in opt_over.items():
        if isinstance(value, dict) and key in opt and opt[key] is not None:
            # parse child options (until leaf nodes are reached)
            opt[key] = overwrite_options(opt[key], value, key_stack=key_stack+[key], safe_check=safe_check)
        else:
            # ensure command line argument to overwrite is also in yaml file
            if safe_check and key not in opt:
                add_new = None
                while add_new not in ["y","n"]:
                    key_str = ".".join(key_stack+[key])
                    add_new = input("\"{}\" not found in original opt, add? (y/n) ".format(key_str))
                if add_new=="n":
                    print("safe exiting...")
                    exit()
            opt[key] = value
    return opt

def process_options(opt):
    # set seed
    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed_all(opt.seed)
    # other default options
    if not getattr(opt, "resume", False) or "version" not in opt:
        opt.version = time.strftime('%Y%m%d-%H%M%S')
    if not getattr(opt, "resume", False) or not hasattr(opt, "output_path"):
        opt.output_path = os.path.join(opt.output_root, opt.data.dataset, opt.data.cat, opt.group, opt.fname, opt.version)
    if not "accelerator" in opt:
        opt.accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    if not "devices" in opt:
        opt.devices = 1
    # opt.device = "cpu" if opt.cpu or not torch.cuda.is_available() else "cuda:{}".format(opt.gpu)

def save_options_file(opt):
    os.makedirs(opt.output_path,exist_ok=True)
    opt_fname = "{}/options.yaml".format(opt.output_path)
    if os.path.isfile(opt_fname) and ("safe_check" not in opt or opt.safe_check):
        with open(opt_fname) as file:
            opt_old = yaml.safe_load(file)
        if opt!=opt_old:
            # prompt if options are not identical
            opt_new_fname = "{}/options_temp.yaml".format(opt.output_path)
            with open(opt_new_fname,"w") as file:
                yaml.dump(to_dict(opt), file, default_flow_style=False, indent=4)
            print("existing options file found (different from current one)...")
            os.system("diff {} {}".format(opt_fname, opt_new_fname))
            os.system("rm {}".format(opt_new_fname))
            overwrite = None
            while overwrite not in ["y","n"]:
                overwrite = input("overwrite? (y/n) ")
            if overwrite=="n":
                print("safe exiting...")
                exit()
        else: print("existing options file found (identical)")
    else: print("(creating new options file...)")
    with open(opt_fname,"w") as file:
        yaml.dump(to_dict(opt),file,default_flow_style=False,indent=4)
