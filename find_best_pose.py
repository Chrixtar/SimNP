import argparse
from collections import defaultdict
from glob import glob
import os
import json

from tqdm import tqdm


METRIC_START = {
    "psnr": -100000000000,
    "ssim": -100000000000,
    "lpips": 100000000000
}


METRIC_AGG = {
    "psnr": max,
    "ssim": max,
    "lpips": min
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parallel predict")
    parser.add_argument("-e", "--experiment", help="Path to experiment (including date)", type=str, required=True)
    parser.add_argument("-np", "--num_poses", help="Number of poses", type=int, default=8)
    parser.add_argument("-rv", "--ref_view", help="Reference view index", type=int, default=64)
    parser.add_argument("-m", "--metric", help="Metric", type=str, default="lpips")
    parser.add_argument("-sh", "--shuffle", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    dump_path = os.path.join(args.experiment, "dump")
    objs = set()
    for p in glob(os.path.join(args.experiment, "dump", "*")):
        if os.path.isdir(p):
            objs.add(os.path.basename(p).split("_")[0])

    best_obj_pose = {}
    view_metrics_per_obj = {}

    for obj in tqdm(objs):
        best_pose_metrics = None
        best_metric = METRIC_START[args.metric]
        best_pose_idx = -1
        for pose in range(args.num_poses):
            with open(os.path.join(dump_path, f"{obj}_{pose}", "metrics.json")) as f:
                cur_pose_metrics = json.load(f)
            if METRIC_AGG[args.metric](best_metric, cur_pose_metrics[args.metric][str(args.ref_view)]) \
                == cur_pose_metrics[args.metric][str(args.ref_view)]:
                best_pose_metrics = cur_pose_metrics
                best_metric = cur_pose_metrics[args.metric][str(args.ref_view)]
                best_pose_idx = pose
        best_obj_pose[obj] = best_pose_idx
        view_metrics_per_obj[obj] = best_pose_metrics

    with open(os.path.join(dump_path, 'best_obj_pose.json'), 'w') as f:
        json.dump(best_obj_pose, f)
    with open(os.path.join(dump_path, 'obj_metrics.json'), 'w') as f:
        json.dump(view_metrics_per_obj, f)

    mean_metrics = {
        "psnr": defaultdict(float),
        "ssim": defaultdict(float),
        "lpips": defaultdict(float)
    }
    for metrics in view_metrics_per_obj.values():
        for metric, view_scores in metrics.items():
            for view, score in view_scores.items():
                mean_metrics[metric][view] += score
    
    for view_scores in mean_metrics.values():
        for view in view_scores.keys():
            view_scores[view] /= len(objs)

    with open(os.path.join(dump_path, 'best_metrics.json'), 'w') as f:
        json.dump(mean_metrics, f)
