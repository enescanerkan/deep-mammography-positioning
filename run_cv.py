"""
10-fold cross-validation runner.

Trains the three models on the shared folds produced by make_folds.py:
    mlo  - MLO landmark detection (CRAUNet, 6 coords)
    cc   - CC landmark detection (CRAUNet, 2 coords)
    cls  - dual-stream MLO/CC classification (ResNet-18)

Each fold gets its own checkpoint and metrics directory under cv_results/.

    python run_cv.py --models mlo cc cls --folds 0-9 --dry-run
    python run_cv.py --models mlo --folds 0-2 --gpu 1

Note: the landmark trainers resume from a fixed '../checkpoints' directory.
It is cleared before every run so folds cannot bleed into each other.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
# Outputs live next to repo/ and drive/, not inside the git tree.
CV_OUT = os.path.abspath(os.path.join(ROOT, "..", "cv_results"))

LANDMARK = {
    "mlo": {
        "main_dir": "rule-based-model/mlo-landmark-detection/code/regression/main",
        "config": "configs/example_config.json",
    },
    "cc": {
        "main_dir": "rule-based-model/cc-landmark-detection/code/regression/main",
        "config": "configs/cc_training_config.json",
    },
}
CLS_DIR = "dual-stream-classification"


def parse_folds(spec):
    """'0-9' or '0,3,7' -> [0,...]"""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in spec.split(",")]


def landmark_config(model, fold, epochs, batch_size):
    """Write a fold-specific config next to the trainer, return its path."""
    spec = LANDMARK[model]
    main_dir = os.path.join(ROOT, spec["main_dir"])
    cfg = json.load(open(os.path.join(main_dir, spec["config"])))

    out_dir = os.path.join(CV_OUT, model, f"fold{fold}")
    os.makedirs(out_dir, exist_ok=True)

    # Paths in the config are relative to main_dir; the outputs are absolute
    # so results land in cv_results/ regardless of cwd.
    cfg["split_file"] = f"../../../../../labels/folds/{model}_fold{fold}.csv"
    cfg["best_model_path"] = os.path.join(out_dir, f"{model}_model.pth")
    # Own resume directory per fold, so parallel runs stay independent.
    cfg["checkpoint_dir"] = os.path.join(out_dir, "checkpoints")
    if epochs:
        cfg["num_epochs"] = epochs
    if batch_size:
        cfg["batch_size"] = batch_size

    cfg_path = os.path.join(out_dir, "config.json")
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    return main_dir, cfg_path, out_dir


def run(cmd, cwd, env, dry, log_path=None):
    print(f"  $ (cd {os.path.relpath(cwd, ROOT)} && {' '.join(cmd)})", flush=True)
    if dry:
        return 0
    t0 = time.time()
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        print(f"  log: {log_path}", flush=True)
        with open(log_path, "w") as log:
            r = subprocess.run(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, cwd=cwd, env=env)
    print(f"  -> exit {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser(description="10-fold CV runner")
    ap.add_argument("--models", nargs="+", default=["mlo", "cc", "cls"],
                    choices=["mlo", "cc", "cls"])
    ap.add_argument("--folds", default="0-9", help="'0-9' or '0,3,7'")
    ap.add_argument("--selection", choices=["f1", "balanced"], default="f1",
                    help="classifier checkpoint criterion; 'balanced' writes to "
                         "*_balanced.pth so the published runs are left alone")
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value")
    ap.add_argument("--epochs", type=int, default=None, help="override num_epochs")
    ap.add_argument("--batch-size", type=int, default=None, help="override batch_size")
    ap.add_argument("--backbone", default="resnet18", help="classification backbone")
    ap.add_argument("--hparams-from", default=None,
                    help="borrow training hyperparameters from this model (cls only)")
    ap.add_argument("--image-size", type=int, default=512,
                    help="classifier input resolution (cls only)")
    ap.add_argument("--lr", type=float, default=None,
                    help="override the classifier learning rate (cls only)")
    ap.add_argument("--dry-run", action="store_true", help="print commands only")
    args = ap.parse_args()

    folds = parse_folds(args.folds)
    py = sys.executable

    env = os.environ.copy()
    if args.gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.gpu

    missing = [
        f for f in folds
        for m in ("mlo", "cc")
        if not os.path.exists(os.path.join(ROOT, "labels", "folds", f"{m}_fold{f}.csv"))
    ]
    if missing:
        sys.exit(f"fold csv missing for folds {sorted(set(missing))} - run make_folds.py first")

    print(f"models={args.models}  folds={folds}  gpu={args.gpu or 'default'}"
          f"{'  [DRY RUN]' if args.dry_run else ''}\n")

    total = len(folds) * len(args.models)
    done = 0
    t_start = time.time()
    failures = []
    for fold in folds:
        for model in args.models:
            done += 1
            elapsed = (time.time() - t_start) / 3600
            print(f"\n[{done}/{total}] fold {fold} / {model}"
                  f"  (elapsed {elapsed:.1f} h)", flush=True)
            if model == "cls":
                tag = args.backbone
                if args.hparams_from and args.hparams_from != args.backbone:
                    tag += f"_hp-{args.hparams_from}"
                if args.image_size != 512:
                    tag += f"_{args.image_size}px"
                if args.lr is not None:
                    tag += f"_lr{args.lr:.0e}"
                if args.selection != "f1":
                    tag += f"_{args.selection}"
                log_path = os.path.join(CV_OUT, "logs", f"cls_{tag}_fold{fold}.log")
                cwd = os.path.join(ROOT, CLS_DIR)
                cmd = [py, "main.py", "--model", args.backbone, "--fold", str(fold)]
                if args.hparams_from:
                    cmd += ["--hparams-from", args.hparams_from]
                if args.image_size != 512:
                    cmd += ["--image-size", str(args.image_size)]
                if args.lr is not None:
                    cmd += ["--lr", str(args.lr)]
                if args.selection != "f1":
                    cmd += ["--selection", args.selection]
            else:
                log_path = os.path.join(CV_OUT, "logs", f"{model}_fold{fold}.log")
                main_dir, cfg_path, out_dir = landmark_config(
                    model, fold, args.epochs, args.batch_size
                )
                # Fresh resume state for this fold (config points here).
                ckpt = os.path.join(out_dir, "checkpoints")
                if os.path.isdir(ckpt) and not args.dry_run:
                    shutil.rmtree(ckpt)
                cwd = main_dir
                cmd = [py, "main.py", "--config", cfg_path]

            if run(cmd, cwd, env, args.dry_run, log_path) != 0:
                failures.append((fold, model))

    print("\n" + "=" * 50, flush=True)
    print(f"total elapsed: {(time.time() - t_start) / 3600:.1f} h")
    if failures:
        print(f"FAILED: {failures}")
        sys.exit(1)
    print("all runs finished" if not args.dry_run else "dry run complete")


if __name__ == "__main__":
    main()
