"""
Evaluate the dual-stream ResNet-18 classifier across the 10 CV folds.

Loads each fold's best checkpoint, runs it on that fold's Test split, and
reports the same metrics as the landmark evaluation so the two approaches
can be compared against manuscript Table 1.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CLS_DIR = os.path.join(ROOT, "dual-stream-classification")
CV_OUT = os.path.abspath(os.path.join(ROOT, "..", "cv_results"))
sys.path.insert(0, CLS_DIR)

from utils.dual_dataloader import DualStreamDataset  # noqa: E402
from utils.dual_models import DualStreamClassifier  # noqa: E402
from model_configs import get_model_config  # noqa: E402


def score(truth, pred):
    """Good is the positive class, matching the manuscript."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    tp = int(((pred == "Good") & (truth == "Good")).sum())
    fp = int(((pred == "Good") & (truth == "Bad")).sum())
    fn = int(((pred == "Bad") & (truth == "Good")).sum())
    tn = int(((pred == "Bad") & (truth == "Bad")).sum())
    n = len(truth)
    acc = (tp + tn) / n * 100
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    # Weighted variants (sklearn average='weighted'), in case Table 1 used them.
    pg, rg = prec, rec
    pb = tn / (tn + fn) * 100 if tn + fn else 0.0
    rb = tn / (tn + fp) * 100 if tn + fp else 0.0
    f1g = f1
    f1b = 2 * pb * rb / (pb + rb) if pb + rb else 0.0
    ng, nb = tp + fn, tn + fp
    wp = (pg * ng + pb * nb) / n
    wr = (rg * ng + rb * nb) / n
    wf = (f1g * ng + f1b * nb) / n
    # Detecting inadequate positioning is the clinically relevant direction,
    # so the Bad class is scored explicitly alongside the manuscript metrics.
    bal_acc = (rg + rb) / 2
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    mcc = ((tp * tn - fp * fn) / denom) if denom else 0.0
    return dict(n=n, accuracy=acc, precision=prec, recall=rec, f1=f1,
                w_precision=wp, w_recall=wr, w_f1=wf,
                bad_precision=pb, bad_recall=rb, bad_f1=f1b,
                balanced_accuracy=bal_acc, mcc=mcc,
                tp=tp, fp=fp, fn=fn, tn=tn)


@torch.no_grad()
def evaluate_fold(fold, ckpt, backbone, device, hparams_from=None, image_size=512,
                  augment="paper", normalize="none"):
    cfg = get_model_config(backbone, fold=fold, hparams_from=hparams_from,
                           image_size=image_size, augment=augment, normalize=normalize)
    ds = DualStreamDataset(
        os.path.join(CLS_DIR, cfg["mlo_dir"]),
        os.path.join(CLS_DIR, cfg["cc_dir"]),
        os.path.join(CLS_DIR, cfg["mlo_labels"]),
        os.path.join(CLS_DIR, cfg["cc_labels"]),
        split_type="Test",
        use_augmentation=False,
        normalize=normalize,
    )
    model = DualStreamClassifier(
        num_classes=2, backbone=backbone, pretrained=False,
        dropout_rate1=cfg["dropout_rate1"], dropout_rate2=cfg["dropout_rate2"],
    ).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    truth, pred, cases = [], [], []
    for i in range(len(ds)):
        mlo, cc, label, meta = ds[i]
        logits = model(mlo.unsqueeze(0).to(device), cc.unsqueeze(0).to(device))
        prob = torch.softmax(logits, dim=1)[0, 1].item()
        p = int(logits.argmax(1).item())
        pred.append("Good" if p == 1 else "Bad")
        truth.append("Good" if label == 1 else "Bad")
        # Per-case rows mirror the landmark evaluator, so the two families can be
        # pooled out-of-fold and bootstrapped the same way.
        cases.append(dict(fold=fold, study=meta["study_id"], side=meta["side"],
                          prob_good=prob, pred=pred[-1], truth=truth[-1]))
    return score(truth, pred), cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--backbone", default="resnet18")
    ap.add_argument("--hparams-from", default=None)
    ap.add_argument("--image-size", type=int, default=512)
    ap.add_argument("--out", default=None)
    ap.add_argument("--augment", choices=["paper", "domain"], default="paper")
    ap.add_argument("--normalize", choices=["none", "tissue"], default="none")
    ap.add_argument("--selection", choices=["f1", "balanced"], default="f1",
                    help="which checkpoint to score; 'balanced' reads "
                         "dual_best_model_balanced.pth from the same fold dir")
    args = ap.parse_args()

    if "-" in args.folds:
        lo, hi = args.folds.split("-")
        folds = list(range(int(lo), int(hi) + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # run_name mirrors the one training used, so evaluation always reads the
    # checkpoints belonging to this exact configuration.
    run_name = get_model_config(args.backbone, fold=0,
                                hparams_from=args.hparams_from,
                                image_size=args.image_size,
                                augment=args.augment,
                                normalize=args.normalize)["run_name"]
    out_dir = args.out or os.path.join(CV_OUT, f"evaluation_cls_{run_name}")
    os.makedirs(out_dir, exist_ok=True)

    rows, case_rows = [], []
    for f in folds:
        weight_name = ("dual_best_model.pth" if args.selection == "f1"
                       else f"dual_best_model_{args.selection}.pth")
        ckpt = os.path.join(CV_OUT, f"cls_{run_name}", f"fold{f}", weight_name)
        if not os.path.exists(ckpt):
            print(f"fold {f}: checkpoint missing, skipped")
            continue
        m, cases = evaluate_fold(f, ckpt, args.backbone, device,
                                 args.hparams_from, args.image_size,
                                 args.augment, args.normalize)
        m["fold"] = f
        rows.append(m)
        case_rows.extend(cases)
        print(f"fold {f}: n={m['n']:>3} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
              f"rec={m['recall']:.2f} f1={m['f1']:.2f} bad_rec={m['bad_recall']:.2f}")

    md = pd.DataFrame(rows)
    md.to_csv(os.path.join(out_dir, "per_fold.csv"), index=False)
    pd.DataFrame(case_rows).to_csv(os.path.join(out_dir, "per_case.csv"), index=False)

    print("\n" + "=" * 60)
    print(f"{run_name} dual-stream — {len(md)}-fold summary (mean +/- SD)")
    summary = {}
    for k in ["accuracy", "precision", "recall", "f1", "w_precision", "w_recall", "w_f1",
              "bad_precision", "bad_recall", "bad_f1", "balanced_accuracy", "mcc"]:
        v = md[k]
        summary[k] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                          min=float(v.min()), max=float(v.max()))
        print(f"  {k:12} {v.mean():6.2f} +/- {summary[k]['sd']:.2f} "
              f"({v.min():.2f}-{v.max():.2f})")
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    print(f"\nwrote per_fold.csv, summary.json to {out_dir}")


if __name__ == "__main__":
    main()
