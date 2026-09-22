"""
Cross-validation evaluation: reproduces the manuscript's internal table.

For each fold's test split, runs the MLO and CC landmark models on the
preprocessed .npy images, converts predictions to millimetres, applies the
PNL rule, and scores against the reference labels.

Decision chain (identical to run_full_evaluation.py):
    PNL_MLO = perpendicular distance nipple -> pectoral line   (512-space px)
    PNL_CC  = nipple_x (L) or 512 - nipple_x (R)               (512-space px)
    mm      = px * pixel_spacing_512
    dD      = |PNL_MLO - PNL_CC|
    quality = Good if dD <= 10 mm else Bad

pixel_spacing_512 (mm per pixel in model space):
    MLO - taken from transformation_details.csv 'adjusted_pixel_spacing'
    CC  - computed as max(crop_h, crop_w)/512 * ImagerPixelSpacing, since the
          CC details file has no spacing column. The same formula reproduces
          the MLO column to within 1.7e-4 mm/px over all 2780 rows.

    python evaluate_cv.py --folds 0-9
    python evaluate_cv.py --folds 0 --mlo-model <path> --cc-model <path>
"""

import argparse
import ast
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
CV_OUT = os.path.abspath(os.path.join(ROOT, "..", "cv_results"))
MLO_MAIN = os.path.join(ROOT, "rule-based-model/mlo-landmark-detection/code/regression/main")
CC_MAIN = os.path.join(ROOT, "rule-based-model/cc-landmark-detection/code/regression/main")
THRESHOLD_MM = 10.0


def load_model(main_dir, ckpt, out_features, device):
    """Import the trainer's own CRAUNet so checkpoints load exactly."""
    sys.path.insert(0, main_dir)
    for mod in [m for m in list(sys.modules) if m.startswith("utils")]:
        del sys.modules[mod]
    from utils.models import CRAUNet

    model = CRAUNet(in_channels=1, out_features=out_features).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    sys.path.pop(0)
    return model


def spacing_512(details, labels):
    """SOPInstanceUID -> mm per pixel in 512-space."""
    def first(v):
        return float(str(v).replace("\\", " ").split()[0])

    out = {}
    if "adjusted_pixel_spacing" in details.columns:
        for _, r in details.iterrows():
            out[r.SOPInstanceUID] = first(r.adjusted_pixel_spacing)
        return out

    # CC: derive from crop extent and the original spacing in the label file.
    orig = labels.drop_duplicates("SOPInstanceUID").set_index("SOPInstanceUID").ImagerPixelSpacing
    for _, r in details.iterrows():
        c = ast.literal_eval(r.crop_coords)
        L = max(c[1] - c[0], c[3] - c[2])
        if r.SOPInstanceUID in orig.index:
            out[r.SOPInstanceUID] = L / 512.0 * first(orig[r.SOPInstanceUID])
    return out


def perpendicular_distance(p1, p2, nipple):
    """Distance from nipple to the line through p1-p2."""
    line = p2 - p1
    vec = nipple - p1
    n = np.linalg.norm(line)
    if n == 0:
        return float(np.linalg.norm(vec))
    unit = line / n
    proj = np.dot(vec, unit) * unit
    return float(np.linalg.norm(vec - proj))


@torch.no_grad()
def predict(model, image_dir, sops, device, batch=32):
    """Run a landmark model over .npy images, returns {sop: coords in 512-space}."""
    preds = {}
    for i in range(0, len(sops), batch):
        chunk = sops[i:i + batch]
        arr = np.stack([np.load(os.path.join(image_dir, f"{s}.npy")) for s in chunk])
        x = torch.from_numpy(arr).float().unsqueeze(1).to(device)
        out = model(x).cpu().numpy()
        for s, o in zip(chunk, out):
            preds[s] = o.reshape(-1, 2) * 512.0
    return preds


def evaluate_fold(fold, mlo_ckpt, cc_ckpt, device):
    folds_dir = os.path.join(ROOT, "labels", "folds")
    mlo_lab = pd.read_csv(os.path.join(folds_dir, f"mlo_fold{fold}.csv"))
    cc_lab = pd.read_csv(os.path.join(folds_dir, f"cc_fold{fold}.csv"))

    mlo_det = pd.read_csv(os.path.join(ROOT, "data/processed/mlo/transformation_details.csv"))
    cc_det = pd.read_csv(os.path.join(ROOT, "data/processed/cc/transformation_details.csv"))
    mlo_sp = spacing_512(mlo_det, mlo_lab)
    cc_sp = spacing_512(cc_det, cc_lab)

    for df in (mlo_lab, cc_lab):
        df["Side"] = df.SeriesDescription.str[0]

    mlo_t = mlo_lab[mlo_lab.Split == "Test"].drop_duplicates(["StudyInstanceUID", "Side"])
    cc_t = cc_lab[cc_lab.Split == "Test"].drop_duplicates(["StudyInstanceUID", "Side"])
    pairs = mlo_t[["StudyInstanceUID", "Side", "SOPInstanceUID"]].merge(
        cc_t[["StudyInstanceUID", "Side", "SOPInstanceUID", "qualitativeLabel"]],
        on=["StudyInstanceUID", "Side"], suffixes=("_mlo", "_cc"),
    )

    mlo_model = load_model(MLO_MAIN, mlo_ckpt, 6, device)
    cc_model = load_model(CC_MAIN, cc_ckpt, 2, device)
    mlo_pred = predict(mlo_model, os.path.join(ROOT, "data/processed/mlo/images"),
                       pairs.SOPInstanceUID_mlo.tolist(), device)
    cc_pred = predict(cc_model, os.path.join(ROOT, "data/processed/cc/images"),
                      pairs.SOPInstanceUID_cc.tolist(), device)

    rows = []
    for _, r in pairs.iterrows():
        m, c = mlo_pred[r.SOPInstanceUID_mlo], cc_pred[r.SOPInstanceUID_cc]
        d_mlo = perpendicular_distance(m[0], m[1], m[2]) * mlo_sp[r.SOPInstanceUID_mlo]
        nx = c[0][0]
        d_cc = (nx if r.Side == "L" else 512 - nx) * cc_sp[r.SOPInstanceUID_cc]
        dd = abs(d_mlo - d_cc)
        rows.append(dict(
            study=r.StudyInstanceUID, side=r.Side,
            pnl_mlo_mm=d_mlo, pnl_cc_mm=d_cc, delta_mm=dd,
            pred="Good" if dd <= THRESHOLD_MM else "Bad",
            truth=r.qualitativeLabel,
        ))
    return pd.DataFrame(rows)


def score(truth, pred, n_total=None, n_abstained=0):
    """Good is the positive class, matching the manuscript's reporting."""
    truth, pred = np.asarray(truth), np.asarray(pred)
    tp = int(((pred == "Good") & (truth == "Good")).sum())
    fp = int(((pred == "Good") & (truth == "Bad")).sum())
    fn = int(((pred == "Bad") & (truth == "Good")).sum())
    tn = int(((pred == "Bad") & (truth == "Bad")).sum())
    n = len(truth)
    acc = (tp + tn) / n * 100 if n else 0.0
    prec = tp / (tp + fp) * 100 if tp + fp else 0.0
    rec = tp / (tp + fn) * 100 if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    out = dict(n=n, accuracy=acc, precision=prec, recall=rec, f1=f1,
               tp=tp, fp=fp, fn=fn, tn=tn)
    if n_abstained:
        out["abstained"] = n_abstained
        out["coverage"] = n / n_total * 100
    return out


def metrics(df, threshold=THRESHOLD_MM):
    """Single fixed threshold applied to every case."""
    pred = np.where(df.delta_mm <= threshold, "Good", "Bad")
    return score(df.truth.values, pred)


def metrics_dual(df, bad_t, good_t):
    """Dual threshold as implemented in the YOLO repo.

    Cases are split by ground truth first, then each group gets its own
    threshold (Bad -> bad_t, Good -> good_t). Because the split uses the
    label being predicted, everything inside the band scores as correct and
    the number rises with band width regardless of model quality. Kept for
    comparison with earlier runs; not a measure of model performance.
    """
    truth = df.truth.values
    pred = np.where(truth == "Bad",
                    np.where(df.delta_mm <= bad_t, "Good", "Bad"),
                    np.where(df.delta_mm <= good_t, "Good", "Bad"))
    out = score(truth, pred)
    out["band_cases"] = int(((df.delta_mm > bad_t) & (df.delta_mm <= good_t)).sum())
    return out


def metrics_abstain(df, lo, hi):
    """Grey zone: cases with lo < dD <= hi are left undecided and excluded.

    Outside the zone the ordinary rule applies. Coverage reports how much of
    the test set was actually decided.
    """
    grey = (df.delta_mm > lo) & (df.delta_mm <= hi)
    kept = df[~grey]
    pred = np.where(kept.delta_mm <= lo, "Good", "Bad")
    return score(kept.truth.values, pred, n_total=len(df), n_abstained=int(grey.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="0-9")
    ap.add_argument("--mlo-model", default=None, help="override checkpoint (single fold)")
    ap.add_argument("--cc-model", default=None)
    ap.add_argument("--out", default=os.path.join(CV_OUT, "evaluation"))
    args = ap.parse_args()

    if "-" in args.folds:
        lo, hi = args.folds.split("-")
        folds = list(range(int(lo), int(hi) + 1))
    else:
        folds = [int(x) for x in args.folds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out, exist_ok=True)

    all_metrics, all_rows = [], []
    for f in folds:
        mlo_ckpt = args.mlo_model or os.path.join(CV_OUT, "mlo", f"fold{f}", "mlo_model.pth")
        cc_ckpt = args.cc_model or os.path.join(CV_OUT, "cc", f"fold{f}", "cc_model.pth")
        if not (os.path.exists(mlo_ckpt) and os.path.exists(cc_ckpt)):
            print(f"fold {f}: checkpoint missing, skipped")
            continue
        df = evaluate_fold(f, mlo_ckpt, cc_ckpt, device)
        df.insert(0, "fold", f)
        all_rows.append(df)
        m = metrics(df)
        m["fold"] = f
        all_metrics.append(m)
        print(f"fold {f}: n={m['n']:>3} acc={m['accuracy']:.2f} prec={m['precision']:.2f} "
              f"rec={m['recall']:.2f} f1={m['f1']:.2f}")

    if not all_metrics:
        sys.exit("no folds evaluated")

    md = pd.DataFrame(all_metrics)
    cases = pd.concat(all_rows)
    cases.to_csv(os.path.join(args.out, "per_case.csv"), index=False)
    md.to_csv(os.path.join(args.out, "per_fold.csv"), index=False)

    def agg(per_fold):
        out = {}
        for k in ["accuracy", "precision", "recall", "f1"]:
            v = pd.Series([m[k] for m in per_fold])
            out[k] = dict(mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else 0.0,
                          min=v.min(), max=v.max())
        cov = [m["coverage"] for m in per_fold if "coverage" in m]
        if cov:
            out["coverage"] = dict(mean=float(np.mean(cov)))
        return out

    def show(title, out):
        print(f"\n{title}")
        for k in ["accuracy", "precision", "recall", "f1"]:
            s = out[k]
            print(f"  {k:10} {s['mean']:6.2f} +/- {s['sd']:.2f} ({s['min']:.2f}-{s['max']:.2f})")
        if "coverage" in out:
            print(f"  {'coverage':10} {out['coverage']['mean']:6.2f}%  (rest left undecided)")

    print("\n" + "=" * 62)
    print(f"{len(md)}-fold summary (mean +/- SD [min-max])")

    summary = {"primary_10mm": agg(all_metrics)}
    show(f"[primary] fixed threshold {THRESHOLD_MM:.0f} mm", summary["primary_10mm"])

    # Sensitivity: one threshold, applied to every case alike.
    print("\n--- threshold sensitivity (same threshold for all cases) ---")
    summary["threshold_sweep"] = {}
    for t in [8, 9, 10, 11, 12]:
        per_fold = [metrics(cases[cases.fold == f], t) for f in sorted(cases.fold.unique())]
        a = agg(per_fold)
        summary["threshold_sweep"][f"{t}mm"] = a
        print(f"  {t:>2} mm: acc={a['accuracy']['mean']:6.2f} +/- {a['accuracy']['sd']:.2f}   "
              f"f1={a['f1']['mean']:6.2f}   prec={a['precision']['mean']:6.2f}   "
              f"rec={a['recall']['mean']:6.2f}")

    # Grey zone: undecided in the band, excluded from scoring.
    summary["grey_zone"] = {}
    for lo, hi in [(9, 11), (8, 12)]:
        per_fold = [metrics_abstain(cases[cases.fold == f], lo, hi)
                    for f in sorted(cases.fold.unique())]
        a = agg(per_fold)
        summary["grey_zone"][f"{lo}-{hi}mm"] = a
        show(f"[grey zone] {lo}-{hi} mm undecided", a)

    # Dual threshold, for comparison with earlier runs. See metrics_dual().
    print("\n--- dual threshold (threshold chosen per ground-truth class) ---")
    print("    comparison only - band cases always score correct, so this")
    print("    tracks band width rather than model performance")
    summary["dual_threshold_gt_dependent"] = {}
    for bad_t, good_t in [(10, 10), (9, 11), (8, 12)]:
        per_fold = [metrics_dual(cases[cases.fold == f], bad_t, good_t)
                    for f in sorted(cases.fold.unique())]
        a = agg(per_fold)
        a["band_cases_mean"] = float(np.mean([m["band_cases"] for m in per_fold]))
        summary["dual_threshold_gt_dependent"][f"{bad_t}-{good_t}mm"] = a
        print(f"  {bad_t:>2}-{good_t:<2} mm: acc={a['accuracy']['mean']:6.2f} +/- {a['accuracy']['sd']:.2f}   "
              f"f1={a['f1']['mean']:6.2f}   (band: {a['band_cases_mean']:.1f} cases/fold)")

    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), indent=2, default=float)
    print(f"\nwrote per_case.csv, per_fold.csv, summary.json to {args.out}")


if __name__ == "__main__":
    main()
