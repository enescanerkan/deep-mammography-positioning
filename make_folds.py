"""
10-fold cross-validation split generator.

Splits are made at the study level so that both breasts (L/R) of the same
study always land in the same fold, and MLO/CC stay synchronized. Folds are
stratified on CC positioning quality (Good/Bad).

Per fold i:  test = fold i, validation = fold (i+1) % 10, train = remaining 8
(~80/10/10, matching the manuscript protocol).

Outputs to labels/folds/:
    mlo_fold{i}.csv, cc_fold{i}.csv   - same schema as labels/*.csv, Split rewritten
    fold_assignment.csv               - breast-side -> fold map
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 10

ROOT = os.path.dirname(os.path.abspath(__file__))
LABELS = os.path.join(ROOT, "labels")
OUT = os.path.join(LABELS, "folds")
MLO_IMG = os.path.join(ROOT, "data", "processed", "mlo", "images")
CC_IMG = os.path.join(ROOT, "data", "processed", "cc", "images")


def side(series_description):
    """'L-MLO' -> 'L'"""
    return series_description.str[0]


def available(image_dir):
    """Base names of real (non-flipped) .npy files present on disk."""
    return {
        f[:-4]
        for f in os.listdir(image_dir)
        if f.endswith(".npy") and "_flipped" not in f
    }


def main():
    mlo = pd.read_csv(os.path.join(LABELS, "mlo_labels.csv"))
    cc = pd.read_csv(os.path.join(LABELS, "cc_labels.csv"))

    for df in (mlo, cc):
        df["Side"] = side(df.SeriesDescription)

    # One row per breast-side, carrying the CC quality label used for stratification.
    mlo_sides = mlo[["StudyInstanceUID", "Side", "SOPInstanceUID"]].drop_duplicates()
    cc_sides = cc[
        ["StudyInstanceUID", "Side", "SOPInstanceUID", "qualitativeLabel"]
    ].drop_duplicates()

    pairs = mlo_sides.merge(
        cc_sides,
        on=["StudyInstanceUID", "Side"],
        suffixes=("_mlo", "_cc"),
    )
    print(f"MLO breast-sides:  {len(mlo_sides)}")
    print(f"CC breast-sides:   {len(cc_sides)}")
    print(f"paired:            {len(pairs)}")

    # Drop pairs whose .npy is missing on disk, so all three models train on
    # exactly the same breast-sides.
    mlo_have, cc_have = available(MLO_IMG), available(CC_IMG)
    ok = pairs.SOPInstanceUID_mlo.isin(mlo_have) & pairs.SOPInstanceUID_cc.isin(cc_have)
    if (~ok).any():
        print(f"\ndropped {(~ok).sum()} pair(s) with missing .npy:")
        for _, r in pairs[~ok].iterrows():
            miss = []
            if r.SOPInstanceUID_mlo not in mlo_have:
                miss.append(f"MLO {r.SOPInstanceUID_mlo}")
            if r.SOPInstanceUID_cc not in cc_have:
                miss.append(f"CC {r.SOPInstanceUID_cc}")
            print(f"  {r.StudyInstanceUID} {r.Side}: {', '.join(miss)}")
    pairs = pairs[ok].reset_index(drop=True)
    print(f"\nusable pairs:      {len(pairs)}")
    print(f"studies:           {pairs.StudyInstanceUID.nunique()}")
    print(f"CC quality:        {pairs.qualitativeLabel.value_counts().to_dict()}")

    # Stratified on CC quality, grouped by study (both breasts stay together).
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    pairs["fold"] = -1
    for fold, (_, test_idx) in enumerate(
        sgkf.split(pairs, pairs.qualitativeLabel, groups=pairs.StudyInstanceUID)
    ):
        pairs.loc[test_idx, "fold"] = fold
    assert (pairs.fold >= 0).all(), "unassigned pair"

    os.makedirs(OUT, exist_ok=True)
    pairs.to_csv(os.path.join(OUT, "fold_assignment.csv"), index=False)

    key = ["StudyInstanceUID", "Side"]
    fold_of = pairs.set_index(key).fold

    print(f"\n{'fold':>4} {'train':>7} {'val':>6} {'test':>6} {'test Bad':>10}")
    for i in range(N_FOLDS):
        val_fold = (i + 1) % N_FOLDS
        split_of = pairs.fold.map(
            lambda f: "Test" if f == i else ("Validation" if f == val_fold else "Train")
        )
        split_of.index = pd.MultiIndex.from_frame(pairs[key])

        for name, df in (("mlo", mlo), ("cc", cc)):
            out = df.set_index(key)
            out = out[out.index.isin(fold_of.index)].copy()
            out["Split"] = split_of.reindex(out.index).values
            out.reset_index().drop(columns=["Side"]).to_csv(
                os.path.join(OUT, f"{name}_fold{i}.csv"), index=False
            )

        n = split_of.value_counts()
        bad = pairs[(pairs.fold == i) & (pairs.qualitativeLabel == "Bad")]
        print(
            f"{i:>4} {n.get('Train', 0):>7} {n.get('Validation', 0):>6} "
            f"{n.get('Test', 0):>6} {len(bad):>10}"
        )

    print(f"\nwrote {2 * N_FOLDS} csv files + fold_assignment.csv to {OUT}")


if __name__ == "__main__":
    main()
