# External test-set labels

Reference landmarks and positioning labels for the two external cohorts of the
paper, in the same CSV schema as `labels/mlo_labels.csv` and `labels/cc_labels.csv`
(one row per annotation; `Split` is always `Test`). Images are not included: they
are obtained from the source databases under their own terms, and every row is
keyed by the original `StudyInstanceUID` / `SOPInstanceUID`.

| Files | Cohort |
|-------|--------|
| `cmmd_cc_labels.csv`, `cmmd_mlo_labels.csv` | Chinese Mammography Database (CMMD, TCIA) |
| `embed_cc_labels.csv`, `embed_mlo_labels.csv` | EMBED (Emory Breast Imaging Dataset, open subset) |

## Common conventions

- `Nipple` rows (`annotationMode = bounding_box`): use the box centre as the
  nipple point.
- `Pectoralis` rows (`annotationMode = line`): `vertices` = [upper end, lower
  end] of the pectoral muscle line, in original pixel coordinates.
- `qualitativeLabel` on the **CC** rows is the reference positioning label of the
  pair: Good when |PNL_MLO - PNL_CC| <= 10 mm from the reference landmarks, where
  PNL_MLO is the perpendicular distance from the nipple to the pectoral line and
  PNL_CC the horizontal distance from the nipple to the posterior (chest-wall)
  image edge. Recomputing the rule from these files reproduces the CC label of
  every pair in both cohorts.
- `ImagerPixelSpacing` is DICOM (0018,1164) as in the internal labels;
  `height`/`width` are the stored matrix size.

## CMMD

Breast-side CC-MLO pairs of the external test set. One breast radiologist
annotated the nipple (box) on CC and the nipple, the pectoral line and the
posterior nipple line (`PNL`, `annotationMode = line`, as drawn) on MLO, and
graded MLO positioning; all pairs have a Good MLO, as the pairing rule requires. Every image is 1914 x 2294 px at 0.0941 mm; the CMMD
headers carry no manufacturer or model tag (`ManufacturerModelName` is left
blank; the CMMD data descriptor reports GE systems), and the chest wall is on
the laterality side of the stored pixels for every image.

## EMBED

**Selection.** 200 examinations (400 breast-sides) were randomly selected from
the open EMBED subset (Hologic and GE units); 250 breast-side CC-MLO pairs were
drawn at random from them, and the 203 eligible pairs evaluated in the paper are
included here (Hologic n = 178, GE n = 25). Devices, as recorded in
`ManufacturerModelName`: Selenia Dimensions (172 pairs), Lorad Selenia (6),
Senographe Essential (14), Senograph 2000D (11).

**Annotation.** One breast radiologist placed, on every image, a nipple point and,
on MLO, the upper and lower end of the pectoral muscle line, under the same
protocol as the internal and first external cohorts. Where the radiologist drew
the nipple outline but no point, the point was set to the area centroid of her
outline. `Nipple` rows carry a 200 x 200 px square centred on the annotated point
(the point is a landmark, not a drawn box). The MLO rows carry `Good` on every
image: MLO positioning was not graded in EMBED, and the value only satisfies the
pairing filter of the data loaders (MLO must be Good).

`ImagerPixelSpacing` is 0.070 mm for the Hologic units; the magnification-corrected
PixelSpacing (0028,0030), 0.065238 mm, is not used.

**Chest-wall side (EMBED only).** Two extra trailing columns are added because
EMBED stores about 40 % of its Hologic images mirrored
(`FieldOfViewHorizontalFlip = YES`), so the chest wall is not always on the side
that laterality implies:

| Column | Values | Meaning |
|--------|--------|---------|
| `ChestWallSide` | `L` / `R` | side of the stored pixel array on which the chest wall (posterior edge) lies |
| `FieldOfViewHorizontalFlip` | `YES` / `NO` | DICOM (0018,7034) as stored |

PNL_CC must be measured towards `ChestWallSide` (x for `L`, width - x for `R`).
The internal VinDr-Mammo and the CMMD labels never need this: there the chest
wall is always on the laterality side, which is what `run_full_evaluation.py`
assumes.
