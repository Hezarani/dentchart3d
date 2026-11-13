# Data

**Dataset.** Teeth3DS+ (3DTeethSeg'22 / 3DTeethLand'24 MICCAI challenges), Ben-Hamadou et al.,
https://crns-smartvision.github.io/teeth3ds , data at https://osf.io/xctdy/ .

**Licence.** CC BY-NC-ND 4.0: non-commercial research and education only. The data is **not**
redistributed here. Run `python scripts/download_teeth3ds.py --parts 1` to download it from the
official source. Trained weights are also not published, because they are derived from the data;
the full pipeline can be re-run with `scripts/train_all.sh`.

**What was used.** Data part 1: 300 intraoral scans (150 upper, 150 lower jaws), each a triangle
mesh of about 80,000-130,000 vertices with per-vertex FDI tooth labels (0 = gingiva).

**Preprocessing.** Area-weighted vertex normals from the mesh; coordinates centred and scaled to
the unit sphere; 16,384 vertices kept per scan by uniform random sampling (seed 42).

**Split.** Per jaw, scans were shuffled with a fixed seed and split 70 / 15 / 15 into training
(210), validation (44) and test (46) scans. Within a jaw each scan belongs to a different patient,
so no patient appears in two subsets of the same jaw. The exact scan IDs are in
`results/splits.json`. The test set was used once, for the final numbers in the README.

**Labels.** 17 classes per jaw (see `dentchart3d/labels.py`): gingiva, plus eight tooth positions
on the patient's right and eight on the patient's left. Right/left classes are aligned across the
two jaws, so, for example, teeth 16 and 46 share one class.
