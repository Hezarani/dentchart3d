# Reporting checklist (TRIPOD+AI and CLAIM, adapted to a technical baseline study)

This project is a **technical development study on a public benchmark**, not a clinical validation.
The checklist below shows, item by item, what is reported and where. Items that need clinical data
are marked as not applicable and listed as limitations.

| Item | Where / status |
|---|---|
| Title and summary state the task, data and design | README (title, "At a glance") |
| Intended use and users | MODEL_CARD.md "Intended use"; research use only, not a medical device |
| Data source, licence and access | DATA.md (Teeth3DS+, CC BY-NC-ND 4.0, OSF link) |
| Eligibility / inclusion | DATA.md: all scans of data part 1 with a mesh and label file (300/300) |
| Reference standard (labels) | Teeth3DS+ expert per-vertex FDI annotations, as provided by the dataset authors |
| Preprocessing | DATA.md: normals, centring, scaling, 16,384-point sampling, seed 42 |
| Data split and leakage control | DATA.md + results/splits.json: fixed-seed 70/15/15 split per jaw; one scan per patient per jaw |
| Sample size | 210 train / 44 validation / 46 test scans; no formal sample-size calculation (benchmark subset) |
| Model description | README "Methods"; dentchart3d/models.py |
| Training procedure and hyper-parameters | README "Methods"; dentchart3d/train.py; scripts/train_all.sh |
| Model selection | Best validation mIoU checkpoint; test set used once |
| Performance measures with definitions | README "Results"; dentchart3d/metrics.py |
| Per-class / subgroup performance | per-tooth IoU (README figure) and upper vs lower jaw results |
| Robustness / stress tests | README "Robustness" (density, noise, rotation) |
| Explainability | dentchart3d/explain.py; README saliency figure |
| Uncertainty | Per-scan results in results/*_test_per_scan.json (distribution, not only the mean) |
| Comparison with baseline | PointNet vs DGCNN under the same split and budget |
| External validation | **Not done** (single public dataset). Listed as limitation |
| Clinical validation against chair-side measurements | **Not applicable** here. Listed as limitation |
| Fairness / subgroups (age, sex, scanner) | **Not available** in the public labels. Listed as limitation |
| Code and reproducibility | Code MIT; fixed seeds; tests in tests/; CI workflow; weights not shared because of the data licence |
| Funding and conflicts | None |
