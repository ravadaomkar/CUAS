# results/

Generated evaluation artifacts — everything here is reproducible by
re-running `scripts/evaluate.py`, so nothing in this folder (except this
README) is meant to be hand-edited.

```
results/
├── predictions/          # per-image YOLO-format prediction .txt files
├── confusion_matrix.png   # single-class (drone) TP/FP/FN counts at --conf
├── pr_curve.png            # precision-recall curve + AP@0.50
└── metrics.csv               # one row per evaluate.py run (mirrors experiments/ but for eval, not train)
```

Generate everything in one shot:

```bash
python scripts/evaluate.py \
    --weights checkpoints/best.pt \
    --data_dir data/real_reference \
    --out results/predictions \
    --plots results/
```

`--plots` is what triggers `utils/visualization.plot_pr_curve` and
`plot_confusion_matrix`, and `metrics.csv` is appended to via
`utils/helper.append_metrics_row` — same pattern as `experiments/`.
