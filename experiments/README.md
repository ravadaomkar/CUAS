# experiments/

One folder per training run, so hyperparameters, resulting mAP, and notes
are tracked over time instead of scattered across terminal scrollback.

```
experiments/
├── exp01/
│   ├── config.yaml    # exact train_config.yaml used for this run (copy, not symlink)
│   ├── metrics.csv     # appended automatically by scripts/train_yolo.py --save_checkpoint
│   └── notes.md         # what you changed vs the previous exp and why
├── exp02/
└── exp03/
```

`scripts/train_yolo.py` auto-creates the next `expNN/` folder and appends a
row to its `metrics.csv` on every run (see `utils/helper.next_experiment_dir`
and `utils/helper.append_metrics_row`). Copying the config in by hand
(`cp configs/train_config.yaml experiments/expNN/config.yaml`) before you
change anything for the *next* run is what makes each exp folder
self-contained and diffable later.

exp01/exp02/exp03 below are seeded as templates — exp01 is the baseline
config already in `configs/train_config.yaml`; exp02/exp03 show the two
highest-leverage things worth sweeping next (image size and augmentation
strength). Duplicate the pattern for exp04+.
