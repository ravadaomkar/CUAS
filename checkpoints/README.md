# checkpoints/

Trained model weights land here, e.g.:

```
checkpoints/
├── best.pt        # best-mAP checkpoint from the current training run
├── last.pt         # most recent epoch, for resuming interrupted training
└── exp01_best.pt    # copy tagged by experiment id, once you're comparing runs
```

`train_yolo.py` (via `ultralytics`) actually writes weights to
`runs/detect/<name>/weights/{best,last}.pt` by default — that's intentional,
`runs/` is git-ignored because it also holds large per-epoch logs/plots you
don't want in version control. This folder is where you copy the two files
you actually want to keep:

```bash
cp runs/detect/cuas_train/weights/best.pt checkpoints/best.pt
cp runs/detect/cuas_train/weights/last.pt checkpoints/last.pt
```

`scripts/train_yolo.py --save_checkpoint` (see script) does this copy
automatically at the end of training and also records the run in
`experiments/<expNN>/metrics.csv` via `utils/helper.append_metrics_row`.

Weight files themselves (`*.pt`) are git-ignored — see `.gitignore` — since
they're large binaries that don't belong in git history. Use Git LFS, a
release asset, or shared cloud storage to hand off `best.pt` to teammates
or the submission pipeline.
