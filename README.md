# Unveil the Mask

Code for **"Unveil the Mask: Jointly-Optimized Diffusion Masked Pretraining for Appearance-Agnostic Operator Verification"** (WBA2026 challenge, ACCV 2026 workshop).

The task is to verify whether two avatar videos are operated by the same person (the *driver*), regardless of the rendered avatar (the *source*). Our system has two branches, both trained from random initialization on the challenge training data only (no external data, pretrained weights or distillation):

- **Landmark branch** (`landmark/`)
  - appearance-normalized landmark sequences grouped into spatio-temporal tokens (Geo, Anat or Motion grouping)
  - **Stage 1**: diffusion masked pretraining (DiMP) with auxiliary behavioral tasks (motion energy, region motion, eye shape), optionally with pseudo-generator randomization
  - **Stage 2**: driver-aware contrastive linear probe on the frozen encoder
- **Video branch** (`video/`): inter-frame differencing model (Chapariniya et al., IEEE FG 2026) on landmark-guided face crops
- **Fusion**: weighted concatenation of the normalized embeddings of several models (`landmark/scripts/fuse.py`)

Final result on the WBA2026 test sets: Same Gen 0.901 / Diff Gen 0.637 / Overall 0.769 (AUC).

## Setup

Python 3.12, PyTorch 2.5 with CUDA, one GPU per run. Each branch has its own environment managed by [uv](https://docs.astral.sh/uv/):

```bash
cd landmark && uv sync && cd ../video && uv sync
```

## Data

Download the WBA2026 data and make `landmark/data` and `video/data` point to it (a symbolic link is enough):

```
data/landmark_2d_106/{train,test-gen1,test-gen2}/...
data/mp4/{train,test-gen1,test-gen2}/...
data/meta/{train-landmark_2d_106-meta.json, train-mp4-meta.json, test_annon-*-meta.json}
data/train_val_splits/{driver,source}_{train,val}_splits.json
```

All models are trained on the official `train` split, and the official `val` split is used for early stopping and model selection.

## Landmark branch

```bash
cd landmark

# token groupings (already provided in stats/; computed from the training split)
uv run python scripts/compute_stats.py --grouping bisect --num-groups 16 --output stats/tube_stats.pt         # Geo
uv run python scripts/compute_stats.py --grouping anatomical             --output stats/tube_stats_anat.pt    # Anat
uv run python scripts/compute_stats.py --grouping motion --num-groups 16 --output stats/tube_stats_motion.pt  # Motion

# Stage 1 (about 7 hours on one GPU) and Stage 2 (about 20 minutes)
uv run python pretrain.py --stats stats/tube_stats_anat.pt --seed 0 --output-dir output/pretrain-anat-s0
uv run python finetune.py --init-from output/pretrain-anat-s0/last_core.pt --output-dir output/probe-anat-s0

# test embeddings
for gen in gen1 gen2; do
  uv run python infer.py --generation $gen --num-gpus 1 --checkpoint output/probe-anat-s0/best_model.pt \
    --output output/submission-anat-s0/embeddings-$gen.bl2
done
```

`pretrain.py` uses all three auxiliary tasks with weight 0.01 by default. The Stage 1 variants used in the paper are selected with:

| Variant | Options |
|---|---|
| Geo / Anat / Motion tokens | `--stats stats/tube_stats.pt` / `stats/tube_stats_anat.pt` / `stats/tube_stats_motion.pt` |
| Condition dropout | `--lambda-mot 10 --motion-cond-dropout 0.15` |
| Velocity reconstruction | `--recon-target velocity` |
| Pseudo-generator randomization | `--lambda-mot 10 --pseudo-gen --pseudo-gen-dro-eta 0.01` (and `--pseudo-gen-aug` in `finetune.py`) |
| Without auxiliary tasks | `--aux-tasks ''` |

## Video branch

```bash
cd video
uv run python scripts/compute_face_boxes.py   # face boxes from the provided landmarks (no face detector)

uv run python train.py --face-crop --diff-mode pixel --seed 0 --output-dir output/video-crop-pixel-s0   # about 9 hours
uv run python train.py --diff-mode pixel --seed 0 --output-dir output/video-pixel-s0                    # full frames

for gen in gen1 gen2; do
  uv run python infer.py --generation $gen --num-gpus 1 --checkpoint output/video-crop-pixel-s0/best_model.pt \
    --output output/submission-video-crop-pixel-s0/embeddings-$gen.bl2
done
```

`--diff-mode feat` gives feature differencing, and `--frame-size 192` a larger input.

## Fusion and submission

Gen 1 (Same Gen) and Gen 2 (Diff Gen) are scored independently, so each file can use its own ensemble. Our final submission used:

- Same Gen: 9 landmark models (Geo; Anat with condition dropout; Anat with velocity reconstruction; seeds 0-2 each) and 9 video models (pixel differencing with face crop, seeds 0-5; without crop, seeds 0-2), with landmark weight 0.5
- Diff Gen: 9 landmark models with pseudo-generator randomization in Stages 1 and 2 (Geo, Anat and Motion tokens; seeds 0-2 each)

```bash
cd landmark
uv run python scripts/fuse.py \
  --gen1-landmark <landmark tags> --gen1-video <video tags> --gen1-alpha 0.5 \
  --gen2-landmark <landmark tags> --gen2-alpha 1.0 --output submission.zip
```

Each tag refers to `output/submission-<tag>/` of its branch.

## Tests

```bash
cd landmark && uv run pytest -q
cd video && uv run pytest -q
```

## License

MIT (see `LICENSE`).
