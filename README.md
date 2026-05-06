# DINOv3 Semantic Segmentation — Coursework Project

This repository is a thin wrapper around Meta's official
[`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3) codebase.
It is submitted as a class project on **DINOv3 for semantic segmentation**.

The project covers two tasks supported by Meta's released code:

1. **Linear segmentation head — full training pipeline** on ADE20K.
   A frozen DINOv3 ViT backbone with a small `Conv2d(1×1)` head trained on top.
2. **Mask2Former (M2F) segmentation — inference only** on ADE20K and on
   custom images. Meta releases a pretrained M2F head; we use it as-is.

> **Note on what is and isn't in this repository.** The actual training,
> inference, and modelling code lives in Meta's repository, which you must
> clone separately (instructions below). This repository contains only:
>
> - A `setup` walkthrough with all the commands you need.
> - One additional script, `scripts/segment_image.py`, for running either
>   head on **your own images** (Meta's repo only ships ADE20K-validation-set
>   inference).
> - One additional config, `configs/config-ade20k-linear-training-1gpu.yaml`,
>   a single-GPU variant of Meta's linear training config for users without
>   an 8-GPU cluster.
> - The LaTeX report at `report/report.tex` (compiled to `report/report.pdf`).
> - A `NOTICE.md` explaining the relationship to the official DINOv3 repo.
>
> See `NOTICE.md` for licensing.

---

## 1. Prerequisites

- Linux (Meta's code is Linux-only).
- Python ≥ 3.11.
- CUDA toolkit installed and matching your PyTorch build (needed because
  Mask2Former uses a custom CUDA op, `MultiScaleDeformableAttention`, that is
  built at install time).
- A GPU. Linear-head training on ADE20K with ViT-L/16 frozen takes ~14 h on
  4×H100 in Meta's recipe; on a single 24 GB GPU you should expect a few
  days, or use a smaller backbone (ViT-S/16) and the 1-GPU config.
- Access to the DINOv3 model weights (gated download form at
  <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>).

## 2. Clone the official DINOv3 repository

```bash
# In the parent directory of this repo:
git clone https://github.com/facebookresearch/dinov3
```

After this step, your directory layout should be:

```
parent/
├── dinov3/                         # Meta's official repository (cloned)
│   ├── dinov3/                     # The Python package
│   ├── dinov3/eval/segmentation/   # The segmentation code we use
│   └── ...
└── dinov3_segmentation/            # This repository
    ├── README.md
    ├── NOTICE.md
    ├── scripts/
    ├── configs/
    └── report/
```

## 3. Install Meta's DINOv3 package

The recommended path (per Meta's README) is the conda environment file shipped
with the official repo:

```bash
cd dinov3
micromamba env create -f conda.yaml      # or: conda env create -f conda.yaml
micromamba activate dinov3
pip install -e .
```

This installs the `dinov3` Python package into your environment so that
`from dinov3.eval.segmentation.models import build_segmentation_decoder`
works from any directory.

If the conda file fails on your system, a minimal pip-only environment also
works for inference:

```bash
python -m venv .venv && source .venv/bin/activate
pip install "torch>=2.7" torchvision numpy Pillow omegaconf tqdm matplotlib opencv-python
cd dinov3 && pip install -e .
```

## 4. Compile the MultiScaleDeformableAttention CUDA op

Required only for **Mask2Former** (the linear head does not use it). From
inside Meta's repo:

```bash
cd dinov3/eval/segmentation/models/utils/ops
python setup.py build_ext --inplace
```

If the compilation fails, the most common causes are CUDA/PyTorch version
mismatch, missing `nvcc` on `PATH`, or `TORCH_CUDA_ARCH_LIST` not set. See
the file `dinov3/eval/segmentation/models/utils/ops/setup.py` for details.

## 5. Download the pretrained checkpoints

After Meta approves your access request you'll receive a list of download URLs.
Save the files anywhere — pass the paths on the command line. You'll typically
want:

| File | Used by |
|---|---|
| `dinov3_vitl16_pretrain_lvd1689m-...pth` | linear-head training, M2F backbone |
| `dinov3_vit7b16_pretrain_lvd1689m-...pth` | M2F backbone (the SOTA variant) |
| `dinov3_vit7b16_ade20k_m2f_head-...pth` | the M2F decoder head, ADE20K-trained |

The smaller backbones (ViT-S/B) are also fine for the linear-head experiment
and are noticeably easier to train on a single GPU.

## 6. Prepare ADE20K

The ADE20K Scene Parsing Challenge data is **not** redistributable. Download
`ADEChallengeData2016.zip` (~900 MB) from the official MIT site and unzip it.
Meta's adapter expects this directory layout:

```
/your/data/path/ADEChallengeData2016/
├── images/
│   ├── training/
│   └── validation/
└── annotations/
    ├── training/
    └── validation/
```

Pass the path on the command line as `datasets.root=/your/data/path` (the
suffix `/ADEChallengeData2016` is appended internally).

---

## 7. Train the linear segmentation head

This uses Meta's training script (`dinov3.eval.segmentation.run`) directly
with Meta's config. Note that Meta's config assumes 8 GPUs.

```bash
# From the official dinov3 repo:
PYTHONPATH=$PWD python -m dinov3.eval.segmentation.run \
  --config-file dinov3/eval/segmentation/configs/config-ade20k-linear-training.yaml \
  --backbone-config <path/to/your/backbone.pth or hub identifier> \
  datasets.root=/your/data/path
```

For single-GPU runs use the 1-GPU config from this repository:

```bash
PYTHONPATH=$PWD python -m dinov3.eval.segmentation.run \
  --config-file ../dinov3_segmentation/configs/config-ade20k-linear-training-1gpu.yaml \
  datasets.root=/your/data/path
```

(The 1-GPU config keeps Meta's optimizer, scheduler, and loss; it only
adjusts `n_gpus`, batch size, and image size for memory. See the file's
header for the full diff.)

## 8. Evaluate Mask2Former on ADE20K

```bash
PYTHONPATH=$PWD python -m dinov3.eval.segmentation.run \
  --config-file dinov3/eval/segmentation/configs/config-ade20k-m2f-inference.yaml \
  --eval-only \
  --backbone-config <path/to/backbone.pth> \
  --decoder-config <path/to/m2f_head.pth> \
  datasets.root=/your/data/path
```

This runs sliding-window inference at 896×896 with TTA scales
`[0.9, 0.95, 1.0, 1.05, 1.1]` and reports mIoU. Expect ≈55.9 mIoU with the
ViT-7B/16 backbone (per Meta's model card).

## 9. Run inference on your own images

This is the only entry point that does not exist in Meta's repository.
The script is at `scripts/segment_image.py`. It loads either head from a
checkpoint, runs sliding-window inference at the recipe-correct settings,
and saves a coloured overlay PNG.

### Linear head:

```bash
python scripts/segment_image.py \
  --head linear \
  --backbone dinov3_vitl16 \
  --backbone-weights /path/to/dinov3_vitl16_pretrain_lvd1689m.pth \
  --decoder-weights  /path/to/your_trained_linear_head.pth \
  --image  examples/my_photo.jpg \
  --output examples/my_photo_seg.png \
  --num-classes 150
```

### Mask2Former:

```bash
python scripts/segment_image.py \
  --head m2f \
  --backbone dinov3_vit7b16 \
  --backbone-weights /path/to/dinov3_vit7b16_pretrain_lvd1689m.pth \
  --decoder-weights  /path/to/dinov3_vit7b16_ade20k_m2f_head.pth \
  --image  examples/my_photo.jpg \
  --output examples/my_photo_seg.png \
  --num-classes 150
```

The script:

1. Builds the backbone via Meta's hub factory (`dinov3.hub.backbones`).
2. Builds the decoder via Meta's `build_segmentation_decoder` (linear or M2F).
3. Loads the decoder checkpoint with `strict=False` for linear, mirroring
   how Meta loads M2F checkpoints in `dinov3/hub/segmentors.py`.
4. Calls `dinov3.eval.segmentation.inference.make_inference` in `slide` mode
   with `crop_size`/`stride` matching Meta's eval configs (512/341 for
   linear, 896/596 for M2F).
5. Colourises the prediction with the ADE20K palette and saves the overlay.

Run `python scripts/segment_image.py --help` for all options.

---

## 10. Reproducibility & limitations

- I have not modified Meta's algorithmic code in any way; everything that
  matters for the model lives in the official repo.
- The ADE20K mIoU number you should be able to reproduce by running step 8
  is the one in Meta's `MODEL_CARD.md`.
- I am unable to redistribute Meta's pretrained weights or the ADE20K data;
  both must be downloaded by the user.
- The MSDeformAttn extension build is the most common point of failure;
  inspect Meta's build logs first if M2F doesn't load.

## 11. Report

A LaTeX report describing the linear and M2F pipelines in detail, with
file/line references into Meta's repo, is in `report/`. Build with

```bash
cd report && pdflatex report.tex && pdflatex report.tex
```

(Two passes for the table of contents.)

## 12. Citation

If you use this work, please cite Meta's DINOv3 paper. The full BibTeX entry
is reproduced in `report/report.tex`.
