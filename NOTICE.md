# NOTICE

## What this repository is

This repository is a **thin wrapper** around the official DINOv3 codebase
released by Meta at <https://github.com/facebookresearch/dinov3>. It was
prepared as an **academic coursework project** on DINOv3 for semantic
segmentation.

## What this repository contains

Only original material authored for the coursework:

- `README.md` — setup and usage walkthrough.
- `NOTICE.md` — this file.
- `scripts/segment_image.py` — a single-image inference utility that does
  not exist in the official DINOv3 repository. It calls into Meta's
  `dinov3` Python package (which the user must install separately) and
  is original code authored for this project.
- `configs/config-ade20k-linear-training-1gpu.yaml` — a single-GPU
  variant of Meta's linear-head training config. The variant adjusts only
  hardware-related fields (number of GPUs, batch size, image size); all
  algorithmic hyperparameters (optimizer, scheduler, loss weights, head
  architecture) are identical to Meta's reference config.
- `report/report.tex` — a LaTeX report describing the pipeline.
- `examples/` — placeholder for user-supplied input images.

## What this repository does NOT contain

- It does **not** re-distribute the DINOv3 source code. Users must clone
  `facebookresearch/dinov3` separately and install it as instructed in
  `README.md`.
- It does **not** re-distribute Meta's pretrained weights. Users must
  request them via Meta's gated download form at
  <https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/>.
- It does **not** re-distribute the ADE20K dataset. Users must download
  it from the original MIT source.

## License of this repository's original contents

The original files authored for this project (listed above) are released
under the same license as the official DINOv3 codebase, the **DINOv3
License Agreement**. A copy of that license can be found in the official
DINOv3 repository at
<https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md>.

This is a deliberate choice for a coursework derivative: when this project
is run, it composes original code with an installed copy of Meta's package,
and the combined behaviour falls under Meta's license terms. Users should
read Meta's license before using this project for anything beyond academic
study.

## Citation

The DINOv3 method is the work of Siméoni et al. (Meta, 2025). If this
repository is used, please cite the DINOv3 paper:

```bibtex
@misc{simeoni2025dinov3,
  title={{DINOv3}},
  author={Sim{\'e}oni, Oriane and Vo, Huy V. and Seitzer, Maximilian and
          Baldassarre, Federico and Oquab, Maxime and Jose, Cijo and
          Khalidov, Vasil and Szafraniec, Marc and Yi, Seungeun and
          Ramamonjisoa, Micha{\"e}l and Massa, Francisco and Haziza, Daniel and
          Wehrstedt, Luca and Wang, Jianyuan and Darcet, Timoth{\'e}e and
          Moutakanni, Th{\'e}o and Sentana, Leonel and Roberts, Claire and
          Vedaldi, Andrea and Tolan, Jamie and Brandt, John and Couprie,
          Camille and Mairal, Julien and J{\'e}gou, Herv{\'e} and Labatut,
          Patrick and Bojanowski, Piotr},
  year={2025},
  eprint={2508.10104},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2508.10104}
}
```

## Authorship

The original components of this repository (the README, the single-image
inference script, the 1-GPU config variant, and the LaTeX report) were
authored as a class assignment. The intellectual content of the segmentation
pipeline (architecture, training procedure, loss, evaluation protocol,
checkpoint files) belongs to Meta and is documented in the cited paper.
