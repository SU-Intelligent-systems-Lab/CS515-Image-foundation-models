# Model Card — Mini DINO (ViT-Tiny, CIFAR-100)

## Summary

This model card documents the from-scratch DINO student/teacher pair produced
by running `scripts/run_mini_dino.py` (or `notebooks/02_dino_mini_training.ipynb`)
with the default configuration `configs/mini_dino_cifar100.yaml`. **This is a
pedagogical artefact, not a production model.**

## Intended use

- **Primary use:** demonstrate end-to-end understanding of the DINO training
  procedure (self-distillation with a momentum teacher, multi-crop
  augmentation, centering + sharpening, cosine schedules).
- **Secondary use:** serve as a correctness check for the modules in
  `src/dino/`. If this model's kNN accuracy significantly exceeds random chance
  on CIFAR-100, the training machinery is confirmed to be working.
- **Out-of-scope uses:** any downstream deployment, transfer learning outside
  this project, or comparison against full-scale DINOv2/v3. The model is too
  small and trained on too little data for those purposes.

## Model details

| Field | Value |
|---|---|
| Architecture | Vision Transformer (ViT-Tiny) |
| Embedding dim | 192 |
| Depth | 12 |
| Heads | 3 |
| Patch size | 4 |
| Input resolution | 32×32 (CIFAR-100) |
| Parameter count | ~5.4M (backbone) + ~10.5M (DINO head) |
| DINO head K | 4096 |
| Head hidden dim | 2048, bottleneck 256 |

## Training

| Field | Value |
|---|---|
| Dataset | CIFAR-100 (50 000 train images, 100 classes, no labels used) |
| Augmentation | 2 global crops (32 px) + 6 local crops (16 px) |
| Loss | DINO cross-entropy with centering + sharpening |
| Optimizer | AdamW |
| Peak LR | 5e-4 (linear warm-up 5 epochs, cosine decay to 1e-5) |
| Weight decay | 0.04 → 0.4 (cosine) |
| Teacher temperature | 0.04 → 0.07 (warm-up 10 epochs) |
| Student temperature | 0.1 |
| Teacher EMA momentum | 0.996 → 1.0 (cosine) |
| Batch size | 128 |
| Epochs | 60 |
| Mixed precision | FP16 AMP |
| Hardware | single ~16 GB GPU |

## Evaluation

**Protocol:** kNN (k=20, cosine similarity, temperature 0.07) on frozen teacher
backbone features of CIFAR-100 train (fit) and test (query). No labels used
during training; labels used only during this evaluation.

**Reference numbers (fill in after running):**
- Random-init baseline (before training): `[to fill]` top-1
- After 60 epochs: `[to fill]` top-1

A healthy run should exceed 30% top-1 on CIFAR-100. Below 10% indicates a bug
in the training machinery (commonly: broken EMA update, wrong teacher temperature
scheduling, or mis-ordered multi-crop forward passes).

## Known limitations

1. **Absolute accuracy is far below the public DINOv2 checkpoints.** Our
   training set (50 k images of 32×32 resolution) is ~3 000× smaller than
   LVD-142M used for DINOv2, and our model is ~200× smaller than DINOv2-g/14.
2. **Attention maps are blurrier.** The emergent-segmentation property DINO
   is famous for is much weaker at this scale.
3. **Positional generalisation is limited.** Because training uses only
   32 px globals and 16 px locals, the positional-embedding interpolation has
   not been stressed at high resolution.

## Ethical considerations

CIFAR-100 is a well-known, standard academic dataset. Training uses only the
images, never the labels. The resulting model has no conceivable application
beyond the pedagogical and diagnostic role described here; it should not be
deployed or transferred to any sensitive domain.

## Reproducibility

Fixed seeds, deterministic config, and pinned dependencies are specified in
`configs/mini_dino_cifar100.yaml` and `requirements.txt`. The complete training
run can be reproduced on any CUDA-capable machine with:

```bash
python scripts/run_mini_dino.py --config configs/mini_dino_cifar100.yaml
```

A history JSON, training-curves figure, and final checkpoint are written to
`results/logs/mini_dino/`.
