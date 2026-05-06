# Benchmarks

This document records the quantitative results produced by the project's
experiments. Numbers marked `[to fill]` will be populated after the
corresponding run has been executed end-to-end.

---

## Part II.A — Feature analysis of pretrained DINOv2

Produced by `notebooks/01_dino_feature_exploration.ipynb` (or
`scripts/run_feature_analysis.py`). Frozen `facebook/dinov2-base` (ViT-B/14,
86 M params), evaluated on CIFAR-100 (images upsampled to 224×224 with
bicubic).

| Probe | Top-1 (%) | Top-5 (%) |
|---|---:|---:|
| kNN (k=20, T=0.07) | `[to fill]` | `[to fill]` |
| Linear probe (100 ep) | `[to fill]` | `[to fill]` |

**Expected ranges** based on the DINOv2 paper's benchmarks on related
fine-grained tasks: kNN in the 80–90 % range, linear probe 2–4 points above
kNN. If the number falls far below that, either (i) the eval transform is
wrong, (ii) the processor normalisation was bypassed, or (iii) features are
not actually being L2-normalised before the kNN vote.

---

## Part II.B — Mini DINO training on CIFAR-100

Produced by `scripts/run_mini_dino.py` with `configs/mini_dino_cifar100.yaml`.
Frozen teacher backbone used for the kNN probe.

| Checkpoint | kNN top-1 (%) | kNN top-5 (%) |
|---|---:|---:|
| Random init (epoch 0)  | `[to fill]` | `[to fill]` |
| After 10 epochs        | `[to fill]` | `[to fill]` |
| After 30 epochs        | `[to fill]` | `[to fill]` |
| After 60 epochs (final)| `[to fill]` | `[to fill]` |

A healthy run should exceed 30 % top-1 at 60 epochs. Below 10 % indicates a
machinery bug (see `MODEL_CARD.md` for the typical culprits).

---

## Part III — Visual Place Recognition

**Not yet run.** Will be filled in with Recall@{1, 5, 10} on Pitts30k,
MSLS-val, and Tokyo24/7 across the three adaptation strategies:

- Frozen DINOv2 + GeM pooling (no training)
- LoRA-adapted DINOv2 fine-tuned on GSV-Cities
- EffoVPR-style zero-shot re-ranking using internal attention features

| Method | Pitts30k R@1 | Pitts30k R@5 | MSLS-val R@1 | MSLS-val R@5 | Tokyo24/7 R@1 | Tokyo24/7 R@5 |
|---|---:|---:|---:|---:|---:|---:|
| Frozen DINOv2 + GeM            | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` |
| LoRA DINOv2 + GeM (trained)    | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` |
| Frozen + EffoVPR-style re-rank | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` | `[to fill]` |

---

## Reproducibility notes

- All random seeds are fixed in the config (`experiment.seed`).
- Each table entry above has a corresponding run record in `results/logs/`
  (JSON history + YAML config + stdout log).
- Figures referenced in the report are in `results/figures/` and are
  regeneratable from the same scripts.
