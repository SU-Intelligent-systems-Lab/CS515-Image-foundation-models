# Methods

This document gives a concise technical description of each method implemented
in this repository, complementing the detailed treatment in the project report.

---

## 1. DINO self-distillation (from scratch, Part II)

**Implementation:** `src/dino/`.

### Model components
- **Backbone:** Vision Transformer (ViT-Tiny / ViT-Small), implemented from
  scratch in `src/dino/model.py`. Standard pre-norm blocks, patch embedding
  via `Conv2d`, learnable [CLS] token, learnable 1-D positional embedding
  with bicubic interpolation to support variable input sizes (required for
  multi-crop).
- **DINO head:** 3-layer MLP with GELU → L2-normalise → weight-normalised
  linear to K prototypes, following Caron et al. 2021 §3. The magnitude of
  the weight-norm is frozen to 1 for training stability.

### Loss
Cross-entropy between sharpened, centered teacher distributions and softmaxed
student distributions, summed over every (teacher global crop, student other
crop) pair with same-view pairs excluded. Centering uses an EMA over the
batch; sharpening uses τ_t ≪ τ_s. See `src/dino/loss.py` for the explicit
formula.

### Optimisation
- **Student:** AdamW, cosine LR schedule with linear warm-up, cosine WD
  schedule (0.04 → 0.4).
- **Teacher:** EMA of student weights, momentum cosine-scheduled from 0.996
  to 1.0 over the run.
- **Teacher temperature:** warmup 0.04 → 0.07.
- **Gradient clipping:** 3.0.
- **First-epoch stabiliser:** freeze the DINO head's last layer to prevent
  early-training instability.
- **Mixed precision:** FP16 AMP.

### Multi-crop augmentation
2 global crops (32 px, scale 0.4–1.0) + 6 local crops (16 px, scale 0.05–0.4).
Global crops use ColorJitter + flips + always-on GaussianBlur on the first
and occasional Solarize on the second. Local crops use weaker blur. The
teacher forward pass uses only the global crops; the student forwards all
crops.

---

## 2. Feature analysis of pretrained DINOv2 (Part II)

**Implementation:** `src/utils/feature_analysis.py`,
`notebooks/01_dino_feature_exploration.ipynb`, `scripts/run_feature_analysis.py`.

### Qualitative probes
- **Attention maps:** final-block CLS→patch attention of HuggingFace
  `facebook/dinov2-base`, extracted with `output_attentions=True`, overlaid on
  the input.
- **Patch-feature PCA:** top-3 principal components of the per-patch features,
  rendered as RGB and upsampled. Optional foreground-mask variant using the
  sign of PC1.
- **Dense patch similarity:** cosine similarity of a chosen query patch
  against every patch of a target image.

### Quantitative probes
- **kNN:** weighted-vote kNN (cosine similarity, k=20, temperature 0.07) on
  L2-normalised features. Follows the DINO / DINOv2 linear-kNN protocol.
- **Linear probe (optional):** single linear classifier on frozen features,
  trained 100 epochs with SGD+cosine schedule.

---

## 3. Visual Place Recognition (Part III — planned)

Three adaptation strategies will be implemented in `src/vpr/`:

### 3.1 Frozen DINOv2 + GeM pooling
- Backbone: `facebook/dinov2-base`, frozen.
- Descriptor: **GeM** (Generalized Mean) pooling of patch tokens, optionally
  combined with the CLS token.
- Eval: cosine similarity against a precomputed reference database.
- No training required.

### 3.2 LoRA-adapted DINOv2 + GeM
- Backbone: DINOv2, frozen weights, LoRA modules inserted on the Q and V
  projections of every attention block (rank 8 by default).
- Descriptor: same GeM head as above.
- Training: contrastive loss on GSV-Cities with triplet mining.
- ~0.5 M trainable parameters vs. 86 M for full fine-tuning.

### 3.3 Zero-shot EffoVPR-style re-ranking
- Stage 1: global retrieval using frozen DINOv2 CLS features.
- Stage 2: re-rank the top-K candidates by mutual-nearest-neighbour count
  of internal-attention key features (no training).
- Baseline from Tzachor et al. 2024.

---

## Evaluation protocol (VPR)

- **Metrics:** Recall@{1, 5, 10}. A retrieval is correct if at least one of
  the top-N ground-truth references lies within 25 m (and within 40° for MSLS)
  of the query.
- **Benchmarks:** Pitts30k, MSLS-val, Tokyo24/7.
- **Hardware budget:** all adaptation methods must run on a single 16 GB GPU.
- **Reporting:** every row of `docs/BENCHMARKS.md` Part III is linked to a
  reproducible config in `configs/vpr_*.yaml`.

---

## Design choices and justifications

| Choice | Justification |
|---|---|
| ViT-Tiny for mini training | A larger backbone makes the loss slower to
converge and obscures the point of the demonstration. Tiny is small enough to
finish in ~20 min on a single GPU yet large enough to show the mechanism works. |
| K = 4096 instead of 65 536 | A high-dimensional prototype space is only
useful when there are millions of training images to populate it. At CIFAR-100
scale, 4 096 is ample and trains much faster. |
| Bicubic upsampling for CIFAR eval with DINOv2 | DINOv2 expects 224-px inputs;
CIFAR's 32-px images must be upsampled. Bicubic is the standard choice. Raw
CIFAR features from DINOv2 at 32 px are not meaningful because the patch size
(14 px) nearly equals the image. |
| HuggingFace `transformers` for DINOv2 loading | Stable, well-maintained API;
avoids hand-porting the backbone. Direct access to attention outputs via
`output_attentions=True`. |
| LoRA on Q, V (not K) | Standard choice from the LoRA paper; V carries
value-level information and Q carries query routing. Matches the convention in
DINOv2-LoRA follow-ups. |
| No distributed training | Project scope. A single-GPU build is easier to
document and reproduce. Distributed training can be added later (`torchrun`)
without touching any module internals. |
