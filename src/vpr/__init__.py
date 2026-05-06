"""
Visual Place Recognition (VPR) adaptation of pretrained DINO models.

This subpackage will be populated in Part III of the project. It will contain:

- backbone.py  : loaders for pretrained DINOv2 checkpoints + frozen inference
- adapter.py   : LoRA and adapter modules for ViT attention blocks
- aggregator.py: GeM / mean / cls pooling for global descriptors
- losses.py    : contrastive / triplet losses for VPR training
- train.py     : VPR fine-tuning loop
- eval.py      : Recall@1/5/10 evaluation on Pitts30k / MSLS / Tokyo24/7

For Phase 2 this module is intentionally empty; importing it is a no-op so that
the ``dinovpr`` package remains well-formed.
"""

# Intentionally empty. Will be filled in Part III.
