# DINOv3 Segmentation Study
# ===========================
# This package provides annotated wrappers around the DINOv3 evaluation pipeline
# for semantic segmentation. It covers:
#
#   - model_loading.py    : Loading pretrained DINOv3 backbones (ViT-L, ViT-7B)
#   - linear_probe.py     : Training a linear segmentation head on frozen features
#   - mask2former_inference.py : Running Mask2Former inference with pretrained decoder
#   - evaluation.py       : Computing mIoU and related segmentation metrics
#   - visualization.py    : PCA feature maps and segmentation overlays
#   - data_utils.py       : Dataset loading utilities for ADE20k and Pascal VOC
