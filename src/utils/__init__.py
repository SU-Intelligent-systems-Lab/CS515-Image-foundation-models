"""
Feature-analysis, visualization, and I/O utilities.

Submodules
----------
feature_analysis
    Extract features from frozen backbones, compute patch-PCA visualizations,
    and run kNN / linear probes on extracted features.
visualization
    Attention-map visualizations, PCA-to-RGB rendering, and general plotting
    helpers used by the notebooks and analysis scripts.
io
    Config loading, checkpoint save/load, seed management.
"""

from .feature_analysis import (  # noqa: F401
    extract_features,
    knn_classify,
    linear_probe,
    patch_pca,
)
from .visualization import (  # noqa: F401
    attention_heatmap,
    overlay_heatmap,
    plot_loss_curves,
    save_figure,
)
from .io import (  # noqa: F401
    load_config,
    save_checkpoint,
    load_checkpoint,
    set_seed,
    get_device,
)
