"""
Spatial Bayesian Prompt Learning (Spatial-BPL)

Extensions to Bayesian Prompt Learning for fine-grained visual classification,
specifically targeting datasets like FGVC-Aircraft where discriminative features
are fine-grained spatial details (engine types, window patterns, wing shapes).

Three complementary improvements over standard BPL:

1. Bayesian Visual Adapter — learns a distribution over image feature residuals
   (analogous to how BPL learns text prompt residuals), implemented as a
   lightweight bottleneck MLP with Bayesian parameters.

2. Patch-Level Spatial Attention — extracts intermediate ViT patch tokens and
   applies learned part-queries via cross-attention to localize discriminative
   spatial regions.

3. Part-Aware Contrastive Loss — encourages the model to distinguish fine-grained
   classes by mining hard negatives and enforcing separation in the patch-feature
   space.

Reference:
  - Derakhshani et al., "Bayesian Prompt Learning for Image-Language Model
    Generalization", ICCV 2023.
  - Gao et al., "CLIP-Adapter: Better Vision-Language Models with Feature
    Adapters", IJCV 2024.
  - Jia et al., "Visual Prompt Tuning", ECCV 2022.
"""

from spatial_bpl.bayesian_visual_adapter import BayesianVisualAdapter
from spatial_bpl.spatial_attention import PatchSpatialAttention
from spatial_bpl.losses import PartAwareContrastiveLoss, SpatialBPLLoss
from spatial_bpl.spatial_bpl_model import SpatialBPL
from spatial_bpl.trainer import SpatialBPLTrainer

__all__ = [
    "BayesianVisualAdapter",
    "PatchSpatialAttention",
    "PartAwareContrastiveLoss",
    "SpatialBPLLoss",
    "SpatialBPL",
    "SpatialBPLTrainer",
]
