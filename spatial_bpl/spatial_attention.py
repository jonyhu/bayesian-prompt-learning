"""
Patch-Level Spatial Attention for fine-grained discrimination.

Problem: CLIP's ViT encodes an image into a [CLS] token + patch tokens, then
only uses the [CLS] token (global average) for the final representation. For
fine-grained tasks like FGVC Aircraft, discriminative cues (engine nacelles,
winglet shapes, window density) are localized to specific spatial regions that
the [CLS] token may not adequately capture.

Solution: Extract intermediate patch tokens from CLIP's ViT, and apply a
lightweight cross-attention mechanism with learned "part queries" that
discover and attend to discriminative spatial regions.

Inspired by:
  - DETR's object queries (Carion et al., 2020)
  - PROMPT-CAM's class-specific prompt-to-patch attention (Chowdhury et al.,
    CVPR 2025)
  - Part-based recognition literature (Zhang et al., "Part-based R-CNNs")

The part queries are analogous to "what engine shape to look for" or "where
are the windows" — learned soft prototypes for discriminative regions.

Architecture:
    patch_tokens [B, N, D]  (N = 196 for 224x224 with patch_size=16)
    part_queries [K, D]     (K learned queries, typically 4-8)

    cross_attention(Q=part_queries, K=patch_tokens, V=patch_tokens)
        -> part_features [B, K, D]

    Aggregate: part_features -> global feature via attention-weighted pooling
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchSpatialAttention(nn.Module):
    """Cross-attention from learned part queries to ViT patch tokens.

    Learns K "part queries" that attend to specific spatial regions of the
    image. Each query discovers a discriminative part (e.g., engine, tail,
    wing shape) via cross-attention over patch tokens.

    The output is a compact part-based representation that supplements
    CLIP's global [CLS] token with spatially-grounded features.

    Args:
        embed_dim: dimension of patch tokens (768 for ViT-B/16)
        num_heads: number of attention heads
        num_parts: number of learned part queries (K)
        proj_dim: output projection dimension (should match CLIP's
                  text projection dim, typically 512)
        dropout: attention dropout
        dtype: parameter dtype
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        num_parts: int = 6,
        proj_dim: int = 512,
        dropout: float = 0.1,
        dtype=torch.float32,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_parts = num_parts
        self.proj_dim = proj_dim
        self.head_dim = embed_dim // num_heads

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        # Learned part queries: [K, D]
        self.part_queries = nn.Parameter(
            torch.randn(num_parts, embed_dim, dtype=dtype) * 0.02
        )

        # Multi-head cross-attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.k_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.v_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)
        self.out_proj = nn.Linear(embed_dim, embed_dim, dtype=dtype)

        self.attn_dropout = nn.Dropout(dropout)
        self.ln_query = nn.LayerNorm(embed_dim, dtype=dtype)
        self.ln_kv = nn.LayerNorm(embed_dim, dtype=dtype)

        # Aggregate part features into a single vector
        # Learned weighted sum over parts
        self.part_importance = nn.Parameter(
            torch.ones(num_parts, dtype=dtype) / num_parts
        )

        # Project from ViT embed_dim to CLIP's output dim
        self.output_proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim, dtype=dtype),
            nn.LayerNorm(proj_dim, dtype=dtype),
        )

    def forward(
        self, patch_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            patch_tokens: [B, N, D] — ViT patch tokens (excluding [CLS])
                          N=196 for 224x224 images with patch_size=16

        Returns:
            part_features: [B, proj_dim] — aggregated part-based representation
            attn_weights: [B, num_heads, K, N] — attention maps for
                          visualization (which patches each part query attends to)
        """
        B, N, D = patch_tokens.shape

        # Normalize inputs
        queries = self.ln_query(
            self.part_queries.unsqueeze(0).expand(B, -1, -1)
        )  # [B, K, D]
        kv = self.ln_kv(patch_tokens)  # [B, N, D]

        # Multi-head cross-attention
        K_parts = self.num_parts
        H = self.num_heads
        d_k = self.head_dim

        Q = self.q_proj(queries).view(B, K_parts, H, d_k).transpose(1, 2)  # [B, H, K, d_k]
        K = self.k_proj(kv).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]
        V = self.v_proj(kv).view(B, N, H, d_k).transpose(1, 2)  # [B, H, N, d_k]

        # Scaled dot-product attention
        scale = math.sqrt(d_k)
        attn_logits = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [B, H, K, N]
        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, H, K, N]
        attn_weights_dropped = self.attn_dropout(attn_weights)

        # Attend to values
        attended = torch.matmul(attn_weights_dropped, V)  # [B, H, K, d_k]
        attended = attended.transpose(1, 2).contiguous().view(B, K_parts, D)

        # Output projection
        part_features = self.out_proj(attended)  # [B, K, D]

        # Aggregate parts into single vector via learned importance weights
        importance = F.softmax(self.part_importance, dim=0)  # [K]
        aggregated = torch.einsum("bkd,k->bd", part_features, importance)  # [B, D]

        # Project to CLIP output dimension
        output = self.output_proj(aggregated)  # [B, proj_dim]

        return output, attn_weights

    def get_part_attention_maps(
        self, patch_tokens: torch.Tensor, image_size: int = 224, patch_size: int = 16
    ) -> torch.Tensor:
        """Get spatial attention maps for visualization.

        Returns attention maps reshaped to spatial grid for each part query.

        Args:
            patch_tokens: [B, N, D]
            image_size: original image size
            patch_size: ViT patch size

        Returns:
            maps: [B, K, H_grid, W_grid] where H_grid = W_grid = image_size // patch_size
        """
        _, attn_weights = self.forward(patch_tokens)
        # attn_weights: [B, num_heads, K, N]
        # Average over heads
        avg_attn = attn_weights.mean(dim=1)  # [B, K, N]

        grid_size = image_size // patch_size
        maps = avg_attn.view(-1, self.num_parts, grid_size, grid_size)
        return maps
