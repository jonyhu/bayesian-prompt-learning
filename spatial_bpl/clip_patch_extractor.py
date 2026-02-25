"""
CLIP Patch Token Extractor — hooks into CLIP's ViT to extract intermediate
patch tokens needed for the spatial attention module.

CLIP's ViT architecture (for ViT-B/16):
    1. PatchEmbed: image [B,3,224,224] -> [B, 196, 768] patch embeddings
    2. Prepend [CLS] token: [B, 197, 768]
    3. Add positional embeddings
    4. 12 Transformer blocks (ResidualAttentionBlock)
    5. LayerNorm
    6. Extract [CLS] token -> project to 512-d output

We need the patch tokens AFTER the transformer blocks but BEFORE the final
projection, since they contain rich spatial information.

Two approaches:
    A) Hook-based: register a forward hook on the desired layer
    B) Manual forward: replicate the forward pass and intercept

We use approach B for maximum compatibility (hooks can be fragile with
torch.no_grad and mixed precision).
"""

import torch
import torch.nn as nn
from typing import Optional


def extract_patch_tokens(
    clip_model: nn.Module,
    images: torch.Tensor,
    layer_index: int = -1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract patch tokens from CLIP's visual transformer.

    Replicates CLIP ViT's forward pass up to a specified transformer layer,
    returning both the [CLS] token and all patch tokens.

    Args:
        clip_model: the CLIP model (from clip.load())
        images: [B, 3, 224, 224] preprocessed images
        layer_index: which transformer layer to extract from.
                     -1 = last layer (default), -2 = second-to-last, etc.

    Returns:
        cls_token: [B, D] the [CLS] token embedding
        patch_tokens: [B, N, D] the patch token embeddings (N=196 for ViT-B/16)
    """
    visual = clip_model.visual
    dtype = images.dtype

    # === Step 1: Patch embedding ===
    # conv1: [B, 3, 224, 224] -> [B, width, grid, grid]
    x = visual.conv1(images.type(visual.conv1.weight.dtype))
    # Reshape to [B, width, N] then transpose to [B, N, width]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 196]
    x = x.permute(0, 2, 1)  # [B, 196, 768]

    # === Step 2: Prepend [CLS] token ===
    # class_embedding: [768] -> [1, 1, 768]
    cls_token = visual.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)  # [B, 197, 768]

    # === Step 3: Add positional embeddings ===
    x = x + visual.positional_embedding.to(x.dtype)

    # === Step 4: Pre-transformer layer norm ===
    x = visual.ln_pre(x)

    # === Step 5: Transformer blocks ===
    # CLIP uses NLD format internally: [N, B, D]
    x = x.permute(1, 0, 2)  # [197, B, 768] (NLD)

    # Get the actual number of transformer layers
    num_layers = len(visual.transformer.resblocks)

    # Convert negative index to positive
    if layer_index < 0:
        target_layer = num_layers + layer_index
    else:
        target_layer = layer_index

    # Run transformer blocks up to target layer (inclusive)
    for i, block in enumerate(visual.transformer.resblocks):
        x = block(x)
        if i == target_layer:
            break

    # === Step 6: Post-transformer layer norm ===
    x = x.permute(1, 0, 2)  # [B, 197, 768] (BND)
    x = visual.ln_post(x)

    # Split [CLS] and patch tokens
    cls_out = x[:, 0, :]  # [B, 768]
    patches_out = x[:, 1:, :]  # [B, 196, 768]

    return cls_out, patches_out


def get_clip_visual_dim(clip_model: nn.Module) -> tuple[int, int]:
    """Get CLIP visual encoder dimensions.

    Returns:
        embed_dim: internal ViT dimension (768 for ViT-B/16)
        output_dim: projected output dimension (512 for ViT-B/16)
    """
    visual = clip_model.visual
    embed_dim = visual.conv1.out_channels  # 768
    output_dim = visual.output_dim if hasattr(visual, 'output_dim') else \
        visual.proj.shape[1] if visual.proj is not None else embed_dim
    return embed_dim, output_dim
