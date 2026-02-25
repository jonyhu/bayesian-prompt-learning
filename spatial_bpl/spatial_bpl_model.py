"""
Spatial Bayesian Prompt Learning (Spatial-BPL) — unified model.

This model extends BPL with three complementary improvements for fine-grained
visual classification:

1. BAYESIAN VISUAL ADAPTER: Learns a distribution over image feature residuals.
   Standard BPL perturbs text embeddings; we additionally perturb image
   embeddings. This is especially important for FGVC Aircraft where the visual
   domain gap is large (all aircraft look similar at global level).

2. PATCH-LEVEL SPATIAL ATTENTION: Extracts ViT patch tokens and applies
   cross-attention with learned part queries to discover discriminative
   spatial regions (engines, windows, tail shapes).

3. DUAL-STREAM FEATURE FUSION: Combines the global [CLS] features (adapted
   via Bayesian visual adapter) with part-based spatial features. The fusion
   is a learned weighted combination that allows the model to dynamically
   balance global context vs local discriminative detail.

Forward pass:
    images -> CLIP ViT -> [CLS] token + patch tokens
                              |              |
                   Bayesian Visual     Patch Spatial
                      Adapter          Attention
                         |              |
                    adapted_global   part_features
                         |              |
                         +-- Fusion ----+
                              |
                         final_image_features [L, B, D]

    text prompts -> BPL text encoder -> text_features [L, C, D]

    logits = image_features @ text_features^T  (per MC sample)
"""

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from spatial_bpl.bayesian_visual_adapter import BayesianVisualAdapter
from spatial_bpl.spatial_attention import PatchSpatialAttention
from spatial_bpl.clip_patch_extractor import extract_patch_tokens, get_clip_visual_dim


class SpatialBPL(nn.Module):
    """Spatial Bayesian Prompt Learning model.

    Combines BPL text-side prompt learning with visual-side adaptation
    via a Bayesian adapter and spatial attention.

    Args:
        classnames: list of class name strings
        clip_model: loaded CLIP model
        n_tokens: number of learnable text prompt tokens
        n_samples: number of MC samples (L)
        device: torch device
        adapter_reduction: bottleneck reduction for visual adapter
        adapter_alpha: residual blend ratio for visual adapter
        num_parts: number of part queries for spatial attention
        fusion_alpha: balance between global and spatial features
                      0 = global only, 1 = spatial only
        patch_layer: which ViT layer to extract patches from (-1 = last)
        use_visual_adapter: enable/disable Bayesian visual adapter
        use_spatial_attention: enable/disable patch spatial attention
    """

    def __init__(
        self,
        classnames: list[str],
        clip_model: nn.Module,
        n_tokens: int = 4,
        n_samples: int = 10,
        device: str = "cuda",
        adapter_reduction: int = 4,
        adapter_alpha: float = 0.2,
        num_parts: int = 6,
        fusion_alpha: float = 0.3,
        patch_layer: int = -1,
        use_visual_adapter: bool = True,
        use_spatial_attention: bool = True,
    ):
        super().__init__()
        self.device = device
        self.n_samples = n_samples
        self.classnames = classnames
        self.clip_model = clip_model
        self.n_tokens = n_tokens
        self.n_classes = len(classnames)
        self.dtype = clip_model.dtype
        self.patch_layer = patch_layer
        self.use_visual_adapter = use_visual_adapter
        self.use_spatial_attention = use_spatial_attention
        self.fusion_alpha = fusion_alpha

        # Get CLIP dimensions
        vit_dim, output_dim = get_clip_visual_dim(clip_model)
        self.vit_dim = vit_dim  # 768 for ViT-B/16
        self.output_dim = output_dim  # 512 for ViT-B/16
        self.d_token = clip_model.token_embedding.weight.shape[1]  # 512

        # ============================================================
        # TEXT SIDE: Standard BPL prompt learning (unchanged)
        # ============================================================

        # Base context (deterministic prompt)
        self.ctx = nn.Parameter(
            torch.randn(n_tokens, self.d_token, device=device, dtype=self.dtype)
            * 0.02
        )

        # Bayesian residual parameters (text side)
        self.text_mu = nn.Parameter(
            torch.zeros(1, self.d_token, device=device, dtype=self.dtype)
        )
        self.text_logvar = nn.Parameter(
            torch.rand(1, self.d_token, device=device, dtype=self.dtype)
        )

        # Text prompt setup
        prompt_prefix = " ".join(["X"] * n_tokens)
        prompts = [prompt_prefix + " " + name + "." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            input_embeddings = clip_model.token_embedding(
                tokenized_prompts.to(device)
            )

        self.register_buffer("prefix", input_embeddings[:, :1, :])
        self.register_buffer("suffix", input_embeddings[:, 1 + n_tokens :, :])
        self.register_buffer("tokenized_prompts", tokenized_prompts)

        # ============================================================
        # VISUAL SIDE: New components
        # ============================================================

        # 1. Bayesian Visual Adapter (on projected features)
        if use_visual_adapter:
            self.visual_adapter = BayesianVisualAdapter(
                feature_dim=output_dim,
                reduction=adapter_reduction,
                alpha=adapter_alpha,
                n_samples=n_samples,
                dtype=torch.float32,  # Use float32 for adapter stability
            )

        # 2. Patch-Level Spatial Attention
        if use_spatial_attention:
            self.spatial_attention = PatchSpatialAttention(
                embed_dim=vit_dim,
                num_heads=8,
                num_parts=num_parts,
                proj_dim=output_dim,
                dropout=0.1,
                dtype=torch.float32,
            )

            # Learnable fusion parameter (logit, sigmoidified)
            self.fusion_logit = nn.Parameter(
                torch.tensor(0.0, dtype=torch.float32)
            )

    def sample_text_residual(self, mu, logvar, n_samples):
        """Sample text prompt residual using reparameterization trick."""
        shape = (n_samples,) + mu.size()
        eps = torch.randn(shape, device=self.device, dtype=self.dtype)
        bias = mu.unsqueeze(0) + eps * (0.5 * logvar).exp().unsqueeze(0)
        return bias  # [L, 1, d_token]

    def encode_text(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode text prompts with BPL sampling.

        Returns:
            text_features: [L, C, output_dim]
            text_mu: [1, d_token]
            text_logvar: [1, d_token]
        """
        L = self.n_samples
        bias = self.sample_text_residual(
            self.text_mu, self.text_logvar, L
        )  # [L, 1, d_token]

        ctx = self.ctx.unsqueeze(0)  # [1, n_tokens, d_token]
        ctx_shifted = ctx + bias  # [L, n_tokens, d_token]

        all_text_features = []
        for l in range(L):
            ctx_l = ctx_shifted[l]  # [n_tokens, d_token]
            ctx_l = ctx_l.unsqueeze(0).expand(
                self.n_classes, -1, -1
            )  # [C, n_tokens, d_token]

            prompts = torch.cat(
                [self.prefix, ctx_l, self.suffix], dim=1
            )  # [C, 77, d_token]

            # Add positional embeddings
            prompts = prompts + self.clip_model.positional_embedding.to(
                self.dtype
            )

            # CLIP text transformer
            x = prompts.permute(1, 0, 2)  # [77, C, d_token]
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)  # [C, 77, d_token]
            x = self.clip_model.ln_final(x).type(self.dtype)

            # Extract EOT features
            eot_indices = self.tokenized_prompts.argmax(dim=-1)
            x = (
                x[torch.arange(self.n_classes), eot_indices]
                @ self.clip_model.text_projection
            )

            all_text_features.append(x)

        text_features = torch.stack(all_text_features, dim=0)  # [L, C, output_dim]
        return text_features, self.text_mu, self.text_logvar

    def encode_image(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Encode images with optional visual adapter and spatial attention.

        Args:
            images: [B, 3, 224, 224]

        Returns:
            image_features: [L, B, output_dim] adapted image features
            part_features: [B, output_dim] or None if spatial attention disabled
        """
        L = self.n_samples
        B = images.shape[0]

        # === Extract patch tokens and CLS from ViT ===
        with torch.no_grad():
            if self.use_spatial_attention:
                cls_token, patch_tokens = extract_patch_tokens(
                    self.clip_model, images, layer_index=self.patch_layer
                )
                # Project CLS token to output dim (like CLIP does)
                if self.clip_model.visual.proj is not None:
                    global_features = cls_token @ self.clip_model.visual.proj
                else:
                    global_features = cls_token
            else:
                # Standard CLIP encoding
                global_features = self.clip_model.encode_image(images)
                patch_tokens = None

        # global_features: [B, output_dim] (e.g., [B, 512])
        global_features = global_features.float()

        # === Bayesian Visual Adapter ===
        if self.use_visual_adapter:
            adapted_global = self.visual_adapter(
                global_features, n_samples=L
            )  # [L, B, output_dim]
        else:
            adapted_global = global_features.unsqueeze(0).expand(
                L, -1, -1
            )  # [L, B, output_dim]

        # === Patch Spatial Attention ===
        part_features = None
        if self.use_spatial_attention and patch_tokens is not None:
            patch_tokens_float = patch_tokens.float()
            part_feat, _ = self.spatial_attention(
                patch_tokens_float
            )  # [B, output_dim]
            part_features = part_feat

            # Fuse global + spatial
            # Dynamic fusion ratio via sigmoid
            alpha = torch.sigmoid(self.fusion_logit)  # scalar in [0, 1]

            # Expand part_features to match [L, B, output_dim]
            part_expanded = part_features.unsqueeze(0).expand(L, -1, -1)

            # Fused features
            image_features = (1 - alpha) * adapted_global + alpha * part_expanded
        else:
            image_features = adapted_global

        return image_features, part_features

    def forward(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Full forward pass.

        Args:
            images: [B, 3, 224, 224]

        Returns:
            logits: [L, B, C] logits for each MC sample
            info: dict with intermediate values for loss computation
        """
        # Encode text (with BPL sampling)
        text_features, text_mu, text_logvar = self.encode_text()
        # [L, C, D]
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Encode image (with visual adapter + spatial attention)
        image_features, part_features = self.encode_image(images)
        # [L, B, D]
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute logits
        logit_scale = self.clip_model.logit_scale.exp()
        # [L, B, D] @ [L, D, C] -> [L, B, C]
        logits = logit_scale * torch.einsum(
            "lbd,lcd->lbc", image_features.type(self.dtype), text_features
        )

        # Visual KL
        visual_kl = (
            self.visual_adapter.kl_divergence()
            if self.use_visual_adapter
            else torch.tensor(0.0, device=self.device)
        )

        info = {
            "text_mu": text_mu,
            "text_logvar": text_logvar,
            "visual_kl": visual_kl,
            "part_features": part_features,
            "image_features": image_features,
            "text_features": text_features,
        }

        return logits, info


class SpatialBPLForNewClasses(SpatialBPL):
    """Evaluation wrapper: transfers learned parameters to unseen classes.

    Analogous to NewBayesianPromptLearner in the original notebook, but
    additionally transfers the visual adapter and spatial attention parameters.
    """

    def __init__(
        self,
        classnames: list[str],
        clip_model: nn.Module,
        trained_model: SpatialBPL,
        device: str = "cuda",
    ):
        super().__init__(
            classnames=classnames,
            clip_model=clip_model,
            n_tokens=trained_model.n_tokens,
            n_samples=trained_model.n_samples,
            device=device,
            use_visual_adapter=trained_model.use_visual_adapter,
            use_spatial_attention=trained_model.use_spatial_attention,
        )

        # Transfer text parameters
        self.ctx = nn.Parameter(
            trained_model.ctx.detach().clone(), requires_grad=False
        )
        self.text_mu = nn.Parameter(
            trained_model.text_mu.detach().clone(), requires_grad=False
        )
        self.text_logvar = nn.Parameter(
            trained_model.text_logvar.detach().clone(), requires_grad=False
        )

        # Transfer visual adapter
        if self.use_visual_adapter:
            self.visual_adapter.load_state_dict(
                trained_model.visual_adapter.state_dict()
            )

        # Transfer spatial attention
        if self.use_spatial_attention:
            self.spatial_attention.load_state_dict(
                trained_model.spatial_attention.state_dict()
            )
            self.fusion_logit = nn.Parameter(
                trained_model.fusion_logit.detach().clone(), requires_grad=False
            )

        # Freeze everything
        for p in self.parameters():
            p.requires_grad = False
