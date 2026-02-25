"""
Bayesian Visual Adapter — learns a distribution over image feature residuals.

Standard BPL only learns residuals on the text prompt side:
    text_features = text_encoder(ctx + r),  r ~ N(mu, sigma^2)

This module adds the image-side counterpart:
    adapted_image = alpha * adapter(image_features) + (1-alpha) * image_features
    where the adapter's weights are Bayesian (we sample from a posterior).

Key insight: For fine-grained classification like FGVC Aircraft, the semantic
gap on the visual side is larger than the text side — different aircraft look
almost identical at the global feature level. Learning to perturb image features
in a Bayesian way allows the model to explore different visual "views" of the
same image, emphasizing different discriminative details.

Architecture:
    image_features [B, D]
        -> Linear(D, D//reduction) + ReLU  (down-project)
        -> Linear(D//reduction, D)         (up-project)
        -> residual blend with alpha

    The up-projection layer has Bayesian parameters:
        W ~ N(mu_W, sigma_W^2),  b ~ N(mu_b, sigma_b^2)

    During training, we sample L weight configurations and produce L adapted
    features per image, matching the L text feature samples from BPL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianLinear(nn.Module):
    """A linear layer with Bayesian (variational) weights.

    Instead of fixed weights W, b, we learn:
        W ~ N(mu_W, exp(logvar_W))
        b ~ N(mu_b, exp(logvar_b))

    The reparameterization trick enables gradient-based learning.
    """

    def __init__(self, in_features: int, out_features: int, dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Posterior mean — initialized like a standard linear layer
        self.mu_weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype)
        )
        self.mu_bias = nn.Parameter(torch.empty(out_features, dtype=dtype))

        # Posterior log-variance — initialized to small values (tight prior)
        self.logvar_weight = nn.Parameter(
            torch.full((out_features, in_features), -6.0, dtype=dtype)
        )
        self.logvar_bias = nn.Parameter(
            torch.full((out_features,), -6.0, dtype=dtype)
        )

        self._init_parameters()

    def _init_parameters(self):
        # Kaiming init for mean weights
        nn.init.kaiming_uniform_(self.mu_weight, nonlinearity="relu")
        bound = 1.0 / (self.in_features ** 0.5)
        nn.init.uniform_(self.mu_bias, -bound, bound)

    def forward(self, x: torch.Tensor, n_samples: int = 1):
        """
        Args:
            x: input tensor [B, in_features] or [L, B, in_features]
            n_samples: number of weight samples (L)

        Returns:
            output: [L, B, out_features] if x is [B, in_features]
                    [L, B, out_features] if x is [L, B, in_features]
        """
        if self.training:
            # Sample weights using reparameterization trick
            # eps_w: [L, out, in],  eps_b: [L, out]
            eps_w = torch.randn(
                n_samples, self.out_features, self.in_features,
                device=x.device, dtype=x.dtype
            )
            eps_b = torch.randn(
                n_samples, self.out_features,
                device=x.device, dtype=x.dtype
            )

            std_w = (0.5 * self.logvar_weight).exp()  # [out, in]
            std_b = (0.5 * self.logvar_bias).exp()  # [out]

            # Sampled weights: [L, out, in]
            W = self.mu_weight.unsqueeze(0) + eps_w * std_w.unsqueeze(0)
            b = self.mu_bias.unsqueeze(0) + eps_b * std_b.unsqueeze(0)

            # x could be [B, in] or [L, B, in]
            if x.dim() == 2:
                # Expand x to [L, B, in]
                x = x.unsqueeze(0).expand(n_samples, -1, -1)

            # Batched matmul: [L, B, in] @ [L, in, out] -> [L, B, out]
            out = torch.bmm(x, W.transpose(1, 2)) + b.unsqueeze(1)
            return out
        else:
            # At eval, use posterior mean (no sampling)
            if x.dim() == 2:
                out = F.linear(x, self.mu_weight, self.mu_bias)
                return out.unsqueeze(0).expand(n_samples, -1, -1)
            else:
                # x is [L, B, in]
                return torch.bmm(
                    x, self.mu_weight.t().unsqueeze(0).expand(x.size(0), -1, -1)
                ) + self.mu_bias.unsqueeze(0).unsqueeze(1)

    def kl_divergence(self) -> torch.Tensor:
        """Compute KL(q(W)||p(W)) where p(W) = N(0, I).

        Analytical KL for diagonal Gaussians:
            KL = 0.5 * sum(sigma^2 + mu^2 - 1 - log(sigma^2))
        """
        kl_w = 0.5 * torch.sum(
            self.logvar_weight.exp() + self.mu_weight.pow(2)
            - 1.0 - self.logvar_weight
        )
        kl_b = 0.5 * torch.sum(
            self.logvar_bias.exp() + self.mu_bias.pow(2)
            - 1.0 - self.logvar_bias
        )
        return kl_w + kl_b


class BayesianVisualAdapter(nn.Module):
    """Bayesian adapter for CLIP image features.

    Learns a distribution over residual transformations of image features.
    Conceptually dual to BPL's text-side residual: instead of perturbing
    the text embedding space, we perturb the image embedding space.

    The adapter uses a bottleneck architecture:
        image_features -> down_proj -> ReLU -> bayesian_up_proj -> residual_add

    The down-projection is deterministic (dimensionality reduction is not
    where uncertainty helps), while the up-projection is Bayesian (this is
    where the model learns which directions in feature space to explore).

    Args:
        feature_dim: dimension of CLIP image features (e.g., 512 for ViT-B/16
                     after projection, or 768 for raw ViT-B/16 features)
        reduction: bottleneck reduction factor (default 4)
        alpha: residual blending ratio (0 = no adaptation, 1 = full replacement)
        n_samples: number of MC samples at train time
        dtype: parameter dtype (match CLIP's dtype, usually float16)
    """

    def __init__(
        self,
        feature_dim: int,
        reduction: int = 4,
        alpha: float = 0.2,
        n_samples: int = 10,
        dtype=torch.float32,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.bottleneck_dim = feature_dim // reduction
        self.alpha = alpha
        self.n_samples = n_samples

        # Deterministic down-projection
        self.down_proj = nn.Linear(
            feature_dim, self.bottleneck_dim, dtype=dtype
        )

        # Bayesian up-projection — this is where uncertainty lives
        self.up_proj = BayesianLinear(
            self.bottleneck_dim, feature_dim, dtype=dtype
        )

        # Layer norm for stability
        self.ln = nn.LayerNorm(feature_dim, dtype=dtype)

    def forward(
        self, image_features: torch.Tensor, n_samples: int = None
    ) -> torch.Tensor:
        """
        Args:
            image_features: [B, D] normalized CLIP image features
            n_samples: override for number of MC samples

        Returns:
            adapted: [L, B, D] adapted image features (L samples)
        """
        L = n_samples if n_samples is not None else self.n_samples

        # Down-project (deterministic): [B, D] -> [B, bottleneck]
        h = F.relu(self.down_proj(image_features))

        # Up-project (Bayesian): [B, bottleneck] -> [L, B, D]
        residual = self.up_proj(h, n_samples=L)

        # Residual blend: alpha * adapter_out + (1-alpha) * original
        # image_features: [B, D] -> [1, B, D] for broadcasting
        original = image_features.unsqueeze(0).expand(L, -1, -1)
        adapted = self.alpha * self.ln(residual) + (1 - self.alpha) * original

        return adapted

    def kl_divergence(self) -> torch.Tensor:
        """Total KL divergence from all Bayesian layers."""
        return self.up_proj.kl_divergence()
