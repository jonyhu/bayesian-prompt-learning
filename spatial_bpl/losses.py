"""
Loss functions for Spatial Bayesian Prompt Learning.

Three loss components:

1. BPL NLL Loss — standard BPL negative log-likelihood with MC averaging
   (unchanged from original BPL)

2. KL Divergence — regularization for both text-side and visual-side
   Bayesian parameters

3. Part-Aware Contrastive Loss — encourages discriminative part features
   by contrasting patch-level representations across fine-grained classes.
   For FGVC Aircraft, this pushes the model to find that e.g. the engine
   region of a Boeing 737 looks different from a Boeing 747, even when
   global features are similar.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PartAwareContrastiveLoss(nn.Module):
    """Contrastive loss on part-level features for fine-grained discrimination.

    Intuition: In fine-grained datasets, global features of different classes
    are very similar (all aircraft look like aircraft). The discriminative
    signal is in local parts (engines, windows, tail shape). This loss
    directly encourages the part-based spatial attention to produce features
    that are discriminative across classes.

    For each image in a batch, we:
    1. Compute part-based features from spatial attention
    2. Find the hardest negative (most similar image from a different class)
    3. Push apart the part features while pulling together same-class features

    This is a triplet-style loss with hard negative mining.

    Args:
        margin: triplet margin
        temperature: scaling temperature for similarity computation
    """

    def __init__(self, margin: float = 0.3, temperature: float = 0.07):
        super().__init__()
        self.margin = margin
        self.temperature = temperature

    def forward(
        self,
        part_features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            part_features: [B, D] part-based features from spatial attention
            labels: [B] class labels

        Returns:
            loss: scalar contrastive loss
        """
        B = part_features.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=part_features.device, dtype=part_features.dtype)

        # Normalize features
        feats = F.normalize(part_features, dim=-1)

        # Pairwise similarity matrix: [B, B]
        sim_matrix = feats @ feats.t() / self.temperature

        # Create masks for positive and negative pairs
        labels = labels.view(-1, 1)
        pos_mask = (labels == labels.t()).float()  # [B, B]
        neg_mask = 1.0 - pos_mask

        # Remove self-similarity
        eye = torch.eye(B, device=feats.device, dtype=feats.dtype)
        pos_mask = pos_mask - eye

        # If no positive pairs exist (all different classes), fall back to
        # standard InfoNCE-style loss
        has_positives = pos_mask.sum(dim=1) > 0

        if not has_positives.any():
            # Pure InfoNCE: each sample vs all others
            # Use cross-entropy over similarity matrix
            # This still encourages separation
            logits = sim_matrix - 1e9 * eye
            # Each sample should be dissimilar to all others
            # Use uniform target (all equally unlikely to be same class)
            loss = -torch.log(
                torch.exp(-sim_matrix * neg_mask).sum(dim=1)
                / torch.exp(sim_matrix).sum(dim=1)
            ).mean()
            return loss.clamp(min=0.0)

        # Hard negative mining: for each sample, find the most similar
        # sample from a different class
        neg_sim = sim_matrix * neg_mask - 1e9 * (1.0 - neg_mask)
        hardest_neg_sim = neg_sim.max(dim=1)[0]  # [B]

        # Hard positive mining: for each sample, find the least similar
        # sample from the same class
        pos_sim = sim_matrix * pos_mask + 1e9 * (1.0 - pos_mask)
        # Only consider samples that have positive pairs
        hardest_pos_sim = pos_sim.min(dim=1)[0]  # [B]

        # Triplet loss: pull positives closer than hardest negative + margin
        losses = F.relu(hardest_neg_sim - hardest_pos_sim + self.margin)

        # Only average over samples that have valid positive pairs
        if has_positives.sum() > 0:
            return losses[has_positives].mean()
        return losses.mean()


class SpatialBPLLoss(nn.Module):
    """Combined loss for Spatial Bayesian Prompt Learning.

    Total loss = NLL + kl_weight * KL_total + part_weight * PartContrastive

    Where:
        NLL: BPL-style negative log-likelihood with MC averaging
        KL_total: KL(text_posterior || prior) + KL(visual_posterior || prior)
        PartContrastive: part-aware contrastive loss for spatial features

    Args:
        kl_weight: weight for KL divergence terms (default 0.001 as in BPL)
        part_weight: weight for part-aware contrastive loss
        contrastive_margin: margin for triplet loss
        contrastive_temp: temperature for contrastive similarity
    """

    def __init__(
        self,
        kl_weight: float = 0.001,
        part_weight: float = 0.1,
        contrastive_margin: float = 0.3,
        contrastive_temp: float = 0.07,
    ):
        super().__init__()
        self.kl_weight = kl_weight
        self.part_weight = part_weight
        self.part_contrastive = PartAwareContrastiveLoss(
            margin=contrastive_margin,
            temperature=contrastive_temp,
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_mu: torch.Tensor,
        text_logvar: torch.Tensor,
        visual_kl: torch.Tensor,
        part_features: torch.Tensor | None = None,
        n_samples: int = 10,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            logits: [L, B, C] logits from L MC samples
            labels: [B] ground truth class indices
            text_mu: [1, D] text posterior mean
            text_logvar: [1, D] text posterior log-variance
            visual_kl: scalar, KL from Bayesian visual adapter
            part_features: [B, D] part features (optional, from spatial attention)
            n_samples: number of MC samples (L)

        Returns:
            dict with keys: 'total', 'nll', 'text_kl', 'visual_kl', 'part_contrastive'
        """
        L, B, C = logits.shape

        # --- 1. NLL Loss (BPL-style MC averaging) ---
        log_probs = torch.log_softmax(logits, dim=-1)  # [L, B, C]

        # Gather log-probs for correct class
        labels_expanded = labels.unsqueeze(0).unsqueeze(-1).expand(L, -1, 1)
        selected_log_probs = torch.gather(
            log_probs, 2, labels_expanded
        ).squeeze(-1)  # [L, B]

        # Log-mean-exp over MC samples
        task_score = torch.logsumexp(selected_log_probs, dim=0) - torch.log(
            torch.tensor(float(L), device=logits.device)
        )
        nll_loss = -task_score.mean()

        # --- 2. Text KL Divergence ---
        text_kl = -0.5 * torch.sum(
            1 + text_logvar - text_mu.pow(2) - text_logvar.exp()
        )
        # Normalize by dimension for stability
        text_kl = text_kl / text_mu.numel()

        # --- 3. Visual KL Divergence ---
        # Already computed by the Bayesian visual adapter
        # Normalize similarly
        visual_kl_normalized = visual_kl / max(
            1, sum(p.numel() for p in [] if True)  # placeholder
        ) if visual_kl.item() > 0 else visual_kl
        # Simpler: just use the raw KL scaled by a factor
        visual_kl_normalized = visual_kl

        # --- 4. Part-Aware Contrastive Loss ---
        if part_features is not None:
            part_loss = self.part_contrastive(part_features, labels)
        else:
            part_loss = torch.tensor(
                0.0, device=logits.device, dtype=logits.dtype
            )

        # --- Total Loss ---
        total_kl = text_kl + visual_kl_normalized
        total = nll_loss + self.kl_weight * total_kl + self.part_weight * part_loss

        return {
            "total": total,
            "nll": nll_loss.detach(),
            "text_kl": text_kl.detach(),
            "visual_kl": visual_kl_normalized.detach(),
            "part_contrastive": part_loss.detach(),
        }
