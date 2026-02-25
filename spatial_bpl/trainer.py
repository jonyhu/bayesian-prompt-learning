"""
Training and evaluation pipeline for Spatial Bayesian Prompt Learning.

Handles:
  - Training loop with all three loss components
  - Evaluation with MC-averaged predictions
  - Base-to-new class generalization protocol
  - Logging and checkpointing
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset

from spatial_bpl.spatial_bpl_model import SpatialBPL, SpatialBPLForNewClasses
from spatial_bpl.losses import SpatialBPLLoss


class SpatialBPLTrainer:
    """Trainer for Spatial Bayesian Prompt Learning.

    Args:
        model: SpatialBPL model instance
        clip_model: frozen CLIP model
        device: torch device
        lr: learning rate
        num_epochs: number of training epochs
        kl_weight: weight for KL divergence terms
        part_weight: weight for part-aware contrastive loss
        contrastive_margin: margin for triplet loss
        contrastive_temp: temperature for contrastive similarity
    """

    def __init__(
        self,
        model: SpatialBPL,
        clip_model: nn.Module,
        device: str = "cuda",
        lr: float = 0.002,
        num_epochs: int = 10,
        kl_weight: float = 0.001,
        part_weight: float = 0.1,
        contrastive_margin: float = 0.3,
        contrastive_temp: float = 0.07,
    ):
        self.model = model
        self.clip_model = clip_model
        self.device = device
        self.num_epochs = num_epochs

        # Loss function
        self.criterion = SpatialBPLLoss(
            kl_weight=kl_weight,
            part_weight=part_weight,
            contrastive_margin=contrastive_margin,
            contrastive_temp=contrastive_temp,
        )

        # Optimizer — only optimize the model's new parameters
        # CLIP stays frozen
        self.optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

    def train_epoch(self, train_loader: DataLoader) -> dict[str, float]:
        """Run one training epoch.

        Returns:
            metrics dict with running averages of all loss components
        """
        self.model.train()

        running = {
            "total": 0.0,
            "nll": 0.0,
            "text_kl": 0.0,
            "visual_kl": 0.0,
            "part_contrastive": 0.0,
        }
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            logits, info = self.model(images)

            # Compute combined loss
            loss_dict = self.criterion(
                logits=logits,
                labels=labels,
                text_mu=info["text_mu"],
                text_logvar=info["text_logvar"],
                visual_kl=info["visual_kl"],
                part_features=info["part_features"],
                n_samples=self.model.n_samples,
            )

            # Backward
            self.optimizer.zero_grad()
            loss_dict["total"].backward()
            self.optimizer.step()

            # Accumulate metrics
            for key in running:
                running[key] += loss_dict[key].item()
            n_batches += 1

        self.scheduler.step()

        # Average metrics
        return {k: v / max(n_batches, 1) for k, v in running.items()}

    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        label_offset: int = 0,
    ) -> float:
        """Evaluate with MC-averaged predictions.

        Args:
            dataloader: evaluation DataLoader
            label_offset: offset for label remapping (for base/new split)

        Returns:
            accuracy: float
        """
        self.model.eval()
        correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            logits, _ = self.model(images)
            # logits: [L, B, C]

            # MC-averaged prediction
            probs = torch.softmax(logits, dim=-1)  # [L, B, C]
            mean_probs = probs.mean(dim=0)  # [B, C]

            preds = mean_probs.argmax(dim=-1)

            labels_local = labels - label_offset
            correct += (preds == labels_local).sum().item()
            total += labels.size(0)

        return correct / total if total > 0 else 0.0

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        test_loader: DataLoader = None,
        verbose: bool = True,
    ) -> dict:
        """Full training loop.

        Returns:
            history: dict with lists of per-epoch metrics
        """
        history = {
            "train_loss": [],
            "train_nll": [],
            "train_text_kl": [],
            "train_visual_kl": [],
            "train_part_loss": [],
            "val_acc": [],
            "test_acc": [],
        }

        for epoch in range(self.num_epochs):
            # Train
            metrics = self.train_epoch(train_loader)
            history["train_loss"].append(metrics["total"])
            history["train_nll"].append(metrics["nll"])
            history["train_text_kl"].append(metrics["text_kl"])
            history["train_visual_kl"].append(metrics["visual_kl"])
            history["train_part_loss"].append(metrics["part_contrastive"])

            if verbose:
                print(
                    f"Epoch {epoch+1}/{self.num_epochs}: "
                    f"loss={metrics['total']:.4f} "
                    f"nll={metrics['nll']:.4f} "
                    f"text_kl={metrics['text_kl']:.4f} "
                    f"vis_kl={metrics['visual_kl']:.4f} "
                    f"part={metrics['part_contrastive']:.4f}"
                )

            # Validate
            if val_loader is not None and (epoch + 1) % 5 == 0:
                val_acc = self.evaluate(val_loader)
                history["val_acc"].append(val_acc)
                if verbose:
                    print(f"  Val accuracy: {val_acc:.4f}")

            if test_loader is not None and epoch == self.num_epochs - 1:
                test_acc = self.evaluate(test_loader)
                history["test_acc"].append(test_acc)
                if verbose:
                    print(f"  Test accuracy: {test_acc:.4f}")

        return history

    def evaluate_base_to_new(
        self,
        new_classes: list[str],
        new_test_loader: DataLoader,
        base_test_loader: DataLoader = None,
        label_offset: int = 0,
    ) -> dict[str, float]:
        """Evaluate generalization from base to new (unseen) classes.

        Creates a SpatialBPLForNewClasses instance and evaluates it.

        Args:
            new_classes: list of new class names
            new_test_loader: DataLoader for new class test set
            base_test_loader: optional DataLoader for base class test set
            label_offset: label offset for new classes

        Returns:
            dict with 'new_acc' and optionally 'base_acc'
        """
        # Create new-class model with transferred parameters
        new_model = SpatialBPLForNewClasses(
            classnames=new_classes,
            clip_model=self.clip_model,
            trained_model=self.model,
            device=self.device,
        ).to(self.device)

        # Temporarily swap model for evaluation
        original_model = self.model
        self.model = new_model

        results = {}
        results["new_acc"] = self.evaluate(
            new_test_loader, label_offset=label_offset
        )

        if base_test_loader is not None:
            self.model = original_model
            results["base_acc"] = self.evaluate(base_test_loader)

        self.model = original_model
        return results


def split_base_new(dataset):
    """Split dataset into base and new class subsets.

    Identical to the notebook's split_base_new function.
    """
    labels = list(dataset._labels)
    all_classes = sorted(set(labels))
    n_classes = len(all_classes)

    m = math.ceil(n_classes / 2)
    base_classes = all_classes[:m]
    new_classes = all_classes[m:]

    base_set = set(base_classes)
    new_set = set(new_classes)

    base_indices = [i for i, y in enumerate(labels) if y in base_set]
    new_indices = [i for i, y in enumerate(labels) if y in new_set]

    base_subset = Subset(dataset, base_indices)
    new_subset = Subset(dataset, new_indices)

    return base_subset, new_subset


def make_fewshot_subset(dataset, shots_per_class, seed=0):
    """Create a few-shot subset with K shots per class.

    Identical to the notebook's make_fewshot_subset function.
    """
    labels = dataset._labels
    n_classes = len(dataset.classes)
    g = torch.Generator().manual_seed(seed)

    subset_indices = []
    for c in range(n_classes):
        cls_indices = [i for i, y in enumerate(labels) if y == c]
        cls_indices = torch.tensor(cls_indices)
        shuffle = torch.randperm(len(cls_indices), generator=g)
        selected = cls_indices[shuffle[:shots_per_class]]
        subset_indices.extend(selected.tolist())

    return Subset(dataset, subset_indices)
