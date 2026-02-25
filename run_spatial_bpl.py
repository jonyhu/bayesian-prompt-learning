"""
Standalone training script for Spatial-BPL on FGVC-Aircraft.

Usage:
    uv run python run_spatial_bpl.py [--epochs 15] [--shots 16] [--n_samples 10]
                                     [--batch_size 4] [--no_visual_adapter]
                                     [--no_spatial_attention] [--seed 2]

Results are printed to stdout and saved to results.json.
"""

import argparse
import json
import math
import os
import random
import time

import clip
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from spatial_bpl import SpatialBPL, SpatialBPLTrainer
from spatial_bpl.trainer import split_base_new, make_fewshot_subset


def parse_args():
    p = argparse.ArgumentParser(description="Spatial-BPL on FGVC-Aircraft")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--shots", type=int, default=16)
    p.add_argument("--n_tokens", type=int, default=4)
    p.add_argument("--n_samples", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--kl_weight", type=float, default=0.001)
    p.add_argument("--part_weight", type=float, default=0.1)
    p.add_argument("--num_parts", type=int, default=6)
    p.add_argument("--adapter_alpha", type=float, default=0.2)
    p.add_argument("--adapter_reduction", type=int, default=4)
    p.add_argument("--no_visual_adapter", action="store_true")
    p.add_argument("--no_spatial_attention", action="store_true")
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--output", type=str, default="results.json")
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    # Load CLIP
    print("\nLoading CLIP ViT-B/16...")
    clip_model, preprocess = clip.load("ViT-B/16", device=device)
    for p in clip_model.parameters():
        p.requires_grad_(False)

    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            224, scale=(0.08, 1.0),
            interpolation=transforms.InterpolationMode.BICUBIC,
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        ),
    ])

    # Load FGVC-Aircraft
    cache_dir = os.path.expanduser("~/.cache")
    print("Loading FGVC-Aircraft dataset...")
    train_dataset = datasets.FGVCAircraft(
        root=cache_dir, download=True, split="train", transform=train_transform)
    val_dataset = datasets.FGVCAircraft(
        root=cache_dir, download=True, split="val", transform=preprocess)
    test_dataset = datasets.FGVCAircraft(
        root=cache_dir, download=True, split="test", transform=preprocess)

    print(f"  Train: {len(train_dataset)} images, {len(train_dataset.classes)} classes")

    # Few-shot + base/new split
    fewshot_train = make_fewshot_subset(train_dataset, args.shots, seed=args.seed)
    m = len(fewshot_train) // 2
    base_train = Subset(fewshot_train.dataset, fewshot_train.indices[:m])
    base_val, _ = split_base_new(val_dataset)
    base_test, new_test = split_base_new(test_dataset)

    base_classes = train_dataset.classes[: len(train_dataset.classes) // 2]
    new_classes = train_dataset.classes[len(train_dataset.classes) // 2 :]

    print(f"  Base classes: {len(base_classes)} | New classes: {len(new_classes)}")
    print(f"  Few-shot train: {len(base_train)} images ({args.shots} shots/class)")

    # DataLoaders
    kw = dict(num_workers=args.workers, drop_last=False, pin_memory=(device == "cuda"))
    train_loader = DataLoader(base_train, batch_size=args.batch_size, shuffle=True, **kw)
    val_loader = DataLoader(base_val, batch_size=32, shuffle=False, **kw)
    test_loader = DataLoader(base_test, batch_size=32, shuffle=False, **kw)
    new_test_loader = DataLoader(new_test, batch_size=32, shuffle=False, **kw)

    # Build model
    use_va = not args.no_visual_adapter
    use_sa = not args.no_spatial_attention
    config_name = []
    if use_va:
        config_name.append("visual_adapter")
    if use_sa:
        config_name.append("spatial_attention")
    if not config_name:
        config_name.append("bpl_baseline")
    config_name = "+".join(config_name)

    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"  visual_adapter={use_va}, spatial_attention={use_sa}")
    print(f"  epochs={args.epochs}, lr={args.lr}, n_samples={args.n_samples}")
    print(f"  kl_weight={args.kl_weight}, part_weight={args.part_weight}")
    print(f"{'='*60}\n")

    model = SpatialBPL(
        classnames=base_classes,
        clip_model=clip_model,
        n_tokens=args.n_tokens,
        n_samples=args.n_samples,
        device=device,
        adapter_reduction=args.adapter_reduction,
        adapter_alpha=args.adapter_alpha,
        num_parts=args.num_parts,
        use_visual_adapter=use_va,
        use_spatial_attention=use_sa,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}")

    # Train
    trainer = SpatialBPLTrainer(
        model=model,
        clip_model=clip_model,
        device=device,
        lr=args.lr,
        num_epochs=args.epochs,
        kl_weight=args.kl_weight,
        part_weight=args.part_weight,
    )

    t0 = time.time()
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        verbose=True,
    )
    train_time = time.time() - t0
    print(f"\nTraining time: {train_time:.1f}s")

    # Evaluate base-to-new
    base_acc = trainer.evaluate(test_loader, label_offset=0)
    results = trainer.evaluate_base_to_new(
        new_classes=new_classes,
        new_test_loader=new_test_loader,
        base_test_loader=test_loader,
        label_offset=len(base_classes),
    )

    hm = 2 * base_acc * results["new_acc"] / (base_acc + results["new_acc"] + 1e-8)

    print(f"\n{'='*60}")
    print(f"RESULTS ({config_name})")
    print(f"{'='*60}")
    print(f"  Base accuracy (seen):     {base_acc:.4f}")
    print(f"  New accuracy (unseen):    {results['new_acc']:.4f}")
    print(f"  Harmonic mean:            {hm:.4f}")
    print(f"  Training time:            {train_time:.1f}s")

    # Save results
    output = {
        "config": config_name,
        "args": vars(args),
        "base_acc": base_acc,
        "new_acc": results["new_acc"],
        "harmonic_mean": hm,
        "train_time_s": train_time,
        "trainable_params": total_params,
        "history": {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
