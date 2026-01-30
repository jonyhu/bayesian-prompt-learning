# Bayesian Prompt Learning for Image–Language Model Generalization (BPL) — Course Project

This repo accompanies my course project report on Bayesian Prompt Learning (BPL), a variational formulation of prompt learning for CLIP that learns a distribution over prompt residuals and averages predictions over samples at inference.

- [Slides](https://docs.google.com/presentation/d/1kT5nuKomYVeE4T044rn1tNPeHtqSSXZqMH2W3dD0vug/edit?usp=sharing) — overview
- [Colab](https://colab.research.google.com/drive/1nrrBTpGJnvLrMQlV8w7rJulOUggLuiG0?usp=sharing) — reproduce
- [Report](./report/report.pdf) — full write-up (PDF)

---

## Summary

CLIP-style vision–language models enable zero-shot image classification by comparing image embeddings to text embeddings generated from class prompts. In practice, downstream adaptation is often needed in the few-shot regime, where labels are scarce and full fine-tuning is undesirable. Prompt learning addresses this by optimizing a small set of continuous context tokens while keeping CLIP’s encoders frozen. However, deterministic prompt optimization (e.g., CoOp/CoCoOp) can overfit, hurting robustness under distribution shift and on unseen classes.

Bayesian Prompt Learning (BPL) mitigates this by learning a variational distribution over prompt residuals and averaging predictions over samples at inference time, with a KL term that regularizes the prompt space.

---

## What’s in the report (high level)

- A compact review that ties together how CLIP enables zero-shot classification, how CoOp/CoCoOp adapt CLIP via learned context tokens, and why BPL reframes prompt learning as variational inference with KL regularization.
- Unconditional BPL reproduced on FGVC-Aircraft under the unseen prompt generalization setting, along with targeted ablations that diagnose:
  - baseline overtraining (epoch sensitivity for CoOp)
  - the effect of Monte Carlo sampling in unconditional BPL, especially in the low-sample regime
- The experimental pipeline is implemented in a single, readable Google Colab notebook to make the end-to-end flow easy to follow. Most components are written from scratch (zero-shot CLIP, CoOp, dataset split, training/evaluation), while the Bayesian prompt learner components (KL term and sampling) are reused/adapted from the original BPL repository.

---

## Reproduce

This project is meant to be reproduced from Colab.

1. Open the notebook: https://colab.research.google.com/drive/1nrrBTpGJnvLrMQlV8w7rJulOUggLuiG0?usp=sharing  
2. Runtime → Run all  
3. The notebook downloads data, runs training/evaluation, and prints results.

---

## Results

Below are a few headline tables; the slides and report include the full context and additional ablations.

<details>
<summary>CoOp in-domain accuracy (train/test on full class set)</summary>

| Context length | CoOp (paper) | Ours |
|---:|---:|---:|
| 4  | 40.27% | 39.67% |
| 8  | 42.57% | 41.90% |
| 16 | 43.53% | 43.78% |

</details>

<details>
<summary>Unconditional BPL: context length on unseen prompt generalization (10 MC samples)</summary>

| Context length | MC samples | BPL (paper) | Ours |
|---:|---:|---:|---:|
| 4  | 10 | 34.17% | 34.35% |
| 8  | 10 | 32.53% | 31.61% |
| 16 | 10 | 31.10% | 30.39% |

</details>

<details>
<summary>Task I: unseen prompt generalization (method comparison)</summary>

| Method | Context length | Original | Ours |
|---|---:|---:|---:|
| Zero-shot CLIP | — | — | 23.13% |
| CoOp | 4 | 22.30% | 25.25% |
| Unconditional BPL | 4 | 33.80% | 34.45% |

</details>

Takeaway: In this setup, unconditional BPL improves over CoOp and zero-shot CLIP on the unseen-class split, and shorter context generalizes better than longer context.

---

## Repo layout

- README.md
- report/
- slides/
- notebooks/

---

## References

- CLIP: Radford et al.
- CoOp / CoCoOp: Zhou et al.
- Bayesian Prompt Learning: Derakhshani et al.
- Codebase used/adapted: https://github.com/saic-fi/Bayesian-Prompt-Learning

---

Course: ECSE 626  
Author: Jonathan Hu  
Term: Fall 2025
