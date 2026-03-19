# Unified Policy-Value Decomposition for Rapid Adaptation

**Cristiano Capone, Luca Falorsi, Andrea Ciardiello, Luca Manneschi**  
*Istituto Superiore di Sanità, Rome · Ospedale Santa Lucia, Rome · University of Sheffield*

[![arXiv](https://img.shields.io/badge/arXiv-2603.17947-b31b1b.svg)](https://arxiv.org/abs/2603.17947)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

We introduce a **bilinear actor–critic decomposition** in which policy and value functions share a low-dimensional coefficient vector — a *goal embedding* — that captures task identity and enables immediate adaptation to novel tasks without retraining representations.

The critic and actor are jointly factorized as:

```
Q(s, a, g)  =  Σ_k  G_k(g) · ψ_k(s, a)
μ(s, g)     =  Σ_k  G_k(g) · Y_k(s)
```

The **same gating vector** `G(g)` modulates both value components `ψ_k` and policy primitives `Y_k`. At test time, the bases are frozen and `G(g)` is estimated zero-shot via a single forward pass — no gradient update required.

This multiplicative gating is reminiscent of gain modulation in Layer 5 pyramidal neurons, where top-down contextual inputs scale sensory-driven responses without altering their tuning.

---

## Key Features

- **Bilinear co-decomposition** — actor and critic share the same gating layer, aligning policy improvement directions with value estimates
- **Zero-shot adaptation** — novel tasks are handled by re-estimating `G` from context, without any gradient update to the bases
- **Online G-space adaptation** — rapid behavioral adaptation via a simple value-based rule: `ΔG_k ∝ r · ψ_k(s, a)`
- **Interpretable latent space** — individual `G_k` components correspond to semantically meaningful control axes (direction, speed, gait)
- **Biologically plausible** — mirrors multiplicative dendritic integration in cortical pyramidal neurons

---

## Method

### Pretraining

During pretraining, `K` value basis functions `ψ_k(s, a)` and `K` policy primitives `Y_k(s)` are jointly learned under a Soft Actor–Critic (SAC) framework. The shared gating network maps goal vectors `g` to coefficients `G(g) ∈ ℝ^K`.

### Adaptation

At test time, only `G` is updated using a linear TD rule:

```
δ_t  =  r_t + γ Σ_k G_k ψ_k(s', a') - Σ_k G_k ψ_k(s, a)
G    ←  G + α · δ_t · ψ(s, a)
```

The updated `G` is immediately reused in the policy, preserving value–policy consistency.

---

## Experiments

We evaluate on **MuJoCo Ant-v4** under a multi-directional locomotion objective. The agent is trained to walk in 8 directions (4 cardinal + 4 diagonal) specified as continuous 2D goal vectors.

**Reward:**
```
r = Δx cos θ + Δy sin θ - 0.1 |Δx sin θ - Δy cos θ|
```

Key results:
- The bilinear structure allows each policy head to specialize to a subset of directions
- The shared coefficient layer generalizes across them, including novel directions via interpolation in `G`-space
- Online adaptation matches near-optimal behavior within a few environment steps


---

## Citation

If you find this work useful, please cite:

```bibtex
@article{capone2025unified,
  title   = {Unified Policy-Value Decomposition for Rapid Adaptation},
  author  = {Capone, Cristiano and Falorsi, Luca and Ciardiello, Andrea and Manneschi, Luca},
  journal = {arXiv preprint arXiv:2603.17947},
  year    = {2025},
  url     = {https://arxiv.org/abs/2603.17947}
}
```

---

## Acknowledgements

Computational Neuroscience Unit, Istituto Superiore di Sanità, Rome, Italy.
