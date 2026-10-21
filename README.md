# V-STRAND: A Universal Multimodal GraphRAG Framework for Long-Context Video Understanding via Structured Reasoning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch >=2.2](https://img.shields.io/badge/PyTorch-%E2%89%A52.2-ee4c2c.svg)](https://pytorch.org/)
![Status: research prototype](https://img.shields.io/badge/status-research%20prototype-orange.svg)

**V-STRAND** = **V**ideo **S**patio-**T**emporal **R**easoning, **A**lignment & **N**eural-graph **D**eployment

---

## 📌 Abstract

V-STRAND is an open-source framework for deep, long-context video understanding. Standard Large Vision-Language
Models (LVLMs) hit context-window limits, pay heavy prefill cost on long videos, and hallucinate when retrieval
returns *hard negatives* (clips that look relevant but are not).

V-STRAND addresses this by (1) unifying image and video features in a language-grounded space **before** projection,
(2) building a **query-independent spatiotemporal knowledge graph offline** (Vgent-style clip–entity indexing),
(3) applying **post-retrieval structured reasoning** that verifies retrieved clips against binary sub-queries,
(4) extracting neural scene-graph triples with ontology-regularised predicate heads, (5) grounding egocentric video
in 3D point-cloud context, and (6) compressing intermediate features to 1-bit codes for edge→cloud offloading.

> **Project status — please read.** This repository is a *runnable prototype*. The complete pipeline executes
> offline on CPU using a **mock LVLM and synthetic data**. Real-model integration and benchmark evaluation are in
> progress (see [Implementation Status](#-implementation-status) and [Evaluation Status](#-evaluation-status)).

---

## 🏗️ System Architecture

```
                         +---------------------------------------+
                         |           RAW VIDEO (+ audio)         |
                         +-------------------+-------------------+
                                             |
          +----------------------------------+----------------------------------+
          v                                  v                                  v
   [Visual stream]                    [Audio stream]                  [Spatial / 3D stream]
   Clip splitting (Decord)            ASR transcripts (Whisper)       Point cloud + xyz encoding
          |                                  |                                  |
   LanguageBind (pre-aligned)                |                       Spatio-temporal aligner
          |                                  |                       (cross-attention)
   Shared projector (MLP)                    |                                  |
          |                                  |                                  |
          +----------------------------------+----------------------------------+
                                             v
                         +---------------------------------------+
                         |   LVLM: per-clip entities + captions  |
                         +-------------------+-------------------+
                                             v
                         +---------------------------------------+
                         |  OFFLINE VIDEO GRAPH (query-agnostic) |
                         |  clip nodes + entity prototype nodes  |
                         |  cosine-similarity entity merging     |
                         +-------------------+-------------------+
              user query --> top-N retrieval | (graph hops)
                                             v
                         +---------------------------------------+
                         |   POST-RETRIEVAL STRUCTURED REASONING |
                         |   decompose -> verify -> filter       |
                         +-------------------+-------------------+
                                             | verified clips
                                             v
                         +---------------------------------------+
                         |        LVLM ANSWER GENERATION         |
                         +---------------------------------------+

   Edge <-> Cloud:  edge 3D-conv encoder -> 1-bit code -> cloud decoder (ApproxBit-style)
   Scene graph:     entity vectors + video encoding -> predicate MLPs -> <subject, predicate, object>
```

---

## ✨ Key Technical Pillars

1. **Alignment-Before-Projection Perception** — image and video tokens come from a pre-aligned encoder
   (LanguageBind) and pass through one **shared** projector into the LLM space (`perception/projector.py`).
2. **Offline GraphRAG Memory (Vgent-style)** — long videos are split into clips; entities from visual descriptions
   and transcripts are merged into global prototype nodes by cosine similarity (threshold in
   `configs/default.yaml`, default `0.80`); clips sharing entities are linked (`graph/`).
3. **Post-Retrieval Structured Reasoning (SR)** — a query is decomposed into binary sub-queries; each top-N clip is
   verified against them and non-qualifying clips (e.g. *closed* laptop for "Did I open the laptop?") are removed
   (`reasoning/structured.py`).
4. **Neural Scene-Graph Extraction** — learnable entity vectors + video encoding → predicate MLPs predict unary
   attributes and binary relations; background rules are enforced as a differentiable penalty, e.g.
   `fold(x,y) ⇒ change(x,y)` (`kg/`).
5. **3D Spatio-Temporal Grounding** — egocentric video tokens cross-attend to allocentric point-cloud tokens with
   xyz positional encoding (`spatial/aligner.py`).
6. **Asymmetric Edge–Cloud Offloading (ApproxBit-style)** — small 3D-conv encoder on the edge, sigmoid + 1-bit
   straight-through threshold, larger decoder in the cloud (`edge/approxbit.py`).

---

## 🧮 Mathematical Foundations

**Implemented in this repo**

*Entity prototype merging (cosine similarity, threshold τ):*
a new entity embedding **v** joins prototype *p* if `cos(v, c_p) ≥ τ`, updating the centroid by a running mean;
otherwise it creates a new prototype.

*Ontology (rule) loss — Łukasiewicz-style implication:*
$$\mathcal{L}_{\text{onto}} = \mathbb{E}_{x,y}\big[\max\big(0,\; P(\text{fold}(x,y)) - P(\text{change}(x,y))\big)\big], \qquad
\mathcal{L}_{\text{KG}} = \mathcal{L}_{\text{BCE}} + \lambda\,\mathcal{L}_{\text{onto}}$$

*1-bit feature coding with straight-through estimator:*
$$\mathbf{c} = \mathbb{1}\big[\sigma(\text{Conv3D}(\mathbf{F})) > 0.5\big], \qquad \frac{\partial \mathbf{c}}{\partial \mathbf{z}} \approx 1$$
$$\mathcal{L}_{\text{codec}} = \lVert \text{Dec}(\mathbf{c}) - \mathbf{F} \rVert_2^2$$

**Planned (not yet implemented)** — included as the intended training objectives for Stage 1–2:

*Angular-margin contrastive loss (MAMA-style), for noisy automatic captions:*
$$L_{v \to t} = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp((s(I_i, T_i) - \mu)/\tau)}{\exp((s(I_i, T_i) - \mu)/\tau) + \sum_{j \neq i}^N \exp(s(I_i, T_j)/\tau)}$$

*Partial optimal-transport alignment between frame and text distributions:*
$$\mathcal{D}_{\text{POT}}(\boldsymbol{\mu}, \boldsymbol{\nu}) = \min_{\mathbf{T} \in \Pi_p(\mathbf{a}, \mathbf{b})} \sum_{i,j} \mathbf{T}_{i,j} \mathbf{C}_{i,j}, \quad
\mathbf{T}\mathbf{1} \le \mathbf{a},\; \mathbf{T}^\top\mathbf{1} \le \mathbf{b},\; \mathbf{1}^\top \mathbf{T} \mathbf{1} = m$$

*Attention-based frame salience:*
$$\mathbf{A} = \text{softmax}\!\left(\tfrac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{D_s}}\right), \qquad w_t = \sum_{j=1}^{T} \mathbf{A}_{t,j}$$

---
