# Research Plan: Project Rayuela

> **Source of Truth** — This document defines the research plan, methodology, and learning roadmap for Project Rayuela. It is designed to be read by both **Claude Code** and by the human researcher (Carlos). All phases include teaching scaffolding so that Carlos learns *while building*, not by watching.

> For Claude Code operational guidance (build commands, behavioral rules, architecture summary), see [CLAUDE.md](CLAUDE.md).

---

## Research Objective

**Question**: Does the internal structure of Julio Cortázar's *Rayuela* — as discovered by embedding models and LLMs analyzing the original Spanish text — reveal emergent patterns that illuminate, contradict, or transcend the author's explicit *Tablero de Dirección* (the prescribed "hopscotch" reading order)?

**What we are NOT doing**: We are not trying to confirm that the *Tablero* is clever. We assume nothing about what the latent space will show. Discovery first, interpretation second.

---

## For Carlos

You are not a spectator. At every phase, you will:
- **Modify** parameters and observe what changes.
- **Hypothesize** before running experiments.
- **Interpret** visualizations before reading Claude's explanation.
- **Journal** your observations in `RESEARCH_LOG.md` (we'll create this together).
- **Ask "why"** relentlessly. If Claude explains something and you don't follow, say so.

---

## Hardware & Environment

### DGX Spark (Primary Compute)

- ARM64 architecture — every container and binary must be aarch64-compatible
- 128 GB unified memory (CPU/GPU shared). Use `free -h` to check, not `nvidia-smi`
- Single-node limit: Llama 3.1 70B (FP8). The 405B requires two stacked Sparks via QSFP/CX-7 cable with NCCL v2.28.3

### Stacking Requirement for 405B

If using Llama 3.1 405B, two DGX Sparks must be connected via the QSFP/CX-7 cable. This requires:
- Netplan configuration on both nodes
- SSH key exchange
- NCCL v2.28.3 for inter-GPU communication
- vLLM tensor parallelism across both nodes

### Connection Pattern

```
MacBook Air (display only)
     │
     ├──AI Workbench sync──▶ Cursor (running ON the DGX Spark)
     │                            └── Claude Code (running ON the DGX Spark)
     │                            └── Terminal (running ON the DGX Spark)
     │
     └──browser──▶ JupyterLab (port 8888 on Spark)
     └──browser──▶ vLLM API   (port 8000 on Spark)

Everything executes on the DGX Spark. The MacBook Air is a viewport.
```

If the Spark is not on the same local network, use Tailscale (there is an official DGX Spark playbook for this) or a VPN to reach it.

---

## Container Environment

**Decision**: Custom Dockerfile (not runtime installs). This ensures reproducibility across sessions.

### Dockerfile

```dockerfile
FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace/rayuela

# Core ML/NLP
RUN pip install --no-cache-dir \
    vllm \
    sentence-transformers \
    transformers \
    tokenizers \
    accelerate

# Embeddings & Dimensionality Reduction
RUN pip install --no-cache-dir \
    umap-learn \
    scikit-learn \
    openTSNE

# Topological Data Analysis (Phase 4+)
RUN pip install --no-cache-dir \
    ripser \
    persim \
    giotto-tda

# Visualization
RUN pip install --no-cache-dir \
    plotly \
    matplotlib \
    seaborn \
    jupyter \
    jupyterlab \
    ipywidgets

# Graph analysis
RUN pip install --no-cache-dir \
    networkx \
    scipy

# Data handling
RUN pip install --no-cache-dir \
    pandas \
    tqdm

# NLP utilities for Spanish text
RUN pip install --no-cache-dir \
    spacy

RUN python -m spacy download es_core_news_lg

EXPOSE 8888 8000

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### Build & Run

```bash
# Build (on the DGX Spark)
docker build -t rayuela:latest .

# Run with GPU access, shared memory, and port forwarding
docker run --gpus all --runtime nvidia \
    --shm-size=32g \
    -p 8888:8888 \
    -p 8000:8000 \
    -v $(pwd)/data:/workspace/rayuela/data \
    -v $(pwd)/notebooks:/workspace/rayuela/notebooks \
    -v $(pwd)/outputs:/workspace/rayuela/outputs \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    rayuela:latest
```

---

## Research Methodology — Three Scales of Analysis

We analyze *Rayuela* at three distinct scales. Each captures a different kind of structure. The literary insight comes from comparing what each scale reveals.

### Scale A: Micro-Textural Flow

> **What we're asking**: Does the *surface* of the language — the rhythm, vocabulary, syntax — naturally cluster into regions that correspond to the book's sections?

**Method**: Sliding-window vectorization of the raw Spanish prose.

- Embed overlapping windows of ~512 tokens using a multilingual model.
- Each window becomes a point in high-dimensional space.
- Color the points by chapter/section and see if clusters emerge *without* being told about sections.

**Embedding Model**: `intfloat/multilingual-e5-large-instruct`
- Chosen over NV-Embed-v2 because Cortázar code-switches between Spanish, French, and occasional English. A multilingual model won't create artificial cluster boundaries at language boundaries.

**Window Calibration Note**: Chapters in *Rayuela* range from a single sentence (Ch. 79) to many pages. Strategy:
1. First pass: one embedding per chapter (weighted by length).
2. Second pass: sliding windows within chapters longer than 1,000 tokens to detect *internal* texture shifts.

> 🧭 **CHECKPOINT A1**: Before running the embeddings, Carlos should read 3 chapters from different sections (e.g., Ch. 1, Ch. 36, Ch. 79) and write down in `RESEARCH_LOG.md` what he *expects* the texture difference to be. After seeing the embeddings, compare predictions to reality.

### Scale B: Semantic Profiling ("Narrative DNA")

> **What we're asking**: If we look at *meaning* rather than surface words — themes, emotions, character dynamics — does a different structure emerge?

**Method**: Use the LLM (70B or 405B) to extract a structured **semantic feature vector** for each chapter.

**Critical Design Decision**: We do NOT ask the model for free-text summaries (which would flatten Cortázar's ambiguity). Instead, we define explicit dimensions and ask the model to score each chapter on each axis.

**Proposed Semantic Dimensions** (v1 — 20 dimensions, to be refined through experimentation):

Each chapter gets a 20-dimensional vector. This is our "Narrative DNA."

> 🧭 **CHECKPOINT B1**: Before running the LLM extraction, Carlos manually scores 3 chapters on these 20 dimensions. After the LLM scores them, compare. Where do you disagree? Why? This builds critical intuition about what the model "sees."

### Scale C: Graph-Theoretic Path Analysis

> **What we're asking**: When we trace Cortázar's two reading paths through the embedding space, do they behave differently? Does the *hopscotch* path create a coherent journey that the linear path doesn't (or vice versa)?

**Method**:

1. Embed all 155 chapters (using both Scale A and Scale B embeddings).
2. Define the two paths:
   - **Linear**: 1 → 2 → 3 → ... → 56 (stop).
   - **Hopscotch** (*Tablero*): 73 → 1 → 2 → 116 → 3 → 84 → ... (all 155 chapters).
3. For each path, compute:
   - **Trajectory smoothness**: Average cosine distance between consecutive chapters. A smooth path stays in nearby regions; a jagged one jumps wildly.
   - **Semantic curvature**: How much does the path "turn" at each step? (Second derivative of the trajectory.)
   - **Cluster visitation order**: If chapters form clusters, in what sequence does each path visit them?
   - **Return patterns**: Does the hopscotch path revisit thematic regions? Does it create loops?
4. Visualize both paths as animated 3D trajectories through the UMAP space.

> 🧭 **CHECKPOINT C1**: Before computing trajectory metrics, Carlos traces the first 15 steps of both paths on a simple 2D UMAP plot *by hand* (connecting dots with numbered arrows). What pattern does each path seem to make? Does the hopscotch path feel purposeful or random?

---

## Control Corpus

**Why**: We need to know whether the patterns we find are specific to *Rayuela* or generic features of any Spanish-language novel.

**Control texts** (at least two):
1. **Cortázar's own**: *62: Modelo para armar* (1968) — a novel that also experiments with non-linear structure but has no prescribed reading order.
2. **Boom baseline**: A roughly contemporary linear novel — e.g., García Márquez's *Cien años de soledad* or Fuentes' *La muerte de Artemio Cruz*.

We run Scale A and Scale B on the control texts and compare the resulting latent spaces.

> 🧭 **CHECKPOINT CONTROL**: After embedding both *Rayuela* and one control novel, look at their UMAP projections side by side. Does *Rayuela* show more structure? Less? Different geometry?

---

## Topological Data Analysis (TDA) — A Learning Module

> **Carlos, this section is for you.** TDA is a relatively new tool in data science that looks at the *shape* of data rather than just distances or clusters. You'll learn it by applying it. Claude Code will teach you step by step.

### What is TDA? (The Intuition)

Imagine you pour water into a landscape. As the water rises:
- First it fills small holes (local features).
- Then it connects separate pools (clusters merge).
- Some holes persist for a long time before being filled — those are "real" topological features.
- Some appear and disappear quickly — those are noise.

The record of what appears and disappears as we sweep through scales is called a **persistence diagram**. Features that persist are structurally important.

### Why TDA for Rayuela?

UMAP gives you a 2D/3D *picture* of high-dimensional data, but it **destroys topology**. It cannot tell you:
- Whether the data forms a **loop** (some chapters might form a thematic cycle).
- Whether there's a **void** (a region in theme-space that Cortázar deliberately avoids).
- Whether two clusters are connected by a **thin bridge** (a transitional chapter) or are truly separate.

TDA detects all of these.

### Key Libraries

- **`ripser`**: Fast persistent homology computation.
- **`persim`**: Persistence diagram visualization and comparison.
- **`giotto-tda`**: Sklearn-compatible TDA pipeline (useful for integration with other ML tools).

### Reference Material

Carlos should skim (not study exhaustively) before Phase 4:
- Carlsson, G. (2009). "Topology and Data." *Bulletin of the AMS*.
- The `giotto-tda` tutorials: https://giotto-ai.github.io/gtda-docs/latest/notebooks/

---

## Phased Workflow

### Phase 1: Environment & Infrastructure

**Goal**: DGX Spark running a reproducible container with all dependencies, accessible from MacBook Air.

**Learning Objective**: Docker containers, SSH tunnels, GPU verification on ARM64.

> 🧭 **CHECKPOINT 1**: Carlos explains in his own words why we need `--shm-size=32g` and what would happen without it.

### Phase 2: Data Parsing

**Goal**: A clean JSON structure of the original Spanish *Rayuela* text.

**Output Schema**:

```json
{
  "metadata": {
    "title": "Rayuela",
    "author": "Julio Cortázar",
    "year": 1963,
    "language": "es",
    "total_chapters": 155
  },
  "reading_paths": {
    "linear": [1, 2, 3, "...", 56],
    "hopscotch": [73, 1, 2, 116, 3, 84, "..."]
  },
  "chapters": [
    {
      "number": 1,
      "section": "Del lado de allá",
      "text": "¿Encontraría a la Maga?...",
      "token_count": 3421,
      "languages_detected": ["es", "fr"],
      "is_expendable": false
    }
  ]
}
```

**Learning Objective**: Data cleaning, JSON schema design, why preprocessing decisions shape everything downstream.

> 🧭 **CHECKPOINT 2**: Carlos picks 5 chapters and manually counts approximate token lengths. Compare with automated counts. Calibrate trust in the pipeline.

### Phase 3: Parallel Vectorization

**Goal**: Two independent embedding sets for all 155 chapters.

#### Track A — Raw Prose Embeddings

Use `intfloat/multilingual-e5-large-instruct` via sentence-transformers.

#### Track B — Semantic Profiles (Narrative DNA)

Use Llama 3.1 70B via vLLM with structured prompts from `prompts/semantic_extraction_v1.txt`.

**Learning Objective**: Embedding models vs. generative models, prompt engineering for structured extraction, what "semantic similarity" actually means in vector space.

> 🧭 **CHECKPOINT 3**: Carlos plots the raw 20D semantic vectors for 10 chapters as a heatmap. Before Claude explains it — what patterns does Carlos see? Are chapters from the same section "warm" in similar columns?

### Phase 4: Unbiased Dimensionality Reduction & TDA

**Goal**: Reduce to 2D/3D for visualization; detect topological features.

**UMAP Hyperparameters to Explore**:
- `n_neighbors`: 5, 15, 30 (local vs. global structure)
- `min_dist`: 0.0, 0.1, 0.5 (tight clusters vs. spread)
- `metric`: cosine (standard for embeddings)

**Learning Objective**: What dimensionality reduction preserves and destroys, UMAP vs. t-SNE trade-offs, persistence diagrams, the concept of "topological features" in data.

> 🧭 **CHECKPOINT 4**: Carlos runs UMAP with three different `n_neighbors` values and describes (in `RESEARCH_LOG.md`) how the visualization changes. What does a small vs. large neighborhood capture?

### Phase 5: Path Analysis & Trajectory Visualization

**Goal**: Trace both reading paths through the embedding spaces and quantify their behavior.

**Learning Objective**: Graph traversals, trajectory analysis, null models (random path baselines), what it means for a literary reading order to be "smooth" in semantic space.

> 🧭 **CHECKPOINT 5**: Before computing statistics — Carlos writes a 1-paragraph hypothesis in `RESEARCH_LOG.md`: "I predict the hopscotch path will be [smoother/rougher/equally jagged] because..."

### Phase 6: Synthesis & Discovery

**Goal**: Compile findings into a coherent narrative. What did we discover?

> 🧭 **CHECKPOINT 6**: Carlos writes the "Limitations" section himself, then Claude Code reviews and adds anything missed.

---

## Open Questions (To Resolve During Research)

These are deliberately left open — answering them *is* the research:

1. Will the *Capítulos prescindibles* (57–155) cluster separately in Scale A (texture), and if so, is the boundary sharp or gradual?
2. Does the hopscotch path create a smoother trajectory through *semantic* space (Scale B) even if it's jagged in *textural* space (Scale A)? That would suggest Cortázar designed the reading order for thematic coherence, not stylistic flow.
3. Are there topological loops in the data? If so, which chapters form them, and can we interpret the loop as a thematic cycle?
4. Do the Morelli chapters (literary theory) function as "bridges" between narrative clusters, or do they form an isolated archipelago?
5. Is there a "Chapter 68" anomaly? (Ch. 68 is written in Glíglico, an invented language — it should be a massive outlier in Scale A but potentially well-integrated in Scale B.)

---

## Learning Objectives Summary

By the end of this project, Carlos will be able to:

- [ ] Explain what an embedding is and why cosine similarity measures semantic relatedness
- [ ] Build and run a Docker container on ARM64 hardware with GPU access
- [ ] Design structured prompts for feature extraction from an LLM
- [ ] Interpret UMAP projections and explain what hyperparameters control
- [ ] Read a persistence diagram and identify significant topological features
- [ ] Compute and interpret trajectory metrics through high-dimensional space
- [ ] Critically evaluate whether computational findings about a literary text are meaningful or artifacts of methodology
- [ ] Maintain a reproducible research pipeline (versioned data, logged experiments, pinned dependencies)

---

*Last updated: 2026-02-28*
*Environment: DGX Spark (GB10), Cursor + Claude Code running locally on Spark, MacBook Air as thin client*
*Status: Phase 1 — Environment Setup*
