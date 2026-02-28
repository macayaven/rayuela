# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Rayuela**: Computational literary analysis of Julio Cortázar's *Rayuela* (1963). We use embedding models, LLMs, and topological data analysis to discover latent structure in the novel's 155 chapters, then compare those structures against the author's prescribed "hopscotch" reading order (*Tablero de Dirección*).

**Current Status**: Phase 1 — Environment Setup. No code has been written yet.

## Behavioral Rules (MANDATORY)

These rules override default Claude Code behavior for this project:

1. **Teaching-First**: Never silently produce code. Before writing any implementation, explain the concept in 2–4 plain sentences with analogies. Ask Carlos if the intuition clicks before proceeding.
2. **Pair Programming**: Write code in small, commented chunks (15–30 lines). After each chunk, pause and ask Carlos to predict output or modify a parameter.
3. **Socratic Checkpoints**: At every `🧭 CHECKPOINT` in the research plan, stop and ask a question that tests *understanding*, not recall.
4. **Error as Pedagogy**: When something breaks, explain *why* before fixing it.
5. **Vocabulary Building**: Bold new technical terms with a one-line definition. Maintain the running glossary in `PROJECT_GLOSSARY.md`.
6. **Progressive Complexity**: Start each phase with the simplest working version, then layer sophistication.

## Hardware Constraints (DGX Spark)

- **ARM64 only**: All containers and binaries must be aarch64-compatible. No x86 images.
- **Unified Memory Architecture (UMA)**: 128 GB shared between CPU and GPU. `nvidia-smi` may report "Not Supported" for memory — this is normal. Use `free -h` instead.
- **Single-node model limit**: Llama 3.1 70B (FP8). The 405B model requires two stacked Sparks.

## Build & Run Commands

```bash
# Build the container
docker build -t rayuela:latest .

# Run with GPU access
docker run --gpus all --runtime nvidia \
    --shm-size=32g \
    -p 8888:8888 \
    -p 8000:8000 \
    -v $(pwd)/data:/workspace/rayuela/data \
    -v $(pwd)/notebooks:/workspace/rayuela/notebooks \
    -v $(pwd)/outputs:/workspace/rayuela/outputs \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    rayuela:latest

# Check memory (NOT nvidia-smi)
free -h
```

- **JupyterLab**: port 8888
- **vLLM API**: port 8000
- **`--shm-size=32g`** is required because PyTorch DataLoader uses shared memory for multiprocessing; without it, large batch operations crash with "bus error."

## Architecture

### Three Scales of Analysis

The project analyzes the novel at three independent scales, then compares findings:

- **Scale A (Micro-Textural)**: Sliding-window embeddings of raw Spanish prose using `intfloat/multilingual-e5-large-instruct`. Captures surface-level rhythm, vocabulary, syntax. Multilingual model chosen because Cortázar code-switches between Spanish, French, and English.
- **Scale B (Semantic/Narrative DNA)**: LLM-extracted 20-dimensional structured feature vectors per chapter (themes, emotions, character dynamics). Uses Llama 3.1 70B via vLLM. Critical: we do NOT use free-text summaries — we score explicit dimensions to preserve ambiguity.
- **Scale C (Graph-Theoretic)**: Trajectory analysis of the two reading paths (linear: Ch. 1–56; hopscotch: all 155 chapters per *Tablero*) through embedding space. Metrics: trajectory smoothness, semantic curvature, cluster visitation order, return patterns.

### Planned File Structure

```
src/
  parsing.py              # Text extraction and JSON structuring
  embeddings.py           # Scale A: prose embedding pipeline
  semantic_extraction.py  # Scale B: LLM feature vector extraction
  trajectory.py           # Scale C: path analysis and metrics
  tda_utils.py            # Topological data analysis (Phase 4+)
  visualization.py        # Plotly/matplotlib visualization helpers
notebooks/
  01–07                   # Phased Jupyter notebooks (one per research phase)
data/
  rayuela_raw.json        # Parsed novel (155 chapters, JSON schema in research plan)
  control_*.json          # Control corpus for comparison
outputs/
  embeddings/             # Saved .npy arrays
  figures/
  persistence_diagrams/
prompts/
  semantic_extraction_v1.txt  # Versioned LLM prompts for Scale B
```

### Key Data Schema

Chapter JSON includes: `number`, `section` ("Del lado de allá" / "Del lado de acá" / "De otros lados"), `text`, `token_count`, `languages_detected`, `is_expendable` (chapters 57–155 are the "expendable chapters").

### Control Corpus

Results are validated against: *62: Modelo para armar* (Cortázar, non-linear but no prescribed order) and a linear Boom novel (e.g., *Cien años de soledad*).

## Key Dependencies

Base image: `nvcr.io/nvidia/pytorch:24.01-py3` (NGC, ARM64-compatible)

Core: vllm, sentence-transformers, transformers, accelerate
Embeddings/DR: umap-learn, scikit-learn, openTSNE
TDA: ripser, persim, giotto-tda
Visualization: plotly, matplotlib, seaborn, ipywidgets
Graph/NLP: networkx, scipy, spacy (with `es_core_news_lg` model)

## Research Phasing

1. Environment & Infrastructure (current)
2. Data Parsing → `rayuela_raw.json`
3. Parallel Vectorization (Scale A + Scale B)
4. Dimensionality Reduction (UMAP/t-SNE) & TDA
5. Path Analysis & Trajectory Visualization
6. Synthesis & Discovery

Each phase has checkpoints (`🧭`) that require stopping for Socratic dialogue with Carlos.

## Detailed Research Plan

Full methodology, checkpoints, TDA learning module, open research questions, and learning objectives are in [RESEARCH_PLAN.md](RESEARCH_PLAN.md). Read it on-demand when entering a new phase — it does not need to be loaded every session.
