# Project Rayuela

Project Rayuela is a computational literary analysis of Julio Cortazar's *Rayuela*. The repository contains the full workflow behind the project: text cleanup, embeddings, classical stylometrics, LLM-based style and semantic scoring, permutation tests, interactive visualizations, and article assets for a two-part series.

This workspace is already well past the planning stage. The analysis code, derived datasets, generated figures, article drafts, and GitHub Pages assets are all committed here.

## Current State

- `Rayuela` has been parsed into [`data/rayuela_raw.json`](data/rayuela_raw.json), with chapter boundaries preserved and editorial navigation markers removed.
- The project currently works across four analysis scales: holistic embeddings (A), content-light stylometrics (A'), LLM-perceived style (B'), and narrative DNA semantics (B).
- The validated semantic instrument is currently 19-dimensional. `temporal_clarity` was excluded after inter-rater replication with Nemotron 70B showed a rubric polarity failure.
- The two long-form article sources live at [`ARTICLE_PART1_MEDIUM.md`](ARTICLE_PART1_MEDIUM.md) and [`ARTICLE_PART2_MEDIUM.md`](ARTICLE_PART2_MEDIUM.md).
- Interactive Plotly outputs are committed in [`docs/`](docs/) for GitHub Pages and in [`outputs/figures/`](outputs/figures/) as source HTML.
- Phase 8 work extends the same methodology to a 10-work Latin American comparison corpus in [`data/corpus/`](data/corpus/).

## Main Findings Captured In The Repo

- The linear reading order (chapters 1-56) is smoother than random across all four scales.
- The hopscotch path keeps moderate structure in the holistic embedding space, but that signal fades in the more explicit stylistic and semantic representations.
- The gap between holistic coherence and explicit semantic disorder is the central interpretive result of the current article series.

## Where To Look First

- [`ARTICLE_PART1_MEDIUM.md`](ARTICLE_PART1_MEDIUM.md): current source for the Part 1 article.
- [`ARTICLE_PART2_MEDIUM.md`](ARTICLE_PART2_MEDIUM.md): current source for the Part 2 article.
- [`ARTICLE_PART1_MEDIUM.html`](ARTICLE_PART1_MEDIUM.html) and [`ARTICLE_PART2_MEDIUM.html`](ARTICLE_PART2_MEDIUM.html): exported HTML copies at repo root.
- [`docs/index.html`](docs/index.html): landing page for the GitHub Pages visualization site.
- [`ANALYSIS_FRAMEWORK.md`](ANALYSIS_FRAMEWORK.md): explicit assumptions, null hypotheses, and interpretive boundaries.
- [`ARTICLE_LOG.md`](ARTICLE_LOG.md): chronological research and writing log.

## Repository Map

- [`src/`](src/): core analysis pipeline, including parsing, embeddings, stylometrics, semantic extraction, replication, trajectory analysis, corpus expansion, and figure generation.
- [`scripts/`](scripts/): article export, deployment, monitoring, EPUB/PDF parsing, and corpus utility scripts.
- [`data/`](data/): source texts, parsed `Rayuela` JSON, calibration passages, and the raw/clean comparison corpus.
- [`outputs/`](outputs/): generated embeddings, semantic and stylistic vectors, audits, review notes, summary JSON, and figure HTML.
- [`docs/`](docs/): GitHub Pages-ready visualizations.
- [`article_images/`](article_images/): static PNGs referenced by the root article markdown.
- [`prompts/`](prompts/): prompt templates for semantic and stylistic extraction, plus article review prompts.
- [`notebooks/`](notebooks/): reserved notebook area; currently empty in this checkout.

## Workspace Notes

- [`CLAUDE.md`](CLAUDE.md) still describes the project as if it were in Phase 1. The codebase and outputs show a much later state, so treat that file as historical guidance rather than the current status document.
- The root `ARTICLE_PART*_MEDIUM.md` files are now the canonical article sources.
- Most expensive outputs are already committed, so you do not need to rerun the full pipeline just to inspect results or deploy the visualization site.

## Environment

The repository was built around an ARM64 NVIDIA DGX Spark workflow. The Docker and Compose setup assumes:

- NVIDIA Container Toolkit / Docker GPU runtime
- an `HF_TOKEN` environment variable for model downloads
- an external Docker volume named `vllm-models`
- GPU access for the vLLM services

The analysis container is defined in [`Dockerfile`](Dockerfile). Multi-service orchestration lives in [`docker-compose.yml`](docker-compose.yml).

## Quick Start

```bash
docker compose build rayuela
docker compose up rayuela
docker compose --profile llm up vllm
docker compose --profile llm-nemotron up vllm-nemotron
```

- `rayuela` exposes JupyterLab on port `8888`.
- `vllm` serves Qwen 3.5 on port `8000`.
- `vllm-nemotron` serves the Nemotron replication model on port `8000` and should not run at the same time as `vllm`.

## Common Commands

```bash
# Rebuild cleaned Rayuela JSON
docker compose run --rm rayuela python src/parsing.py

# Scale A: chapter embeddings
docker compose run --rm rayuela python src/embeddings.py

# Scale A': classical stylometrics
docker compose run --rm rayuela python src/stylometrics.py

# Scale B': LLM-perceived style
docker compose run --rm rayuela python src/stylistic_extraction.py --resume

# Scale B: narrative DNA semantics
docker compose run --rm rayuela python src/semantic_extraction.py --resume

# Cross-scale analysis and figure generation
docker compose run --rm rayuela python src/scale_comparison.py
docker compose run --rm rayuela python src/article_figures.py

# Prepare GitHub Pages output from generated figure HTML
docker compose run --rm rayuela python scripts/prepare_ghpages.py
```

## Publication Helpers

- [`scripts/prepare_ghpages.py`](scripts/prepare_ghpages.py): copies selected figure HTML from `outputs/figures/` into `docs/` and swaps inline Plotly bundles for the CDN version.
- [`scripts/md_to_html.py`](scripts/md_to_html.py): converts the root Medium article markdown files into self-contained HTML. This script uses the Python `markdown` package.
- [`scripts/export_article_pngs.py`](scripts/export_article_pngs.py): exports static PNGs from Plotly figures for article use. It requires `kaleido`.

## Corpus Extension

The repo is no longer only about `Rayuela`. [`src/corpus_cleanup.py`](src/corpus_cleanup.py), [`src/corpus_stylometrics.py`](src/corpus_stylometrics.py), and [`src/corpus_semantic.py`](src/corpus_semantic.py) extend the same methods to a broader Latin American corpus stored in [`data/corpus/`](data/corpus/).

## Data Note

This repository includes raw and derived literary texts inside [`data/`](data/). Before redistributing the project or reusing the source texts elsewhere, verify that you have the rights to do so.
