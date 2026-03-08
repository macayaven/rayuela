# Project Rayuela

[![Pre-commit](https://github.com/macayaven/rayuela/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/pre-commit.yml)
[![Type Check](https://github.com/macayaven/rayuela/actions/workflows/type-check.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/type-check.yml)
[![Tests](https://github.com/macayaven/rayuela/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/tests.yml)
[![Documentation Coverage](https://github.com/macayaven/rayuela/actions/workflows/documentation-coverage.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/documentation-coverage.yml)
[![Coverage Threshold](https://img.shields.io/badge/coverage%20threshold-85%25-brightgreen)](./pyproject.toml)
[![Docstring Threshold](https://img.shields.io/badge/docstrings-85%25%2B-blue)](./pyproject.toml)

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
- [`scripts/md_to_html.py`](scripts/md_to_html.py): converts the root Medium article markdown files into self-contained HTML. The helper depends on the Python `markdown` package, which is included in [`requirements-dev.txt`](requirements-dev.txt) for local QA but is not preinstalled in the base analysis container. Install it before running the export helper inside Docker with `docker compose run --rm rayuela pip install markdown`, or locally with `python3 -m pip install markdown`.
- [`scripts/export_article_pngs.py`](scripts/export_article_pngs.py): exports static PNGs from Plotly figures for article use. It depends on `kaleido`; install it the same way with `docker compose run --rm rayuela pip install kaleido` or `python3 -m pip install kaleido`.

## Quality Gates

- GitHub Actions enforces four checks on pull requests: pre-commit, type checking, tests with an 85% coverage threshold, and docstring coverage with an 85% threshold.
- The CI-safe quality scope currently covers [`src/parsing.py`](src/parsing.py), [`src/project_config.py`](src/project_config.py), [`src/reconstruction_contract.py`](src/reconstruction_contract.py), [`scripts/md_to_html.py`](scripts/md_to_html.py), and [`scripts/prepare_ghpages.py`](scripts/prepare_ghpages.py).
- Local setup:

```bash
python3 -m pip install -r requirements-dev.txt
python3 -m pre_commit install
python3 -m pre_commit run --all-files
python3 -m mypy
python3 -m pytest
python3 -m interrogate src/parsing.py src/project_config.py src/reconstruction_contract.py scripts/md_to_html.py scripts/prepare_ghpages.py
```

- The source-controlled GitHub ruleset definition lives at [`.github/rulesets/main-quality-gate.json`](.github/rulesets/main-quality-gate.json).

## Reconstruction Contract

Part 3 reconstruction runs are rooted under [`outputs/reconstruction/`](outputs/reconstruction/). Phase 0 establishes an immutable run policy:

- each run writes its manifest to `outputs/reconstruction/runs/<run_id>/manifest.json`
- each manifest is mirrored to `outputs/reconstruction/manifests/<run_id>.json`
- failed runs are retained in place and their run IDs are never reused
- all manifest paths are stored project-relative so the run can be replayed on another checkout

The required manifest fields are:

- `schema_version`
- `run_id`
- `phase`
- `status`
- `created_at`
- `updated_at`
- `git_sha`
- `model_id`
- `prompt_template_id`
- `seed`
- `config_hash`
- `corpus_manifest`
- `split_manifest`
- `paths`
- `error_message`

Phase 0 dry run:

```bash
python3 src/reconstruction_contract.py --run-id phase0-dry-run --phase phase-0-quality-envelope
```
