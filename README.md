# Project Rayuela

[![Pre-commit](https://github.com/macayaven/rayuela/actions/workflows/pre-commit.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/pre-commit.yml)
[![Type Check](https://github.com/macayaven/rayuela/actions/workflows/type-check.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/type-check.yml)
[![Tests](https://github.com/macayaven/rayuela/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/tests.yml)
[![Documentation Coverage](https://github.com/macayaven/rayuela/actions/workflows/documentation-coverage.yml/badge.svg?branch=main)](https://github.com/macayaven/rayuela/actions/workflows/documentation-coverage.yml)
[![Coverage Threshold](https://img.shields.io/badge/coverage%20threshold-85%25-brightgreen)](./pyproject.toml)
[![Docstring Threshold](https://img.shields.io/badge/docstrings-85%25%2B-blue)](./pyproject.toml)

Project Rayuela is a computational literary analysis of Julio Cortazar's *Rayuela*. The repository contains the full workflow behind the project: text cleanup, embeddings, classical stylometrics, LLM-based style and semantic scoring, permutation tests, interactive visualizations, and article assets for a two-part series.

This workspace is already well past the planning stage. The analysis code and article source markdown are versioned here, while raw corpus files, derived outputs, and generated publishing assets stay local-only so the repo carries reproducible code rather than copyrighted or heavyweight artifacts.

## Current State

- `Rayuela` parsing outputs live in the local `data/` working tree and are intentionally not versioned.
- The project currently works across four analysis scales: holistic embeddings (A), content-light stylometrics (A'), LLM-perceived style (B'), and narrative DNA semantics (B).
- The validated semantic instrument is currently 19-dimensional. `temporal_clarity` was excluded after inter-rater replication with Nemotron 70B showed a rubric polarity failure.
- The two long-form article sources live at [`ARTICLE_PART1_MEDIUM.md`](ARTICLE_PART1_MEDIUM.md) and [`ARTICLE_PART2_MEDIUM.md`](ARTICLE_PART2_MEDIUM.md).
- Interactive Plotly outputs are generated locally into `outputs/figures/`, and the published GitHub Pages bundle is tracked in `docs/`.
- Phase 8 work extends the same methodology to a 10-work Latin American comparison corpus stored in the local `data/corpus/` working tree.

## Main Findings Captured In The Repo

- The linear reading order (chapters 1-56) is smoother than random across all four scales.
- The hopscotch path keeps moderate structure in the holistic embedding space, but that signal fades in the more explicit stylistic and semantic representations.
- The gap between holistic coherence and explicit semantic disorder is the central interpretive result of the current article series.

## Where To Look First

- [`ARTICLE_PART1_MEDIUM.md`](ARTICLE_PART1_MEDIUM.md): current source for the Part 1 article.
- [`ARTICLE_PART2_MEDIUM.md`](ARTICLE_PART2_MEDIUM.md): current source for the Part 2 article.
- Regenerate the root HTML exports and `docs/` publishing bundle locally with [`scripts/md_to_html.py`](scripts/md_to_html.py) and [`scripts/prepare_ghpages.py`](scripts/prepare_ghpages.py).
- [`ANALYSIS_FRAMEWORK.md`](ANALYSIS_FRAMEWORK.md): explicit assumptions, null hypotheses, and interpretive boundaries.
- [`ARTICLE_LOG.md`](ARTICLE_LOG.md): chronological research and writing log.

## Repository Map

- [`src/`](src/): core analysis pipeline, including parsing, embeddings, stylometrics, semantic extraction, replication, trajectory analysis, corpus expansion, and figure generation.
- [`scripts/`](scripts/): article export, deployment, monitoring, EPUB/PDF parsing, and corpus utility scripts.
- `data/`: local-only source texts, parsed `Rayuela` JSON, calibration passages, and the raw/clean comparison corpus.
- `outputs/`: local-only embeddings, semantic and stylistic vectors, audits, review notes, summary JSON, and figure HTML.
- `docs/`: published GitHub Pages bundle generated from `outputs/figures/`.
- `article_images/`: local-only PNG exports referenced by the root article markdown.
- [`prompts/`](prompts/): prompt templates for semantic and stylistic extraction, plus article review prompts.
- [`notebooks/`](notebooks/): reserved notebook area; currently empty in this checkout.

## Local-Only Asset Policy

- `data/`, `outputs/`, `article_images/`, and the root `ARTICLE_PART*_MEDIUM.html` exports are working artifacts and are intentionally ignored by git.
- `docs/` remains versioned because the current custom-domain GitHub Pages site publishes from `main:/docs`.
- Keep copyrighted corpus material and regenerated publishing bundles in your own local storage.
- Use the source markdown and the helper scripts in this repo to regenerate local artifacts when needed.

## Workspace Notes

- [`CLAUDE.md`](CLAUDE.md) still describes the project as if it were in Phase 1. The codebase and outputs show a much later state, so treat that file as historical guidance rather than the current status document.
- The root `ARTICLE_PART*_MEDIUM.md` files are now the canonical article sources.
- Most expensive outputs are intentionally local-only, so avoid rerunning the full pipeline unless you actually need refreshed artifacts.

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

- [`scripts/prepare_ghpages.py`](scripts/prepare_ghpages.py): copies selected figure HTML from the local `outputs/figures/` tree into the local `docs/` publishing bundle and swaps inline Plotly bundles for the CDN version.
- [`scripts/md_to_html.py`](scripts/md_to_html.py): converts the root Medium article markdown files into self-contained HTML. The helper depends on the Python `markdown` package, which is included in [`requirements-dev.txt`](requirements-dev.txt) for local QA but is not preinstalled in the base analysis container. Install it before running the export helper inside Docker with `docker compose run --rm rayuela pip install markdown`, or locally with `python3 -m pip install markdown`.
- [`scripts/export_article_pngs.py`](scripts/export_article_pngs.py): exports static PNGs from Plotly figures for article use. It depends on `kaleido`; install it the same way with `docker compose run --rm rayuela pip install kaleido` or `python3 -m pip install kaleido`.

## Quality Gates

- GitHub Actions enforces four checks on pull requests: pre-commit, type checking, tests with an 85% coverage threshold, and docstring coverage with an 85% threshold.
- The CI-safe quality scope currently covers [`src/parsing.py`](src/parsing.py), [`src/project_config.py`](src/project_config.py), [`src/reconstruction_contract.py`](src/reconstruction_contract.py), [`scripts/md_to_html.py`](scripts/md_to_html.py), and [`scripts/prepare_ghpages.py`](scripts/prepare_ghpages.py).
- Part 3 reconstruction coverage now also includes [`src/reconstruction_audit.py`](src/reconstruction_audit.py), [`src/reconstruction_metrics.py`](src/reconstruction_metrics.py), [`src/reconstruction_dataset.py`](src/reconstruction_dataset.py), and [`src/reconstruction_baselines.py`](src/reconstruction_baselines.py).
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
- `config_payload`
- `error_message`

Phase 0 dry run:

```bash
python3 src/reconstruction_contract.py --run-id phase0-dry-run --phase phase-0-quality-envelope
```

## Reconstruction Audit

Phase 1 adds [`src/reconstruction_audit.py`](src/reconstruction_audit.py) to verify that the cleaned comparison corpus and its derived outputs remain operationally decoupled from stale or orphaned artifacts.

Targeted Phase 1 verification:

```bash
python3 -m pytest tests/test_reconstruction_audit.py -q
python3 src/reconstruction_audit.py
```

Primary outputs:

- `outputs/corpus/corpus_metadata.json`: work- and author-level segment counts from the cleaned corpus
- `outputs/reconstruction/analysis/corpus_sync_audit.json`: machine-readable synchronization report

## Reconstruction Metrics

Phase 2 adds [`src/reconstruction_metrics.py`](src/reconstruction_metrics.py) to lock the measurement contract before any rewrite generation starts.

The module provides:

- typed stylometric and semantic baseline reports under `outputs/reconstruction/baselines/`
- deterministic rewrite scoring against source semantics and target stylistic envelopes
- identity, copy-source, and random-target control diagnostics

Targeted Phase 2 verification:

```bash
python3 -m pytest tests/test_reconstruction_metrics.py -q
python3 src/reconstruction_metrics.py
```

The CLI intentionally refuses to lock live baselines if the Phase 1 corpus outputs are stale or
incomplete.

Primary outputs:

- `outputs/reconstruction/baselines/stylometric_baseline.json`
- `outputs/reconstruction/baselines/semantic_baseline.json`
- `outputs/reconstruction/baselines/control_diagnostics.json`

## Reconstruction Dataset

Phase 3 adds [`src/reconstruction_dataset.py`](src/reconstruction_dataset.py) to lock the
pilot design before prompt baselines or training.

The module provides:

- deterministic 128-256 word source windows with chapter-level reference vectors
- chapter-aware train/validation/test split manifests with automated leakage checks
- target work envelopes with explicit train-split provenance
- serialized success criteria for the locked pilot under operational decoupling language

Targeted Phase 3 verification:

```bash
python3 -m pytest tests/test_reconstruction_dataset.py -q
python3 src/reconstruction_dataset.py
```

Primary outputs:

- `outputs/reconstruction/pilots/source_windows.json`
- `outputs/reconstruction/pilots/target_envelopes.json`
- `outputs/reconstruction/pilots/split_manifest.json`
- `outputs/reconstruction/pilots/success_criteria.json`

## Reconstruction Baselines

Phase 4 adds [`src/reconstruction_baselines.py`](src/reconstruction_baselines.py) to
establish a prompt-only baseline before any fine-tuning.

The module provides:

- versioned identity, paraphrase, style-shift, and revise prompt templates
- traceable generate-score-revise histories with raw responses and per-iteration scores
- immutable Phase 4 run manifests under `outputs/reconstruction/runs/`
- baseline case, summary, and markdown report artifacts for the locked pilot

Targeted Phase 4 verification:

```bash
python3 -m pytest tests/test_reconstruction_baselines.py -q
python3 src/reconstruction_baselines.py --run-id phase4-dry-run --dry-run --max-cases 2
```

Primary outputs:

- `outputs/reconstruction/runs/<run_id>/prompt_baseline_cases.json`
- `outputs/reconstruction/runs/<run_id>/prompt_baseline_summary.json`
- `outputs/reconstruction/runs/<run_id>/prompt_baseline_report.md`

The live prompt-control path expects the local Qwen vLLM service. If the host is
currently serving another model (for example the Nemotron replication endpoint),
use `--dry-run` to validate the Phase 4 contract without broadening the claim.

## Reconstruction Training

Phase 5 adds [`src/reconstruction_train.py`](src/reconstruction_train.py) and
[`src/reconstruction_infer.py`](src/reconstruction_infer.py) to prepare the
training envelope before real adapter fine-tuning.

The modules provide:

- deterministic `identity_smoke` dataset assembly from the locked pilot split
- immutable run directories with training config, tokenizer config, metrics, and
  checkpoint metadata
- placeholder-adapter handling that is explicit in metadata and rejected by
  inference when a real adapter is not present yet

Targeted Phase 5 verification:

```bash
python3 -m pytest tests/test_reconstruction_training.py -q
python3 src/reconstruction_train.py --run-id phase5-smoke --split-manifest-path outputs/reconstruction/pilots/split_manifest.json --target-envelopes-path outputs/reconstruction/pilots/target_envelopes.json
```

Primary outputs:

- `outputs/reconstruction/runs/<run_id>/training_config.json`
- `outputs/reconstruction/runs/<run_id>/training_metrics.json`
- `outputs/reconstruction/runs/<run_id>/checkpoint_metadata.json`

## Reconstruction Analysis

Phase 6 adds [`src/reconstruction_analysis.py`](src/reconstruction_analysis.py)
to turn immutable Phase 4+ run artifacts into synthesis-ready summaries.

The module provides:

- complete cross-run case aggregation from `prompt_baseline_cases.json`
- explicit failure-mode labeling from the saved scoring contract
- source-side bias slices by work and author
- pairwise run-comparison summaries over overlapping case identities
- provenance comparability gates over run invariants such as model, prompt,
  corpus/pilot artifacts, backend, and generation seed
- paired bootstrap intervals for mean run-to-run objective deltas
- explicit whether the paired mean-delta interval excludes zero
- failure-transition summaries that separate persistent, resolved, and
  introduced failure labels across overlapping cases
- explicit promotion recommendations that stay separate from scheduler
  keep/discard decisions
- article-ready summary/report artifacts and a close-reading queue with stable
  `run_id` links
- optional W&B analysis logging for aggregate counts, failure-mode totals,
  run-summary tables, run-comparison tables, failure-transition tables,
  promotion tables, provenance tables, close-reading queues, and attached
  analysis artifacts

Targeted Phase 6 verification:

```bash
.venv/bin/python -m pytest tests/test_reconstruction_analysis.py -q
.venv/bin/python src/reconstruction_analysis.py --wandb-project rayuela --wandb-mode offline
```

Promotion criteria are configurable at analysis time, for example:

```bash
.venv/bin/python src/reconstruction_analysis.py \
  --schedule-summary-path outputs/reconstruction/analysis/schedules/<schedule_id>/schedule_summary.json \
  --promotion-min-overlapping-cases 4 \
  --promotion-min-mean-delta 0.005 \
  --promotion-min-median-delta 0.0 \
  --promotion-min-non-negative-share 0.5 \
  --promotion-max-failure-case-delta 0 \
  --promotion-require-comparable-provenance
```

Primary outputs:

- `outputs/reconstruction/analysis/reconstruction_analysis_summary.json`
- `outputs/reconstruction/analysis/reconstruction_analysis_report.md`
- `outputs/reconstruction/analysis/reconstruction_article_inputs.json`

## Guided Scheduler

[`src/reconstruction_scheduler.py`](src/reconstruction_scheduler.py) adds a
finite experiment scheduler inspired by the keep/discard discipline of
Andrej Karpathy's `autoresearch`, but adapted to the reconstruction run contract.

The module provides:

- JSON-defined experiment queues with explicit commands, timeouts, and metric paths
- append-only scheduler results under `outputs/reconstruction/analysis/schedules/`
- automatic `keep` / `discard` / `failed` decisions based on a metric key extracted
  from produced run artifacts
- schedule summaries that record kept/discarded/failed `run_id`s for direct
  handoff into Phase 6 analysis
- optional W&B per-experiment logging for run metadata, scheduler decisions,
  extracted reconstruction metrics, and attached immutable run/scheduler artifacts

Targeted scheduler verification:

```bash
.venv/bin/python -m pytest tests/test_reconstruction_scheduler.py -q
.venv/bin/python src/reconstruction_scheduler.py --plan-path plans/reconstruction_guided_schedule.example.json --wandb-project rayuela --wandb-mode offline
.venv/bin/python src/reconstruction_analysis.py --schedule-summary-path outputs/reconstruction/analysis/schedules/<schedule_id>/schedule_summary.json --wandb-project rayuela --wandb-mode offline --wandb-group <schedule_id>
```

W&B logging is operationally decoupled from experiment execution. The scheduler
logs one run per planned experiment with the explicit keep/discard/failed
decision, the chosen advancement metric, and attached local artifacts so the
research loop can compare runs without rereading raw filesystem state. The
analysis step logs aggregate failure counts, run summaries, run comparisons,
comparability gates, paired uncertainty intervals, failure-transition tables,
promotion recommendations, and close-reading queues so the third article can
cite stable synthesis outputs rather than hand-maintained notes.
Those paired intervals should be read as approximate uncertainty summaries, not
as a stronger significance claim than the saved case table supports.

This separation is deliberate: scheduler decisions are local operational
decisions about one finite queue, while promotion recommendations are
research-facing judgments derived later from the saved case table and explicit
criteria.

Live prompt baselines now pass the configured reconstruction seed through the
OpenAI-compatible backend request surface. That does not make every upstream
serving stack perfectly reproducible, but it closes the avoidable gap where the
manifest claimed a seed while the generation call ignored it.

For detached execution, use [`src/reconstruction_launcher.py`](src/reconstruction_launcher.py)
or the thin shell shims in [`scripts/launch_reconstruction_schedule.sh`](scripts/launch_reconstruction_schedule.sh),
[`scripts/status_reconstruction_schedule.sh`](scripts/status_reconstruction_schedule.sh),
and [`scripts/stop_reconstruction_schedule.sh`](scripts/stop_reconstruction_schedule.sh).
The launcher validates the plan, checks the backend and required environment
keys, writes non-secret launch metadata, starts a tmux session keyed by
`schedule_id`, and keeps `scheduler.log` and `analysis.log` separate under the
schedule directory.

Minimal plan shape:

```json
{
  "schedule_id": "guided-20260310a",
  "experiments": [
    {
      "experiment_id": "baseline-a",
      "run_id": "phase4-live-20260310a",
      "phase": "phase-4-prompt-baselines",
      "command": [
        ".venv/bin/python",
        "src/reconstruction_baselines.py",
        "--run-id",
        "{run_id}"
      ],
      "timeout_seconds": 3600,
      "metric_path_template": "outputs/reconstruction/runs/{run_id}/prompt_baseline_summary.json",
      "metric_key": "controls.style_shift.mean_weighted_objective",
      "higher_is_better": true
    }
  ]
}
```

Primary outputs:

- `outputs/reconstruction/analysis/schedules/<schedule_id>/schedule_results.jsonl`
- `outputs/reconstruction/analysis/schedules/<schedule_id>/schedule_summary.json`

A ready-to-edit example lives at
[`plans/reconstruction_guided_schedule.example.json`](plans/reconstruction_guided_schedule.example.json).
Treat scheduler plan files as trusted executable specifications: the scheduler
passes the declared command directly to `subprocess.run()` and does not sandbox
or validate contributor-authored plans.

## Corpus Extension

The repo is no longer only about `Rayuela`. [`src/corpus_cleanup.py`](src/corpus_cleanup.py), [`src/corpus_stylometrics.py`](src/corpus_stylometrics.py), and [`src/corpus_semantic.py`](src/corpus_semantic.py) extend the same methods to a broader Latin American corpus stored in [`data/corpus/`](data/corpus/).

## Data Note

Raw and derived literary texts belong in the local-only `data/` working tree, not in version control. Before redistributing the project or reusing the source texts elsewhere, verify that you have the rights to do so.
