# Article Log: Project Rayuela

> This file captures the research journey for a future Medium article. Each session appends dated entries with discoveries, visualizations, key decisions, and "aha moments." Written in a narrative voice that can be adapted into article prose.

---

## Article Working Title

*"What Does a Novel Look Like From the Inside? Using AI to Map the Hidden Structure of Cortázar's Rayuela"*

## Article Outline (evolves as research progresses)

1. **The Hook**: Cortázar wrote a novel you can read two ways. We asked: does AI see why?
2. **The Setup**: What are embeddings, and what does it mean to "map" a book?
3. **The Texture Layer**: What the surface of the language reveals (Scale A)
4. **The Meaning Layer**: What the themes and emotions reveal (Scale B)
5. **The Path**: Tracing the hopscotch reading order through latent space (Scale C)
6. **The Shape**: What topology tells us that clustering can't (TDA)
7. **The Surprise**: What we didn't expect to find
8. **The Reflection**: What AI literary analysis can and can't tell us

---

## Session Log

<!-- Each session adds an entry below in this format:
### YYYY-MM-DD — Session Title
**Phase**: [current phase]
**What happened**: [2-3 sentences]
**Key finding/decision**: [the most important thing from this session]
**For the article**: [1-2 sentences written in article-ready prose]
**Visuals**: [any figures generated, with paths]
-->

### 2026-02-28 — Project Genesis
**Phase**: Phase 1 — Environment Setup
**What happened**: Defined the full research plan, set up the GitHub repository, and established the three-scale analysis framework (micro-textural, semantic DNA, graph-theoretic paths).
**Key finding/decision**: Chose `intfloat/multilingual-e5-large-instruct` over NV-Embed-v2 because Cortázar code-switches between Spanish, French, and English — a monolingual model would create artificial boundaries.
**For the article**: Before writing a single line of analysis code, we had to make a decision that would shape everything: how do you turn Spanish prose — prose that slips into French mid-sentence, that invents words, that breaks every rule — into numbers a computer can compare? The choice of embedding model is the choice of what "similarity" means.
**Visuals**: None yet.

### 2026-02-28 — The Container Wars
**Phase**: Phase 1 — Environment Setup
**What happened**: Built the Docker container on the DGX Spark (ARM64/Blackwell). Hit three cascading dependency failures: (1) NVIDIA's base image has local `.whl` installs that create ghost file paths in `pip freeze`, (2) vLLM is incompatible with NVIDIA's custom PyTorch version string (`2.11.0a0+...nv26.2`), and (3) giotto-tda has no ARM64 wheel. Solved by splitting vLLM into a separate NVIDIA container and deferring giotto-tda.
**Key finding/decision**: Two-service architecture — analysis container (our Dockerfile) + vLLM container (NVIDIA's `nvcr.io/nvidia/vllm:26.01-py3`). They communicate via OpenAI-compatible HTTP API. This avoids all PyTorch version conflicts and lets us start/stop the 70B model independently.
**For the article**: The first battle wasn't with literature — it was with dependency management. NVIDIA ships custom PyTorch builds with version numbers that no third-party package recognizes. The solution was to stop fighting and split the system in two: one container for analysis, one for the language model. They talk over HTTP, like microservices. Sometimes the best engineering decision is to stop trying to make things fit in one box.
**Visuals**: None yet.
