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

### 2026-02-28 — The Poison in the Footnotes
**Phase**: Phase 2 — Data Parsing (proofreading & source analysis)
**What happened**: Began proofreading the OCR'd text of Rayuela and discovered that each chapter ends with a parenthesized navigation number pointing to the next chapter in the hopscotch sequence — editorial apparatus baked into the prose. Carlos identified these as a source of circular analysis: if left in, our embeddings would encode the very structure we're trying to discover. We then switched from the OCR'd text (17 corrupted chapter headers, systematic `5`→`9` digit errors, ~15 missing markers) to a clean ePub source where each chapter is a separate XHTML file. Also discovered that Chapter 55 is the "phantom chapter" — the only one excluded from the hopscotch reading sequence entirely.
**Key finding/decision**: All editorial apparatus must be stripped — chapter numbers at the start, navigation markers at the end — to prevent data leakage. The ePub is vastly superior to the OCR text as a source: chapter boundaries are explicit (separate files), navigation markers are reliably tagged (`<p class="derecha">`), and there are zero OCR errors. Parsing will use the ePub, not the text dump.
**For the article**: Before we could feed a single word to the AI, we had to perform surgery on the text itself. Every chapter of Rayuela ends with a small number in parentheses — a signpost telling the reader where to jump next. Leave those numbers in, and any pattern the AI finds would be a forgery: we'd have embedded the answer inside the question. The technical term is circular analysis; the human term is cheating. We also found a ghost: Chapter 55, the only chapter Cortázar's hopscotch path never visits. A chapter that exists only if you read the book the "wrong" way.
**Visuals**: None yet.

### 2026-02-28 — The Container Wars
**Phase**: Phase 1 — Environment Setup
**What happened**: Built the Docker container on the DGX Spark (ARM64/Blackwell). Hit three cascading dependency failures: (1) NVIDIA's base image has local `.whl` installs that create ghost file paths in `pip freeze`, (2) vLLM is incompatible with NVIDIA's custom PyTorch version string (`2.11.0a0+...nv26.2`), and (3) giotto-tda has no ARM64 wheel. Solved by splitting vLLM into a separate NVIDIA container and deferring giotto-tda.
**Key finding/decision**: Two-service architecture — analysis container (our Dockerfile) + vLLM container (NVIDIA's `nvcr.io/nvidia/vllm:26.01-py3`). They communicate via OpenAI-compatible HTTP API. This avoids all PyTorch version conflicts and lets us start/stop the 70B model independently.
**For the article**: The first battle wasn't with literature — it was with dependency management. NVIDIA ships custom PyTorch builds with version numbers that no third-party package recognizes. The solution was to stop fighting and split the system in two: one container for analysis, one for the language model. They talk over HTTP, like microservices. Sometimes the best engineering decision is to stop trying to make things fit in one box.
**Visuals**: None yet.

### 2026-02-28 — First Blood: The Embeddings Speak
**Phase**: Phase 3 — Scale A (Micro-Textural Embeddings)
**What happened**: Embedded all 155 chapters using `intfloat/multilingual-e5-large-instruct` (1024 dimensions per chapter). Ran diagnostics on the full pairwise similarity distribution (11,935 pairs) and computed trajectory smoothness for both reading paths (linear and hopscotch). Created `ANALYSIS_FRAMEWORK.md` to make all assumptions explicit.
**Key finding/decision**: Three findings emerged. (1) The novel's three editorial sections (Del lado de allá / de acá / de otros lados) do NOT cluster by texture — Cortázar's voice is consistent across the whole novel, and the sections differ in content, not style. (2) The linear reading path (Ch. 1→56) is texturally smoother than the hopscotch path (mean consecutive similarity 0.9477 vs. 0.9293), suggesting the Tablero was designed for *semantic* coherence, not stylistic flow. (3) The expendable chapters (57–155) are internally more diverse than the main narrative — computationally confirming their experimental, heterogeneous character. Chapter 106 is the most texturally alien chapter in the entire novel, appearing in 4 of the 5 most dissimilar pairs.
**For the article**: The first thing the AI told us was something Cortázar scholars might find surprising: the three parts of Rayuela don't sound different. "Del lado de allá" (Paris) and "Del lado de acá" (Buenos Aires) and the "expendable chapters" — they all share the same textural DNA. The divisions readers feel are about *what* is being said, not *how*. But trace the two reading paths through this texture space, and something else appears: the linear path glides; the hopscotch path jolts. Every few steps, the Tablero throws you into a completely different register. If Cortázar designed the hopscotch order for continuity, it wasn't the continuity of style — it was the continuity of meaning. That's a question for a different model.
**Visuals**: None yet (UMAP projections coming in Phase 4).

### 2026-02-28 — The Reader Who Shuffles the Deck
**Phase**: Phase 3 — Scale A (Permutation Baselines)
**What happened**: Ran 10,000 random permutation baselines to test whether the trajectory smoothness of both reading paths is intentional or an artifact of the chapter pool. Compared each path against random orderings of the same chapters.
**Key finding/decision**: Both paths are intentionally ordered. The linear path is +6.37σ above its random baseline (smoother than all 10,000 random orderings). The hopscotch path is +2.77σ above random (smoother than 99.7% of permutations). This overturns our earlier claim that "the Tablero was not designed for textural flow." It *was* — just less aggressively than the linear sequence. The imagined reader Cortázar describes — who picks their own chapter order — would produce a rougher texture path than either designed sequence.
**For the article**: Cortázar told his readers they could make their own novel by choosing their own path through the chapters. So we tested it: we generated ten thousand imaginary readers, each shuffling the deck and reading in a random order. Every single one produced a rougher textural journey than Cortázar's own two paths. Both the linear reading and the hopscotch reading are smoother than chance — not by a little, but by a lot. The linear path is smoother than all ten thousand random orderings. The hopscotch path beats 99.7% of them. Cortázar didn't just write 155 chapters and suggest two routes through them. He *shaped* both routes for textural continuity — giving the linear reader a silk road and the hopscotch reader something rougher, wilder, but still guided.
**Visuals**: None yet.

### 2026-02-28 — The Texture Inside the Chapter
**Phase**: Phase 3A — Scale A (Sliding-Window Second Pass)
**What happened**: Built and ran the sliding-window embedding pipeline. 81 chapters (those over 512 words) were split into overlapping 512-token windows with 50% overlap, producing 888 window-level embeddings. Measured "internal drift" — how much the texture changes within each chapter — and identified where the biggest register shifts occur. Tested Carlos's prediction that Ch. 36 would show no special texture transition.
**Key finding/decision**: Three findings. (1) Carlos's prediction about Ch. 36 was essentially correct — it ranked 41st of 81 chapters, right at the median. Cortázar's authorial voice is consistent enough that content-mode shifts (dialogue vs. description) don't create outlier drift. (2) Chapters 34, 23, and 56 all show their biggest texture jump near the *end* (92%, 96%, 18% respectively) — consistent with a shift from mimetic mode (showing action/dialogue) to diegetic mode (narrating outcomes, describing character states). Carlos identified this pattern: the endings shift from the intensity of interaction to a narrative of aftermath. (3) Short chapters with 3-4 windows dominate the high-drift rankings due to small-sample variance — a confound we'll need to normalize for.
**For the article**: We zoomed in. The first pass had given each chapter a single number — a coordinate in texture space. Now we slid a magnifying glass across the long chapters, watching how the texture shifts sentence by sentence. The result was surprisingly uniform: Cortázar's voice holds. Even in chapters where the *content* shifts dramatically — from philosophical debate to intimate confession, from dialogue to description — the surface of the language stays consistent. The model detects something the reader feels but might not articulate: Cortázar writes everything in the same register. What changes at the end of his great ensemble chapters isn't the voice but the *mode* — from showing to telling, from the heat of the scene to the cool summary of its aftermath. The AI can see the moment the camera pulls back.
**Visuals**: None yet (per-chapter drift curves saved for later plotting).

### 2026-03-01 — Teaching a Machine to Read Like a Critic
**Phase**: Phase 3B — Scale B (Semantic Profiling / Narrative DNA)
**What happened**: Set up the second scale of analysis: LLM-based semantic profiling. After a multi-hour infrastructure battle (model license gates, OOM errors, Docker image incompatibilities, systemd-managed containers blocking ports, DNS failures inside containers), settled on Qwen 3.5 27B FP8 via vLLM's nightly CUDA 13.0 image — the only combination that supports this new model architecture on DGX Spark's Blackwell GPU. Designed a 20-dimension scoring rubric covering five facets of the novel (thematic, emotional, character dynamics, narrative mode, formal/experimental). Validated on three chapters (Ch. 1, 36, 68) before launching full extraction on all 155 chapters.
**Key finding/decision**: The test chapters produced strikingly defensible scores. Chapter 68 — the famous Glíglico chapter written in invented language — scored language_experimentation: 10, love_and_desire: 9, spatial_grounding: 1, dialogue_density: 1. The model understood that a 186-word passage in a made-up language is about love, not space, not dialogue, and that the language itself is the experiment. Chapter 1 scored oliveira_centrality: 10, la_maga_presence: 10, interiority: 10 — correctly identifying the deep stream-of-consciousness opening. The model uses the full 1–10 range and the scores differentiate meaningfully across chapters. Carlos proposed a future extension: extracting explicit text evidence for each score to enable "generative remixing" — replacing period-specific elements to create Rayuela for different generations.
**For the article**: Scale A told us what the surface of the language looks like. Now we needed to go deeper — into meaning. We designed twenty questions for the AI to answer about each chapter: How much existential questioning? How present is La Maga? How experimental is the language? The test case was Chapter 68, the passage Cortázar wrote in a language he invented. The AI scored it 10/10 for language experimentation and 9/10 for love and desire — which is exactly right. The Glíglico passage is a love scene rendered in sounds rather than words. The machine couldn't read the invented language (no one can), but it understood what the invented language was *for*.
**Visuals**: None yet (20-dim heatmap and radar plots planned for Phase 4).

### 2026-03-01 — The Hopscotch Illusion
**Phase**: Phase 3B (completing) + Phase 4 (UMAP & Article Visualizations)
**What happened**: v1 extraction is completing (120/155 chapters scored). Built the v2 prompt with 3 calibration examples from *62: Modelo para armar* and *Un tal Lucas* — external anchoring to prevent overfitting. Meanwhile, launched Phase 4: UMAP projections of both Scale A (1024-dim texture) and Scale B (20-dim narrative DNA) into 2D, plus a full suite of 13 interactive article visualizations. The permutation test (5,000 random orderings) produced the session's central finding.
**Key finding/decision**: The linear reading order (Ch. 1→56) is **-7.7σ smoother than random** in both texture and narrative space — virtually impossible by chance. But the hopscotch path (Tablero de Dirección) is **statistically indistinguishable from a random ordering** (z ≈ 0 in both spaces). This overturns the Phase 3A finding (which used only texture): with the full 20-dimensional narrative profile, the hopscotch path is no longer smoother than chance. The earlier +2.77σ result was an artifact of texture-only measurement. Cross-scale correlation is ρ = 0.336 — the two scales capture partially overlapping but substantially independent structure. The hopscotch path is designed not for smooth transitions but for **productive discontinuity**: systematic interleaving of the novel's three sections.
**For the article**: Here's the twist that surprised us. When we tested only the surface texture of the language, the hopscotch path seemed designed for smooth flow — smoother than 99.7% of random orderings. But when we added the semantic layer — twenty dimensions measuring theme, emotion, character dynamics, narrative mode — the smoothness advantage vanished entirely. The hopscotch path is as jumpy as a random shuffle. The linear reading of Rayuela is like a river: each chapter flows naturally into the next, both in sound and meaning. The hopscotch reading is like channel-surfing: Paris, Buenos Aires, philosophical essay, love scene, literary theory, grocery list — the Tablero deliberately scrambles the texture. Cortázar didn't design the hopscotch for continuity. He designed it for *collision*. The jarring transitions are the point. You're not supposed to settle in. You're supposed to be thrown, again and again, from one register into another — to feel, in the reading itself, the disorientation of Oliveira wandering between cities and selves.
**Visuals**: 13 interactive HTML figures in `outputs/figures/` — highlights: `article_dual_heatmap.html` (same data, two reading orders side-by-side), `article_permutation.html` (the statistical proof), `article_weaving.html` (section interleaving pattern), `article_radar.html` (fingerprints of Ch.1, 28, 36, 56, 68, 73, 93).

### 2026-03-02 — From Research to Publication
**Phase**: Article drafting, copyright review, GitHub Pages, commit history
**What happened**: Wrote the Part 1 article draft (`ARTICLE_PART1.md`, ~1,600 words) — a tighter, narrative-driven version with accessible statistics and a strong dramatic arc (question → false lead → twist). Fact-checked every statistical claim against the overnight audit report; corrected three errors (Ch. 68 scores were love=9 not 8, spatial=2 not 1; "middle of the pack" was imprecise for hopscotch percentiles; the probability framing needed a footnote). Researched copyright for quoting Rayuela (copyrighted until ~2054, short quotes for criticism = fair use). Sanitized the v2 prompt to remove embedded copyrighted excerpts before committing. Created 5 logically grouped commits telling the project's story from Phase 2 through Phase 4. Built GitHub Pages companion site: stripped inline Plotly.js from 19 visualizations (92.7 MB → 0.8 MB via CDN), created index page, deployed to `docs/`. v2 extraction confirmed at 153/155 chapters — Ch. 28 and 41 (the two longest) exceed the 24,576-token context window with the few-shot prompt.
**Key finding/decision**: The v2 (few-shot + evidence) extraction confirms the core finding with independent scores: Linear z = −8.81σ, Hopscotch z = +0.47σ. Cross-scale ρ increased from 0.336 (partial data) to 0.505 (full 155 chapters). One dimension — `temporal_clarity` — has low v1-v2 correlation (r = 0.227), meaning the two prompt versions interpret it differently. This dimension should be flagged as unreliable or excluded from cross-version claims.
**For the article**: We fact-checked every number in the article against the raw data. Three corrections. The kind of corrections that don't change the story but make it bulletproof — the difference between a finding that survives peer review and one that doesn't. The central claim held: the linear path is designed for smooth flow (smoother than every one of 5,000 random shuffles, on both measurement scales independently); the hopscotch path falls within the normal range of random orderings. Cortázar built a river and a kaleidoscope from the same 155 chapters.
**Visuals**: 6 new 3D UMAP explorations in `outputs/figures/3d_*.html`. GitHub Pages companion: `docs/` with 19 interactive charts + index at `carloscrespomacaya.com/rayuela`.
**Domain**: Migrated GitHub Pages custom domain from `carlos-crespo.com` to `carloscrespomacaya.com` (Cloudflare registrar). DNS config: 4 A records to GitHub Pages IPs + CNAME `www` → `macayaven.github.io`.

### 2026-03-03 — Medium Prep, Scale A Rethink, Corpus Planning
**Phase**: Article publication prep + new research direction (Scale A')
**What happened**: Generated static PNGs for all 10 article figures (Playwright headless Chromium screenshots of Plotly HTML). Created Medium-ready versions of Part 1 (~1,600 words) and Part 2 (~2,200 words, methodology deep-dive including v1/v2 sensitivity analysis, evidence strings, dimension correlations). Fixed HTTPS enforcement on both `macayaven.github.io` and `macayaven/rayuela` GitHub Pages — SSL certificates now working. Prepared a download package with articles, PNGs, and a context file with verified statistics for Claude co-work editorial polishing. **Critical methodological insight from Carlos**: Scale A (E5 embeddings) cannot be claimed to capture only "texture" — sentence transformers are trained via contrastive learning on semantic similarity, so they encode meaning as well as form. The honest framing: Scale A is a "holistic embedding" (unstructured mix of surface + meaning), Scale B is an "explicit semantic decomposition" (20 named dimensions). This led to planning Scale A' — classical stylometric features (sentence length, vocabulary richness, function word frequencies, punctuation density, code-switching frequency) that are by construction content-free. Also designed a corpus preparation pipeline: Gemini prompt for parsing borrowed Internet Archive novels from 10+ Latin American authors into structured JSON, for future style/content disentanglement training.
**Key finding/decision**: The "texture vs meaning" framing in the articles is imprecise and needs correction. The finding itself is stronger with the honest framing — two representations that BOTH include semantic content independently agree. The planned Scale A' (stylometrics) would complete the triangulation with a provably content-free measurement.
**For the article**: We caught our own imprecision before publication. Scale A doesn't capture "how the prose sounds" — it captures everything, in an unstructured way. Scale B captures what we explicitly asked it to measure. The fact that both agree makes the finding stronger, not weaker. The correction: stop saying "texture vs meaning" and start saying "holistic embedding vs structured semantic decomposition." Less poetic, more defensible. The kind of precision that matters when someone tries to replicate this.
**New prompts**: `prompts/corpus_preparation_v1.txt` — Gemini prompt for parsing Latin American literary corpus (14 priority works across 10 authors).
**Next**: Fix article language for Scale A framing, build stylometric feature extractor (Scale A'), integrate corpus when Gemini parsing completes.

### 2026-03-03 — The Four-Scale Rewrite

**Phase**: Phase 5 (QA) + Article revision

**What happened**: Rewrote both articles from 2-scale to 4-scale, incorporating all feedback from Gemini 3 Flash Preview and Codex GPT-5.3. Key changes: (1) "Two Microscopes" → "Four Ways of Listening" — introduced A' (26 content-free stylometric features) and B' (12 LLM-perceived style dimensions) alongside the original A (holistic) and B (semantic). (2) Added new "Shape of Disruption" section presenting curvature findings: hopscotch is -4.7σ on Scale B — not just random, but actively jagged, an "anti-path." (3) Refined "collision" framing to "collision-dominant with formal undertow." (4) Added 4-scale results table with Bonferroni annotation. (5) Softened overstatements: "statistically impossible" → permutation language, "stochastic noise — by design" → "near-random," "every row different" → "alternates much more abruptly," "non-borderline" → "moderate and representation-dependent." (6) Harmonized Gliglico to v2 values (love=8, spatial=1). (7) Added "Gap of the Unnamed" section to Part 2 with Mantel correlation matrix, introducing "form-content resonance" as the technical term for the irreducible holistic signal. (8) Introduced "atmospheric coherence" (Part 1, accessible) and "form-content resonance" (Part 2, technical) for the unnamed quality in Scale A.

**Key finding/decision**: The four-scale gradient is the strongest new rhetorical and analytical device. As measurement moves from holistic to explicit, the hopscotch signal fades: A(+2.8σ) → B'(+1.3σ) → A'(+1.1σ) → B(-0.4σ). Only Scale A survives Bonferroni. The gradient itself constrains interpretation more powerfully than any single z-score. The Mantel matrix reveals that A↔A' = 0.38 (lowest correlation) despite both including "form" — because the E5 model is dominated by meaning. The 46% of holistic variance unexplained by semantics is where the atmospheric coherence lives.

**For the article**: The articles grew from two lenses to four, and the story became more honest. We no longer claim the hopscotch is merely random — it's an anti-path, rougher than chance in its semantic curvature. And we no longer claim the holistic signal is unexplainable — the gradient from +2.8σ to -0.4σ traces the contour of something we can name but not fully decompose. We call it atmospheric coherence in the accessible version and form-content resonance in the technical one. It's the quality that emerges when all the elements of prose — rhythm, vocabulary, theme, mood — combine into something greater than their sum. Cortazar's hopscotch preserves that quality while shattering everything else.

**Visuals**: No new figures — same 10 PNGs, but captions updated to reflect 4-scale framing.

**Reviewer feedback**: Both reviewers called the revised articles "ready for publication." Gemini highlighted the gradient argument and curvature as the strongest additions. Codex called the thesis "genuinely memorable" and noted Part 2's mid-section density as the only pacing concern.

### 2026-03-03 — Publication Package and GitHub Push

**Phase**: Publication prep + repo cleanup

**What happened**: Created a publication package (`publication/part1/` and `publication/part2/`) with clean articles and 10 figure PNGs. Removed obsolete files (old drafts, duplicates). Fixed .gitignore to cover PDFs that were exposed. Removed all future teasers from articles — no more "Part 3 will explore..." promises. Added collaboration attribution crediting Claude (implementation/drafting), GPT-5.3 Codex and Gemini 3 Flash Preview (code reviews). Updated GitHub Pages index with 4-scale framing and figure descriptions. Pushed two commits: code (20 files) + articles/assets (16 files). Zero copyrighted data in repo.

**Key finding/decision**: The articles now stand as self-contained, complete pieces — no open-ended promises, no implementation noise (Docker, DGX Spark, overnight audits), just the research and its findings. The collaboration attribution is explicit and honest: this was built as a team.

**For the article**: The microscope is not a judge. It reveals structure that was always there but invisible to the unaided eye. What we make of that structure — whether we find it beautiful, significant, or worth arguing about — remains irreducibly human.

**Visuals**: 10 PNGs in `publication/part1/` (figures 1-5) and `publication/part2/` (figures 6-10). GitHub Pages live at `carloscrespomacaya.com/rayuela` with updated index.

### 2026-03-03 — Data Leakage Audit (3/3 Clean) and Replication Strategy

**Phase**: Verification & methodology hardening

**What happened**: Audited the entire Scale A pipeline for circular analysis — does the E5 embedding model ever see chapter numbers, section labels, or hopscotch navigation markers? Three independent reviewers examined the code: Claude (static data-flow trace across `src/parsing.py`, `src/embeddings.py`, `src/embeddings_windowed.py`), Gemini 3 Flash Preview (structural code analysis), and GPT-5.3 Codex (dynamic verification — ran Python scripts against the actual ePub to confirm zero metadata leakage in all 155 chapters). All three returned clean verdicts. The pipeline extracts only `ch["text"]` (the stripped prose), applies a uniform instruction prefix, and never exposes chapter numbers, section names, or reading path information to the model.

**Key finding/decision**: The +2.8σ hopscotch smoothness finding is not an artifact of metadata leakage. Three independent reviewers confirmed the model receives only prose content. Codex's contribution was especially strong: it didn't just read the code, it tested the actual parsed output against the original XHTML to verify zero residual navigation markers.

**Replication strategy**: Carlos proposed convergent validity — re-running Scale A with a completely different embedding model. Found `nvidia/llama-nemotron-embed-1b-v2` (NVIDIA, Llama 3.2-based bidirectional encoder, 1.24B params, multilingual including Spanish/French/English). This is maximally independent from E5: different company, different architecture (decoder-made-bidirectional vs encoder-only), different training pipeline, different base data. Embedding dimension is configurable to 1024 to match E5. Compatible with sentence-transformers on our DGX Spark. If both models detect hopscotch smoothness above random, the finding is model-independent — it's in the text, not an artifact of any single model's quirks.

**For the article**: We didn't just claim our pipeline was clean — we proved it with three independent auditors, including one that tested the actual data. Then we asked: what if the model itself is the bias? So we planned a replication with a completely different model. This is how you build an argument that can withstand scrutiny.

### 2026-03-03 — Convergent Validity: Nemotron Replication

**Phase**: Verification & methodology hardening

**What happened**: Replicated Scale A with `nvidia/llama-nemotron-embed-1b-v2` — a fundamentally different model (Llama 3.2 decoder-made-bidirectional, 1.2B params, 2048 dims) from our original E5 (encoder-only, 560M params, 1024 dims). Different company (NVIDIA vs Microsoft), different architecture, different training pipeline. Ran the full 155-chapter embedding pipeline and the same trajectory permutation test (5,000 permutations, seed 42).

**Key finding**: Partial convergence — the most honest result we could have hoped for.

| Path | E5 (1024d) | Nemotron (2048d) | Direction? |
|------|------------|------------------|------------|
| Linear | +6.40σ | +9.73σ | Both strongly significant |
| Hopscotch | +2.80σ | +1.01σ | Both positive, but only E5 >2σ |

The linear finding is rock-solid: two independent models agree the sequential path is extraordinarily smooth. The hopscotch finding is more nuanced — both models detect above-random smoothness (both positive z), but the signal strength is model-sensitive. E5's semantic similarity training captures more of Cortazar's thematic threading than Nemotron's retrieval-oriented training.

**For the article**: We replicated our analysis with a completely different AI model — different company, different architecture, different training. The linear finding survived perfectly. The hopscotch finding survived in direction (both positive) but not magnitude. This is what real science looks like: not a perfect echo, but a consistent signal that depends on what you're measuring. The instrument matters — and reporting that honestly is what distinguishes research from advocacy.

---

### Session 2026-03-03 (Evening) — Scale B Inter-Rater Reliability: Setup

**The question**: Our Scale B "Narrative DNA" scores were extracted by a single LLM (Qwen 3.5 27B). Are those scores measuring something real in the text, or just one model's idiosyncrasies? This is the "inter-rater reliability" problem from psychometrics — if two independent raters agree, the signal is real.

**The second rater**: We selected `RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic` — NVIDIA's 70B instruction-tuned model, FP8 quantized. Independence vectors: different company (NVIDIA vs Alibaba), different base architecture (Llama 3.1 vs Qwen 3.5), different parameter count (70B vs 27B), different training pipeline.

**External review**: Gemini 3 Flash and GPT-5.3 Codex both reviewed the plan. Key catches:
- Codex flagged that our `--max-model-len 16384` was too tight for Chapter 28 (12,332 words × ~1.4 tokens/word = ~17K tokens). Increased to 24576.
- Codex caught a silent model-fallback bug in `semantic_extraction.py` that could invalidate the replication by accidentally running Qwen while labeling outputs as Nemotron. Fixed to fail hard on model mismatch.
- Both recommended adding quadratic weighted Cohen's kappa (chance-corrected ordinal agreement) alongside Spearman/Pearson. Added via scikit-learn.
- Gemini confirmed guided JSON decoding is model-agnostic in vLLM — the same v1 prompt works unchanged.

**Files created**: `src/replication_scale_b.py` (comparison metrics), `vllm-nemotron` service in `docker-compose.yml`.

**Status**: Nemotron 70B vLLM server is compiling CUDA kernels (first-time cost, ~20-30 min). Once ready, extraction runs detached: `docker compose run --rm -d --name rayuela-extract-nemotron rayuela python src/semantic_extraction.py --model "RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic" --output-dir outputs/semantic_nemotron --resume`

**For the article**: We're applying the same rigor to our AI-generated scores that psychometricians apply to human raters. If two completely independent AI models — from different companies, with different architectures — read the same Cortázar chapter and score it similarly on "existential questioning" or "humor," then that score reflects something real in the prose. If they disagree, we learn which literary dimensions are robustly measurable and which are more in the eye of the (silicon) beholder.

---

### 2026-03-04 — Inter-Rater Reliability Results & Dimension Exclusion

**The data is in.** Nemotron 70B scored all 155 chapters overnight with zero failures. The inter-rater reliability comparison:

- **Overall**: ρ = 0.844, mean weighted κ = 0.753 ("substantial" on the Landis & Koch scale)
- **18/20 dimensions** show strong agreement (ρ ≥ 0.7), including the most narratively important ones: Existential Questioning (0.91), Love/Desire (0.93), Dialogue (0.94), Oliveira (0.92)
- **1 moderate**: Language Experimentation (ρ = 0.50) — weak but reliably positive
- **1 anti-correlated**: Temporal Clarity (ρ = -0.30, κw = -0.16) — the models systematically disagree

**The failed dimension is the most interesting finding.** `temporal_clarity` uses a rubric where "1 = Clear timeline, 10 = Fragmented," but the label says "Clarity." One model appears to follow the label (high clarity = high score), the other follows the rubric (fragmented = high score). Bootstrap CIs confirm the disagreement is systematic, not noisy: 95% CI [-0.47, -0.12]. Reverse-coding doesn't fix it (ρ only reaches +0.30). This is exactly the kind of dimension where *Rayuela*'s famous temporal play makes LLM scoring genuinely ambiguous.

**Decision after consulting three reviewers (Gemini, Codex, Claude):** Drop `temporal_clarity` only. Keep `language_experimentation` as lower-confidence but real signal. The validated Scale B is now 19 dimensions.

**Re-run results (19D):** Linear +4.46σ (slightly stronger), Hopscotch -0.28σ (still indistinguishable from random), Curvature -4.67σ (still maximally jagged). The core findings are robust — removing the noisy dimension barely changed the z-scores. Cross-scale Mantel correlation A↔B: 0.532 (up from 0.505 with 20D).

**For the article**: The failure of `temporal_clarity` is itself a finding worth reporting. In a novel where time is famously fractured — where Chapter 34 interleaves two narratives, where the hopscotch path jumps across decades — asking "how clear is the timeline?" may not have a single correct answer. The models aren't wrong; they're disagreeing about something genuinely ambiguous. This is where quantitative analysis meets the irreducible complexity of literature.

---

### 2026-03-04 (Evening) — Implementation, Figure Regeneration, and Commit

**What happened**: Implemented the dimension exclusion in `project_config.py` (`DIMS_EXCLUDED`, `DIMS_ORDERED_ALL` for replication, `DIMS_ORDERED` 19D for analysis, `filter_excluded_dims()` helper). Updated `replication_scale_b.py` to report all 20 dims with exclusion markers, and `trajectory_stylometrics.py` / `scale_comparison.py` to filter the `.npy` columns. Regenerated all 10 article HTML figures and 8 PNGs with the 19D instrument. Updated GitHub Pages. Committed as "Phase 7: Inter-rater reliability and dimension validation" (28 files, +1008/-113 lines). Prepared self-contained Claude App prompt for article text updates (20→19D, add IRR section).

**Key finding/decision**: The 19D re-run confirmed all findings are robust to the exclusion. The z-scores barely moved (Hopscotch -0.4σ → -0.28σ, Curvature -4.7σ → -4.67σ). Cross-scale Mantel A↔B improved slightly (0.505 → 0.532) — removing noise strengthened the signal.

**For the article**: We didn't just flag a bad dimension and move on — we re-ran everything. The z-scores held. The figures held. The cross-scale correlations actually improved. That's the difference between a finding that depends on a specific configuration and one that reflects something real in the text.

**Figures regenerated**: `article_images/figure{3,4,5,6,7,8,9,10}_*.png`, all `docs/*.html` (GitHub Pages).

**Session end note**: Commit `e0fb4e9` pushed to origin. Articles pending text update (20→19D references + new IRR section) — Claude App prompt prepared for Carlos.

### 2026-03-04 — Corpus Cleanup and the Disentanglement Question

**Phase**: Phase 8A — Corpus Cleanup + External Review

**What happened**: Built `src/corpus_cleanup.py` to clean 10 Latin American literary works (García Márquez, Sábato, Cortázar ×2, Borges ×2, Bolaño, Cabrera Infante, Rulfo, Quiroga) from various source formats (ePub, PDF, scanned). Each work required unique cleanup rules: Borges stories were segmented from single blobs using dual-signal detection (year markers + paragraph boundaries), Bolaño's PDF had broken words ("M exicanos"), Cabrera Infante had 428 running headers, and Rulfo's text was 60% critical apparatus. Final corpus: 152 chapters, 832K words, zero artifacts. Sent the Phase 8 plan to Gemini 3 Flash Preview and GPT-5.3 Codex for independent review.

**Key finding/decision**: Both reviewers converged on a critical reframing: what we have is "operational decoupling" (A' measures style, B measures content), not mathematical disentanglement (Mantel r=0.42 shows they're partially coupled). Both recommended Path 1 (prompt-based transfer with generate-score-revise loop) as the right first approach, with tolerance bands instead of exact numeric targets. The corpus is sufficient for a pilot but weak for per-author generalization claims — single-work authors confound author style with book/topic/period.

**For the article**: Before we could transfer anyone's style, we had to solve a more mundane problem: getting the text into shape. Ten novels, ten different data quality nightmares. Borges's stories were concatenated into a single wall of text. Bolaño's PDF had split words across line breaks. Rulfo's file was two-thirds critical apparatus and one-third novel. The cleanup took longer than the analysis — as it should. In computational literary analysis, data quality isn't a preliminary step. It's the foundation.

**Visuals**: None (infrastructure phase).

---

### 2026-03-08 — Part 3 Phase 0: Quality Envelope and Observability

**Phase**: Part 3 — Phase 0 (Quality Envelope and Observability)

**What happened**: Implemented the first reconstruction module, `src/reconstruction_contract.py`, to centralize the Part 3 output tree, enforce project-relative paths, seed Python/NumPy/torch/splitter state deterministically, and persist immutable run manifests under `outputs/reconstruction/`. Added `tests/test_reconstruction_contract.py` first, then wired the new module into the coverage and mypy scopes in `pyproject.toml`, documented the manifest contract in `README.md`, and wrote a dry-run manifest to `outputs/reconstruction/runs/phase0-dry-run-20260308a/manifest.json`.

**Key decision**: Failed runs must remain inspectable and run IDs must be immutable. The contract mirrors each run manifest into `outputs/reconstruction/manifests/` for global lookup, but the per-run directory under `outputs/reconstruction/runs/<run_id>/` remains the source of truth and is never reused.

**Verification**:
- `pytest tests/test_reconstruction_contract.py -q` passed
- full `pytest -q` passed
- `ruff check src/reconstruction_contract.py tests/test_reconstruction_contract.py` passed
- repo-wide `ruff check src tests scripts` was still red at the end of the initial Phase 0 implementation because of pre-existing lint debt outside the reconstruction files; the baseline was restored in the follow-up cleanup recorded alongside Phase 1
- full `mypy` passed
- dry-run manifest write passed without generation

**For Part 3**: The experiment now has a concrete run contract before any generation begins. Every later phase can inherit the same metadata envelope: git SHA, seed bundle, config hash, corpus manifest, split manifest, and run-local artifact paths.

### 2026-03-08 — Part 3 Phase 1: Corpus Synchronization Audit Baseline

**Phase**: Part 3 — Phase 1 (Corpus Synchronization and Audit)

**What happened**: Added `src/reconstruction_audit.py` and `tests/test_reconstruction_audit.py` test-first to compare the cleaned comparison corpus against the committed stylometric and semantic outputs. The audit generates a machine-readable corpus manifest and flags stale coverage, missing aggregates, orphan outputs, and profile drift without collapsing the result into a single pass/fail sentence. The implementation keeps the claim narrow: this is an operational decoupling audit of the measurement stack, not a claim that the corpus is now permanently synchronized.

**Current audit baseline**:
- cleaned corpus total: 10 works / 495 segments
- stale stylometric outputs: `bolano_detectivessalvajes` (3 vs 194), `cabrerainfante_trestistestigres` (1 vs 40), `cortazar_62modelo` (1 vs 44), `rulfo_pedroparamo` (1 vs 71)
- missing aggregate: `outputs/corpus/author_profiles_semantic.json`
- stale aggregate drift in `author_profiles_stylo.json`, including a Cortázar profile that still mixes `Rayuela` into the corpus-only aggregate

**Key decision**: Corpus author profiles should be corpus-only by default. `src/corpus_stylometrics.py` and `src/corpus_semantic.py` now reserve `Rayuela` inclusion for an explicit `--include-rayuela` path instead of silently folding it into the comparison corpus aggregates.

**Constraint encountered**: The canonical `outputs/corpus/` tree is owned by `nobody:nogroup` in this sandbox, so the Phase 1 audit could only write its generated artifacts under `outputs/reconstruction/analysis/` during this session. The code paths for `outputs/corpus/corpus_metadata.json` remain implemented, but in-place regeneration of the stale corpus outputs is blocked here by filesystem permissions, not by missing audit coverage.

**Verification**:
- `pytest tests/test_reconstruction_audit.py -q` passed
- repo-wide `ruff check src tests scripts` passed after baseline cleanup
- full `pytest -q` passed
- full `mypy` passed

### 2026-03-09 — Part 3 Phase 2: Measurement Contract and Control Harness

**Phase**: Part 3 — Phase 2 (Measurement Contract, Controls, and Baselines)

**What happened**: Added `src/reconstruction_metrics.py` and `tests/test_reconstruction_metrics.py` test-first to formalize the measurement layer that later rewrite experiments will rely on. The new module loads corpus-aligned stylometric and semantic outputs, filters excluded semantic dimensions, computes typed per-dimension baselines, scores candidate rewrites against source semantics and target stylistic envelopes, and writes deterministic identity/copy-source/random-target control diagnostics under `outputs/reconstruction/baselines/`.

**Key decision**: Phase 2 should fail loudly when the measurement layer is stale instead of silently building baselines from mismatched files. The implementation therefore treats synchronized chapter counts and dimension orders as part of the contract, not as advisory metadata.

**Constraint encountered**: The Phase 2 CLI cannot lock the live corpus baselines yet because `outputs/corpus/` is still root-owned in this environment and the known stale stylometric work outputs remain unreadable as a clean synchronized set. The code path is implemented and the synthetic fixture path is green, but the live baseline artifact write is blocked by the unresolved Phase 1 filesystem state rather than by missing Phase 2 tests.

**Verification**:
- `pytest tests/test_reconstruction_metrics.py -q --no-cov` passed
- full `pytest -q` passed
- full `ruff check src tests scripts` passed
- full `mypy` passed

**For Part 3**: The experiment now has a concrete scoring contract before prompt baselines or training. Later phases can compare identity, copy-source, prompt-only, and fine-tuned rewrites against the same baseline registry and tolerance vocabulary instead of inventing evaluation on the fly.

### 2026-03-09 — Part 3 Phase 3: Leakage-Safe Dataset and Pilot Design

**Phase**: Part 3 — Phase 3 (Leakage-Safe Dataset and Pilot Design)

**What happened**: Added `src/reconstruction_dataset.py` and `tests/test_reconstruction_dataset.py` test-first to turn the synchronized comparison corpus and locked measurement layer into deterministic pilot artifacts. The new module extracts non-overlapping 128-256 word chapter windows, assigns chapter-aware train/validation/test splits, audits same-chapter and near-duplicate leakage, builds target work envelopes from train-split provenance, selects held-out source windows, and serializes explicit success criteria under operational decoupling language.

**Key decision**: The pilot should be reproducible from saved manifests rather than from ad hoc notebook state. Phase 3 therefore writes the source windows, target envelopes, split manifest, and success criteria as first-class JSON artifacts under `outputs/reconstruction/pilots/`, and it refuses to proceed if the split audit finds leakage.

**Verification**:
- `pytest tests/test_reconstruction_dataset.py -q --no-cov` passed
- full `pytest -q` passed
- full `ruff check src tests scripts` passed
- full `mypy` passed

**For Part 3**: The experiment now has a locked pilot definition before prompt-only baselines or adapter training. Later phases can run against a fixed evaluation set, fixed target envelopes, and fixed tolerance bands instead of drifting the task definition between runs.

### 2026-03-09 — Part 3 Phase 4: Prompt Baseline Driver and Revision Harness

**Phase**: Part 3 — Phase 4 (Prompt Baselines and Generate-Score-Revise Controls)

**What happened**: Added `src/reconstruction_baselines.py` and `tests/test_reconstruction_baselines.py` test-first to implement the Phase 4 prompt-only baseline harness. The new module defines versioned identity/paraphrase/style-shift/revise prompt templates, runs traceable generate-score-revise loops against the locked Phase 3 pilot, computes the weighted objective from the Phase 2 scoring contract, and persists full case histories plus summary/report artifacts inside immutable run directories under `outputs/reconstruction/runs/`.

**Key decision**: Phase 4 should keep the prompt-control contract executable even when the live control model is not currently serving. The implementation therefore supports both a real OpenAI-compatible vLLM backend and a deterministic `--dry-run` path that exercises the exact manifest, history, and artifact envelope without claiming live prompt-baseline quality.

**Current runtime baseline**:
- targeted dry run completed as `phase4-dry-run-20260309a`
- run artifacts written under `outputs/reconstruction/runs/phase4-dry-run-20260309a/`
- dry-run summary for 2 locked style-shift cases: mean weighted objective `0.5655`, stop reason `no_objective_improvement`
- local `http://localhost:8000/v1/models` is currently serving `RedHatAI/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic`, not the locked Qwen prompt-control model

**Constraint encountered**: A live Phase 4 prompt baseline should run against the Qwen vLLM service specified in the plan. During this session the host endpoint was occupied by the Nemotron replication service, so only the dry-run contract path was executed here. This is a serving-state prerequisite, not a missing Phase 4 implementation.

**Verification**:
- `pytest tests/test_reconstruction_baselines.py -q --no-cov` passed
- `python src/reconstruction_baselines.py --run-id phase4-dry-run-20260309a --dry-run --max-cases 2 --max-iterations 2` passed
- full `pytest -q` passed
- full `ruff check src tests scripts` passed
- full `mypy` passed

**For Part 3**: The experiment now has the full prompt-baseline control harness before adapter training. Once the Qwen service is active again, the same locked pilot, scoring contract, and run-manifest envelope can be used for the real no-training baseline without redefining the task.
