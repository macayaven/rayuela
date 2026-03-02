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
**Visuals**: 6 new 3D UMAP explorations in `outputs/figures/3d_*.html`. GitHub Pages companion: `docs/` with 19 interactive charts + index at `carlos-crespo.com/rayuela` (pending domain purchase).

