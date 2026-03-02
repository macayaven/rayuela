# Analysis Framework: Project Rayuela

> **Purpose**: This document makes explicit every assumption, method, and interpretive boundary in our analysis. Each phase states what it can and cannot conclude, what its null hypothesis is, and what confounds exist. If we cannot trace a conclusion back to a stated assumption and a defined method, we do not make that conclusion.

---

## Foundational Principles

### 1. Discovery First, Interpretation Second

We compute structure in the data *before* interpreting it. If we find a cluster, we first describe it numerically (which chapters, how tight, how separated from others). Only then do we ask whether it aligns with known literary structure. We never reverse this order.

### 2. Explicit Null Hypotheses

Every analysis has a stated null hypothesis — the "boring" explanation. We only consider a finding meaningful if it exceeds what the null would predict. Where possible, we quantify this with a baseline (random permutation, control corpus, or shuffled path).

### 3. Assumptions Are Stated, Not Hidden

Every tool we use (embedding model, similarity metric, dimensionality reduction) makes choices that shape results. We state those choices and, where feasible, test sensitivity to them (e.g., different prompts, different metrics).

### 4. Two Scales, Two Lenses

Scale A (texture) and Scale B (semantics) measure genuinely different things. A finding in one scale is not evidence for the other. When both scales agree, that is stronger evidence. When they disagree, that is *also* a finding — it tells us which dimension (surface vs. meaning) drives the structure.

---

## Phase 3A: Micro-Textural Embeddings

### Method

Each of the 155 chapters is embedded as a single vector using the `intfloat/multilingual-e5-large-instruct` model (1024 dimensions), run on the full chapter text (truncated to the model's 512-token window).

### Assumptions

| # | Assumption | Justification | Risk | Mitigation |
|---|---|---|---|---|
| A1 | The E5 model captures "texture" (vocabulary, syntax, rhythm) in its embedding | E5 is trained on diverse multilingual text; surface features dominate its representations for clustering tasks | The model may weight semantic content more than surface form, especially for short texts | Compare with explicit stylometric features (type-token ratio, sentence length distribution) as a sanity check in Phase 4 |
| A2 | Cosine similarity is an appropriate distance metric for comparing chapter embeddings | Standard for L2-normalized dense embeddings; invariant to vector magnitude | Other metrics (Euclidean, Manhattan) may reveal different structure | Test sensitivity: rerun key analyses with Euclidean distance |
| A3 | Truncation to 512 tokens (~350 words) captures the chapter's texture | The opening of a chapter establishes its register; consistent truncation ensures comparability | Long chapters may shift register mid-chapter; short chapters are fully captured but long ones are not | Second pass with sliding windows addresses this; also compare truncated vs. mean-pooled embeddings |
| A4 | The instruction prefix ("stylistic and thematic clustering") steers the embedding usefully | E5-instruct was trained to use task instructions to select relevant feature subspaces | Different instructions may produce meaningfully different embeddings | Experiment: rerun with 2-3 different prompts and compare cluster stability |
| A5 | The multilingual model treats code-switching naturally | E5-large-instruct was trained on multilingual data including mixed-language text | Language boundaries might still create artificial clustering | Check whether chapters with detected French/English cluster differently; compare with monolingual model |

### What This Phase CAN Tell Us

- Whether chapters form clusters based on surface-level linguistic features
- The relative textural distance between any two chapters
- Whether specific chapters are textural outliers
- Whether a reading path creates smooth or jagged transitions through texture space

### What This Phase CANNOT Tell Us

- Whether clusters correspond to *meaningful* literary categories (that requires human interpretation + Scale B comparison)
- Whether textural similarity implies thematic similarity (that's Scale B's domain)
- Whether Cortázar *intended* any patterns we find (intention requires biographical/archival evidence, not computational analysis)
- Whether the patterns are specific to Rayuela or generic to any novel (that requires the control corpus)

### Null Hypotheses

| Finding | Null Hypothesis | How to Test |
|---|---|---|
| Sections cluster together | Chapters cluster by proximity (nearby chapters are similar regardless of section) because they share context, characters, and setting | Permutation test: randomly assign chapters to 3 groups of the same sizes; do they show the same within-group similarity? |
| Linear path is smoother than hopscotch | Any sequential reading of 56 chapters from a 155-chapter pool will be smoother than a path that visits all 155, because it covers less territory | Compare hopscotch smoothness against random 155-chapter permutations; compare linear smoothness against random 56-chapter subsequences |
| **TESTED**: Both paths smoother than random | Compositional proximity (consecutive chapters written close in time share style) | Permutation test (`trajectory_baselines.py`, 10K permutations): Linear at +6.37σ (p ≈ 0), Hopscotch at +2.77σ (p < 0.003). Both paths are intentionally ordered for textural smoothness — linear more strongly so. Compositional proximity remains a confound for the linear path but cannot explain the hopscotch result (which jumps between distant chapters). |
| Chapter X is an outlier | Some chapters are short; short texts produce unreliable embeddings. The outlier may be a length artifact, not a texture anomaly | Check correlation between chapter length and mean distance to other chapters |
| Expendable chapters are more diverse | There are 99 expendable chapters vs. 56 non-expendable; larger groups have more internal variance by construction | Compare variance per chapter (mean distance to group centroid), not total group variance |

### Known Confounds

1. **Compositional proximity**: Consecutive chapters were likely written in temporal proximity, sharing the author's stylistic state. Smoothness of the linear path may reflect writing process, not design intent.

2. **Chapter length bias**: Short chapters (e.g., Ch. 79: 18 words) produce embeddings dominated by a few tokens. Long chapters are truncated to 512 tokens. These are not comparable on equal footing.

3. **Model training data**: E5 was not trained on literary text specifically. Its notion of "similarity" is weighted toward web-scale patterns, which may not align with literary-critical notions of texture.

---

## Phase 3A — Second Pass: Sliding-Window Embeddings

### Method

Chapters with >512 words are split into overlapping windows of 512 model tokens (stride 256, 50% overlap) and each window is embedded independently using the same model and instruction as the first pass. This produces a sequence of embeddings per chapter, revealing internal texture changes.

81 of 155 chapters exceeded the threshold → 888 total windows embedded.

### Assumptions

| # | Assumption | Justification | Risk | Mitigation |
|---|---|---|---|---|
| A6 | 512-token windows capture meaningful texture | This matches the model's trained sequence length; the first pass used the same window implicitly (via truncation) | Very short passages (<100 tokens) in the final window may produce less reliable embeddings | Check whether final-window artifacts inflate drift for borderline-length chapters |
| A7 | 50% overlap (stride 256) provides sufficient resolution for detecting register shifts | Standard in NLP sliding-window analysis; each token appears in ~2 windows | Finer shifts (within a single paragraph) may fall below resolution | Can re-run with stride 128 for high-priority chapters |
| A8 | "Internal drift" (1 - mean consecutive similarity) is a valid measure of intra-chapter texture variation | Directly analogous to trajectory smoothness from Phase 3A first pass, applied within chapters instead of between them | Short chapters with 3-4 windows can show extreme drift from a single noisy transition (small-sample effect) | Report drift alongside window count; weight or flag chapters with n_windows < 5 |
| A9 | Tokenize → split → decode → re-encode preserves embedding fidelity | The round-trip through the tokenizer may introduce minor artifacts at window boundaries (split subwords, lost whitespace) | Could create artificial dissimilarity at boundaries | Overlap mitigates this: boundary artifacts in one window are interior tokens in the adjacent window |

### What This Pass CAN Tell Us

- Whether specific chapters contain measurable internal texture shifts
- *Where* in a chapter the largest register change occurs (max jump position)
- Whether long chapters end in a different texture than they start (overall span)
- Whether narrative mode transitions (mimetic → diegetic) produce detectable texture signatures

### What This Pass CANNOT Tell Us

- *What* the texture shift consists of (vocabulary change? syntax shift? language switch?)
- Whether the shift is literarily significant (a high-drift chapter may be unremarkable; a low-drift chapter may be artistically subtle)
- Whether observed drift patterns are specific to Cortázar or generic to any novelist

### Key Results

| Finding | Result | Interpretation |
|---|---|---|
| Carlos predicted Ch. 36 shows no special texture transition | Ch. 36 drift = 0.0237 (rank 41/81, at the median) | **Partially confirmed**: Ch. 36 is texturally average, not an outlier. Cortázar's authorial voice is consistent enough that register shifts (dialogue → description) don't produce extreme drift. |
| Ch. 34 (interleaved two-column) max jump at 92% | Max jump = 0.0474 near chapter's end, not at interleaving boundaries | The interleaving maintains a composite texture; the shift happens at the close, likely where narrative mode changes from mimetic (showing) to diegetic (telling). |
| Ch. 23 (Club de la Serpiente) max jump at 96% | Max jump = 0.0342 at the very end | Consistent with Ch. 34 pattern: the long interactive scene shifts texture at its conclusion when the narrative summarizes outcomes. |
| Ch. 41 (Berthe Trépat concert) max jump at 8% | Max jump = 0.0435 near the start | The chapter opens in a different mode before settling into the extended scene. |
| Short chapters dominate high-drift rankings | Ch. 132 (3 windows) tops drift chart at 0.0354 | Small-sample effect: chapters with few windows are more susceptible to single-transition noise. Confound A8. |
| Longest chapters have lower drift | Ch. 28 (74 windows): 0.0228; Ch. 56 (50 windows): 0.0218 | Regression to the mean: more windows → more averaging. This is expected but means drift comparisons across very different chapter lengths require normalization. |

### Null Hypotheses (Second Pass)

| Finding | Null Hypothesis | How to Test |
|---|---|---|
| High-drift chapters have real texture shifts | Short chapters produce high drift due to small-sample variance, not genuine texture change | Correlate drift with n_windows; compare only chapters with >10 windows |
| Max jump position at chapter end indicates mimetic→diegetic shift | Window boundary artifacts concentrate at chapter edges (start/end of text may have different token patterns) | Check distribution of max_jump_position across all chapters; if uniformly distributed, end-concentration is real signal |
| Overall span measures genuine start-to-end texture change | The first and last windows of long chapters are simply far apart in sequence, and any long text drifts by random walk | Compare observed span against mean span of random 512-token samples from the same chapter |

### Known Confounds (Second Pass)

4. **Small-sample effect**: Chapters with 3-4 windows can show extreme drift from a single unusual transition. Rankings mixing 3-window and 74-window chapters are not apples-to-apples.

5. **Tokenizer round-trip artifacts**: Splitting token IDs at window boundaries may occasionally break subword units, introducing slight differences when decoded and re-encoded. The 50% overlap mitigates this (boundary tokens in one window are interior tokens in the next).

6. **Regression to the mean in long chapters**: Chapters with many windows will tend toward the population mean drift simply because they sample more of the distribution. This means long chapters appearing "low-drift" may not indicate genuine textural consistency.

---

## Phase 3B: Semantic Profiling (Narrative DNA)

### Method

Each of the 155 chapters is scored on 20 semantic dimensions (integer scale 1–10) by **Qwen 3.5 27B FP8** via vLLM. A structured system prompt defines each dimension with explicit anchors (1 = low, 10 = high). The model receives the full chapter text (up to ~24K tokens) and returns a JSON object with all 20 scores, constrained by vLLM's JSON schema-guided decoding.

**Model**: `Qwen/Qwen3.5-27B-FP8` (dense, 27B parameters, FP8 quantization ~27 GB)
**Server**: vLLM 0.16.1rc1 with `--language-model-only --enable-prefix-caching`
**Prompt**: `prompts/semantic_extraction_v1.txt` (versioned)
**Output**: 155 × 20 matrix (integer scores) + JSON metadata

### The 20 Dimensions

Organized into 5 groups:

| Group | Dimensions |
|---|---|
| **Thematic** | existential_questioning, art_and_aesthetics, everyday_mundanity, death_and_mortality, love_and_desire |
| **Emotional** | emotional_intensity, humor_and_irony, melancholy_and_nostalgia, tension_and_anxiety |
| **Character** | oliveira_centrality, la_maga_presence, character_density, interpersonal_conflict |
| **Narrative mode** | interiority, dialogue_density, metafiction |
| **Formal/experimental** | temporal_clarity, spatial_grounding, language_experimentation, intertextual_density |

**Why these 20?** They were chosen to cover: (a) Cortázar-specific themes (existential questioning, art/aesthetics, metafiction), (b) general narrative dimensions any novel has (emotion, character, setting), (c) Rayuela-specific features (language experimentation, intertextual density, the Oliveira/La Maga axis). The mix of Cortázar-specific and generic dimensions lets us compare Rayuela against the control corpus on shared dimensions.

### Assumptions

| # | Assumption | Justification | Risk | Mitigation |
|---|---|---|---|---|
| B1 | Qwen 3.5 27B can reliably score Spanish literary text on nuanced dimensions | Qwen 3.5 is multilingual with strong performance on Spanish; the scoring task is well-defined with explicit rubric | The model may have biases toward certain score ranges, or may misunderstand Cortázar's more experimental passages (Glíglico, stream of consciousness) | Check score distributions: if all dimensions cluster around 5, the model may be hedging. Compare against Carlos's manual scores (CHECKPOINT B1). |
| B2 | 20 dimensions are sufficient to capture the semantic profile of each chapter | 20 covers the five major facets (theme, emotion, character, mode, form); additional dimensions risk correlation without new information | Important semantic aspects may be missed (e.g., political content, gender dynamics, philosophical schools) | Review the 20-dim vectors after Phase 4 (UMAP): if chapters that should differ cluster together, add discriminating dimensions in v2 |
| B3 | Integer 1–10 scores provide meaningful granularity | 10 levels allow differentiation without false precision; the model can reliably distinguish "low" from "medium" from "high" | Adjacent scores (e.g., 6 vs 7) may not be meaningfully different; the model may not use the full range | Report score distributions per dimension. If variance is low, the scale may need recalibration. Treat scores as ordinal, not interval. |
| B4 | JSON-schema-guided decoding produces the same scores as unconstrained generation | Guided decoding constrains token selection to valid JSON, but should not change the model's semantic judgments | Constraint on the first token ("{\") may slightly alter the generation trajectory | Test: extract 10 chapters with and without guided decoding and compare scores (deferred to sensitivity analysis) |
| B5 | Scoring based on chapter text alone (no cross-chapter context) is appropriate | Each chapter is a self-contained unit of the novel; scoring in isolation avoids context contamination | Some chapters depend heavily on context from previous chapters; the model may score based on its pre-training knowledge of Rayuela | The prompt explicitly instructs: "Score what the TEXT contains, not what you know about the novel from outside this chapter." |
| B6 | FP8 quantization does not degrade scoring quality vs. BF16 | FP8 compresses from 16-bit to 8-bit, ~2× memory savings; for structured scoring tasks, the precision loss is minimal | Extreme quantization could affect nuanced literary judgment | Could compare a subset against the BF16 variant (if memory allows) or against a different model |
| B7 | Temperature=0 produces the most consistent scores | Greedy decoding eliminates randomness, making results deterministic and reproducible | The model may be more accurate with slight temperature (0.1-0.3) allowing it to "deliberate" | For reproducibility, keep temperature=0 in the primary run; test temperature sensitivity in a future pass |

### What This Phase CAN Tell Us

- The semantic "fingerprint" of each chapter across 20 dimensions
- Which chapters are semantically similar (cosine similarity in 20-dim space) vs. texturally similar (Scale A)
- Whether the novel's sections differ semantically (not just texturally)
- Dimensional profiles: which dimensions dominate which chapters/sections
- Whether Scale A and Scale B agree or diverge on chapter similarity

### What This Phase CANNOT Tell Us

- Whether the 20 dimensions fully capture a chapter's meaning (they are a projection, not a complete representation)
- Whether the LLM scores agree with human literary judgment (requires CHECKPOINT B1 validation)
- The "correct" semantic profile of a chapter (different models, prompts, or rubrics would produce different profiles)
- Whether high scores on a dimension indicate literary quality (high existential_questioning is not "better" than low)

### Null Hypotheses

| Finding | Null Hypothesis | How to Test |
|---|---|---|
| Sections have different semantic profiles | Score differences reflect chapter length, not section identity (expendable chapters are shorter and may score differently due to length) | Compare distributions controlling for chapter length; permutation test on section labels |
| Scale A and Scale B agree on chapter clustering | Both scales capture the same underlying feature (chapter length or vocabulary size) rather than complementary features | Check correlation between A and B similarity matrices; if r > 0.9, they may be redundant |
| Specific dimensions differentiate sections | Any 20 random features would show some significant differences by chance across 3 groups | Bonferroni correction for 20 tests × 3 comparisons; also check effect sizes, not just p-values |
| The LLM scores capture meaningful literary features | The model assigns scores based on surface cues (word frequency) rather than genuine semantic understanding | Compare against a bag-of-words baseline: if keyword counts predict scores with r > 0.8, the LLM may not add value beyond keyword matching |

### Known Confounds

7. **LLM pre-training knowledge**: Qwen was likely trained on analysis of Rayuela. Its scores may reflect learned literary criticism rather than independent assessment of the text. Mitigation: the prompt instructs to score only what the text contains, and guided decoding limits output to scores only.

8. **Score range compression**: LLMs tend to avoid extreme scores (1 or 10), compressing the effective range. If most scores fall in 4–7, the signal-to-noise ratio decreases. Monitor: check score distributions after full extraction.

9. **Dimension correlation**: Some dimensions may be inherently correlated (e.g., love_and_desire and la_maga_presence). High correlation is not a confound per se, but it means the effective dimensionality is less than 20. Measure: compute the correlation matrix and effective rank.

10. **Thinking mode leakage**: Qwen 3.5 has a "thinking" mode that generates internal reasoning before producing output. With JSON-schema-guided decoding, this is suppressed, but it may subtly affect the generated scores compared to free-form generation.

### Future Extension: Evidence Extraction

A planned second pass will extract explicit text evidence for each score — verbatim quotes and identified elements (characters, locations, time markers) that justify each score. This structured evidence would enable:
- Score auditability (verify the model's reasoning)
- Generative remixing (replace period-specific elements to create "Rayuela for a new generation")
- Fine-grained literary element database

---

## Phase 4: Dimensionality Reduction & TDA

> To be completed when we begin Phase 4.

### Assumptions to Document

- UMAP hyperparameter choices and their effect on visual structure
- Whether topological features are stable across parameter ranges
- The distinction between features of the data vs. artifacts of the method

---

## Phase 5: Path Analysis & Trajectory

> To be completed when we begin Phase 5.

### Assumptions to Document

- Definition of "smooth" and "rough" trajectories
- Baseline for comparison (random paths)
- Statistical significance of trajectory differences

---

## Interpretation Boundaries

### We Will Say

- "Chapters X, Y, Z form a cluster in texture space" (factual, about the data)
- "The linear path has higher mean consecutive similarity than the hopscotch path" (factual, about the metric)
- "This is consistent with the hypothesis that..." (explicit about the inferential step)

### We Will NOT Say

- "Cortázar designed the hopscotch path to..." (we cannot infer intention from computation)
- "The novel's structure is..." (our analysis captures one representation, not the structure itself)
- "The embedding model proves that..." (the model is a lens, not a proof system)

---

*This document is updated as we enter each new phase. Assumptions discovered during analysis are added retroactively with a note.*

*Last updated: 2026-02-28 — Phase 3A complete*
