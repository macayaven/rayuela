# What Does a Novel Look Like From the Inside?

## Using AI to Map the Hidden Structure of Cortázar's *Rayuela*

---

*In 1963, Julio Cortázar published a novel with two reading orders. Sixty years later, we fed it to an AI and asked: does the machine see what the author designed?*

---

### 1. The Question

In the first pages of *Rayuela* — before you read a single chapter — Cortázar gives you a choice. You can read the novel straight through, chapters 1 to 56, like any other book. Or you can follow the *Tablero de Dirección*, a hopscotch path that leaps between all 155 chapters in an order the author prescribes: 73, 1, 2, 116, 3, 84, 4, 71...

Readers and scholars have argued about these two paths for six decades. Does the hopscotch order reveal deeper connections? Is it designed for thematic continuity, or for something else entirely? Nobody had the tools to answer this empirically — until now.

We took the full text of *Rayuela*, converted every chapter into a high-dimensional mathematical representation using AI embedding models, then traced both reading paths through that mathematical space. We measured the texture of the prose, the semantic content of each chapter, and the statistical significance of each reading order against thousands of random alternatives.

What we found surprised us.

> **`[Figure: umap_scale_a.html]`** — *This is what Rayuela looks like from the inside: 155 chapters projected from 1024 dimensions into two. Each dot is a chapter. The colors represent the novel's three sections. The lines trace the two reading paths through this space.*

Before writing a single line of analysis code, we had to make a decision that would shape everything: how do you turn Spanish prose — prose that slips into French mid-sentence, that invents words, that breaks every rule — into numbers a computer can compare? The choice of embedding model is the choice of what "similarity" means. We chose a multilingual model (`intfloat/multilingual-e5-large-instruct`) specifically because Cortázar code-switches between Spanish, French, and English. A monolingual model would have created artificial boundaries where Cortázar saw none.

But turning prose into numbers is only half the battle. The other half is making sure you don't accidentally encode the answer inside the question.

---

### 2. The Surgery

Before we could feed a single word to the AI, we had to perform surgery on the text itself.

Every chapter of *Rayuela* ends with a small number in parentheses — a signpost telling the reader where to jump next in the hopscotch sequence. Leave those numbers in, and any pattern the AI finds would be a forgery: we'd have embedded the answer inside the question. The technical term is **circular analysis**; the human term is cheating.

We also found a ghost. Chapter 55 is the only chapter Cortázar's hopscotch path never visits — a phantom chapter that exists only if you read the book the "wrong" way, straight through. And we discovered that the OCR scan of the novel we started with was riddled with systematic errors: corrupted chapter headers, digit substitutions, missing markers. We switched to a clean ePub source where each chapter is a separate file, navigation markers are reliably tagged, and there are zero OCR artifacts.

This kind of data hygiene is invisible to the reader of the final analysis, but it's where computational literary studies succeed or fail. If you don't strip the editorial apparatus, you find structure that the editor created, not the author.

---

### 3. The Surface: What the Texture of Language Reveals

The first scale of analysis captures what we call **texture**: the surface-level rhythm, vocabulary, and syntax of Cortázar's prose. We used a multilingual embedding model to convert each chapter into a 1024-dimensional vector — think of it as a fingerprint of the chapter's linguistic surface.

The first thing the AI told us was something Cortázar scholars might find surprising: **the three parts of *Rayuela* don't sound different.** "Del lado de allá" (Paris) and "Del lado de acá" (Buenos Aires) and the expendable chapters — they all share the same textural DNA. The divisions readers feel are about *what* is being said, not *how*.

> **`[Figure: umap_comparison.html]`** — *Left: chapters in texture space (1024 dimensions). Right: chapters in narrative space (20 dimensions). Notice how sections overlap in texture but separate in meaning.*

But trace the two reading paths through this texture space, and something else appears: the linear path glides; the hopscotch path jolts. We quantified this by measuring the distance between consecutive chapters along each path, then testing both against 10,000 random orderings of the same chapters.

The result was striking. The linear path is smoother than all 10,000 random orderings — a +6.37σ outlier, statistically impossible by chance. The hopscotch path is smoother than 99.7% of random orderings (+2.77σ). Both paths are intentionally ordered for textural continuity.

> **`[Figure: article_smoothness.html]`** — *Step-by-step distances along each reading path. The linear path (green) traces small, regular jumps. The hopscotch path (red) is wilder but still smoother than random.*

We then zoomed in further. For the 81 chapters over 512 words, we slid a 512-token window across the text and measured how texture shifts *within* each chapter. The result was surprisingly uniform: Cortázar's voice holds. Even in chapters where the content shifts dramatically — from philosophical debate to intimate confession, from dialogue to description — the surface of the language stays consistent. What changes at the end of his great ensemble chapters isn't the voice but the **mode** — from showing to telling, from the heat of the scene to the cool summary of its aftermath. The AI can see the moment the camera pulls back.

---

### 4. Teaching a Machine to Read Like a Critic

Texture tells us how the prose *sounds*. But to understand what it *means*, we needed a different instrument.

We designed a 20-dimensional scoring rubric — five facets of the novel, each broken into measurable dimensions:

| Facet | Dimensions |
|-------|-----------|
| **Thematic** | Existential questioning, Art & aesthetics, Everyday mundanity, Death & mortality, Love & desire |
| **Emotional** | Emotional intensity, Humor & irony, Melancholy & nostalgia, Tension & anxiety |
| **Character** | Oliveira centrality, La Maga presence, Character density, Interpersonal conflict |
| **Narrative** | Interiority, Dialogue density, Metafiction, Temporal clarity |
| **Formal** | Spatial grounding, Language experimentation, Intertextual density |

A 27-billion-parameter language model (Qwen 3.5 27B) scored every chapter on every dimension, producing a **narrative DNA** profile — a 20-number fingerprint that captures the semantic essence of each chapter.

The validation test was Chapter 68 — the famous Glíglico chapter, written entirely in a language Cortázar invented. The model scored it: language_experimentation = 10, love_and_desire = 9, spatial_grounding = 1, dialogue_density = 1. It understood that a 186-word passage in a made-up language is about love, not space, not dialogue, and that the language itself *is* the experiment. The machine couldn't read the invented language (no one can), but it understood what the invented language was *for*.

> **`[Figure: article_radar.html]`** — *Narrative fingerprints of notable chapters. Each spoke is one of the 20 dimensions. Chapter 68 (Glíglico) makes a distinctive star shape — almost all its energy concentrated in two dimensions. Compare it to Chapter 1 (the expansive opening) or Chapter 93 (Morelli's literary theory).*

> **`[Figure: article_heatmap.html]`** — *The full 155 × 20 heatmap: every chapter, every dimension. Red means high, blue means low. You can see the expendable chapters (57–155) diversify dramatically — they are the laboratory of the novel.*

The 20 dimensions also reveal what's structurally different about the three sections:

- **Paris (Allá)**: Dominated by Oliveira's interiority (8.6), existential questioning (7.9), and La Maga's presence. This is the philosophical core of the novel.
- **Buenos Aires (Acá)**: Similar profile but lower La Maga presence (she's absent), more spatial grounding, more dialogue. The real world pressing in.
- **Expendable chapters (Otros lados)**: Break free of Oliveira's dominance. Intertextual density rises; metafiction spikes. This is where Cortázar experiments most freely.

> **`[Figure: article_sections.html]`** — *Box plots showing how each dimension distributes across the three sections. The top dimensions that differentiate the sections are La Maga presence, character density, and dialogue.*

> **`[Figure: article_correlation.html]`** — *Which dimensions co-occur? This correlation matrix reveals the "grammar" of Cortázar's writing modes. Oliveira centrality and interiority are tightly linked (r ≈ 0.7). Humor and existential questioning are surprisingly independent.*

---

### 5. The Twist: Cortázar Designed Collision, Not Continuity

Here is where the story turns.

Remember the texture-only finding from Section 3: the hopscotch path was smoother than 99.7% of random orderings. When we tested it with only the 1024-dimensional surface features, the hopscotch path looked intentionally designed for textural flow — rougher than the linear path, but still meaningfully smooth.

**That finding was an illusion.**

When we ran the same permutation test with the full 20-dimensional narrative DNA — capturing not just how the prose sounds but what it means — the hopscotch path's smoothness advantage vanished entirely.

The numbers, from 5,000 random permutations:

| Path | Scale A (Texture) | Scale B (Narrative) |
|------|-------------------|---------------------|
| **Linear (1→56)** | z = −7.7σ (smoother than all) | z = −8.6σ (smoother than all) |
| **Hopscotch (Tablero)** | z = −0.6σ (indistinguishable from random) | z = +0.4σ (indistinguishable from random) |

The linear reading of *Rayuela* is like a river: each chapter flows naturally into the next, both in sound and meaning. The hopscotch reading is like channel-surfing: Paris, Buenos Aires, philosophical essay, love scene, literary theory, grocery list — the Tablero deliberately scrambles everything.

> **`[Figure: article_permutation.html]`** — *The hero figure. The grey histogram shows the distribution of 5,000 random reading orders. The green line marks where the linear path falls (far left — extremely smooth). The red line marks the hopscotch path (dead center — indistinguishable from random). This is true in both texture space and narrative space.*

Cortázar didn't design the hopscotch for continuity. He designed it for **collision**. The jarring transitions are the point. You're not supposed to settle in. You're supposed to be thrown, again and again, from one register into another — to feel, in the reading itself, the disorientation of Oliveira wandering between cities and selves.

The weaving pattern makes this visible. In the linear reading, you progress through Paris (chapters 1–36), then Buenos Aires (37–56), in clean blocks. In the hopscotch reading, Cortázar interleaves all three sections from the very first step — Paris, expendable, Buenos Aires, expendable, Paris — deliberately fragmenting the reader's sense of location and time.

> **`[Figure: article_weaving.html]`** — *Section color strips. Top: the linear reading (solid blocks of blue, then orange). Bottom: the hopscotch reading (a kaleidoscope of alternating colors). The hopscotch path is designed to never let you stay in one world for long.*

> **`[Figure: article_dual_heatmap.html]`** — *The same 20-dimensional data, reordered by the two reading paths. The linear heatmap shows smooth gradients — similar chapters next to each other. The hopscotch heatmap shows rapid alternation — maximally different chapters placed side by side.*

The two reading experiences produce genuinely different emotional arcs. Trace any single dimension — say, tension and anxiety — along both paths, and you see: the linear path builds and releases in long waves; the hopscotch path spikes and drops in an arrhythmic pulse.

> **`[Figure: article_journey.html]`** — *Emotional arc sparklines: six key dimensions traced along both reading paths. The linear path (top) builds in waves. The hopscotch path (bottom) is stochastic noise.*

> **`[Figure: article_dual.html]`** — *Detailed dimension-by-dimension comparison: 8 key dimensions × 2 reading paths. Raw scatter (light dots) + smoothed trend lines. The structural difference is visible in every dimension.*

The cross-scale correlation between texture and narrative is ρ = 0.505 — the two measurement scales capture partially overlapping but substantially different structure. This means that the texture of the prose and the meaning of the prose are related but not redundant: chapters that sound similar often (but not always) mean similar things.

---

### 6. [PLACEHOLDER] The Evidence Layer: What the Machine Actually Sees

> *This section will be completed when the v2 extraction finishes (~155 chapters with evidence strings).*

Our first extraction was zero-shot: the model scored each chapter cold, with no examples. Our second extraction used **few-shot calibration** — three carefully chosen chapters from other Cortázar novels (*62: Modelo para armar* and *Un tal Lucas*) scored by hand and included as reference examples. This prevents the model from overfitting to Rayuela's internal norms: it has an external anchor for what "tension = 10" or "humor = 1" actually looks like.

The v2 extraction also asks the model to provide **one sentence of textual evidence** for each score — a citation from the chapter that justifies the rating. This turns the scoring from a black box into an auditable system.

**Early comparison (20/155 chapters)**:
- 87% of scores are within ±1 point between v1 and v2
- 98% within ±2
- The main effect: v2 scores average 0.48 points lower — the calibration examples taught the model what a "real 1" looks like, reducing ceiling compression
- 17 of 20 dimensions correlate at r > 0.65 between the two versions
- One problematic dimension: `temporal_clarity` (r = −0.30) — the model interprets this differently across prompt versions

**[TODO: Full v1 vs v2 comparison with 155 chapters]**

**[TODO: Example evidence strings for notable chapters — Ch. 1 (Oliveira's existential monologue), Ch. 68 (Glíglico), Ch. 36 (the bridge), Ch. 93 (Morelli's literary theory)]**

**[TODO: What the evidence strings reveal about the model's "reasoning" — does it cite the right passages? Does it understand subtext?]**

Here is a taste of what the evidence looks like. For Chapter 1, the model wrote:

- **Existential questioning (8)**: *"Oliveira reflects on the nature of chance encounters, the meaning of searching, and the 'signs' that govern his life, questioning the logic of existence."*
- **Love and desire (9)**: *"The entire chapter is an address to La Maga, exploring their relationship, intimacy, and the 'terrible mirror' of their love."*
- **Spatial grounding (8)**: *"Specific Parisian locations: Pont des Arts, Quai de Conti, Marais, boulevard de Sébastopol, Parc Montsouris, rue des Lombards."*
- **Melancholy and nostalgia (8)**: *"Deep nostalgia for Buenos Aires (shoes, tea, mother) and the lost moments with La Maga suffuse the text."*

These evidence strings are not just validation — they are the raw material for a new kind of literary analysis. When the model tells you *why* it scored a chapter, you can ask: what if we changed the evidence?

---

### 7. [PLACEHOLDER] The Smoothest Way to Read *Rayuela*

> *This section requires a computational optimization that will be run next.*

If the linear reading order is the smoothest of all possible orderings — smoother than 99.99% of random permutations — is it *the* smoothest? Or could we find a reading order that is even smoother?

We can answer this computationally. Using a greedy nearest-neighbor algorithm (a heuristic for the traveling salesman problem), we constructed:

1. **The smoothest texture path**: Starting from Chapter 1, always jump to the most texturally similar unvisited chapter (Scale A, cosine distance).
2. **The smoothest narrative path**: Starting from Chapter 1, always jump to the most semantically similar unvisited chapter (Scale B, Euclidean distance).

**[TODO: The optimal reading orders for Scale A and Scale B]**

**[TODO: How these compare to the actual linear and hopscotch paths — are they smoother? By how much?]**

**[TODO: What chapters does the algorithm choose to follow each other? Does the AI-optimized reading order make literary sense? Does it accidentally reconstruct the linear order, or does it discover a third path through the novel that neither Cortázar nor any human reader has traced?]**

**[TODO: Visualization of the three paths — linear, hopscotch, and AI-optimal — through UMAP space]**

This is a parlor trick with a serious point. If the AI-optimal path resembles the linear order, it confirms that Cortázar organized his first 56 chapters with extraordinary care. If it departs significantly, it suggests there are hidden affinities between chapters that the linear order ignores — connections that only become visible when you let a machine read without the constraint of narrative sequence.

---

### 8. [PLACEHOLDER] Remixing *Rayuela*: Rewriting the Novel Across Generations

> *This section depends on the v2 evidence extraction being complete.*

The evidence strings from our v2 analysis do something no previous computational literary study has done: they decompose each chapter into a set of **identifiable elements** — the specific locations, cultural references, emotions, and character dynamics that the model cites as justification for each score.

This decomposition opens a radical possibility: **generative remixing**.

Consider Chapter 1. The model identifies:
- **Spatial anchors**: Pont des Arts, Quai de Conti, Marais, boulevard de Sébastopol
- **Cultural references**: Braque, Ghirlandaio, Max Ernst, Klee, Harold Lloyd, Fritz Lang
- **Emotional register**: nostalgia for Buenos Aires, longing for La Maga, existential anxiety
- **Everyday texture**: eating hot dogs, riding bicycles, smoking on trash heaps

What if we kept the emotional register and narrative structure intact but *substituted the entities*? Replace 1950s Paris with 2020s Mexico City. Replace the jazz clubs with electronic music warehouses. Replace the Old Masters with Instagram artists. Replace the Pont des Arts with the Puente de la Mujer.

**[TODO: Select 2-3 short subchapters and perform entity substitution using the evidence strings as a map]**

**[TODO: Score the rewritten passages with the same 20-dimension rubric — do they preserve the narrative DNA?]**

**[TODO: What changes and what stays the same? Which dimensions are robust to entity substitution (probably: interiority, emotional intensity) and which collapse (probably: spatial grounding, intertextual density)?]**

The hypothesis is that Cortázar's genius lives in the *structure* of his chapters — the interplay between interiority and spatial grounding, between existential questioning and everyday mundanity — not in the specific cultural furniture of 1960s Paris. If we can swap the furniture and keep the structure, we've identified what is essential to the novel and what is contingent.

This is not a claim that the remix is as good as the original. It is an experiment in literary decomposition: separating the skeleton from the skin.

---

### 9. The Shape of the Novel

There is one more scale of analysis we haven't discussed: topology.

**Topological Data Analysis (TDA)** asks a different question than clustering or dimensionality reduction. Instead of "which chapters are near each other?", it asks: "what is the *shape* of the space these chapters inhabit?" Are there holes, loops, or voids in the narrative landscape? Are there paths through the novel that circle back on themselves?

**[TODO: TDA results — persistence diagrams, Betti numbers, and what they reveal about the topological structure of Rayuela's embedding space. This analysis is planned for Phase 5.]**

The tools exist: `ripser` for persistent homology, `persim` for diagram distances. The question is whether the shape of *Rayuela*'s chapter space differs meaningfully from the shape of a conventional linear novel — and whether the hopscotch path traces a topologically interesting trajectory through that space.

---

### 10. Reflection: What AI Literary Analysis Can and Can't Tell Us

The central finding of this project is a single sentence: **Cortázar designed the hopscotch reading order for semantic collision, not for continuity.** The jarring transitions between chapters in the Tablero de Dirección are not a flaw or a side effect — they are the design. The reader who follows the hopscotch path is meant to experience the disorientation of Oliveira himself, wandering between cities and selves and modes of being.

This is not a new insight in literary criticism. Scholars have argued versions of this for decades. What's new is the evidence: a statistical test, reproducible, against 5,000 random orderings, across two independent measurement scales, showing that the linear path is −8.6σ smoother than chance while the hopscotch path sits at z ≈ 0.

But we should be precise about what the AI can and cannot tell us.

**What it can tell us**: Whether a reading order is statistically smoother or rougher than chance. Whether chapters cluster by section. Which dimensions of meaning co-occur. Where the biggest register shifts happen within a chapter. Whether two chapters are more similar in texture, in meaning, or in both.

**What it cannot tell us**: Whether the collision is *aesthetically successful*. Whether the disorientation the reader feels is productive or merely annoying. Whether Cortázar was conscious of the statistical properties we measured, or whether his literary intuition produced them emergently. Whether *Rayuela* is a great novel.

The AI is a microscope, not a judge. It reveals structure that was always there but invisible to the unaided eye. What we do with that structure — whether we find it beautiful, or significant, or worth arguing about — remains irreducibly human.

---

### Methodology Note

This project analyzed 155 chapters of *Rayuela* (Julio Cortázar, 1963) at two independent scales:

- **Scale A (Texture)**: 1024-dimensional embeddings using `intfloat/multilingual-e5-large-instruct`, a multilingual sentence transformer. Cosine similarity captures surface-level linguistic resemblance.
- **Scale B (Narrative DNA)**: 20-dimensional semantic profiles extracted by `Qwen/Qwen3.5-27B-FP8` (27 billion parameters), scoring each chapter on thematic, emotional, character, narrative, and formal dimensions.

Statistical significance was established via permutation testing (5,000–10,000 random orderings). Cross-scale correlation (Spearman ρ = 0.505) confirms the two scales capture substantially overlapping but not redundant structure.

All code and data (excluding the copyrighted novel text) are available at [github.com/macayaven/rayuela](https://github.com/macayaven/rayuela).

The analysis was conducted on an NVIDIA DGX Spark (ARM64, GB10 GPU, 128 GB unified memory) with a two-service architecture: an analysis container for embeddings and visualization, and a separate vLLM container for LLM inference.

---

*Carlos Crespo Macaya & Claude · 2026*
