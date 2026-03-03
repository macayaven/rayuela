# How Do You Teach a Machine to Read Like a Critic?

## The Methodology Behind Our AI Analysis of Cortazar's Rayuela

*Carlos Crespo Macaya & Claude · 2026*

*This is Part 2 of a series. [Part 1: What Does a Novel Look Like From the Inside?](LINK_TO_PART1) presented the finding: Cortazar designed the hopscotch reading order for collision — collision-dominant, but with a formal undertow that the AI detects but cannot name. This article goes deeper into how we built the four instruments that revealed it — and what happens when you probe their limits.*

---

In Part 1, we made a claim: the hopscotch reading path through *Rayuela* is near-random in its explicit semantic transitions and actively jagged in its curvature, while the linear path is smoother than chance to an extraordinary degree. That claim rests on four complementary instruments — a holistic lens, a content-free stylometric lens, an LLM stylistic lens, and a semantic lens — and a statistical test that compares both reading orders against 5,000 random alternatives.

This article pulls those instruments apart and shows you what's inside.

## The 20-Dimensional Rubric

The "semantic lens" from Part 1 is actually a structured scoring system. A 27-billion-parameter language model (Qwen 3.5 27B) read each of the 155 chapters and scored it on 20 dimensions, grouped into five facets:

**Thematic**: existential questioning, art & aesthetics, everyday mundanity, death & mortality, love & desire

**Emotional**: emotional intensity, humor & irony, melancholy & nostalgia, tension & anxiety

**Character**: Oliveira centrality, La Maga presence, character density, interpersonal conflict

**Narrative mode**: interiority, dialogue density, metafiction, temporal fragmentation

**Formal**: spatial grounding, language experimentation, intertextual density

Each dimension is scored 1-10 with an explicit rubric. For example, *humor and irony* = 1 means "completely serious"; = 10 means "pervasive humor, satire, or ironic play." The rubric was designed before any model ran — we didn't tune it to produce interesting results.

We call the resulting 20-number profile of each chapter its **narrative DNA**.

![The full 155-chapter x 20-dimension heatmap: every chapter scored on every dimension.](figure6_heatmap.png)
*Figure 6 — Narrative DNA. Each row is a chapter; each column is a dimension. Red means high, blue means low. The expendable chapters (57-155) diversify dramatically — they are the laboratory of the novel. ([Interactive version](https://carloscrespomacaya.com/rayuela/article_heatmap.html))*

## The Gliglico Test

Every instrument needs calibration. Ours had Chapter 68.

Chapter 68 is the famous Gliglico chapter — 186 words written entirely in a language Cortazar invented. Nobody can "read" it in the conventional sense. The words are nonsense: *"Apenas el le amalaba el noema, a ella se le agolpaba el clemiso y caian en hidromurias..."*

If our scoring model understood literature and not just language, it should recognize that Gliglico is about love (the invented words mime a sexual encounter) and that it is the most radical experiment in the novel.

The model scored it:
- **Language experimentation**: 10 — *"The entire text is written in Gliglico, a private invented language with neologisms and no standard Spanish meaning."*
- **Love and desire**: 8 — *"The context of the invented language in Rayuela is known to describe sexual intimacy, and phrases like 'aproximara suavemente sus orfelunios' suggest erotic contact."*
- **Spatial grounding**: 1 — *"No clear physical setting is described; the text is abstract and focused on sensation."*
- **Dialogue density**: 1 — *"No dialogue is present; the text is entirely narrative description."*

The model couldn't read the invented language. But it understood what the invented language was *for*. That's the kind of comprehension a literary scoring system needs.

## What the Novel Looks Like in 20 Dimensions

The heatmap reveals structure that no human reader could hold in mind simultaneously.

The three sections of the novel — Paris ("Del lado de alla"), Buenos Aires ("Del lado de aca"), and the expendable chapters ("De otros lados") — differ in characteristic ways:

- **Paris**: dominated by Oliveira's interiority, existential questioning, and La Maga's presence. This is the philosophical and emotional core.
- **Buenos Aires**: similar emotional register but lower La Maga presence (she's absent), more spatial grounding, more dialogue. The real world pressing in.
- **Expendable chapters**: break free of Oliveira's dominance. Intertextual density rises; metafiction spikes; humor becomes more prevalent. This is where Cortazar experiments most freely.

![Box plots showing how each dimension distributes across the three sections.](figure8_sections.png)
*Figure 8 — Section Profiles. The dimensions that most differentiate the three sections are La Maga presence, character density, and dialogue density. ([Interactive version](https://carloscrespomacaya.com/rayuela/article_sections.html))*

The dimensions also reveal a hidden grammar — which traits co-occur in Cortazar's writing. Oliveira's centrality and interiority are tightly linked (r ~ 0.7): when Oliveira dominates, the prose turns inward. Humor and existential questioning are surprisingly independent: Cortazar can be deeply philosophical and deeply funny at the same time.

![Dimension correlation matrix.](figure9_correlation.png)
*Figure 9 — Dimension Correlation Matrix. The "grammar" of Cortazar's writing modes. ([Interactive version](https://carloscrespomacaya.com/rayuela/article_correlation.html))*

## Completing the Gradient: Scales A' and B'

Part 1 presented four scales. Here we explain why the two intermediate scales — A' (content-free stylometrics) and B' (LLM-perceived style) — exist, and what they reveal.

**Scale A' (Stylometric)** is our strongest methodological weapon. It measures 26 features that are, by construction, immune to meaning: sentence length distribution (mean, median, standard deviation, maximum, coefficient of variation), vocabulary richness (MATTR with 50-word window, hapax legomena ratio), function word frequencies (articles, prepositions, conjunctions, pronouns per 1,000 words), punctuation profile (semicolons, em-dashes, ellipses, exclamation marks, question marks, parentheses, colons), syntactic complexity (dependency parse depth, subordination ratio via spaCy), code-switching frequency (French and English markers), and readability proxies (mean word length, syllable density, paragraph length). These are computed directly from the text using classical NLP — no neural network, no semantic representation, no possibility of content leaking in.

A' is the control. If the holistic lens (Scale A) detects smoothness in the hopscotch ordering, A' tells us whether that smoothness lives in form. The answer is ambiguous: A' gives the hopscotch +1.1σ — a weak positive signal, not statistically significant after correction for multiple testing. The formal undertow is real but faint at the level of measurable surface features.

But A' is devastating on the linear path: +3.9σ, meaning the linear ordering is smooth even when measured on content-free features alone. Cortazar didn't just arrange chapters for thematic flow — he arranged them for *formal* flow. Sentence rhythms, punctuation habits, and vocabulary register transition gradually from chapter to chapter. This is invisible to the naked eye and would be nearly impossible to engineer consciously across 56 chapters. It suggests a deep consistency in Cortazar's compositional process: chapters that are thematically adjacent were also written in adjacent registers.

**Scale B' (LLM Stylistic)** fills the gap between A' and B. The same 27-billion-parameter model that scored narrative content (Scale B) also assessed each chapter on 12 dimensions of perceived style: sentence complexity, vocabulary register, punctuation expressiveness, prose rhythm, descriptive density, dialogue vs. narration balance, code-switching intensity, paragraph density, narrative distance, repetition patterns, typographic experimentation, and syntactic variety. These are the AI's subjective impressions of *how* the prose reads — closer to a human critic's stylistic commentary than to a feature count.

B' gives the hopscotch +1.3σ — again, suggestive but not significant after correction. The four-scale gradient for the hopscotch is the finding:

| Scale | Hopscotch z | What it captures |
|-------|------------|------------------|
| A (Holistic) | **+2.8σ** | Everything mixed together |
| B' (LLM Stylistic) | +1.3σ | Perceived style |
| A' (Stylometric) | +1.1σ | Pure form |
| B (Narrative DNA) | −0.4σ | Explicit meaning |

As measurement moves from holistic to explicit, the hopscotch signal fades. The coherence lives at the level of undifferentiated impression — what we call **form-content resonance** — and dissolves when you try to name its components.

## Running It Twice: The Sensitivity Test

A single model run proves nothing. What if the scores are arbitrary — dependent on the model's mood, the prompt phrasing, or the phase of the moon?

We ran the semantic analysis twice.

**Version 1** (zero-shot): The model scored each chapter cold, with only the rubric as guidance.

**Version 2** (few-shot with evidence): The model received three calibration examples from *other* Cortazar novels — a passage from *62: Modelo para armar* anchoring interiority at 10, a passage from *Un tal Lucas* anchoring humor at 10, and a second passage from *62* anchoring emotional intensity at 10 and death at 9. We used external novels deliberately: calibrating with passages from *Rayuela* itself would risk circularity. The model also had to provide one sentence of textual evidence for each score — a justification for the rating.

The results across 153 chapters (v2 missed two chapters that exceeded the model's context window):

- **83.6%** of all 3,060 dimension-scores were within +/-1 point between v1 and v2
- **95.6%** within +/-2 points
- **45.9%** were exact matches
- **19 of 20 dimensions** correlated above r = 0.85

One dimension failed: *temporal clarity* (r = 0.23). The two prompt versions interpreted it differently — v1 read it as "is the timeline clear?" while v2 read it as "is time fragmented?" These are nearly opposite. This is a genuine finding: it tells us that temporal structure is the most ambiguous dimension in our rubric, the one most sensitive to prompt engineering. We kept it in the analysis with this caveat.

The mean shift was -0.48 points — v2 scored slightly lower across the board. The calibration examples taught the model what a "real 1" looks like from an external reference point, reducing ceiling compression. This is exactly what external anchoring is supposed to do in psychometric design.

The critical question: **does the central finding survive?** If the hopscotch path looked random under v1 but not under v2, our conclusion would collapse. We ran the permutation test on v2 scores. The finding survived both runs: the linear path remained extreme, the hopscotch remained near-random in semantic space. The conclusion is robust to prompt variation.

## What the Evidence Strings Reveal

The v2 extraction produced 153 x 20 = 3,060 evidence sentences — one for each score. These are LLM-generated justifications: the model's explanation of the textual basis for each rating. An important caveat: independent verification shows that only about 54% of quoted spans (3+ words) are exact substring matches in the source chapter. The rest are paraphrases or hallucinated phrasings. These are best understood as *the model's reasoning*, not as precise textual citations.

Here are selected evidence strings for Chapter 1 (Oliveira's opening monologue about searching for La Maga):

- **Existential questioning (8)**: *"Oliveira reflects on the nature of chance encounters, the meaning of searching, and the 'signs' that govern his life, questioning the logic of existence."*
- **Love and desire (9)**: *"The entire chapter is an address to La Maga, exploring their relationship, intimacy, and the 'terrible mirror' of their love."*
- **Spatial grounding (8)**: *"Specific Parisian locations: Pont des Arts, Quai de Conti, Marais, boulevard de Sebastopol, Parc Montsouris, rue des Lombards."*
- **Melancholy and nostalgia (8)**: *"Deep nostalgia for Buenos Aires (shoes, tea, mother) and the lost moments with La Maga suffuse the text."*

And for Chapter 93 (Oliveira's meditation on love and language — the first chapter in the hopscotch order that "responds" to Chapter 1):

- **Love and desire (10)**: *"The entire chapter is a meditation on love, desire, and the impossibility of possession, addressing 'Amor mio' directly."*
- **Metafiction (8)**: *"Oliveira explicitly critiques the act of writing ('artificios de escriba', 'fabricaciones') and the limitations of language."*
- **Intertextual density (9)**: *"Dense allusions to Nashe, Octavio Paz, Puttenham, Morelli, Atala, and architectural figures like Wright and Le Corbusier."*

These evidence strings do something rare in computational literary study: they make the model's reasoning partially auditable. When the model says "love and desire = 9," you can read its justification and decide whether you agree. The AI becomes a transparent interlocutor, not a black box — with the caveat that its "quotes" are often paraphrases rather than exact citations.

They also open a radical possibility: if you know *which* textual elements drive each score, you could surgically replace them — swapping the cultural furniture of 1960s Paris with another time and place while preserving the narrative DNA.

## Chapter Fingerprints

Each chapter's 20-dimensional profile is a unique fingerprint. Compare a few:

![Radar charts showing the narrative fingerprints of notable chapters.](figure7_radar.png)
*Figure 7 — Narrative Fingerprints. Each spoke is one of the 20 dimensions. Chapter 68 (Gliglico) concentrates all its energy in two dimensions. Chapter 1 is expansive. Chapter 93 (Morelli's literary theory) lights up metafiction and intertextuality. ([Interactive version](https://carloscrespomacaya.com/rayuela/article_radar.html))*

The visual difference between chapters is the difference the hopscotch path exploits. When Cortazar places Chapter 1 (Oliveira searching for La Maga) next to Chapter 93 (a philosophical meditation on love and language), the jump between those two fingerprints is what creates the collision. The reader's mind has to bridge the gap. That bridging — that work — is the hopscotch experience.

## The Same Novel, Two Heatmaps

The most direct way to see the collision principle is to lay the entire novel out twice — once in linear order, once in hopscotch order — and look at the patterns.

![Side-by-side heatmaps: linear order (smooth gradients) vs hopscotch order (rapid alternation).](figure10_dual_heatmap.png)
*Figure 10 — Two Reading Orders, Same Data. Left: linear order shows smooth gradients — similar chapters adjacent. Right: hopscotch order shows rapid color alternation — maximally different chapters placed side by side. ([Interactive version](https://carloscrespomacaya.com/rayuela/article_dual_heatmap.html))*

The linear heatmap looks like a landscape: rolling hills, gradual transitions, coherent regions. The hopscotch heatmap alternates much more abruptly — starkly different profiles juxtaposed step after step. The data makes visible what readers have always felt.

## The Gap of the Unnamed

The most provocative finding is not any single z-score — it's the gradient.

When we measure the hopscotch ordering through the holistic lens (Scale A, +2.8σ), there is clear intentional structure. When we decompose that measurement into content-free form (A', +1.1σ) and perceived style (B', +1.3σ), the signal weakens but persists as a trend. When we isolate explicit narrative content (B, -0.4σ), it vanishes.

What does this mean? The holistic embedding captures something that the sum of its decomposed parts does not. We call this **form-content resonance** — the quality that emerges from the interaction of formal and semantic features but belongs to neither alone. It is not sentence length. It is not theme. It is not vocabulary register. It is the way all of these elements combine into what a reader might call "atmosphere" or "voice" — a holistic quality that defies decomposition.

The Mantel correlation matrix — which measures how well each scale's pairwise distance structure predicts another's — makes this concrete:

| | A | A' | B' | B |
|---|---|---|---|---|
| A (Holistic) | — | 0.38 | 0.44 | 0.54 |
| A' (Stylometric) | | — | 0.61 | 0.42 |
| B' (LLM Stylistic) | | | — | 0.61 |
| B (Narrative DNA) | | | | — |

The style-focused scales correlate most with each other (A'↔B': 0.61, B↔B': 0.61). The holistic embedding is closest to the semantic scale (A↔B: 0.54) and farthest from the content-free stylometrics (A↔A': 0.38) — confirming that the embedding model's "gut impression" is dominated by meaning. Yet the holistic scale detects hopscotch structure that the semantic scale misses. The residual — the 46% of holistic variance unexplained by semantic content — is where the atmospheric coherence lives.

This gap is not a limitation of our methodology. It is the finding. The hopscotch ordering has a quality that resists decomposition — and that irreducibility may be precisely what makes it *literary*.

## What This Methodology Proves

The methodology described here has four properties that, together, make the finding defensible:

1. **Convergence across four representations**: The holistic lens (1,024-dimensional embeddings), the content-free stylometrics (26 surface features), the LLM stylistic lens (12 perceived-style dimensions), and the semantic lens (20 explicit narrative dimensions) use different models, different feature spaces, and different levels of human interpretability. All four agree that the linear path is intentionally ordered (+3.2σ to +6.4σ, all surviving Bonferroni correction). They form a gradient on the hopscotch: signal fading from +2.8σ (holistic) through +1.1σ to +1.3σ (formal/stylistic) to -0.4σ (semantic). This convergent pattern is far more informative than any single measurement.

2. **Reproducibility**: Running the semantic lens twice — with different prompts, different calibration strategies, and an independent evidence layer — produced correlations above 0.85 for 19 of 20 dimensions. The finding survived both runs.

3. **Statistical rigor**: The permutation test compares each reading order against 5,000 random alternatives, with separate null distributions for each path (shuffled Ch.1-56 for linear; shuffled all 155 for hopscotch). All linear z-scores survive Bonferroni correction for 8 tests (threshold ~2.24σ). The hopscotch result at +2.8σ on Scale A is moderate and representation-dependent — it survives correction, but the three remaining hopscotch scores do not. We report this gradient transparently rather than cherry-picking the strongest result.

4. **Methodological controls**: The four-scale design functions as its own control. Scale A' proves that the linear path's smoothness is partly formal (+3.9σ on content-free features), not just semantic. The curvature analysis shows the hopscotch is not merely neutral but actively jagged (-4.7σ on Scale B). The gradient across scales constrains interpretation: the hopscotch ordering cannot be dismissed as meaningless noise (the holistic signal is real) nor elevated to deliberate thematic design (the semantic signal is absent).

What the methodology cannot do is tell you what Cortazar intended. It can show that the hopscotch order is anti-smooth in its explicit thematic transitions but intentionally coherent in its holistic impression — suggesting a craft that operates below the level of nameable narrative dimensions. The interpretation — that this gap between semantic collision and holistic coherence is the design, that the irreducible residual *is* the literary quality — is ours. But the statistical scaffolding is as solid as we know how to build.

---

*All interactive visualizations: [carloscrespomacaya.com/rayuela](https://carloscrespomacaya.com/rayuela). Code: [github.com/macayaven/rayuela](https://github.com/macayaven/rayuela).*

---

**Methodology note**: Four scales — A: 1024-dim embeddings (multilingual-e5-large-instruct); A': 26 content-free stylometric features (via spaCy es_core_news_lg + regex); B': 12 LLM-perceived style dimensions (Qwen 3.5 27B, zero-shot); B: 20 narrative-semantic dimensions (Qwen 3.5 27B, v1 zero-shot + v2 few-shot with evidence). Scale B v1-v2 agreement: 83.6% within +/-1, 95.6% within +/-2, 19/20 dims r > 0.85. Temporal clarity excluded from sensitivity claims (r = 0.23). Smoothness z-scores: Linear +3.2σ to +6.4σ (all survive Bonferroni at ~2.24σ); Hopscotch +2.8σ (A), +1.3σ (B'), +1.1σ (A'), -0.4σ (B). Curvature z-scores (positive = gentler): Linear +1.3σ to +1.7σ; Hopscotch -1.9σ (A'), -4.1σ (A, d=1024 caveat), -4.7σ (B). Cross-scale Mantel correlations: A↔B 0.535, A'↔B' 0.609, B↔B' 0.611, A'↔B 0.420, A↔B' 0.441, A↔A' 0.376. Permutation tests use separate null distributions per path. Evidence strings: ~54% span-level fidelity (LLM paraphrases, not citations). 153/155 chapters scored in v2 (2 exceeded 24,576-token context window). Single LLM scorer (Qwen 3.5 27B); inter-rater reliability not established. Analysis on NVIDIA DGX Spark (ARM64, GB10 GPU, 128 GB unified memory).
