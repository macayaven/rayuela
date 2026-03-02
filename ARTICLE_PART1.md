# What Does a Novel Look Like From the Inside?

## AI Reveals the Hidden Architecture of Cortázar's *Rayuela*

*Carlos Crespo Macaya & Claude · 2026*

---

In the first pages of *Rayuela*, before you read a single chapter, Cortázar gives you a choice. You can read the novel straight through — chapters 1 to 56 — like any other book. Or you can follow the *Tablero de Dirección*, a hopscotch path that leaps between all 155 chapters in an order the author prescribes: 73, 1, 2, 116, 3, 84, 4, 71...

Readers and scholars have argued about these two paths for sixty years. Is the hopscotch order designed for thematic continuity — deeper connections that the linear order misses? Or is it something else entirely?

We set out to answer this empirically. We converted every chapter of *Rayuela* into mathematical representations using AI, then traced both reading paths through that mathematical space. We measured two things independently: the *texture* of the prose (how it sounds) and the *meaning* of the prose (what it says). Then we tested both reading orders against thousands of random alternatives to determine whether they could have arisen by chance.

What we found overturned our own initial hypothesis.

> **[Figure 1 — The Novel in 3D]** Each of the 155 chapters is a point in space; chapters that sound similar sit close together. The green line traces the linear path; the red dashed line traces the hopscotch. *[Interactive version at GitHub Pages]*

---

### The Invisible Surgery

Before feeding a single word to the AI, we had to operate on the text.

Every chapter of *Rayuela* ends with a small number in parentheses — a signpost telling the hopscotch reader where to jump next. Leave those numbers in, and any pattern the AI finds would be contaminated: we'd have embedded the reading order inside the data we were using to evaluate it. This is what statisticians call **circular analysis**. In plain language: cheating.

We also found a ghost. Chapter 55 is the only chapter the hopscotch path never visits — a phantom chapter that exists only for linear readers.

We switched from a corruption-prone OCR scan to a clean digital source, stripped every navigation marker, and verified the resulting 155 chapters against the published text. This kind of data hygiene is invisible in the final analysis, but it's where computational literary studies succeed or fail. If you don't strip the editorial apparatus, you find structure the editor created, not the author.

---

### Two Microscopes

We examined the novel through two independent lenses.

**The texture lens** captures how the prose *sounds*: rhythm, vocabulary, syntax. A multilingual AI model converts each chapter into a list of 1,024 numbers — a fingerprint of its linguistic surface. We chose a multilingual model because Cortázar slips between Spanish, French, and English mid-sentence. A monolingual model would have created artificial boundaries where Cortázar saw none.

**The meaning lens** captures what the prose *says*. A 27-billion-parameter language model scored every chapter on 20 semantic dimensions — existential questioning, love and desire, humor and irony, interiority, spatial grounding, language experimentation, and 14 others — producing what we call a **narrative DNA** profile: a 20-number fingerprint of each chapter's thematic content.

We validated the meaning lens with Chapter 68, the famous Glíglico chapter, written entirely in a language Cortázar invented. The model scored it: *language experimentation = 10, love and desire = 9, spatial grounding = 2, dialogue density = 1*. It understood that a passage in a made-up language is about love, not place. The machine couldn't read the invented language — nobody can — but it understood what the invented language was *for*.

> **[Figure 2 — Two Ways to See the Same Novel]** Left: chapters in texture space. Right: chapters in meaning space. In texture space, the novel's three sections overlap completely — Paris, Buenos Aires, and the expendable chapters all sound alike. In meaning space, they begin to separate. The divisions readers feel are about *what* is being said, not *how*. *[Interactive version at GitHub Pages]*

---

### The False Lead

With two microscopes calibrated, we asked the question: are the two reading paths designed, or could they be random?

We measured the "smoothness" of each path — how similar consecutive chapters are as you read. Then we generated 5,000 random orderings of the same chapters and measured their smoothness. This gave us a baseline: if a reading order is no smoother than a random shuffle, it wasn't designed for flow.

The texture lens told a seductive story. The linear path was extraordinarily smooth — smoother than all 5,000 random orderings. And the hopscotch path was also smooth — rougher than the linear, but still smoother than most random alternatives.

Our first interpretation: both paths are designed for textural continuity, just at different intensities.

**This was wrong.**

---

### The Twist

When we ran the same test with the meaning lens — the 20-dimensional narrative DNA — the result changed completely.

The linear path remained extraordinarily smooth. But the hopscotch path's signal vanished. Against 5,000 random orderings measured by narrative content, the hopscotch fell squarely within the range of random orderings — statistically indistinguishable from a shuffle.

Here is the simplest way to understand the numbers. Imagine shuffling the novel's chapters into a random order 5,000 times and measuring how smoothly each shuffled order reads. The linear path is smoother than *every single one* of those 5,000 shuffles — both in sound and in meaning. The probability of this happening by chance is astronomically small.[^1] The hopscotch path, measured the same way, falls within the normal range of random orderings on both scales — never more than half a standard deviation from the average. It is not statistically distinguishable from a random shuffle.

| | Texture (how it sounds) | Meaning (what it says) |
|---|---|---|
| **Linear (1→56)** | Smoother than all 5,000 shuffles | Smoother than all 5,000 shuffles |
| **Hopscotch (Tablero)** | Within normal range of random | Within normal range of random |

[^1]: The linear path's smoothness is 7.7 to 8.6 standard deviations below the random mean, depending on the measurement scale. For reference, a result 5 standard deviations from the mean is the threshold particle physicists use to declare a discovery. Ours exceeds that by a wide margin.

This result holds across both measurement scales independently. The cross-scale correlation between texture and meaning smoothness is ρ = 0.505 — the two lenses capture partially overlapping but distinct structure, yet they agree on the central finding.

> **[Figure 3 — Was the Reading Order Designed?]** The grey histogram shows the smoothness of 5,000 random chapter orderings. The green line marks the linear path (far left — smoother than everything). The red line marks the hopscotch (well within the random range — indistinguishable from chance). This is the finding in one picture. *[Interactive version at GitHub Pages]*

---

### Collision, Not Continuity

What does this mean?

Cortázar didn't design the hopscotch for smooth reading. He designed it for **collision**. The jarring transitions are not a flaw or a side effect — they are the architecture.

The weaving pattern makes this visible. In the linear reading, you progress through Paris (chapters 1–36), then Buenos Aires (37–56), in clean blocks. In the hopscotch, Cortázar interleaves all three sections from the first step — Paris, expendable, Buenos Aires, expendable, Paris — deliberately fragmenting the reader's sense of place and time.

> **[Figure 4 — Section Weaving]** Top: the linear path (solid blocks of blue, then orange). Bottom: the hopscotch (a kaleidoscope of alternating colors). The hopscotch never lets you stay in one world for long. *[Interactive version at GitHub Pages]*

Trace any single dimension of meaning along both paths and the difference is striking. Take *tension and anxiety*: the linear path builds and releases in long waves, like a tide. The hopscotch path spikes and drops in arrhythmic pulses, like a broken heart monitor.

> **[Figure 5 — Emotional Arcs]** Six dimensions of meaning traced along both reading paths. The linear path builds in waves. The hopscotch is stochastic noise — by design. *[Interactive version at GitHub Pages]*

The reader who follows the hopscotch is meant to experience the disorientation of Oliveira himself — wandering between cities and selves and modes of being. The collisions between chapters are the message. You're not supposed to settle in. You're supposed to be thrown.

---

### What the Microscope Can and Cannot See

We should be precise about what this analysis proves and what it doesn't.

**What it proves**: The linear reading order is designed for smooth flow across both the surface and the meaning of the text — to a degree that is statistically impossible by chance. The hopscotch order is not designed for smooth flow; its smoothness is indistinguishable from a random shuffle. These are empirical facts, reproducible, tested against 5,000 random orderings across two independent measurement scales.

**What it suggests**: Cortázar designed the hopscotch for deliberate discontinuity — for collision between chapters that differ in theme, tone, and setting. The interleaving of sections supports this interpretation.

**What it cannot tell us**: Whether the collision is *aesthetically successful*. Whether the disorientation is productive or merely annoying. Whether Cortázar consciously calculated these statistical properties or whether his literary intuition produced them emergently. Whether *Rayuela* is a great novel.

The AI is a microscope, not a judge. It reveals structure that was always there but invisible to the unaided eye. What we make of that structure — whether we find it beautiful, significant, or worth arguing about — remains irreducibly human.

---

*This is Part 1 of a series. Part 2 will examine the methodology in depth: the 20-dimensional scoring rubric, the sensitivity analysis between two independent model runs, and what the AI's own "evidence" strings reveal about how it reads literature. Part 3 will explore what happens when you computationally remix Rayuela — substituting the cultural furniture of 1960s Paris with other times and places while preserving the novel's narrative DNA.*

*All interactive visualizations are available at [carloscrespomacaya.com/rayuela](https://carloscrespomacaya.com/rayuela). Code at [github.com/macayaven/rayuela](https://github.com/macayaven/rayuela).*

---

**Methodology note**: 155 chapters analyzed at two scales — Scale A: 1024-dim embeddings (multilingual-e5-large-instruct); Scale B: 20-dim semantic profiles (Qwen 3.5 27B). Statistical significance via permutation testing (5,000 random orderings). Cross-scale Spearman ρ = 0.505. Analysis conducted on NVIDIA DGX Spark (ARM64, GB10 GPU, 128 GB unified memory).
