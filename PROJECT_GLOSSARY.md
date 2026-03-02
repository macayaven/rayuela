# Project Glossary

Technical terms introduced during the project, defined in context. New terms are added as they come up — this is a living document.

---

| Term | Definition | First Introduced |
|------|-----------|-----------------|
| **Embedding** | A fixed-length vector of numbers that represents the "meaning" of a piece of text. Texts with similar meaning have vectors that point in similar directions. | Phase 1 |
| **Cosine Similarity** | A measure of how similar two vectors are, based on the angle between them. 1.0 = identical direction, 0.0 = perpendicular, -1.0 = opposite. | Phase 1 |
| **Latent Space** | The high-dimensional "landscape" where embedded texts live. Structure in this space (clusters, paths, loops) reveals structure in the texts themselves. | Phase 1 |
| **Container (Docker)** | A lightweight, isolated environment that packages code and all its dependencies. Like a virtual machine but faster and more portable. | Phase 1 |
| **UMA (Unified Memory Architecture)** | A hardware design where CPU and GPU share the same physical memory pool, instead of having separate RAM and VRAM. The DGX Spark uses this. | Phase 1 |
| **Sliding Window** | A technique for processing long texts by sliding a fixed-size frame across the text, embedding each frame independently. Produces a sequence of vectors that reveals how texture changes *within* a document. | Phase 3A |
| **Stride** | How far the sliding window advances between steps, measured in tokens. A stride smaller than the window size creates overlap — each token appears in multiple windows, smoothing transitions. | Phase 3A |
| **Internal Drift** | A measure of how much the texture changes within a single chapter: 1 minus the mean cosine similarity between consecutive windows. Higher drift = more internal texture variation. | Phase 3A |
| **Overall Span** | The cosine similarity between the first and last window of a chapter. Low span means the chapter ends in a very different texture than it started. | Phase 3A |
| **Mimetic Mode (Mimesis)** | Narrative that *shows* action directly — dialogue, scene, real-time events. Produces a characteristic texture of short turns, interruptions, and present-tense rhythm. | Phase 3A |
| **Diegetic Mode (Diegesis)** | Narrative that *tells* about events — summary, reflection, character states. Produces longer sentences, more introspective vocabulary, and a slower rhythm. | Phase 3A |
