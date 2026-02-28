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
