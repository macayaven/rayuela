# Research Log

> Carlos's observations, predictions, and interpretations throughout the project.
> Write in your own voice. Be honest about what surprises you, what confuses you, and what you think you understand.

---

## Phase 1 — Environment Setup

### 2026-02-28 — Project Start

**What I did today:**
- Defined the research plan and three-scale analysis framework
- Set up the GitHub repository
- Created the Dockerfile and docker-compose.yml

**What I expect to happen next:**
I think it will run smoothly, worst case some dependency incompatibility failure, but completely solvable.

**Questions I have:**
The topological data analysis dependencies, what each of them will do? Are they suposed to run in GPU or CPU?

What is the best way to manage your context, so you don't need to auto-compact and forget important things? Are there any memory systems officially recommended for this? And any plugin, hook, or similar?

---

## Phase 3 — Scale A: Micro-Textural Embeddings

### 2026-02-28 — CHECKPOINT A1: Texture Predictions

**Chapters read**: 1, 8, 36 (replaced the suggested Ch. 79 with Ch. 8, which I've actually read)

**My texture predictions:**

- **Chapter 1** (Del lado de allá): Stream of consciousness. Introduces many exterior observations without explicit dialogue. The narrator wanders Paris and wanders mentally — long, flowing prose.

- **Chapter 8** (Del lado de allá): Like a poetry artwork. Very personal, directed from subject A to subject B. No dialogues either, but the register is completely different from Ch. 1 — concentrated, intense, intimate address.

- **Chapter 36** (Del lado de allá): Very rich in dialogues and interactions. Has descriptive passages that feel similar to Ch. 1, but NOT similar to Ch. 8. However, shares emotional intensity with Ch. 8.

**Prediction for embedding space:** Ch. 36 should sit somewhere between Ch. 1 and Ch. 8 — pulled toward Ch. 1 by its narrative/descriptive texture, and toward Ch. 8 by its emotional intensity. Ch. 1 and Ch. 8 should be relatively far apart despite both lacking dialogue, because their surface patterns (sentence structure, vocabulary, mode of address) are very different.

**To verify after embeddings**: Do the three chapters form a triangle? Or does the model prioritize one dimension (e.g., dialogue vs. no-dialogue) over the subtler texture differences?

### 2026-02-28 — Embedding Results & Trajectory Analysis

**CHECKPOINT A1 results:**
- Ch.1 ↔ Ch.36: 0.9497 (most similar — shared narrative texture)
- Ch.1 ↔ Ch.8:  0.9434
- Ch.8 ↔ Ch.36: 0.9287 (least similar — poetic vs. dialogue registers)
- My prediction was partially right: Ch.8 is the outlier. The model sees texture (surface patterns) not emotional intensity (which would be Scale B).

**Key findings from diagnostics:**
- Similarity range across all 11,935 pairs: 0.807 – 0.972 (enough spread for clustering)
- The three editorial sections do NOT cluster by texture (within-section sim ≈ between-section sim)
- Expendable chapters (57–155) are more internally diverse than main narrative (1–56)
- Chapter 106 is the most texturally alien chapter in the entire novel

**Trajectory analysis:**
- Linear path (1→56): mean consecutive similarity = 0.9477
- Hopscotch path (Tablero): mean consecutive similarity = 0.9293
- Both paths are smoother than random permutation baselines (10,000 tests):
  - Linear: +6.37σ above random (smoother than all 10,000 random orderings)
  - Hopscotch: +2.77σ above random (smoother than 99.7% of permutations)
- Both paths are intentionally ordered for textural continuity — linear more strongly

**What surprised me:**
- The sections not clustering was unexpected — I assumed the Paris/Buenos Aires divide would show in texture
- The hopscotch path being smoother than random was also unexpected — I assumed it was purely semantic design