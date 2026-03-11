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

### 2026-03-10 — Phase 5 Training Scaffold Decisions

**Decision**: Phase 5 starts with a scaffold that writes the full training envelope (config, checkpoint metadata, adapter placeholder, tokenizer config) before any real fine-tuning. The default `identity_smoke` dataset mode keeps the run safe and deterministic while we validate the infrastructure.

**Experiment logging**: We will use Weights & Biases for experiment metadata and metrics when running real training. It is optional and defaults to offline mode; runs should only upload when we deliberately enable it.

**Why this matters**: The scaffold forces every run to be reproducible and auditable before we spend GPU time, which keeps the operational decoupling claims grounded in saved artifacts instead of ad hoc training output.

### 2026-03-10 — Phase 6 Analysis Contract Decisions

**Decision**: Phase 6 starts by aggregating immutable Phase 4 run artifacts instead of inventing a separate notebook-only synthesis layer. The new analysis contract reads `prompt_baseline_cases.json` plus each run manifest, then emits a complete case table, explicit failure labels, source-side bias slices, and a close-reading queue with stable `run_id` links.

**Failure taxonomy**: We are labeling only operationally defensible failure modes that are already present in the saved scoring contract: `semantic_drift`, `target_miss`, `length_guardrail`, `lexical_overlap`, and `stalled_revision`. No broader interpretive claim should be attached until the underlying runs exist in sufficient number.

**First aggregation result**: Running the new analysis module against the saved Phase 4 dry run (`phase4-dry-run-20260309a`) produced a 2-case synthesis bundle. Both cases ended in `stalled_revision`, and one also carried `target_miss`. The highest dry-run objective was Borges -> Bolaño at `0.5807`; the weakest was Borges -> García Márquez at `0.5504`.

**Why this matters**: This keeps Phase 6 operationally decoupled from ad hoc qualitative memory. The article-facing materials now derive from immutable run outputs, which means tomorrow's real training or live baselines can drop into the same synthesis path without rewriting the analysis layer.

### 2026-03-10 — Guided Scheduler Decisions

**Decision**: We are adding a finite guided scheduler, not an unconstrained autonomous loop. Each scheduled experiment is a fixed command with a timeout, an expected metric artifact, and an explicit advancement rule (`keep`, `discard`, or `failed`) based on the measured output.

**Implementation**: `src/reconstruction_scheduler.py` reads a JSON plan, executes each command in order, extracts a dotted metric key from the produced artifact, and writes append-only per-experiment logs plus a schedule summary under `outputs/reconstruction/analysis/schedules/`.

**Attribution**: The scheduler structure is explicitly inspired by Andrej Karpathy's `autoresearch` project, particularly its keep/discard experiment loop and emphasis on machine-readable experiment state. The current repository adopts that operational pattern only at the scheduling layer; it does not adopt autonomous code-editing or open-ended self-direction.

**Why this matters**: This is the right adaptation of the `autoresearch` idea for the current repository. Our bottleneck is not code mutation; it is disciplined experiment execution against immutable run manifests and analysis artifacts. The scheduler therefore keeps the experimentation loop operationally decoupled from manual babysitting without broadening the claim into autonomous research.

### 2026-03-10 — Phase 6 W&B Instrumentation Decisions

**Decision**: W&B logging now sits on top of the scheduler and analysis layers rather than inside an ad hoc notebook or dashboard-only path. The logging remains optional and defaults to offline mode so observability does not become a hidden execution dependency.

**Logging schema**:
- Scheduler experiments log one W&B run per planned experiment with `schedule_id`, `experiment_id`, `run_id`, phase, command, timeout, metric selector, and explicit attribution to Andrej Karpathy's `autoresearch` as scheduler inspiration.
- Each scheduler run logs the operational decision (`keep`, `discard`, `failed`), incumbent comparison context, execution duration, return code, the extracted reconstruction objective, and compact control-level summary metrics from the produced artifact.
- Each scheduler run attaches the immutable local artifacts that matter for auditability: scheduler stdout/stderr/result logs plus any available run manifest, baseline summary, baseline cases, and baseline report.
- Phase 6 analysis logs one aggregate W&B run with total runs/cases, failure-mode counts, a run-summary table, and the close-reading queue, then attaches the saved analysis summary, Markdown report, and article-input JSON.

**Why these metrics and artifacts matter for the research loop**: The scheduler metrics let us compare candidate runs by the same saved advancement signal without rereading raw JSON by hand. The attached artifacts keep every keep/discard claim operationally anchored to immutable run products, which preserves auditability when we later inspect surprising results or failed runs.

**Why they matter for Part 3**: The analysis-side W&B outputs capture exactly the synthesis objects the article needs: failure distributions, traceable per-run summaries, and a stable queue for close reading. That keeps the Part 3 narrative operationally decoupled from memory or dashboard screenshots because the narrative can be reconstructed from saved artifacts and logged tables.

### 2026-03-10 — External Review Gate Before Experiments

**Review sources**: Gemini CLI and Claude Code CLI were both asked to review the Phase 6 analysis and scheduler changes before any real experiment runs. The reviews were run in non-interactive mode against the current implementation surface rather than after results existed.

**Adopted feedback**:
- Added a scheduler immutability guard so reusing the same `schedule_id` now fails instead of overwriting prior schedule logs.
- Extended scheduler summaries with kept/discarded/failed `run_id`s.
- Added a direct scheduler-to-analysis handoff so `src/reconstruction_analysis.py` can aggregate kept runs from `--schedule-summary-path`.
- Documented the trust boundary that scheduler plan files are executable specifications and must be treated as trusted input.

**Deferred or declined feedback**:
- We kept the current explicit failure taxonomy in Phase 6 because it is intentionally aligned to the present scoring contract; if new failure labels are added later, the taxonomy should be expanded deliberately rather than inferred implicitly.
- We kept dotted metric resolution narrow in the scheduler because the current artifact contract is intentionally simple JSON; richer selectors can be added later if the metric artifacts actually require them.
- We did not broaden the close-reading queue yet. For the current small-run state, best/worst salience is enough; this should be revisited once live runs produce a materially larger case table.

**Why this matters**: The experiment gate is now not only test-green but externally reviewed before execution. That reduces the chance that the first scheduled live runs will expose avoidable workflow mistakes rather than genuine model behavior.

### 2026-03-11 — Overnight Guided Run Findings and Follow-Up Fix Plan

**Observed overnight result**: The detached overnight schedule completed cleanly. On the 6-case comparison, moving from 2 iterations to 3 iterations increased the mean weighted objective only slightly (`0.0861` -> `0.0876`) while leaving the failure pattern largely unchanged and producing large case-level variance.

**Interpretation**: The overnight batch was operationally useful, but not yet a clean causal comparison. The manifest recorded `seed=42`, yet the live OpenAI-compatible generation path was not forwarding that seed into the actual chat completion request. That means part of the observed difference between 2 and 3 iterations can still come from avoidable sampling variance rather than from the iteration budget alone.

**Fix ordering**: We should fix experiment validity before we harden the detached wrapper further. The immediate change is to pass the generation seed through the live backend request and test that the request surface receives it. After that, the detached launcher can be hardened around tmux session naming, prerequisite checks, separate scheduler/analysis logs, persisted non-secret launch metadata, and explicit status/stop helpers.

**External review**: Gemini CLI reviewed that ordering and agreed with it. Claude CLI was not available in this environment, so we could not obtain the second external review pass from the same machine.

### 2026-03-11 — Phase 6 Cross-Run Comparison and Promotion Criteria

**Decision**: Research promotion should be an analysis-layer decision, not a scheduler side effect. The scheduler still records operational `keep` / `discard` / `failed`, but Phase 6 now computes separate run-to-run deltas and explicit promotion recommendations from the immutable case table.

**Implementation**: `src/reconstruction_analysis.py` now builds pairwise run comparisons over overlapping case identities, measures mean/median objective deltas plus case-level improvement shares, and applies explicit promotion criteria (`min_overlapping_cases`, `min_mean_delta`, `min_median_delta`, `min_non_negative_share`, `max_failure_case_delta`). These criteria are configurable at analysis time and are logged into the saved analysis artifacts and W&B summaries.

**Why this matters**: This gives the research loop a disciplined answer to a different question than the scheduler asks. The scheduler answers “did this run beat the current queue rule?” The analysis layer now answers “is this run strong enough, on explicit criteria, to promote as the new research incumbent?” That separation reduces post hoc interpretation and makes seeded vs unseeded comparisons more defensible.

### 2026-03-11 — Phase 6 Methodology Hardening for Cross-Run Claims

**Decision**: Cross-run deltas should not be interpreted without three additional checks: provenance comparability, paired uncertainty, and explicit failure transitions. These belong in analysis, not in the scheduler, because they are judgments over saved artifacts rather than queue-control logic.

**Implementation**: Phase 6 analysis now loads compact run provenance from each manifest and gates comparisons on invariant fields (`git_sha`, phase, prompt template, model, corpus/pilot artifact paths, backend, and generation seed). It also computes deterministic paired bootstrap intervals for mean objective deltas, records whether the paired interval excludes zero, and adds per-label failure-transition summaries (`persistent`, `resolved`, `introduced`) across overlapping cases. These surfaces are persisted in the JSON/Markdown/article artifacts and logged to W&B as comparison, transition, and provenance tables.

**Why this matters**: This materially improves research discipline without changing experiment execution. A small delta can now be interpreted against approximate paired uncertainty rather than raw means alone; a run can be held back if the compared artifacts are not provenance-comparable; and failure movement can be described as “resolved” or “introduced” instead of flattened into one total count. That makes the next seeded-vs-unseeded and 2-vs-3-iteration narratives much harder to overstate.

### 2026-03-11 — Phase 6 Experiment Interpretation Surfaces

**Decision**: Phase 6 should explain how to read a batch, not only emit metrics. The analysis layer now carries a compact reading guide, concrete source/output examples, and an explicit reasoning-leak summary so the research loop can interpret runs without reverse-engineering the W&B tables every time.

**Implementation**: `src/reconstruction_analysis.py` now extracts short source and output excerpts from each saved case, flags outputs that still look like process text (`Thinking Process:` and similar markers), and publishes weakest/strongest concrete examples into the summary JSON, Markdown report, article inputs, and W&B tables. The detached launcher now calls analysis with `--schedule-run-selection nonfailed`, so future schedule-level synthesis includes both kept and discarded candidates instead of only the incumbent.

**Why this matters**: This improves research throughput without changing the execution contract. We can now judge whether a scalar objective corresponds to an actually usable rewrite, spot reasoning leakage quickly, and compare discarded candidates against the incumbent in the same analysis batch. That should reduce idle GPU time lost to human confusion rather than to model runtime.

### 2026-03-11 — Hidden Reasoning Instead of Visible Chain-of-Thought

**Decision**: For Qwen-based reconstruction runs, the preferred containment path is not “disable thinking.” It is “allow reasoning, but keep the reasoning channel separate from the final passage.” That preserves model capability while stopping the saved rewrite from filling with `Thinking Process:` scaffolding.

**Implementation**: The tracked Qwen vLLM service now enables `--reasoning-parser qwen3`, and Phase 4 run configs can record `--reasoning-parser qwen3` in their manifest payload. `parse_generated_text()` also strips leading `<think>...</think>` blocks when a server returns them inline instead of through a separated reasoning channel.

**Why this matters**: This is the right experimental control for the current problem. The objective is not to make the model shallower; it is to stop visible reasoning text from contaminating the rewrite artifact and the downstream score. That keeps the experiment closer to the later fine-tuning target, which will also care about final passage quality rather than exposed chain-of-thought.

### 2026-03-11 — Generation Budget Guard for Hidden-Reasoning Runs

**Decision**: The completion token budget for Phase 4 live generation is now an explicit experiment parameter rather than an implicit backend default. Hidden reasoning only helps if the model still reaches a final passage inside the same response.

**Implementation**: `src/reconstruction_baselines.py` now exposes `--generation-max-tokens`, records it in the immutable run manifest, and forwards it into the live prompt backend. The OpenAI-compatible path also now fails fast when the serving stack returns a non-empty reasoning channel but no final `content`, with an error that points directly to the generation-budget/output-contract problem.

**Why this matters**: This improves research throughput and result quality at the same time. We stop wasting GPU time on runs that would only score empty candidates, and we make future seeded comparisons auditable because the token budget is part of the saved experiment contract instead of a hidden default.
