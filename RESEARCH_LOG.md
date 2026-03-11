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
