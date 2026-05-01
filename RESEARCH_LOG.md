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

### 2026-03-11 — Semantic Evaluator Guard for Hidden-Reasoning Runs

**Decision**: The structured semantic-evaluation path must be parser-aware too. Hidden reasoning is acceptable only if the evaluator still receives the final JSON payload; otherwise a “semantic failure” may really be an execution-contract failure.

**Implementation**: `src/semantic_extraction.py` now uses the same parser-aware message-content helper as the prompt generator. It accepts an explicit `max_tokens` budget, defaults that budget higher under the Qwen reasoning parser, and raises a clear error when the model returns reasoning without final JSON. `src/reconstruction_baselines.py` now exposes `--semantic-generation-max-tokens` and records it in the run manifest so the measurement budget is part of the saved experiment contract.

**Why this matters**: The failed hidden-reasoning run showed that fixing only the rewrite generator was not enough. The evaluator can also starve before it emits structured output. Making that budget explicit closes a real gap in the methodology and avoids wasting GPU time on retries that can never yield a score.

### 2026-03-11 — Prompt Contract Tightening for Hidden Reasoning

**Decision**: Before swapping models or adding more orchestration complexity, we should try a narrower intervention: keep Qwen as the active lane but tighten the visible-output contract in the rewrite prompts. That is cheaper and more interpretable than immediately pivoting to a different model family.

**Implementation**: The default rewrite templates are now versioned as `style_shift_v2` and `revise_v2`. They explicitly tell the model to think silently, keep reasoning private, and begin the visible answer immediately with the rewritten passage itself, with no labels, markdown, XML tags, quotes, or explanatory text. The Phase 4 manifest now records the new template ID instead of silently reusing the old `v1` label.

**Why this matters**: This is the lowest-complexity change that could plausibly fix the current blocker. If it works, we keep the research lane simple and move toward the fine-tuning prerequisites faster. If it fails, we will know the remaining problem is not just sloppy prompt contract wording.

### 2026-03-11 — Visible Meta-Suffix Trimming for Prompt Baselines

**Decision**: Some models now satisfy the basic rewrite call but still append visible commentary after the passage, for example `**Nota:**` plus a bullet list describing the stylistic changes. That suffix should not be scored as part of the literary rewrite.

**Implementation**: `src/reconstruction_baselines.py` now trims obvious post-passage meta-commentary markers such as `Nota:`, `Note:`, `Explicación:`, `Commentary:`, `Justificación:`, `Cambios realizados:`, and `Changes made:` after normalization. The saved iteration record keeps audit fields indicating whether a visible meta suffix was trimmed and which marker triggered it. The Phase 4 summary/report also exposes a per-control trimmed-case count.

**Why this matters**: This is a low-complexity way to keep the measured object aligned with the research target. We still record that the model violated the output contract, but we no longer let an explanatory suffix dominate the score or the close-reading sample.

### 2026-03-11 — Official Spark Nemotron Lane Reframed Around llama.cpp

**Decision**: The clean official Nemotron reasoning-control lane for this workstation is not the local Nano-vLLM plugin experiment. It is the DGX Spark playbook path based on Nemotron 3 Nano through `llama.cpp`.

**Implementation**: Added `src/reconstruction_spark_nemotron.py` to codify the published Spark commands into a testable helper that can print the build/bootstrap steps, install the dedicated Hugging Face CLI venv, clone and compile `llama.cpp` for `SM_121`, download the official GGUF artifact, and write a bounded Phase 4 launchcheck plan targeting the local OpenAI-compatible endpoint. The dead-end `vllm-nemotron-nano` compose service was removed so the repo no longer presents that experimental workaround as a first-class path.

**Why this matters**: This reduces methodological ambiguity. If we test Nemotron as a reasoning-aware generation lane on DGX Spark, we should do it through the surface NVIDIA actually documents for Spark rather than through a fragile local adaptation that already proved too heavy and slow for the current box. That keeps the added complexity justified by a clearer experiment contract.

### 2026-04-30 — Unattended Fast-Signal Nemotron Schedule

**Decision**: Start the unattended run on the healthy Spark Nemotron `llama.cpp` endpoint at `http://localhost:30000/v1` instead of waiting for the Qwen endpoint. At launch time, port `8000` was occupied by a failing vLLM vision container that reset `/v1/models`, while the Nemotron server returned a valid model list for `Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf`.

**Runtime fact**: A direct 16-token chat probe returned hidden reasoning but no final content; the same probe with 256 completion tokens returned final `content`. This confirms that the generation budget is an execution-contract variable for the `llama.cpp` reasoning lane, not a cosmetic setting.

**Plan**: `plans/reconstruction_guided_schedule.nemotron-fast-20260430.json` starts with a 1-case/1-iteration launchcheck, then 2-case 1-vs-2 iteration comparisons, then a 4-case 2-iteration confirmation, with a 4-case 3-iteration optional tail if the earlier runs finish inside the budget. The first useful artifact should be `outputs/reconstruction/runs/phase4-nemotron-fast-20260430a/prompt_baseline_summary.json`.

**Why this matters**: The schedule is optimized for early evidence rather than maximum batch size. It should reveal quickly whether the hidden-reasoning contract produces scoreable rewritten passages before spending the rest of the night on broader comparison.

### 2026-05-01 — Nemotron Schedule Result and Retry Fix

**Observed result**: `guided-phase4-nemotron-fast-20260430a` completed quickly. The 1-case launchcheck succeeded with mean weighted objective `0.1581` and no visible reasoning leak, but all larger experiments failed because the model sometimes returned hidden reasoning without final `content`.

**Interpretation**: The Spark Nemotron lane is viable enough to produce a clean scoreable passage, but the Phase 4 runner was too brittle: one first-iteration generation failure crashed the whole run, and one revision failure discarded the already scored first iteration.

**Implementation**: Phase 4 now records first-iteration case failures separately under `case_failures` and continues the remaining cases when at least one case is scoreable. If a revision generation fails after an earlier scoreable iteration, the run preserves the best previous iteration with `stop_reason=prompt_generation_failed` and stores the error message. The prompts also now tell the model to keep private reasoning short and reserve most of the completion budget for the final passage.

**Next retry**: `plans/reconstruction_guided_schedule.nemotron-fast-retry-20260501.json` repeats the fast-signal shape with fresh immutable run IDs so we can distinguish model quality failures from runner brittleness.

**Follow-up fix**: The first retry showed that an all-failed 1-case launchcheck still exited before writing the failure artifact. Phase 4 now writes completed artifacts even when every case in a run fails, with `failed_cases` populated and the relevant control metric set to `0.0`. This lets the scheduler continue and keeps all-failure runs visible to Phase 6 instead of disappearing into stderr.

**Retry2 result**: `guided-phase4-nemotron-fast-retry2-20260501a` completed all three experiments with no scheduler failures. The all-failed 1-case launchcheck now persisted correctly with metric `0.0`. The 2-case / 1-iteration run produced one scoreable case and one first-iteration generation failure, with mean objective `0.1662`. The 2-case / 2-iteration run again produced one scoreable case and one generation failure, with mean objective `0.1713`.

**Interpretation**: Revision produced a small paired uplift on the one overlapping scoreable case (`+0.0051`), but both the 1- and 2-iteration outputs still failed `semantic_drift` and `target_miss`. There were no visible reasoning leaks. The persistent failing target was Borges -> Bolaño; the scoreable target was Borges -> García Márquez.

**Next decision**: Do not scale Nemotron prompt baselines yet. The runner is now robust enough, but the evidence says the current prompt-only Nemotron lane is unstable for some targets and weak even when scoreable. The next research step should either add a target-specific final-answer rescue pass or move to the Phase 5 fine-tuning scaffold rather than simply increasing batch size.

### 2026-05-01 — Final-Answer Rescue Pass

**Decision**: Add a narrow rescue-generation path before moving to Phase 5 fine-tuning. When a prompt call returns hidden reasoning without final content, Phase 4 now sends one strict fallback request that asks only for the final rewritten passage and records `rescue_used` plus the original error in the iteration artifact.

**Why this comes first**: The previous runs showed a mixed failure: serving works and some passages are scoreable, but missing final content can still erase whole cases. The rescue pass separates execution-contract failure from true literary reconstruction failure before we spend time on adapter training.

**Retry plan**: `plans/reconstruction_guided_schedule.nemotron-rescue-20260501.json` repeats the small 1-case and 2-case comparison with fresh immutable run IDs, so we can measure whether rescue improves completion rate and whether the extra recovered outputs remain weak or become useful.

**Result**: `guided-phase4-nemotron-rescue-20260501a` completed all three experiments without scheduler failures. The rescue path recovered the Borges -> Bolaño case that previously failed to emit final content, producing a scoreable passage with objective `0.1425`. However, the output still failed `semantic_drift`, `target_miss`, and `length_guardrail`. The 2-case runs still failed to recover the Borges -> García Márquez case, and the 2-iteration rescue run introduced `stalled_revision` with no objective improvement.

**Conclusion**: Rescue is useful as instrumentation and completion recovery, but it does not solve literary quality. The evidence now strongly favors moving to Phase 5 fine-tuning or a different generation strategy rather than continuing to tune prompt-only Nemotron runs.

### 2026-05-01 — Phase 5 Contract-Smoke Dataset

**Decision**: Start Phase 5 fine-tuning with an output-contract dataset before attempting target-author style transfer. The new `contract_smoke` dataset mode keeps `target_text` equal to `source_text`, but wraps every example in an instruction that requires a Spanish final passage only, with no reasoning, notes, headings, markdown, or explanation.

**Implementation**: `src/reconstruction_train.py` now writes split-specific JSONL datasets under each immutable training run directory. The first scaffold run, `phase5-contract-smoke-20260501a`, produced `3,240` examples from the locked pilot artifacts: `2,270` train, `507` validation, and `463` test examples. The run remains `scaffold_only` and writes placeholder adapter metadata rather than claiming real training.

**Why this matters**: The prompt-only Nemotron evidence showed two separable problems: final-answer reliability and literary quality. Training the contract first attacks the cheaper, more measurable reliability problem and reduces the risk that a later style-transfer adapter is blamed for failures that are really output-format instability.

### 2026-05-01 — Phase 5 Seq2Seq Smoke Training Node

**Decision**: Start the fine-tuning node with a bounded real training smoke run before adding PEFT/QLoRA complexity. The local development venv does not include the ML training stack, but the Rayuela Docker image has PyTorch, Transformers, Datasets, and Accelerate. It does not currently include PEFT or bitsandbytes.

**Implementation**: `src/reconstruction_train.py` now supports `--training-mode seq2seq_smoke`, which tokenizes the contract dataset, runs a bounded `Seq2SeqTrainer` job, saves a non-placeholder model artifact, and records training metrics inside the same immutable Phase 5 envelope. The scaffold mode remains the default.

**First run**: `phase5-seq2seq-smoke-20260501a` used `hf-internal-testing/tiny-random-t5`, `contract_smoke`, `2` train examples, `1` validation example, and `1` optimizer step. It produced a non-placeholder `seq2seq_full_model_smoke` artifact with train loss about `7.006`. This is not a quality claim; it proves the training node can execute and persist a real checkpoint-shaped artifact.

**Next risk**: The next escalation is not literary evaluation yet. It is adding the actual adapter path, probably PEFT/QLoRA, to the training container while keeping this smoke run as the fast health check.

### 2026-05-01 — DGX Spark Fine-Tuning Playbook Review

**Source review**: Reviewed NVIDIA's official DGX Spark playbooks for PyTorch fine-tuning and Unsloth. The PyTorch playbook uses the NGC PyTorch container plus `transformers`, `peft`, `datasets`, `trl`, and `bitsandbytes`, with example scripts for full SFT, LoRA, and QLoRA. The Unsloth playbook uses the same Spark premise but adds optimized kernels, 4-bit model loading, and a compact validation script.

**Decision**: Adapt the PyTorch PEFT/TRL path first. It is closer to Rayuela's current `seq2seq_smoke` runner and introduces fewer new moving parts. Unsloth remains a second lane for throughput once PyTorch LoRA proves the dataset, adapter artifact, and evaluation contract.

**Implementation note**: Added `plans/phase5_dgx_spark_finetune_playbook.md` and `scripts/bootstrap_dgx_spark_finetune.sh` so the official playbook assumptions are captured in repo-native form instead of living only as shell history.

**Hardware guardrail**: DGX Spark should be treated as a narrow NVIDIA-supported stack, not as a generic CUDA workstation. The bootstrap now defaults to NVIDIA's playbook PyTorch image, checks `nvidia-smi`, `nvcc`, and PyTorch CUDA visibility before installing dependencies, and constrains pip installs so the NGC Torch/CUDA/Triton stack is not silently replaced.

### 2026-05-01 — Phase 5 LoRA SFT Smoke

**Decision**: Add `training_mode=lora_sft` as the first real adapter lane, following NVIDIA's PyTorch PEFT/TRL recipe shape while preserving Rayuela's immutable run contract.

**Compatibility finding**: An unpinned PEFT/TRL install inside `nvcr.io/nvidia/pytorch:25.11-py3` failed before training because `peft 0.19.1` rejected the image's bundled `torchao 0.14.0+git`. We did not upgrade Torch or `torchao`; instead, the bootstrap pins the Hugging Face/PEFT/TRL userland layer to avoid breaking the NVIDIA stack.

**Successful smoke**: `phase5-lora-sft-smoke-20260501b` completed one LoRA SFT optimizer step on `hf-internal-testing/tiny-random-LlamaForCausalLM` with `2` contract examples, rank `4`, and a non-placeholder adapter artifact under `adapter/adapter_model.safetensors`. Train loss was about `10.29`. This is still an execution proof, not a quality result.

### 2026-05-01 — Phase 5 Qwen 0.5B Contract Adapter

**Run**: `phase5-lora-contract-qwen05b-20260501a` trained a bounded LoRA contract adapter on `Qwen/Qwen2.5-0.5B-Instruct` inside `nvcr.io/nvidia/pytorch:25.11-py3`, using the pinned PyTorch PEFT/TRL dependency lane. The run used `128` contract-smoke training examples, rank `8`, gradient accumulation `4`, and `20` optimizer steps. Final train loss was about `2.81`.

**Contract probe**: Added a repo-native contract probe in `src/reconstruction_infer.py` and ran it on the first `8` validation examples. The fine-tuned adapter produced no empty outputs, no prompt-scaffold echoes, and `1/8` outputs with forbidden contract markers. The unfine-tuned base model on the same examples also produced no empty outputs or scaffold echoes, but had `5/8` forbidden-marker outputs.

**Interpretation**: This is the first positive Phase 5 signal at the reliability layer. It does not establish literary quality or style transfer, but it suggests the contract adapter reduces visible output-contract violations compared with the base model under the same prompt.

### 2026-05-01 — Phase 5 Scaled Contract Adapter Probe

**Run**: `phase5-lora-contract-qwen05b-20260501b` scaled the same validated DGX Spark PyTorch PEFT/TRL lane to `512` contract-smoke training examples and `80` optimizer steps on `Qwen/Qwen2.5-0.5B-Instruct`. The run stayed inside `nvcr.io/nvidia/pytorch:25.11-py3`, used rank `8`, gradient accumulation `4`, and recorded git SHA `6f02db352e65b1a76b7439bd6e1f0e86c984c103`. Final train loss was about `2.62`, with runtime about `49` seconds.

**Contract probe**: Ran the repo-native adapter/base comparison on the first `32` validation examples. The adapter produced `0/32` empty outputs, `0/32` prompt-scaffold echoes, and `1/32` outputs with forbidden contract markers. The unfine-tuned base model produced `0/32` empty outputs, `0/32` prompt-scaffold echoes, and `10/32` outputs with forbidden contract markers.

**Interpretation**: The reliability signal strengthened from the first `8`-example probe to a larger `32`-example probe. This still does not establish literary reconstruction quality, but it makes the next experiment clearer: move from contract reliability toward controlled target-style evaluation while preserving the same container and probe discipline.
