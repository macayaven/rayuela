# Phase 5 Style-Transfer Adapter Plan

Date: 2026-05-01

## Target Outcome

Build a reproducible style-transfer adapter that rewrites a source passage into
a target literary style envelope while preserving the source scene, facts,
characters, order, and approximate length.

The reader-facing artifact is not a generic model demo. It is a controlled
gallery of cross-author transmutations, with metrics and commentary showing
what changed stylistically, what stayed semantically stable, and where the
method failed.

## Research Claim Ladder

1. Output-contract reliability: the model emits only the final passage.
2. Distilled style-transfer training: prompt-generated, scored examples can be
   converted into supervised adapter data.
3. Style movement: generated passages move measurably toward the target
   stylometric envelope.
4. Semantic preservation: the transformation preserves the source event well
   enough to be interpretable.
5. Article claim: the pipeline applies a learned literary style field to
   external narrative material under an auditable measurement contract.

Do not claim imitation of a named author until both style movement and semantic
preservation pass on held-out examples.

## Artifacts

- `outputs/reconstruction/runs/<teacher_run>/prompt_baseline_cases.json`: scored
  teacher candidates from Phase 4.
- `outputs/reconstruction/style_distill/<dataset_id>/training_dataset/*.jsonl`:
  distilled supervised examples.
- `outputs/reconstruction/runs/<style_adapter_run>/adapter/`: LoRA adapter
  trained on distilled style-transfer data.
- `outputs/reconstruction/runs/<style_adapter_run>/style_probe.json`: held-out
  probe comparing base, contract adapter, and style adapter.
- `outputs/reconstruction/runs/<style_adapter_run>/article_gallery.json`: small
  curated set of examples with metrics and failure labels.

## Execution Steps

1. Distill Phase 4 teacher examples into SFT JSONL.
   - Use only scoreable teacher generations.
   - Record teacher run ID, case ID, target envelope, score history, and split.
   - Keep generated text in ignored `outputs/`, not in tracked files.
2. Add train-from-distilled-data support to `src/reconstruction_train.py`.
   - Preserve the existing immutable run contract.
   - Use the same DGX Spark PyTorch PEFT/TRL container lane.
   - Keep `contract_smoke` as the reliability baseline.
3. Train the first style adapter on a small distilled dataset.
   - Start with Qwen 0.5B, rank `8`, `80-200` steps.
   - Compare against the base model and the latest contract adapter.
4. Add a style probe.
   - Measure contract failures, length ratio, lexical overlap, style-target
     distance, and semantic-source distance.
   - Produce paired base/contract/style outputs on the same held-out examples.
5. Scale only after the style probe is useful.
   - Increase teacher data volume before model size.
   - Prefer better teacher filtering over longer training if semantic drift is
     the main failure.
6. Build the article gallery.
   - Select `3-5` examples.
   - Include at least one failure.
   - Phrase claims as measured stylistic displacement, not author recreation.

## Near-Term Run Order

1. Create a distilled dataset from the best available Phase 4 rescue runs.
2. Train a small style adapter using the validated DGX Spark container lane.
3. Probe `32` held-out examples against base, contract adapter, and style
   adapter.
4. If semantic preservation is poor, improve teacher filtering before scaling.
5. If contract behavior regresses, mix contract examples into the style dataset.

## Risk Controls

- Copyright: do not track generated/source text examples in Git.
- Leakage: final evaluation must use held-out cases that were not teacher
  examples for the style adapter.
- Runtime drift: official training and evaluation remain in the NVIDIA PyTorch
  DGX Spark container.
- Overclaiming: a good qualitative example is not enough; the article needs the
  metric table and the failure analysis.
- Teacher weakness: low-scoring prompt outputs should be treated as provisional
  data, not ground truth.
