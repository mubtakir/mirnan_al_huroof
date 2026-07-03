# Post-Training Quality Pass - 2026-07-03

This note records the final cleanup pass after the latest paragraph-level
training run.

## Training Launcher

- `models/mirnan/run_train.bat` now keeps the terminal open at the end of the
  pipeline, both on success and failure.
- The one-click pipeline runs:
  - training with `MIRNAN_SEGMENT_LEVEL=paragraph`
  - `rebuild_istinbat_memory.jl`
  - `run_evolution.jl`
  - `build_semantic_scenes.jl`
  - `rebuild_quantity_memory.jl`
  - `post_training_smoke.jl`

## RelationFrame Phrasing

- Conditional Arabic answers now use smoother result wording:
  - `إذا زاد العلم؛ فالنتيجة: زاد الفهم.`
- Spatial Arabic answers normalize the learned marker `حيث` into natural
  answer phrasing:
  - `كان مكان جلوس الطفل في الحديقة.`

## Quantity Memory Cleanup

- Quantity extraction now cleans dangling target prepositions before storing a
  frame. This prevents targets such as `صيغ المبالغة أوزان سماعية في`.
- The rebuilt trained quantity memory contains 373 frames:
  - comparison: 110
  - count: 178
  - measure: 9
  - quantifier_scope: 50
  - vague_quantity: 26
- The selected vague-quantity probe is now:
  - target: `صيغ المبالغة أوزان سماعية`
  - value: `من الأحيان`

## Verification

- `test_relation_frame.jl`: passed, including the new target-preposition guard.
- `trained_quantity_memory_probe.jl`: passed and shows cleaned quantity targets.
- `post_training_smoke.jl`: passed:
  - response polisher
  - semantic scene / purpose bridge
  - quantity memory
