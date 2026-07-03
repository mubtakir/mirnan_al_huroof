# SemanticScene Memory - Phase 2

This note documents the second phase of Mirnan semantic imagination.

## Scope

Phase 2 adds an independent memory and selection layer for semantic scenes. It does not affect generation, answer strategies, `yesno_relations.jl`, or `relation_strategy.jl`.

## New API

- `SemanticSceneMemory`
- `learn_semantic_scene_from_text!(mem, calculus, text)`
- `train_semantic_scenes_from_texts!(mem, calculus, texts)`
- `has_semantic_scenes(mem)`
- `select_semantic_scene(mem, prompt)`
- `semantic_scene_diagnostic(mem, prompt)`

## Behavior

- Stores scenes that have a central action and semantic effects.
- Ignores quiet declarative sentences that do not produce a sensory/event scene.
- Selects the closest scene by overlap between the prompt and the scene actor, action, patient, and effect terms.
- Keeps a bounded scene list through `max_scenes`.
- Provides a diagnostic report for developers.

## Architecture Rule

This phase follows the agreed development sequence:

1. Extract.
2. Store.
3. Select.
4. Diagnose.
5. Only later consider answer generation behind a gate.

## Tests

`test_semantic_imagination.jl` now covers:

- Scene extraction.
- Scene memory storage.
- Empty-memory behavior.
- Scene selection.
- Diagnostic output.
- Capacity limiting.
