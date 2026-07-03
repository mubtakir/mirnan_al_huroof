# SemanticScene Comparison - Phase 3

This note documents the third phase of Mirnan semantic imagination.

## Scope

Phase 3 adds a diagnostic comparison between:

- `SemanticSceneMemory`: the sensory/event scene layer.
- `al_hisban_al_dalali`: the Clifford semantic-calculus guidance layer.

It does not affect `generate!`, strategies, question routes, or ready-answer boundaries.

## New API

- `SemanticSceneComparison`
- `compare_semantic_scene_with_calculus(scene_mem, calculus, prompt)`
- `semantic_scene_comparison_diagnostic(scene_mem, calculus, prompt)`

## Comparison Result

`SemanticSceneComparison` records:

- the original prompt
- the selected scene, if any
- guidance terms from semantic calculus
- scene effect terms
- overlap score
- agreement label
- scene confidence
- guidance confidence
- guidance relation

## Agreement Labels

- `aligned`: scene effects and calculus guidance share strong overlap.
- `partial`: they share weak but real overlap.
- `divergent`: both exist but do not share terms.
- `scene_only`: only the sensory scene is available.
- `calculus_only`: only semantic calculus guidance is available.
- `none`: neither side has usable evidence.

## Tests

`test_semantic_imagination.jl` now covers:

- aligned or partial comparison.
- scene-only comparison.
- calculus-only comparison.
- diagnostic text output.

## Next Step

The next safe phase is not generation yet. A good next phase is to add a small comparison corpus/probe that prints scene-vs-calculus agreement for hand inspection.
