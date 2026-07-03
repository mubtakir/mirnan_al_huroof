# SemanticScene Strategy Comparison - Phase 4.5

This note documents the comparison layer for Mirnan semantic imagination answers.

## Scope

Phase 4.5 adds:

- `SemanticSceneAnswerComparison`
- `compare_semantic_scene_strategies(scene_mem, calculus, generate_func, prompt)`

The function compares the standalone semantic-scene answer with a supplied general generation function.

## Why

The semantic imagination layer should not enter `generate!` directly before we can observe how it behaves beside the existing answer path. This comparison record gives us a stable inspection point.

## Record Fields

- `prompt`
- `scene_answer`
- `generate_answer`
- `memory_has_scene`
- `question_allowed`
- `agreement`
- `overlap_score`
- `scene_confidence`
- `guidance_confidence`

## Boundary

This phase does not add a generation strategy and does not modify `generate!`.

The general answer path is injected as a function:

```julia
compare_semantic_scene_strategies(mem, calculus, p -> generate!(gen, p), prompt)
```

That keeps the semantic imagination engine independent from the generator.

## Failure Behavior

If the supplied generation function throws an error, `generate_answer` becomes an empty string. The semantic-scene answer is still measured independently.

## Tests

The added tests verify:

- comparison record creation
- independent scene answer presence
- injected general answer capture
- non-scene prompts remain blocked by the scene gate
- generation-function failure does not break the comparison

