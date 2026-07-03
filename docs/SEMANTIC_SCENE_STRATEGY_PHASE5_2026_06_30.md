# SemanticScene Strategy - Phase 5

This note documents the first gated generation strategy for Mirnan semantic imagination.

## Scope

Phase 5 adds:

- `SemanticSceneStrategy <: GenerationStrategy`
- `strategies/semantic_scene_strategy.jl`
- `_LEARNED_SEMANTIC_SCENE_MEMORY`
- optional insertion into `generate!` behind:

```text
MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=1
```

## Default Behavior

The strategy is disabled by default. With no environment variable, `generate!` keeps the previous behavior.

## Answer Source

The strategy calls:

```julia
semantic_scene_answer(scene_mem, gen.hisban, prompt)
```

It therefore still depends on:

- semantic scene memory
- `al_hisban_al_dalali`
- the scene-question gate from Phase 4

## Boundary

This strategy does not create ready-made knowledge answers. It only verbalizes semantic event effects when:

- a scene memory exists
- the prompt asks about event effect
- the scene/calculus comparison is aligned or partial
- overlap is positive

It should not answer yes/no, definition, or why questions.

## Tests

The added test covers:

- direct `try_generate(SemanticSceneStrategy(), ...)`
- yes/no prompt rejection
- empty scene memory rejection

The test uses a temporary scene memory and restores the previous global memory after completion.

