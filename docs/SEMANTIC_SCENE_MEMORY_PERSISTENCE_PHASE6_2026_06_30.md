# SemanticScene Memory Persistence - Phase 6

This note documents the transition of semantic imagination scenes from a test-time memory to a trained model memory.

## Scope

Phase 6 adds official persistence for semantic scene memory:

- `semantic_scenes_to_dict(mem)`
- `save_semantic_scenes(mem, path)`
- `load_semantic_scenes(path)`
- model artifact: `semantic_scenes.json`

## Training

`train.jl` now builds semantic scene memory after `al_hisban_al_dalali`:

```text
5.0i building semantic scene imagination memory...
```

This order matters because scene extraction can use semantic-calculus guidance when available.

## Loading

`MirnanGenerator` now loads:

```text
semantic_scenes.json
```

The loaded memory is stored in:

```julia
gen.semantic_scenes
```

The existing `_LEARNED_SEMANTIC_SCENE_MEMORY` remains available as a temporary override for tests and experiments, but the formal path is now the generator field.

## Strategy Behavior

`SemanticSceneStrategy` first checks the temporary override. If it is absent or empty, it falls back to:

```julia
gen.semantic_scenes
```

The strategy remains gated by:

```text
MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=1
```

## Summary

Semantic imagination is now a model memory, not only an isolated probe. It can be trained, saved, loaded, summarized, and used by the gated strategy.

