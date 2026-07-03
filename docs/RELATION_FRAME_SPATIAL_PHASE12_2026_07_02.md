# RelationFrame Spatial Bridge - Phase 12

This phase adds spatial RelationFrame answering and a gated strategy.

## Independent Answer

New API:

```julia
spatial_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String
```

Example:

```text
memory:
  جلس الطفل حيث الحديقة.

prompt:
  أين جلس الطفل؟

answer:
  جلس الطفل حيث الحديقة.
```

The internal memory keeps normalized terms. Spatial output restores readable
Arabic, including common place endings such as `الحديقه` -> `الحديقة`.

## Guards

`spatial_answer` returns empty when:

- the prompt is a yes/no question;
- the prompt is not spatial;
- no spatial frame exists;
- the selected frame is not `spatial`;
- the overlap with the prompt is too weak;
- the place side is empty.

## Comparison

New API:

```julia
SpatialComparisonRecord
compare_spatial_strategies(mem::IstinbatAttentionMemory,
                           generate_func::Function,
                           prompt::AbstractString)
```

The comparison is diagnostic only.

## Gated Strategy

New strategy:

```text
SpatialFrameStrategy
```

Environment gate:

```text
MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY=1
```

Default behavior is unchanged. When enabled, the strategy is inserted before the
general `AqlStrategy`/`RelationStrategy` path.

Live probe:

```text
models/mirnan/scripts/spatial_frame_strategy_live_probe.jl
```

Observed behavior:

```text
controlled spatial
  gate off: generic fallback terms
  gate on:  جلس الطفل حيث الحديقة.

yes/no guard
  gate off == gate on

non-spatial guard
  gate off == gate on
```

## Files

- `models/mirnan/src/physics/engines/al_istinbat.jl`
- `models/mirnan/src/physics/groups/arabic_group.jl`
- `models/mirnan/src/physics/engines/strategies/base.jl`
- `models/mirnan/src/physics/engines/strategies/spatial_frame_strategy.jl`
- `models/mirnan/src/physics/engines/generator.jl`
- `models/mirnan/test/test_relation_frame.jl`
- `models/mirnan/scripts/spatial_frame_strategy_live_probe.jl`

## Verification

```text
test_relation_frame.jl
  spatial_answer: matches spatial question | 4/4
  spatial_answer: guards | 3/3
  compare_spatial_strategies: returns SpatialComparisonRecord | 9/9
  compare_spatial_strategies: guards | 5/5
  SpatialFrameStrategy | 5/5

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```

## Next Step

The remaining RelationFrame type in this series is `state`. After that, the
RelationFrame family should get a small shared helper to reduce duplication
between purpose, conditional, temporal, spatial, and state.
