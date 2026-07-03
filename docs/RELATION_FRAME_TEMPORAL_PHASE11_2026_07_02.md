# RelationFrame Temporal Bridge - Phase 11

This phase adds the first temporal answer layer on top of RelationFrame.

The scope is deliberately narrow:

- add an independent `temporal_answer`;
- add a diagnostic comparison record;
- keep temporal answers out of `generate!` for now.

## Independent Answer

New API:

```julia
temporal_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String
```

Example:

```text
memory:
  سافر الطالب قبل الفجر.

prompt:
  متى سافر الطالب؟

answer:
  سافر الطالب قبل الفجر.
```

The internal memory keeps normalized terms, so formatting restores readable
Arabic where possible. For example, an internal `فجر` is rendered as `الفجر` in
the temporal answer.

## Guards

`temporal_answer` returns empty when:

- the prompt is a yes/no question;
- the prompt is not temporal;
- no temporal frame exists;
- the selected frame is not `temporal`;
- the overlap with the prompt is too weak;
- the time side is empty.

## Comparison

New API:

```julia
TemporalComparisonRecord
compare_temporal_strategies(mem::IstinbatAttentionMemory,
                            generate_func::Function,
                            prompt::AbstractString)
```

The comparison is diagnostic only. It does not change generation.

## Files

- `models/mirnan/src/physics/engines/al_istinbat.jl`
  - exported `temporal_answer`;
  - added temporal question detection;
  - added temporal formatting;
  - added `TemporalComparisonRecord`;
  - added `compare_temporal_strategies`.

- `models/mirnan/src/physics/groups/arabic_group.jl`
  - re-exported temporal APIs.

- `models/mirnan/test/test_relation_frame.jl`
  - added temporal answer tests;
  - added temporal comparison tests.

## Verification

```text
test_relation_frame.jl
  temporal_answer: matches temporal question | 4/4
  temporal_answer: guards | 3/3
  compare_temporal_strategies: returns TemporalComparisonRecord | 9/9
  compare_temporal_strategies: guards | 5/5

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```

## Gated Strategy

New strategy:

```text
TemporalFrameStrategy
```

Files:

- `models/mirnan/src/physics/engines/strategies/base.jl`
- `models/mirnan/src/physics/engines/strategies/temporal_frame_strategy.jl`
- `models/mirnan/src/physics/engines/generator.jl`

Environment gate:

```text
MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY=1
```

Default behavior is unchanged. When enabled, the strategy is inserted before the
general `AqlStrategy`/`RelationStrategy` path, matching the conditional strategy.

Live probe:

```text
models/mirnan/scripts/temporal_frame_strategy_live_probe.jl
```

Observed behavior:

```text
controlled temporal
  gate off: generic fallback terms
  gate on:  سافر الطالب قبل الفجر.

yes/no guard
  gate off == gate on

non-temporal guard
  gate off == gate on
```

## Additional Verification

```text
test_relation_frame.jl
  TemporalFrameStrategy | 5/5

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```

## Next Step

The same pattern can now be repeated for `spatial` frames, then `state` frames.
