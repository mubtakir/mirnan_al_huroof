# RelationFrame State Bridge - Phase 13

This phase completes the first RelationFrame question-family pass by adding
state/condition answering and a gated generation strategy.

## Independent Answer

New API:

```julia
state_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String
```

The function targets state markers such as `hal`/state phrases represented by
RelationFrame records. It answers questions such as "how did the event happen?"
when the memory has a matching `state` frame.

Example:

```text
memory:
  entered child state afraid

prompt:
  how entered child?

answer:
  entered child in state afraid
```

Arabic output preserves the marker phrase when available, for example:

```text
دخل الطفل حال خائف.
```

## Guards

`state_answer` returns empty when:

- the prompt is a yes/no question;
- the prompt is not a state/how question;
- no state frame exists;
- the selected frame is not `state`;
- the prompt overlap is too weak;
- the state side is empty.

## Comparison

New diagnostic API:

```julia
StateComparisonRecord
compare_state_strategies(mem::IstinbatAttentionMemory,
                         generate_func::Function,
                         prompt::AbstractString)
```

This compares `state_answer` with a caller-supplied generation function without
changing normal generation.

## Gated Strategy

New strategy:

```text
StateFrameStrategy
```

Environment gate:

```text
MIRNAN_ENABLE_STATE_FRAME_STRATEGY=1
```

Default behavior is unchanged. When enabled, the strategy is inserted before the
general `AqlStrategy`/`RelationStrategy` path.

Live probe:

```text
models/mirnan/scripts/state_frame_strategy_live_probe.jl
```

Observed behavior:

```text
controlled state
  gate off: generic fallback terms
  gate on:  دخل الطفل حال خائف.

yes/no guard
  gate off == gate on

non-state guard
  gate off == gate on
```

## Files

- `models/mirnan/src/physics/engines/al_istinbat.jl`
- `models/mirnan/src/physics/groups/arabic_group.jl`
- `models/mirnan/src/physics/engines/strategies/base.jl`
- `models/mirnan/src/physics/engines/strategies/state_frame_strategy.jl`
- `models/mirnan/src/physics/engines/generator.jl`
- `models/mirnan/test/test_relation_frame.jl`
- `models/mirnan/scripts/state_frame_strategy_live_probe.jl`

## Verification

```text
test_relation_frame.jl
  state_answer: matches state question | 4/4
  state_answer: guards | 3/3
  compare_state_strategies: returns StateComparisonRecord | 9/9
  compare_state_strategies: guards | 5/5
  StateFrameStrategy | 5/5

state_frame_strategy_live_probe.jl
  controlled state gate works
  yes/no guard unchanged
  non-state guard unchanged

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```

## Next Step

The RelationFrame family now has purpose, conditional, temporal, spatial, and
state answer paths. The next cleanup should extract the repeated selection,
comparison, and strategy boilerplate into small shared helpers before adding
QuantityFrame answer paths.
