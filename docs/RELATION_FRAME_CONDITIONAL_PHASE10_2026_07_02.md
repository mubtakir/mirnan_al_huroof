# RelationFrame Conditional Bridge - Phase 10

This phase starts the conditional bridge after the purpose bridge.

The goal is deliberately narrow:

- improve `RelationFrame` extraction for conditional markers;
- add an independent `conditional_answer`;
- keep it out of `generate!` for now.

## Conditional Splitting

Before this phase, a sentence such as:

```text
إذا زاد العلم زاد الفهم.
```

could be detected as `conditional`, but the post-marker side was treated as one
undivided side.

Now initial conditional frames split the post-marker terms into:

```text
condition: زاد علم
result:    زاد فهم
```

The internal terms remain normalized keys. Surface formatting restores readable
Arabic in the answer.

## Independent Answer

New API:

```julia
conditional_answer(mem::IstinbatAttentionMemory, prompt::AbstractString) -> String
```

Example:

```text
prompt:
  ماذا يحدث إذا زاد العلم؟

answer:
  إذا زاد العلم، فالنتيجة زاد الفهم.
```

## Guards

`conditional_answer` returns empty when:

- the prompt is a yes/no question;
- no conditional frame exists;
- the selected frame is not `conditional`;
- the overlap with the prompt is too weak;
- either condition or result is empty.

It does not enter generation strategy ordering yet.

## Files

- `models/mirnan/src/physics/engines/al_istinbat.jl`
  - exported `conditional_answer`;
  - added ordered tokenization for conditional splitting;
  - added `_conditional_split_terms`;
  - added independent `conditional_answer`.

- `models/mirnan/src/physics/groups/arabic_group.jl`
  - re-exported `conditional_answer`.

- `models/mirnan/test/test_relation_frame.jl`
  - added conditional split tests;
  - added `conditional_answer` behavior and guard tests.

## Verification

```text
test_relation_frame.jl
  extract_relation_frames: conditional split | 3/3
  conditional_answer: matches conditional question | 4/4
  conditional_answer: guards | 3/3

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```

## Trained Selection Guard

The trained response-polisher probe exposed a selection problem that was not a
surface-polishing issue: a conditional prompt such as:

```text
Ù…Ø§Ø°Ø§ ÙŠØ­Ø¯Ø« Ø¥Ø°Ø§ Ø²Ø§Ø¯ Ø§Ù„Ø¹Ù„Ù…ØŸ
```

could match a distant trained conditional record merely because the record
contained partial overlap with `Ø¹Ù„Ù…` and `Ø²Ø§Ø¯`. The answer could then import an
unrelated domain.

The conditional selector now extracts the prompt condition terms after the
conditional marker (`Ø¥Ø°Ø§`, `Ø§Ø°Ø§`, `Ø¥Ù†`, `Ù„Ùˆ`, `if`, `when`) and requires those
terms to match the condition side of the candidate record, not just anywhere in
the record. This keeps the answer layer conservative: if the memory does not
contain a matching condition, it returns empty instead of borrowing a loosely
related record.

Added guard coverage:

```text
conditional_answer: repairs shortened trained record from example | 4/4
conditional_answer: rejects distant partial overlap               | 1/1
```

Verified with the trained-memory probe:

```text
CASE: conditional
PROMPT: Ù…Ø§Ø°Ø§ ÙŠØ­Ø¯Ø« Ø¥Ø°Ø§ Ø²Ø§Ø¯ Ø§Ù„Ø¹Ù„Ù…ØŸ
-- polisher off --
Ø¥Ø°Ø§ Ø²Ø§Ø¯ Ø§Ù„Ø¹Ù„Ù…ØŒ ÙŠØªØ±ØªØ¨ Ø¹Ù„Ù‰ Ø°Ù„Ùƒ Ù‚Ù„Ù‘Øª Ø§Ù„Ø®Ø±Ø§ÙØ§Øª.
-- polisher on --
Ø¥Ø°Ø§ Ø²Ø§Ø¯ Ø§Ù„Ø¹Ù„Ù…Ø› ÙŠØªØ±ØªØ¨ Ø¹Ù„Ù‰ Ø°Ù„Ùƒ Ù‚Ù„Ù‘Øª Ø§Ù„Ø®Ø±Ø§ÙØ§Øª.
```

## Next Step

Done in the same phase as a narrow diagnostic extension:

```text
compare_conditional_strategies(mem, generate_func, prompt)
```

The gated strategy was then added as the final step of this phase.

## Conditional Comparison

New API:

```julia
ConditionalComparisonRecord
compare_conditional_strategies(mem::IstinbatAttentionMemory,
                               generate_func::Function,
                               prompt::AbstractString)
```

The record mirrors the purpose comparison layer:

- `conditional_answer`
- `generate_answer`
- `memory_has_conditional`
- `conditional_confidence`
- `overlap_score`
- `has_marker`
- `relation_type`

This is still diagnostic only. It does not change `generate!`, strategy ordering,
or yes/no behavior.

## Probe

New script:

```text
models/mirnan/scripts/conditional_relation_probe.jl
```

It loads trained `al_istinbat.json`, reports conditional record count, runs a
controlled sanity example, scans for one trained conditional example, and checks
the yes/no guard.

Latest observed probe:

```text
Conditional RelationFrame probe
istinbat_records: 26943
conditional_records: 10534

controlled conditional sanity
  prompt: ماذا يحدث إذا زاد العلم؟
  answer: إذا زاد العلم، فالنتيجة زاد الفهم.

trained conditional
  memory_has_conditional: true
  conditional_confidence: 0.8
  overlap: 0.857

yes/no guard
  prompt: هل زاد العلم؟
  conditional_answer: empty
```

The trained example can still be linguistically noisy because it reflects the
source corpus sentence. The diagnostic proves the bridge works; polishing trained
conditional phrasing is a later quality pass.

## Gated Strategy

New strategy:

```text
ConditionalFrameStrategy
```

Files:

- `models/mirnan/src/physics/engines/strategies/base.jl`
- `models/mirnan/src/physics/engines/strategies/conditional_frame_strategy.jl`
- `models/mirnan/src/physics/engines/generator.jl`

Environment gate:

```text
MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY=1
```

Default behavior is unchanged. When enabled, the strategy is inserted before the
general `AqlStrategy`/`RelationStrategy` path, because otherwise a generic
conditional answer can consume the prompt before the precise RelationFrame answer
is tried.

Live probe:

```text
models/mirnan/scripts/conditional_frame_strategy_live_probe.jl
```

Observed behavior:

```text
controlled conditional
  gate off: إذا زاد العلم، ظهرت النتيجة، لأن الشرط يفتح طريق النتيجة...
  gate on:  إذا زاد العلم، فالنتيجة زاد الفهم.

yes/no guard
  gate off == gate on

definition guard
  gate off == gate on
```

## Additional Verification

```text
test_relation_frame.jl
  compare_conditional_strategies: returns ConditionalComparisonRecord | 9/9
  compare_conditional_strategies: empty memory | 4/4
  compare_conditional_strategies: non-conditional question | 3/3
  compare_conditional_strategies: yesno question ignored | 3/3
  ConditionalFrameStrategy | 5/5

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29
```
