# RelationFrame Selection Guards - 2026-07-03

This note records the quality guards added after trained-memory probes exposed
loose selection in non-purpose RelationFrame answers.

## Problem

The direct RelationFrame answer layers were correct on controlled tests, but
trained memories can contain noisy or loosely related records. A prompt could
match a record only because one or two words overlapped, then import an unrelated
answer.

Examples seen in probes:

- conditional prompts could match a distant medical-style record through partial
  overlap with `علم` and `زاد`;
- temporal answers could accept glue fragments such as `اللم`;
- spatial probes could build generic prompts such as `أين كان؟`;
- ambiguous temporal anchors such as `عند` could be used for non-time context.

## Fix

The answer selectors now distinguish between general overlap and relation-side
agreement.

- `conditional_answer` extracts the condition terms after `إذا/لو/if/when` and
  requires them to match the condition side of the record.
- `temporal_answer`, `spatial_answer`, and `state_answer` use typed selection
  instead of the generic `select_relation_frame_attention`.
- typed selection rejects:
  - distant partial overlap,
  - generic event-only prompts such as `أين كان؟`,
  - trained glue fragments such as `اللم`, `اللكن`, `يل`, `تي`,
  - ambiguous temporal anchors unless the time side contains a clear time word.
- the trained response-polisher probe can now fall back to an auto-built trained
  prompt for temporal/spatial/state, but it skips one-word subjects.

## Important Boundary

These guards are conservative. They do not invent a clean answer when the
trained memory is poor. If a trained temporal/spatial/state record is too noisy,
the correct behavior is to return empty or skip it, not to polish it into a false
answer.

The trained probe still shows that temporal/spatial/state need better corpus
examples, similar to the clean seed examples already added for scene-purpose and
quantity.

## Tests

Added guard coverage in `test_relation_frame.jl`:

```text
conditional_answer: rejects distant partial overlap
temporal_answer: rejects distant partial overlap
temporal_answer: rejects trained glue noise
temporal_answer: rejects ambiguous non-time anchor
spatial_answer: rejects distant partial overlap
spatial_answer: rejects generic event prompt
state_answer: rejects distant partial overlap
```

Verification:

```text
test_relation_frame.jl passed
trained_response_polisher_probe.jl ran with 6 trained/direct cases
```

## Next Step

Add small, clean training seeds for:

- temporal: before/after/during/since/when with explicit time words;
- spatial: where/inside/above/under/near with clear event-place pairs;
- state: while/حال/وهو/وهي with clear event-state pairs.

Then retrain or rebuild the relevant memory and rerun the trained probe.
