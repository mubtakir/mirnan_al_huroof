# SemanticScene Rich Fields - Phase 8

This phase extends `SemanticScene` from a minimal event frame into a richer sensory scene frame.

## New Fields

`SemanticScene` now stores:

- `instrument`
- `place`
- `time_marker`
- `state_before`
- `state_after`
- `affect_tone`

The original core remains:

- `actor`
- `action`
- `patient`
- `effect_candidates`

## Compatibility

The old constructor shape remains supported:

```julia
SemanticScene(sentence, actor, action, patient, effect_candidates,
              confidence, guidance_relation, source)
```

Old `semantic_scenes.json` files also load safely because missing new fields default to empty strings.

## Extraction

The first extraction pass is deliberately conservative:

- instrument from markers like `with`, `using`, `by`
- place from markers like `in`, `inside`, `at`, `on`
- time from markers like `before`, `after`, `during`, `when`
- before/after state from the action family
- affect tone from the action family

Known action-family states:

```text
motion -> stable -> moved and changed position -> neutral
break  -> whole  -> damaged and separated      -> disruptive
light  -> unclear -> visible and clear         -> revealing
```

Arabic equivalents are also generated for the same action families.

## Persistence

The new fields are included in:

- `semantic_scenes_to_dict`
- `save_semantic_scenes`
- `load_semantic_scenes`

This means future trained scene memories keep the richer sensory structure.

## Answer Wording

When a selected scene has rich fields, `semantic_scene_answer` can now include them in the answer:

```text
When hit affects ball, using bat, in yard, before/around sunset,
the semantic effect includes movement, away, position change,
and the affected thing shifts from stable to moved and changed position.
```

Arabic answers use the same structure with Arabic labels and state wording.

The gate and answer conditions did not change. The richer wording is used only after a semantic scene was already selected and accepted.

## Diagnostics

`semantic_scene_comparison_diagnostic` now reports:

- `scene_instrument`
- `scene_place`
- `scene_time`
- `scene_state_before`
- `scene_state_after`
- `scene_affect_tone`

## Test Coverage

Added and expanded a focused testset:

```text
semantic imagination rich scene fields | 21/21
```

Latest focused semantic imagination result:

```text
scene extraction                 17/17
rich scene fields                21/21
probe matrix fixture             33/33
calculus comparison              34/34
scene memory                     34/34
independent answer               22/22
answer comparison                16/16
strategy gate                    13/13
```

Total: `190/190`.

The extra assertion verifies that scene `patient` stops before rich detail markers such as instrument/place/time. Example:

```text
Khalid hit the ball with bat in yard before sunset
patient = ball
instrument = bat
place = yard
time = sunset
```

## Status

This phase is structural and safe. It does not change the generation gate, question routes, yes/no logic, definition logic, or relation strategies.

The next step is interaction between semantic imagination and `RelationFrame`, especially for why/purpose questions that also contain physical scene effects.
