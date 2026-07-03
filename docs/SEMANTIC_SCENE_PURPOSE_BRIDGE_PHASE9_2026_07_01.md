# SemanticScene Purpose Bridge - Phase 9

This phase adds the first explicit bridge between semantic imagination and rational inference.

## Goal

Mirnan now has two separate layers:

- `SemanticScene`: sensory/event imagination
- `RelationFrame`: rational relation inference

Phase 9 does not merge them into generation yet. It adds an independent comparison layer so developers can inspect how both layers behave on the same prompt.

## New Record

```julia
ScenePurposeComparisonRecord
```

Fields:

- `prompt`
- `scene_answer`
- `scene_summary`
- `purpose_answer`
- `scene_has_event`
- `memory_has_purpose`
- `agreement`
- `scene_confidence`
- `purpose_confidence`
- `scene_relation`
- `purpose_relation`

## New Function

```julia
compare_scene_purpose_strategies(scene_mem, calculus, istinbat_mem, prompt)
```

It compares:

- the selected sensory scene from `SemanticSceneMemory`
- semantic-calculus guidance from `al_hisban_al_dalali`
- the selected `RelationFrame` purpose answer from `IstinbatAttentionMemory`

## Agreement Values

```text
cooperative   -> scene and purpose are both present
scene_only    -> sensory scene exists, no purpose frame
purpose_only  -> purpose frame exists, no sensory scene
none          -> neither layer has usable evidence
```

## What It Does Not Do

This phase does not:

- change `generate!`
- change `RelationFrameStrategy`
- change `SemanticSceneStrategy`
- change yes/no answers
- change definition answers
- make discovered purpose rules canonical

It is an inspection/bridge layer only.

## Quality Guard

Purpose records are not considered usable by the bridge unless they produce a non-empty `purpose_answer`.

The purpose formatter now rejects overly broad frames:

```text
left side  > 12 words -> reject
right side > 16 words -> reject
missing left side     -> reject
missing right side    -> reject
```

This prevents long paragraph fragments from being presented as a precise purpose answer.

## Example

Scene memory:

```text
Khalid hit the ball with bat in yard before sunset.
```

Purpose memory:

```text
Khalid hit the ball لكي يتحرك ball.
```

Prompt:

```text
لماذا Khalid hit the ball؟
```

Expected bridge result:

```text
agreement = cooperative
scene_summary contains action=hit, instrument=bat
purpose_relation = purpose
```

## Tests

New tests in `test_relation_frame.jl`:

```text
compare_scene_purpose_strategies: cooperative bridge | 13/13
compare_scene_purpose_strategies: isolated sides      |  7/7
```

Total new assertions: `20/20`.

Latest user-run result: all relation frame, quantity frame, relation strategy, and bridge tests passed.

## Next Step

The next safe step is a diagnostic command or script that prints the bridge result for real trained memory. After that, the bridge can feed a gated strategy, still off by default.
