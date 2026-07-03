# SemanticScene Purpose Composite - Phase 9.3

This phase adds a diagnostic composite answer that combines:

- the sensory semantic scene layer (`SemanticSceneMemory` + semantic calculus);
- the rational purpose layer (`RelationFrame` purpose memory in `IstinbatAttentionMemory`).

The new function is:

```julia
scene_purpose_answer(scene_mem, calculus, istinbat_mem, prompt)
```

## Behavior

The function returns a non-empty answer only when the bridge state is:

```text
cooperative
```

That means:

- the semantic scene layer found an event/scene;
- the purpose layer found a valid purpose frame;
- both are available for the same prompt.

If the result is `scene_only`, `purpose_only`, or `none`, the function returns an empty
string. These isolated states remain diagnostic and do not become final answers.

## Formatting

When the scene layer has a ready `semantic_scene_answer`, the composite uses it.

When the scene answer is empty but the comparison has a scene summary, the composite
uses the scene summary instead. This matters for `why` prompts, where the scene is
often detected but the standalone scene-answer gate may not fire.

Raw diagnostic summaries such as:

```text
action=push | patient=stone | before=stillness | after=motion
```

are converted into readable sentences before entering the composite answer.

Example shape:

```text
scene/effect part. In terms of purpose: purpose part.
```

For Arabic output:

```text
... ومن جهة الغاية: ...
```

If the purpose answer already starts with `the purpose of ... is ...`, the composite
keeps only the purpose fragment after `is`. This avoids repeating the word "purpose"
twice in Arabic output.

## Safety

This phase does not:

- enter `generate!`;
- change strategy order;
- change yes/no answers;
- change definition answers;
- make learned purpose records canonical.

It is an inspection/composition layer only.

## Tests

Added test group:

```text
scene_purpose_answer: cooperative composite only
```

The tests cover:

- cooperative scene + purpose -> non-empty composite answer;
- scene-only -> empty;
- purpose-only -> empty.
- no raw `action=...` diagnostic fields leak into the composite answer;
- Arabic purpose text does not repeat `the purpose of ... is ...` after the phrase
  "from the side of purpose".

Latest observed result:

```text
scene_purpose_answer: cooperative composite only | 9/9
```

## Probe Display

The live bridge probe now prints:

```text
-- composite answer --
```

This field is shown only from the diagnostic result. It is empty unless the case is
`cooperative`.

Latest live probe shape:

```text
CASE: trained cooperative
AGREEMENT: cooperative
-- composite answer --
when pushing the stone, the affected thing shifts from stillness to motion and
position change. From the side of purpose: moving the stone away from the road.

CASE: trained scene_only
-- composite answer --
(empty)

CASE: trained purpose_only
-- composite answer --
(empty)

CASE: trained none
-- composite answer --
(empty)
```
