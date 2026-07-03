# SemanticScene Purpose Strategy - Phase 9.4

This phase adds an experimental generation strategy for the composite bridge between:

- `SemanticSceneMemory`: sensory semantic scene/effect memory;
- `IstinbatAttentionMemory`: rational `RelationFrame` purpose memory.

The strategy is:

```julia
ScenePurposeStrategy <: GenerationStrategy
```

It calls:

```julia
scene_purpose_answer(scene_mem, gen.hisban, istinbat_mem, prompt)
```

## Gate

The strategy is off by default.

It is inserted into `generate!` only when:

```text
MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY=1
```

This keeps normal generation unchanged.

## Order

When enabled, `ScenePurposeStrategy` is inserted before `SemanticSceneStrategy` and
`CodeStrategy`.

Reason:

- the composite answer is more specific than a scene-only answer;
- it should get the first chance only when both sensory scene and purpose memory
  cooperate;
- if it returns `nothing`, the rest of Mirnan continues normally.

## Safety

The strategy returns `nothing` unless:

- semantic scene memory exists;
- istinbat purpose memory exists;
- `scene_purpose_answer` returns a non-empty cooperative answer.

The cooperative answer is now stricter than "scene exists + purpose exists":

- the selected scene must overlap the prompt through action/patient-like terms;
- the selected scene must also overlap the selected purpose record.

This prevents unrelated sensory scenes from being combined with an unrelated purpose
record in long procedural prompts.

It does not answer yes/no questions and does not affect definitions or social replies.

## Tests

Added test group:

```text
ScenePurposeStrategy
```

The tests cover:

- direct `try_generate(ScenePurposeStrategy(), ...)` returns the composite answer;
- the answer does not leak raw `action=...` diagnostic fields;
- yes/no prompts are ignored;
- `generate!` uses the strategy only when the gate is enabled.
- unrelated scene + purpose memories do not produce a composite answer.

Latest observed result:

```text
ScenePurposeStrategy | 6/6
```

The wider `test_relation_frame.jl` run also kept the previous bridge checks green:

```text
scene_purpose_answer: cooperative composite only | 9/9
RelationFrameStrategy                           | 9/9
```

## Live Probe

Added:

```text
models/mirnan/scripts/scene_purpose_strategy_live_probe.jl
```

This script loads the trained model and prints:

- independent `scene_purpose_answer`;
- `generate!` with the gate off;
- `generate!` with the gate on.

During the comparison it explicitly disables:

```text
MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=0
MIRNAN_ENABLE_RELATION_FRAME_STRATEGY=0
```

so the observed difference comes from `ScenePurposeStrategy` itself.

Run:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\scene_purpose_strategy_live_probe.jl
```

## Latest Live Result

The live probe confirmed:

```text
trained cooperative:
  gate off -> normal fallback answer
  gate on  -> composite scene + purpose answer

known event/purpose:
  gate on  -> composite answer when both memories match

purpose only guard:
  independent answer -> empty
  gate on            -> same as gate off

yes/no guard:
  independent answer -> empty
  gate on            -> same as gate off

definition guard:
  independent answer -> empty
  gate on            -> same as gate off
```

This verifies that the strategy no longer combines an unrelated scene with an unrelated
purpose record in the food-storage prompt.
