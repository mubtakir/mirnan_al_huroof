# Recent Changes - 2026-07-01

This note summarizes the latest Mirnan work around semantic imagination and gated generation.

## Semantic Imagination Stabilized

The `SemanticScene` layer is now a practical sensory/event layer for Mirnan. It extracts simple event scenes:

```text
actor | action | patient | semantic effects
```

Example:

```text
Khalid hit the ball
-> action=hit, patient=ball
-> movement, away, position change
```

This layer is intentionally separate from strict rational inference. It gives Mirnan a physical-semantic image of events before the rational/question layers decide how to answer.

## New/Updated Files

- `src/physics/engines/semantic_imagination.jl`
- `src/physics/engines/strategies/semantic_scene_strategy.jl`
- `scripts/build_semantic_scenes.jl`
- `scripts/semantic_scene_strategy_live_probe.jl`
- `test/test_semantic_imagination.jl`
- `docs/SEMANTIC_SCENE_STRATEGY_STABILIZATION_PHASE7_2026_07_01.md`

## Builder Improvements

`build_semantic_scenes.jl` now rebuilds only `semantic_scenes.json` for an already trained model.

It now avoids common noise sources:

- code files
- quarantined data
- PowerShell captures
- API/server code reports
- agent experience corpus by default

It also adds a small calibrated seed set for physical events:

- hit/push
- break
- illuminate
- Arabic equivalents

## Strategy Gate

The scene strategy is still off by default.

Enable it with:

```text
MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=1
```

When enabled, it is inserted before `CodeStrategy`, so event/effect prompts can be answered by the semantic scene layer before broad code memories intercept them.

## Safety

The strategy answers only event/effect questions. It returns empty for:

- `هل` / yes-no questions
- definition questions
- social replies
- unrelated conceptual prompts
- prompts with incompatible scene actions

Important guard verified:

```text
What happens when Khalid hit the ball?
```

cannot be answered from a `broke cup` scene.

## Live Probe Result

Latest successful live probe showed:

```text
semantic_scenes: 531

What happens when Khalid hit the ball?
gate off -> generic noisy continuation
gate on  -> When hit affects ball, the semantic effect includes movement, away, position change.

Does Khalid hit the ball?
semantic scene answer -> empty

ما معنى السلام؟
semantic scene answer -> empty
definition route preserved
```

## Focused Test Result

Latest focused semantic imagination test:

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

## Rich Scene Fields

After the strategy stabilization, `SemanticScene` was extended with richer sensory fields:

- instrument
- place
- time marker
- state before
- state after
- affect tone

This is a structural extension only. It does not change the default gate behavior or the existing question routes.

The semantic answer formatter now uses these fields when they are present, so a selected scene can mention the instrument, place, time, and state transition. The selection gate itself did not change.

## SemanticScene Purpose Bridge

Added the first inspection bridge between the sensory scene layer and the rational purpose layer:

- `ScenePurposeComparisonRecord`
- `compare_scene_purpose_strategies(scene_mem, calculus, istinbat_mem, prompt)`

The bridge reports one of:

- `cooperative`
- `scene_only`
- `purpose_only`
- `none`

It does not enter generation yet. It is a diagnostic/comparison layer for safely studying how semantic imagination and `RelationFrame` purpose inference cooperate.

User-run tests:

```text
compare_scene_purpose_strategies: cooperative bridge | 13/13
compare_scene_purpose_strategies: isolated sides      |  7/7
```

Added a live diagnostic script:

```text
models/mirnan/scripts/semantic_scene_purpose_bridge_probe.jl
```

It loads trained memories directly and prints bridge cases for:

- cooperative
- scene_only
- purpose_only
- none

The bridge/probe now includes a light quality guard so overly long purpose frames are not treated as usable answers in the bridge display.

Phase 9.2 stabilized the probe further:

- indexed bridge search is now enabled in `semantic_scene_purpose_bridge_probe.jl`
- `scene_only` has its own scan limit via `MIRNAN_BRIDGE_SCENE_ONLY_LIMIT`
- controlled `scene_only` sanity output is printed when no trained example is found
- noun-only purpose records are rejected before becoming `purpose_answer` output
- a small cooperation seed corpus was added at:

```text
models/mirnan/data/toy_corpus/aaa0_mafahem/000_semantic_scene_purpose_cooperation.txt
```

The dedicated phase note is:

```text
models/mirnan/docs/SEMANTIC_SCENE_PURPOSE_BRIDGE_PHASE9_2_2026_07_01.md
```

## Developer Commands

Build semantic scene memory:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\build_semantic_scenes.jl
```

Run semantic imagination tests:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_semantic_imagination.jl
```

Run live gate probe:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_strategy_live_probe.jl
```

## Next Work

The next development step is to use the richer scene fields without breaking the gate:

- include instrument/place/time in semantic-scene diagnostics
- optionally include state-before/state-after in event/effect answers
- connect scene effects with `RelationFrame` when a question asks why or for what purpose
- connect scene quantities with `QuantityFrame` when the prompt includes count, measure, or comparison

This keeps the sensory scene layer separate from strict inference while preparing both layers to cooperate.
