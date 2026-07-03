# SemanticScene Strategy Stabilization - Phase 7

This note documents the stabilization work after `semantic_scenes.json` became a trained model memory.

## Purpose

The semantic imagination layer is the sensory/event layer before strict rational inference. It lets Mirnan answer event/effect questions from semantic scenes such as:

```text
Khalid hit the ball -> movement, away, position change
```

It remains non-neural and non-statistical in the neural-network sense. It uses explicit event scenes and semantic-calculus guidance from `al_hisban_al_dalali`.

## Memory Build

Semantic scenes can be rebuilt without full model retraining:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\build_semantic_scenes.jl
```

The builder now:

- reads only `.txt` and `.md` training sources
- skips code-like and quarantined paths
- skips PowerShell/API/report-like files
- excludes agent experience by default
- can include agent experience only with `MIRNAN_SEMANTIC_SCENE_INCLUDE_AGENT=1`
- prints progress while collecting and processing texts
- writes `models/mirnan/model/semantic_scenes.json`

Default limits:

```text
MIRNAN_SEMANTIC_SCENE_TEXT_LIMIT=3000
MIRNAN_SEMANTIC_SCENE_LIMIT=20000
MIRNAN_SEMANTIC_SCENE_FILE_CHARS=200000
```

## Seed Scenes

The standalone builder adds a small, explicit set of seed event pairs before reading the corpus. These are not ready-made answers; they are calibration examples for physical event effects:

- hit/push -> movement, away, position change
- break -> separation, damage, shape change
- illuminate -> visibility, reveal, clarity

Arabic equivalents are included for:

- ضرب الكرة
- دفع الحجر
- كسر الكأس
- إضاءة الغرفة

## Strategy Gate

Generation uses the strategy only when:

```text
MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY=1
```

When enabled, `SemanticSceneStrategy` is inserted before `CodeStrategy`. This matters because code memories can otherwise answer broad English prompts before the semantic scene layer gets a chance.

The default remains off, so ordinary generation is unchanged unless the gate is explicitly enabled.

## Answer Guards

`semantic_scene_answer` answers only event/effect prompts, such as:

- `What happens when Khalid hit the ball?`
- `ماذا يحدث عندما يضرب خالد الكرة؟`
- `ما أثر ...؟`

It returns empty for:

- yes/no questions
- definition questions
- unrelated conceptual questions
- prompts without a compatible scene

This protects previous question behavior and social/definition routes.

## Compatibility Rules

Scene selection now checks action compatibility. A break scene cannot answer a hit question, and a light scene cannot answer a motion question.

Known action families:

- `motion`: hit, push, move, ضرب, دفع
- `break`: break, broke, كسر
- `light`: illuminate, reveal, أضاء

For known families, canonical effects are preferred over noisy corpus fragments:

```text
motion -> movement, away, position change
break  -> separation, damage, shape change
light  -> visibility, reveal, clarity
```

Arabic answers use Arabic canonical effects.

## Live Probe

Use this command to compare gate-off and gate-on behavior:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_strategy_live_probe.jl
```

Expected behavior:

- `semantic_scenes` is greater than zero
- event/effect prompt improves when the gate is on
- yes/no prompt does not receive a semantic scene answer
- `ما معنى السلام؟` remains on the definition route

## Regression Tests

Focused test:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_semantic_imagination.jl
```

Recent verified result:

```text
semantic imagination scene extraction        17/17
semantic imagination probe matrix fixture    33/33
semantic imagination calculus comparison     34/34
semantic imagination scene memory            34/34
semantic imagination independent answer      16/16
semantic imagination answer comparison       16/16
semantic imagination strategy gate           13/13
```

Total focused coverage: `163/163`.

## Current Status

Phase 7 makes the sensory/event layer practically usable behind a gate:

- trained memory can be built and loaded
- live generation can use scene answers when enabled
- event/effect answers are cleaner and more physical
- code memories no longer steal gated event/effect prompts
- strict question routes are preserved

The next natural step is to expand the same pattern from simple physical scenes into richer semantic scenes: agent, action, patient, instrument, place, time, state before, state after, and emotional/affective tone.
