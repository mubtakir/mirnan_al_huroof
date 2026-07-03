# Mirnan Current Developer Map - 2026-07-02

This note is the current developer map after the recent RelationFrame,
SemanticScene, ScenePurpose, QuantityFrame, HoloPRNN Vision, and one-button
training work.

It is intentionally practical: where the code lives, which memories must be
rebuilt, which gates are experimental, and which checks should be run after
training.

## Current Principle

Mirnan now has two cooperating non-statistical layers:

1. **Reasoning / relation layer**: extracts linguistic keys such as purpose,
   condition, time, place, state, and quantity into explicit frames.
2. **Semantic imagination layer**: extracts event scenes and effects, then
   compares them through semantic calculus.

Ready-made answers are still limited to near social memory only, such as
greetings and simple social exchanges. Conceptual answers should come from
memory, relation frames, semantic scenes, or the generator, not from broad
hard-coded answer lists.

## Main Code Paths

- `src/physics/engines/al_istinbat.jl`
  - `RelationFrame`
  - `QuantityFrame`
  - purpose, conditional, temporal, spatial, and state frame extraction
  - purpose answers
  - scene-purpose composite answers
  - quantity answers

- `src/physics/engines/semantic_imagination.jl`
  - semantic scene extraction
  - scene memory
  - semantic-calculus comparison
  - independent scene answers

- `src/physics/engines/generator.jl`
  - loads model memories
  - registers gated strategies
  - keeps default behavior stable unless gates are enabled

- `src/physics/engines/strategies/`
  - `relation_frame_strategy.jl`
  - `semantic_scene_strategy.jl`
  - `scene_purpose_strategy.jl`
  - `quantity_frame_strategy.jl`
  - conditional / temporal / spatial / state frame strategies

- `models/mirnan/HoloPRNN_Vision/`
  - independent physical vision prototype
  - semantic-scene image demos
  - no backpropagation or neural-network training loop

## Runtime Memories

These files are generated or refreshed by training and rebuild scripts:

- `model/al_istinbat.json`
  - learned relation attention, including purpose records after retraining

- `model/semantic_scenes.json`
  - semantic event scenes and effects

- `model/quantity_memory.json`
  - trained count, measure, comparison, scope, and vague quantity frames

- `model/al_hisban_al_dalali.json`
  - Clifford semantic-calculus memory

- `model/rapg_kb.db`
  - RAPG knowledge base

- `model/paragraph_centroids.json`
  - paragraph gravity centers

When extractor logic changes, rebuild the corresponding memory before judging
runtime behavior.

## One-Button Training

The main button is:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon\models\mirnan"
.\run_train.bat
```

The root convenience button is also available:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
.\run_train.bat
```

Both run the paragraph-level training pipeline:

1. `train.jl`
2. `run_evolution.jl`
3. `scripts/build_semantic_scenes.jl`
4. `scripts/rebuild_quantity_memory.jl`
5. `scripts/post_training_smoke.jl`

The default post-training smoke intentionally skips heavy gate probes. To include
the slower generator-backed gate checks:

```powershell
$env:MIRNAN_POST_SMOKE_HEAVY_GATES="1"
```

## Useful Direct Commands

Post-training smoke:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\post_training_smoke.jl
```

Rebuild only semantic scenes:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\build_semantic_scenes.jl
```

Rebuild only quantity memory:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\rebuild_quantity_memory.jl
```

Light quantity-memory probe:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\trained_quantity_memory_probe.jl
```

Semantic scene probe:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_probe.jl
```

Scene-purpose bridge probe with bounded search:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
$env:MIRNAN_BRIDGE_PROBE_LIMIT="200"
$env:MIRNAN_BRIDGE_SCENE_ONLY_LIMIT="60"
$env:MIRNAN_BRIDGE_PROBE_SECONDS="45"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_purpose_bridge_probe.jl
```

## Strategy Gates

The strategies are gated so the default generator remains stable.

- `MIRNAN_ENABLE_RELATION_FRAME_STRATEGY`
- `MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY`
- `MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY`
- `MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY`
- `MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY`
- `MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY`
- `MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY`
- `MIRNAN_ENABLE_STATE_FRAME_STRATEGY`
- `MIRNAN_STRICT_NO_TEMPLATES`

Use `=1` to enable a gate for a probe or controlled experiment.

## Question Behavior

- Social replies may be ready answers.
- `هل` questions are guarded and should not be hijacked by purpose, scene, or
  quantity strategies.
- `لماذا` may use purpose memory, scene-purpose composites, or normal generation
  depending on evidence.
- `ماذا يحدث عندما...` may use semantic scene effects.
- `كم`, `ما عدد`, `كم طول`, `كم وزن`, `كم مدة`, `كم مسافة`, `أيهما أكثر`, and
  `أيهما أقل` may use quantity memory.
- Definition questions such as `ما معنى...` should stay in definition routes.

## Verified Examples

Scene + purpose:

```text
Q: لماذا دفع اللاعب الحجر؟
A: دفع اللاعب الحجر؛ فانتقل الحجر من سكون إلى حركة وتغير موضع؛ وظهرت آثار مثل حركة، ابتعاد، تغير موضع؛ وكانت الغاية إبعاد الحجر عن الطريق.
```

Scene only:

```text
Q: ماذا يحدث عندما يضرب خالد الكرة؟
A: ضرب خالد الكرة؛ فتغير حال الكرة من سكون إلى حركة وتغير موضع؛ وظهرت آثار مثل حركة، ابتعاد، تغير موضع.
```

Quantity:

```text
Q: كم عدد جمع القلة يدل على؟
A: يدل جمع القلة على من ثلاثة إلى عشرة.
```

Supported quantity paths include count, measure, comparison, quantifier scope,
and vague quantity:

```text
كم عدد الطلاب؟
كم طول الجسر؟
أيهما أكثر خمس نجمات أم ثلاث نقاط؟
أيهما أقل السكر أم الملح؟
```

## Main Tests

- `test/test_relation_frame.jl`
  - relation frames
  - purpose / conditional / temporal / spatial / state behavior
  - quantity frames
  - scene-purpose bridge

- `test/test_semantic_imagination.jl`
  - scene extraction
  - semantic calculus comparison
  - scene memory
  - independent scene answers
  - strategy gate checks

- `test/test_question_type_matrix.jl`
  - question behavior guards

- `test/test_social_reply_memory_boundary.jl`
  - ready-answer boundary

- `scripts/post_training_smoke.jl`
  - post-training health check

## HoloPRNN Vision Notes

`models/mirnan/HoloPRNN_Vision` is a separate physical vision prototype.
It now has core safety tests and semantic-scene demos that can render simple
scene images from Mirnan's scene extraction.

Current useful demo:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon\models\mirnan\HoloPRNN_Vision"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=. demos\demo_semantic_scene_from_text.jl
```

## Known Cautions

- Heavy generator-backed probes can take a long time. They are not part of the
  default post-training smoke.
- If a terminal shows PowerShell continuation prompt `>>`, the command is
  incomplete, usually because of an unmatched quote or brace.
- Some older docs have mixed legacy encoding. Prefer this map and `LATEST.md`
  for the current state.
- After retraining, rebuild semantic scenes and quantity memory before judging
  the new strategies.

## Next Useful Work

1. Add more trained quantity measure examples, especially length, weight,
   duration, and distance.
2. Improve English scene-only answer phrasing.
3. Polish conditional, temporal, spatial, and state answers the same way
   purpose and quantity were polished.
4. Share formatting helpers across frame strategies.
5. Continue cleaning and consolidating old developer documentation.
