# Mirnan to HoloPRNN Vision Bridge - 2026-07-03

This note records the current text-to-visual bridge between Mirnan semantic
imagination and `HoloPRNN_Vision`.

## What Was Added

- `semantic_text_to_visual_scene(text; extractor=nothing, width=32, height=32)`
  in `models/mirnan/HoloPRNN_Vision/src/HoloPRNN_Vision.jl`.
- The function can work in two modes:
  - lightweight local parsing only;
  - optional Mirnan extractor mode, where `extractor(text)` may return a
    Mirnan `SemanticScene`.
- If the Mirnan extractor returns a partial scene, the bridge fills missing
  visual fields from the text itself.

## Arabic Visual Context

The bridge now recognizes richer Arabic visual context:

- actor/action/patient, for example: `خالد / ضرب / الكرة`
- instrument, for example: `بالمضرب` -> `instrument:bat`
- place, for example: `في الحديقة` -> `place:garden`
- time, for example: `قبل الفجر` -> `time:dawn`

Example:

```julia
visual = semantic_text_to_visual_scene(
    "ضرب خالد الكرة بالمضرب في الحديقة قبل الفجر"
)
```

Expected visual objects include:

```text
place:garden, time:dawn, actor:person, instrument:bat, patient:ball
```

## Demo

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\demo_semantic_scene_from_text.jl "ضرب خالد الكرة بالمضرب في الحديقة قبل الفجر"
```

The demo writes a rendered image to:

```text
models/mirnan/HoloPRNN_Vision/output/
```

## Tests

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\test\runtests.jl
```

Current result:

```text
HoloPRNN Vision core safety | 82 / 82 passed
```

## Remaining Notes

- The visual bridge is still opt-in and separate from normal answer generation.
- It should not answer definition, yes/no, or relation questions by itself.
- Future improvement: move old demo-only helper code out once all scripts rely
  on `semantic_text_to_visual_scene`.
