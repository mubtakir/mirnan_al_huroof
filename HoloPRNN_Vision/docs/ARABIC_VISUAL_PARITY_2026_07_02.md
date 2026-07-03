# Arabic Visual Parity - 2026-07-02

This note records the Arabic parity pass for `HoloPRNN_Vision`.

## Why

The first semantic-scene visual bridge worked, but the core tests and many visual
shape rules were still more explicit for English examples than Arabic ones. This
made Arabic support depend too much on a narrow set of hints.

## What Changed

The visual bridge now folds common Arabic variants before classifying visual
scene families:

- hamza forms: `أ`, `إ`, `آ` -> `ا`
- final `ى` -> `ي`
- `ة` -> `ه`
- Arabic diacritics are removed

This helps cases such as `أضاء` and `اضاء` resolve to the same visual family.

## Arabic Action Families

Arabic cues were expanded for:

- motion: `ضرب`, `دفع`, `حرك`, `تحرك`, `ابتعاد`, `ابعاد`
- break/damage: `كسر`, `تلف`, `انفصال`, `شظايا`, `قطع`, `زجاج`
- light/reveal: `أضاء`, `اضاء`, `إضاءة`, `نور`, `ضوء`, `وضوح`, `ظهور`, `انكشاف`, `مصباح`

## Arabic Object Shapes

Arabic object names now map to visual shapes more explicitly:

- `الكرة` -> `ball`
- `الكأس` / `الكاس` -> `cup`
- `الغرفة` -> `room`
- `الحجر` -> `stone`
- `المصباح` -> `lamp`

Actors are no longer always forced to be a person. If the actor is a lamp, it
uses a lamp/source-like shape; otherwise human actors still use `person`.

## Demo Coverage

The text demo now includes Arabic examples by default:

- `ضرب خالد الكرة`
- `دفع اللاعب الحجر`
- `كسر الطفل الكأس`
- `أضاء المصباح الغرفة`

Generated files include:

- `output/text_scene_4_ضرب_خالد_الكرة.png`
- `output/text_scene_5_دفع_اللاعب_الحجر.png`
- `output/text_scene_6_كسر_الطفل_الكأس.png`
- `output/text_scene_7_أضاء_المصباح_الغرفة.png`

## Tests

Verified with:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\test\runtests.jl
```

Result:

```text
HoloPRNN Vision core safety | 70 / 70 passed
```

The test suite now includes Arabic visual-scene checks for motion, stone,
breakage, cup fragments, lamp, room, and glow.

## 2026-07-03 Addendum

Arabic parity now includes richer text-to-vision context:

- `بالمضرب` -> `instrument:bat`
- `في الحديقة` -> `place:garden`
- `قبل الفجر` -> `time:dawn`

The public bridge is:

```julia
semantic_text_to_visual_scene(text; extractor=nothing, width=32, height=32)
```

It can use a Mirnan semantic-scene extractor when supplied, while filling missing
visual fields from the original Arabic text.

Current verification:

```text
HoloPRNN Vision core safety | 82 / 82 passed
```
