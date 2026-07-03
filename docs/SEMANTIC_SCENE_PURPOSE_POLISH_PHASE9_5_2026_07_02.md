# SemanticScene Purpose Polish - Phase 9.5

This phase stabilizes the visible wording around the SemanticScene/Purpose bridge
after the live probe confirmed that the strategy itself is gated correctly.

## What Changed

### Arabic composite wording

`scene_purpose_answer` still returns a non-empty answer only for cooperative
scene + purpose cases, but the Arabic surface form is now cleaner:

```text
عند دفع الحجر ينتقل المتأثر من سكون إلى حركة وتغير موضع، وتظهر آثار مثل تحرك، مكانه، حركة؛ والغاية إبعاد الحجر عن الطريق.
```

Instead of the earlier diagnostic-style form:

```text
... ومن جهة الغاية: ...
```

The change is purely presentational:

- the cooperative gate is unchanged;
- scene-only and purpose-only cases still return empty from the composite path;
- yes/no and definition guards are unchanged;
- English composite wording remains unchanged.

### Arabic list separators

Arabic composite answers now normalize comma-separated effect lists from:

```text
تحرك, مكانه, حركة
```

to:

```text
تحرك، مكانه، حركة
```

This keeps the sensory-scene answer closer to normal Arabic prose without changing
the underlying scene memory.

### Yes/No initial-event negation

The yes/no relation formatter now distinguishes between:

- nominal relations, e.g. `الظلم يحفظ السلام` -> `الظلم لا يحفظ السلام`;
- initial event clauses, e.g. `دفع اللاعب الحجر` -> `لم يدفع اللاعب الحجر`.

This prevents broken output such as:

```text
لا، دفع لا اللاعب الحجر.
```

The live probe now shows:

```text
هل دفع اللاعب الحجر؟
لا، لم يدفع اللاعب الحجر.
```

### Scene extraction cleanup

The semantic scene extractor now has stronger guards around Arabic event clauses:

- purpose boundaries such as `لكي` stop patient extraction;
- attached instrument forms such as `بالمضرب` are read as instruments, not as
  part of the patient;
- weak generic effect terms such as `على`, `لا`, and `جواب` are filtered out of
  scene effects.

Example:

```text
ضرب خالد الكرة بالمضرب لكي تتحرك الكرة بعيداً
```

extracts:

```text
patient=الكرة
instrument=المضرب
```

not:

```text
patient=الكرة بالمضرب لكي تتحرك...
```

## Tests

Verified locally:

```text
generation quality guard | 131/131
learned opposition yes/no route | 24/24
relation and difference guarded fallback | 25/25
test_semantic_imagination.jl all groups passed
semantic imagination purpose boundary cleanup | 10/10
test_relation_frame.jl core groups all passed
scene_purpose_answer: cooperative composite only | 11/11
RelationFrameStrategy | 9/9
ScenePurposeStrategy | 6/6
```

Live probe:

```text
models/mirnan/scripts/scene_purpose_strategy_live_probe.jl
```

Observed guards:

- cooperative case: gate on returns scene + purpose composite;
- purpose-only guard: gate on matches gate off;
- yes/no guard: `لا، لم يدفع اللاعب الحجر.`;
- definition guard: unchanged.

## Rebuilt Semantic Scene Memory

After the extraction cleanup, `semantic_scenes.json` was rebuilt with:

```text
MIRNAN_SEMANTIC_SCENE_TEXT_LIMIT=500
MIRNAN_SEMANTIC_SCENE_LIMIT=5000
```

Observed build result:

```text
texts: 473
seed_scenes: 8
learned_scenes: 538
saved: models/mirnan/model/semantic_scenes.json
```

The bridge probe then loaded:

```text
semantic_scenes: 546
istinbat_records: 26943
purpose_records: 3476
```

The trained cooperative case is now clean:

```text
prompt: لماذا دفع اللاعب الحجر؟
scene summary:
  action=دفع
  patient=الحجر
  effects=تحرك, مكانه, حركة
composite:
  عند دفع الحجر ينتقل المتأثر من سكون إلى حركة وتغير موضع،
  وتظهر آثار مثل تحرك، مكانه، حركة؛ والغاية إبعاد الحجر عن الطريق.
```

The previous leakage patterns were not observed:

- no `لكي` in `patient`;
- no `على/لا/جواب` in the displayed trained scene effects;
- purpose-only procedural prompt still returns no composite answer.

## Developer Notes

Implementation points:

- `_format_scene_purpose_answer` handles only final answer formatting;
- `_arabic_list_separators` normalizes visible Arabic comma lists;
- `_yesno_negated_relation_statement` delegates initial event clauses to
  `_yesno_negate_initial_event`.

The next recommended phase is not more formatting. It is semantic expansion:

1. add conditional bridge behavior for `إذا/إن/لو`;
2. rebuild `semantic_scenes.json` so the trained memory benefits from the cleaner
   extraction rules;
3. expand the Arabic seed scenes for motion, breaking, light, carrying, opening,
   falling, closing, and lifting.
