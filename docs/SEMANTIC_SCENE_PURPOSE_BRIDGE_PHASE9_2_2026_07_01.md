# SemanticScene Purpose Bridge Probe - Phase 9.2

Date: 2026-07-01

This phase stabilizes the live diagnostic bridge between:

- `SemanticSceneMemory`: sensory/event scene memory.
- `al_hisban_al_dalali`: semantic-calculus guidance.
- `IstinbatAttentionMemory`: rational `RelationFrame` purpose memory.

## What Changed

The probe script now uses an indexed search instead of relying only on raw scans.

Script:

```text
models/mirnan/scripts/semantic_scene_purpose_bridge_probe.jl
```

New diagnostic header:

```text
indexed_bridge_search: enabled
```

## Indexed Search

The probe builds a small in-memory index:

```text
purpose left-side terms -> purpose records
```

Then it compares each semantic scene only with purpose records that share an action,
patient, or actor term.

This makes the diagnostic result more meaningful:

- if a trained cooperative case is found, it is a real memory intersection;
- if none is found, the model likely lacks natural trained overlap between scene and purpose;
- the probe no longer spends most of its time trying unrelated purpose records.

Before the indexed scan, the probe also tests a short fixed list of known cooperation
seed prompts, such as:

```text
لماذا ضرب خالد الكرة؟
لماذا دفع اللاعب الحجر؟
لماذا كسر الطفل الكأس؟
لماذا أضاء المصباح الغرفة؟
لماذا Khalid hit the ball؟
```

This makes retraining feedback immediate: if the new cooperation seed corpus entered
both memories correctly, `trained cooperative` should appear without waiting for a
large scan.

## Scene-Only Limit

The probe now has a separate scan limit for `scene_only`:

```text
MIRNAN_BRIDGE_SCENE_ONLY_LIMIT
```

Default:

```text
60
```

This prevents the scene-only case from consuming the full scan window.

## Controlled Sanity Cases

The probe now prints controlled cases when trained examples are not found quickly.

### Controlled cooperative

Checks that the sensory scene and purpose memory can cooperate:

```text
Khalid hit the ball with bat in yard before sunset.
Khalid hit the ball لكي يتحرك ball.
```

Expected:

```text
AGREEMENT: cooperative
SCENE_HAS_EVENT: true
PURPOSE_HAS_MEMORY: true
```

### Controlled scene_only

Checks that the sensory scene layer works without purpose memory:

```text
The child broke the cup.
```

Expected:

```text
AGREEMENT: scene_only
SCENE_HAS_EVENT: true
PURPOSE_HAS_MEMORY: false
```

## Purpose Quality Filter

Purpose answers now reject noun-only purpose records.

Example rejected:

```text
موسيقي كلاسي لكي عربيه بمقاماتها متعدده.
```

Reason:

The left side is not a clear event/action. It is a nominal phrase, so it should not be
used as a purpose answer.

## Semantic Scene Cleanup

After the first trained cooperative examples appeared, the probe exposed two scene
extraction issues:

- purpose markers such as `لكي` could leak into `patient` or `instrument`;
- noisy guidance words such as `علي`, `لا`, and `جواب` could appear in
  `effect_candidates`.

The semantic imagination extractor now treats purpose markers as scene-field
boundaries and filters those noisy effect terms.

New test:

```text
semantic imagination purpose boundary cleanup | 4/4
```

Note: existing `semantic_scenes.json` files must be rebuilt before this cleanup appears
in live trained probes.

Example accepted:

```text
فتح علبه انقل محتوي متبقي فوراً وعاء زجاجي بلاستي لكي محكم اغلاق واحفظه ثلاجه واستهلكه خلال يومين.
```

Reason:

The left side contains an action (`فتح`, `انقل`) and the right side gives a usable
purpose/outcome.

## Latest Observed Probe Result

With:

```text
MIRNAN_BRIDGE_PROBE_LIMIT=1000
MIRNAN_BRIDGE_SCENE_ONLY_LIMIT=60
MIRNAN_BRIDGE_PROBE_SECONDS=45
```

The probe showed:

```text
controlled cooperative sanity -> cooperative
controlled scene_only sanity  -> scene_only
trained purpose_only          -> purpose_only
trained none                  -> none
trained cooperative           -> not found in current trained memory
```

Interpretation:

The bridge itself works. The current trained memories do not yet contain many natural
cases where the same prompt activates both a semantic scene and a purpose frame.

## Training Seed Added

A small corpus file was added:

```text
models/mirnan/data/toy_corpus/aaa0_mafahem/000_semantic_scene_purpose_cooperation.txt
```

It contains short examples where a sensory event and an explicit purpose appear in the
same sentence.

These are not ready-answer templates. They are training seeds for the bridge:

```text
event/action + purpose marker + effect/goal
```

After retraining and rebuilding semantic scenes, the probe should have a better chance
of finding a trained `cooperative` case without relying only on controlled examples.

## Post-Retrain Probe Update

After retraining and rebuilding `semantic_scenes.json`, the live bridge probe found a
real trained cooperative case:

```text
trained cooperative -> cooperative
prompt              -> why did the player push the stone?
scene               -> action=push | patient=stone | before=stillness | after=motion and position change
purpose             -> the purpose of pushing the stone is moving the stone away from the road
```

The same probe still reports:

```text
trained scene_only   -> scene_only
trained purpose_only -> purpose_only
trained none         -> none
```

This confirms that the bridge can now find all four diagnostic states from the trained
model/memory, with a controlled sanity case still present as a fallback.

## Purpose Answer Phrasing Guard

Purpose answers now have a narrow Arabic phrasing smoother for short event frames:

```text
push player stone -> push the player the stone
move-away stone road -> moving the stone away from the road
```

The smoother is intentionally disabled for longer procedural phrases. This prevents
outputs such as long food-storage instructions from being over-definitized word by word.

New tests:

```text
purpose_answer: smooths Arabic purpose phrasing              | 2/2
purpose_answer: leaves long procedural phrases unsmoothed    | 2/2
```

## Status

This phase remains diagnostic. It does not change:

- normal generation,
- yes/no answers,
- definition answers,
- strategy order,
- default gates.

It only improves inspection, quality filtering, and training data for future memory
intersection.
