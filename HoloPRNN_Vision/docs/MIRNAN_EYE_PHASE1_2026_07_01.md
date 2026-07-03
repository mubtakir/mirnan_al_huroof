# Mirnan Eye Phase 1 - HoloPRNN Vision

Date: 2026-07-01

This document records the first working bridge between Mirnan semantic imagination and
HoloPRNN visual wave rendering. The goal of this phase was not photorealistic image
generation. The goal was to give Mirnan a first "eye": a physical, wave-based visual
surface that can receive a semantic scene and turn it into a visible field without
neural networks, backpropagation, or statistical image models.

## Position In Mirnan

Mirnan now has two complementary layers under active development:

1. `RelationFrame` / `QuantityFrame`:
   a stricter reasoning layer that detects linguistic keys such as purpose,
   condition, time, place, state, and quantity.
2. `SemanticScene` / `HoloPRNN_Vision`:
   a sensory-imaginative layer that turns event meaning into felt scene effects:
   movement, separation, illumination, damage, direction, place, and change.

This HoloPRNN phase belongs to the second layer. It gives the semantic-calculus output
a visible wave body.

## Implemented Pipeline

```text
text
  -> Mirnan semantic scene extraction
  -> VisualScene
  -> complex wave field
  -> RGB image
```

The practical bridge is:

```julia
semantic_scene_to_visual_scene(scene; width, height)
visual_scene_to_wavefield(visual_scene)
render_visual_scene(visual_scene)
```

The demo bridge is:

```julia
demo_semantic_scene_from_text.jl
```

It loads Mirnan's semantic scene extractor when available and falls back to a small
local parser only if Mirnan is unavailable.

## Core Additions

### 1. Safety And Stability

The HoloPRNN core now checks:

- image and matrix dimensions before simulation,
- finite wave-field values after oscillator updates,
- valid classifier dimensions,
- valid `StyleProfile` contents after loading,
- deterministic seeds for reproducible experiments.

This prevents silent NaN cascades and unclear dimension errors.

### 2. StyleProfile

`StyleProfile` separates style extraction from style application:

```julia
profile = extract_style_profile(style_image, params)
styled = apply_style(content_wave, profile)
save_style_profile(profile, path)
loaded = load_style_profile(path)
```

This is important for later Mirnan work because a style can become a reusable visual
temperament: calm, sharp, luminous, turbulent, sparse, dense, etc.

### 3. Classifier Cleanup

The classifier path now has:

- `compute_classification_loss(model, x, target)`,
- `predict_batch(model, X)`,
- loss history,
- optional early stopping,
- configurable weight decay,
- cleaner demos without duplicated forward-pass code.

This keeps the physical CHL classifier usable as a small experimental component.

### 4. VisualScene Bridge

The new bridge introduces:

```julia
VisualObject
VisualScene
semantic_scene_to_visual_scene
visual_scene_to_wavefield
render_visual_scene
```

It maps semantic actions into visual primitives:

- hit / push: motion trail and displaced patient,
- break: fragments and damaged object,
- illuminate: glow and revealed area,
- actor: simple person-like primitive,
- ball / cup / room: shape-specific rendering.

This is intentionally symbolic and wave-native. It is a first visual grammar, not a
photographic renderer.

## Verified Demos

### Hardcoded Semantic Scenes

Command:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\demo_semantic_scene.jl
```

Outputs:

- `models\mirnan\HoloPRNN_Vision\output\semantic_scene_hit_ball.png`
- `models\mirnan\HoloPRNN_Vision\output\semantic_scene_broke_cup.png`
- `models\mirnan\HoloPRNN_Vision\output\semantic_scene_lit_room.png`

### Text To Semantic Visual Scene

Command:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\demo_semantic_scene_from_text.jl "Khalid hit the ball" "The child broke the cup" "The lamp illuminated the room"
```

Verified text effects:

- `Khalid hit the ball` -> `حركة`, `ابتعاد`, `تغير موضع`
- `The child broke the cup` -> `انفصال`, `تلف`, `تغير هيئة`
- `The lamp illuminated the room` -> `ظهور`, `انكشاف`, `وضوح`

Outputs:

- `models\mirnan\HoloPRNN_Vision\output\text_scene_1_khalid_hit_the_ball.png`
- `models\mirnan\HoloPRNN_Vision\output\text_scene_2_the_child_broke_the_cup.png`
- `models\mirnan\HoloPRNN_Vision\output\text_scene_3_the_lamp_illuminated_the_room.png`

## Tests

Main command:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan\HoloPRNN_Vision models\mirnan\HoloPRNN_Vision\test\runtests.jl
```

Latest reported result:

```text
HoloPRNN Vision core safety | 59 passed / 59 total
```

The tests cover:

- wave/image conversion,
- Laplacian and simulation safety,
- non-finite update guards,
- inpainting and crystallization seeds,
- classifier loss, batch prediction, and early stopping,
- style profile save/load/apply,
- semantic scene to visual scene mapping,
- action-specific visual shapes for motion, breakage, and illumination.

## Current Limits

- The visual output is symbolic and physical, not photorealistic.
- The bridge is currently demo-level; it is not yet a default Mirnan answer strategy.
- The visual grammar has only a small set of action families.
- Arabic and English extraction depend on Mirnan's current semantic-scene extractor.
- There is no persistent visual memory yet for reusing rendered scenes.

## Recommended Next Steps

1. Add a persistent visual-scene memory:
   save extracted `VisualScene` records beside semantic scenes and reuse them.
2. Expand the visual grammar:
   add container, cutting, burning, falling, opening, closing, growing, shrinking,
   giving, taking, and social interaction primitives.
3. Add style temperament:
   connect `StyleProfile` to semantic tone, for example calm, bright, damaged,
   turbulent, joyful, fearful.
4. Add a probe matrix:
   fixed text prompts should produce stable visual scene categories and expected
   effects, similar to the semantic-scene probe matrix.
5. Keep the gate explicit:
   HoloPRNN visual generation should remain opt-in until its outputs are stable
   enough to become part of normal Mirnan responses.

## Design Principle

The eye of Mirnan should remain consistent with Mirnan's general philosophy:

- no neural image generator,
- no backpropagation-based vision model,
- no statistical imitation as the core mechanism,
- semantic calculus first,
- physical wave rendering second,
- then gradual interaction between reason and sensory imagination.

