# SemanticScene Guidance Filter - Phase 3.6

This note documents the guidance-filter calibration for Mirnan semantic imagination.

## Scope

Phase 3.6 improves diagnostic comparison only. It does not call `generate!`, add a strategy, or change answer behavior.

## Problem

The semantic-calculus guidance can return broad terms from several learned transitions. This made probe output look partially aligned even when terms belonged to another scene.

Example:

- prompt: `The child broke the cup`
- selected scene: `broke / cup`
- raw guidance could still include terms from `ball` or `room`.

## Change

`SemanticSceneComparison` now stores both:

- `raw_guidance_terms`: all terms returned by `al_hisban_al_dalali`.
- `guidance_terms`: terms filtered against the selected scene actor/action/patient/effects.

Agreement and overlap use the filtered terms. Diagnostics can still show the raw terms.

## Probe Output

`semantic_scene_probe.jl` now prints:

- `GUIDANCE_TERMS_FILTERED`
- `GUIDANCE_TERMS_RAW`

This keeps the probe honest: developers can see both what the calculus produced and what the scene-aware comparison accepted.
