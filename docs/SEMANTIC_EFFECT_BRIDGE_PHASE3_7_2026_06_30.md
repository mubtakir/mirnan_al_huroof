# Semantic Effect Bridge - Phase 3.7

This note documents the diagnostic semantic-effect bridge for Mirnan semantic imagination.

## Scope

Phase 3.7 is diagnostic only. It does not generate answers, add ready knowledge, or change `generate!`.

## Purpose

The scene layer can extract Arabic effects such as:

- `حركة`
- `ابتعاد`
- `تغير موضع`

The semantic-calculus guidance may return English terms such as:

- `moved`
- `away`
- `changed`
- `position`

Without a bridge, the comparison can report `scene_only` even when both sides describe the same effect.

## Change

`semantic_imagination.jl` now has small effect bridge groups used only by:

- scene/calculus term overlap
- scene-aware guidance filtering

The bridge maps effect families such as:

- movement: `moved` / `movement` / `حركة` / `تحرك`
- distance: `away` / `ابتعاد`
- position change: `changed` / `position` / `تغير` / `موضع`
- visibility: `clear` / `visible` / `وضوح` / `ظهور` / `انكشاف`
- breaking: `separated` / `pieces` / `lost shape` / `انفصال` / `تلف`

## Boundary

This is not an answer template. It is a diagnostic comparison bridge. It should not be used as a source of final answers without later gated phases and tests.
