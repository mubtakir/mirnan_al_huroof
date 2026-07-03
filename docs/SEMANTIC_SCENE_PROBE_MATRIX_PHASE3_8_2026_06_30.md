# SemanticScene Probe Matrix - Phase 3.8

This note documents the wider diagnostic probe matrix for Mirnan semantic imagination.

## Scope

Phase 3.8 is diagnostic only. It does not call `generate!`, add a strategy, or change answer behavior.

## Files

- `models/mirnan/test/fixtures/semantic_scene_probe_matrix.tsv`
- `models/mirnan/scripts/semantic_scene_probe_matrix.jl`

## Matrix Format

The TSV columns are:

```text
source<TAB>target<TAB>prompt<TAB>expected
```

`source` and `target` build the semantic-calculus pair and the scene memory.
`prompt` is the probe question or sentence.
`expected` is a diagnostic label:

- `aligned`
- `partial`
- `scene_only`
- `calculus_only`
- `divergent`
- `none`

## Output

The script prints one row per case:

- status: `OK`, `REVIEW`, or `MISS`
- expected label
- actual agreement
- overlap
- selected scene
- filtered guidance terms

It ends with a summary count.

## Purpose

This step helps detect:

- correct alignment
- excessive semantic bridging
- false alignment
- scene-only cases
- missing calculus support

It is the inspection step before writing any independent semantic-scene answer function.
