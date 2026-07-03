# SemanticScene Answer - Phase 4

This note documents the first independent answer function for Mirnan semantic imagination.

## Scope

Phase 4 adds a standalone function only:

- `semantic_scene_answer(scene_mem, calculus, prompt)`

It does not call or modify `generate!`, does not add a strategy, and does not create ready knowledge answers.

## Gate

The function returns an empty string unless:

- the prompt asks about event effect, such as:
  - `What happens when ...`
  - `What is the effect ...`
  - `ماذا يحدث عندما ...`
  - `ما أثر ...`
  - `كيف يؤثر ...`
- a semantic scene is selected
- scene/calculus comparison is `aligned` or `partial`
- overlap is positive
- there are effect terms to verbalize

## Output

Arabic example:

```text
عند ضرب الكرة يظهر أثر دلالي مثل حركة، ابتعاد، تغير موضع.
```

English example:

```text
When hit affects ball, the semantic effect includes moved, away, changed.
```

## Boundary

This is not yet a generation strategy. It is an independent probe-level answer function. It should be compared against `generate!` in a later phase before any gated strategy is added.
