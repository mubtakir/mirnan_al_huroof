# Bilingual Marker Parity Phase - 2026-07-02

This note records the parity pass that gives English prompts and markers a
clearer share in the same non-statistical mechanisms already used for Arabic.

## Scope

The pass focused on the independent inference layers in `al_istinbat.jl`:

- RelationFrame marker extraction.
- Purpose question detection and yes/no guards.
- QuantityFrame marker extraction.
- Quantity answer routing for English comparison questions.

It does not change canonical yes/no relation rules, relation strategy ranking,
or the default generation behavior beyond the existing gated strategies.

## English RelationFrame Markers

Added English markers beside the Arabic relation keys:

- purpose: `in order to`, `so that`, `for the purpose of`
- conditional: `if`, `unless`, `whenever`
- temporal: `before`, `after`, `during`, `since`, `when`
- spatial: `where`, `above`, `over`, `under`, `below`, `inside`
- state: `while`, `being`, `as`

These markers are used by `relation_type_for_marker` and
`extract_relation_frames` with confidence levels matching the existing Arabic
marker design.

## English Purpose Questions

The purpose gate now recognizes English purpose prompts such as:

- `why ...`
- `for what purpose ...`
- `what purpose ...`
- `what goal ...`
- `what benefit ...`

The yes/no guard now also protects English yes/no forms such as `does`, `do`,
`did`, `is`, `are`, `was`, `were`, `can`, and `should`.

For purely English purpose frames, `purpose_answer` can produce English phrasing:

```text
The purpose of student studies is succeed.
```

This is intentionally simple and remains an independent RelationFrame answer,
not a ready-answer template.

## English Relation Answers

The parity pass also covers independent English answers for non-purpose
RelationFrame paths:

- conditional: `if student studies then student succeeds` can answer
  `what happens if student studies?`
- temporal: `student traveled before dawn` can answer `when student traveled?`
- spatial: `child sat where garden` can answer `where child sat?` as
  `The place of child sat is garden.`
- state: `child entered while afraid` can answer `how child entered?`

The English yes/no guard protects these paths from direct yes/no prompts such
as `does student study?`, `did student travel?`, and `did child enter?`.

For conditional extraction, `if ... then ...` now splits explicitly at `then`,
so the condition and result do not collapse into one side of the frame.

## English QuantityFrame Markers

Added English quantity markers:

- count: `how many`, `number`, `number of`
- measure: `how much`, `amount`, `quantity`, `length`, `weight`, `duration`, `distance`
- comparison: `more than`, `less than`, `greater than`, `equal to`, `half`, `double`
- scope: `all`, `some`
- vague quantity: `many`, `few`

The quantity question gate now accepts English comparison prompts:

- `which is more ...`
- `which is greater ...`
- `which is less ...`

This lets an extracted frame such as `five stars more than three points` answer
a comparison prompt such as `which is more stars or points?`.

## Tests

Verified with:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_relation_frame.jl
```

Relevant added/covered testsets:

- `relation frames: English marker parity`
- `purpose_answer: English why question`
- `conditional_answer: English if then phrasing`
- `temporal_answer: English when phrasing`
- `spatial_answer: English where phrasing`
- `state_answer: English how phrasing`
- `quantity_answer: English extraction and comparison`

The full `test_relation_frame.jl` run passed after the parity fixes.
