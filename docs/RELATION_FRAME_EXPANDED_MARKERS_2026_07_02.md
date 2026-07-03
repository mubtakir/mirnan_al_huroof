# RelationFrame Expanded Markers - 2026-07-02

This phase extends non-purpose RelationFrame marker coverage.

## Added Markers

Temporal:

- `حين`
- `عندما`

Spatial:

- `فوق`
- `تحت`
- `داخل`
- `في` with low confidence because it is highly ambiguous

State:

- `وهو`
- `وهي`

## Priority Rule

`relation_type_for_marker` now checks the RelationFrame marker table before the
legacy default marker table. This matters for `حين` and `عندما`, which existed
as old `causal_anchor` markers but are temporal markers in RelationFrame.

The legacy extraction paths are not removed; this priority only affects the
RelationFrame marker classification API.

## Verification

Ran:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_relation_frame.jl
```

Result: all `test_relation_frame.jl` testsets passed.
