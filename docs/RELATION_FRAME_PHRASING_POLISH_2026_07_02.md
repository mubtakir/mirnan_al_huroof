# RelationFrame Phrasing Polish - 2026-07-02

This note records a small phrasing-only polish for the non-purpose
RelationFrame answer paths.

## Scope

The change affects formatting only. It does not change:

- memory storage
- frame extraction
- frame selection
- generator gates
- yes/no guards

## Polished Paths

- `conditional_answer`
  - before: `إذا X، فالنتيجة Y.`
  - after: `إذا X، يترتب على ذلك Y.`

- `temporal_answer`
  - before: `سافر الطالب قبل الفجر.`
  - after: `كان سفر الطالب قبل الفجر.`

- `spatial_answer`
  - before: `جلس الطفل حيث الحديقة.`
  - after: `كان مكان جلوس الطفل حيث الحديقة.`

- `state_answer`
  - before: `دخل الطفل حال خائف.`
  - after: `دخل الطفل، وكان على حال خائف.`

These are still conservative templates, but they are less raw. Temporal and
spatial paths also use a small Arabic event nominalizer for common simple verbs
such as `سافر -> سفر` and `جلس -> جلوس`.

## Verification

Ran:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_relation_frame.jl
```

Result: all `test_relation_frame.jl` testsets passed, including the new
phrasing guard tests for conditional, temporal, spatial, and state answers.
