# QuantityFrame Answer Layer - Phase 14

This phase turns the existing QuantityFrame extraction layer into a measurable
independent answer path, a lightweight quantity memory, and an off-by-default
gated strategy.

## Existing Base

QuantityFrame already existed as an extraction-only structure:

```julia
struct QuantityFrame
    marker::String
    quantity_type::String
    target::String
    value::String
    polarity::String
    confidence::Float64
end
```

Supported quantity types:

- `count`
- `measure`
- `comparison`
- `quantifier_scope`
- `vague_quantity`

## New Selection API

```julia
select_quantity_frame(frames::Vector{QuantityFrame},
                      prompt::AbstractString;
                      min_score::Float64=0.15)

select_quantity_frame(mem::QuantityFrameMemory,
                      prompt::AbstractString;
                      min_score::Float64=0.15)
```

The selector compares prompt terms with frame target, value, and marker terms.
It is diagnostic only and does not read or mutate Istinbat memory.

## New Answer API

```julia
quantity_answer(frames::Vector{QuantityFrame},
                prompt::AbstractString) -> String

quantity_answer(mem::QuantityFrameMemory,
                prompt::AbstractString) -> String
```

It answers quantity prompts such as:

- "how many"
- "how much"
- "what amount"
- Arabic prompts containing count/amount markers such as count, number,
  amount, quantity, and comparison markers.
- Arabic scope prompts such as "ma nitaq ..." / "ما نطاق ..." for
  `quantifier_scope` records.

The answer path is guarded:

- yes/no prompts return empty;
- non-quantity prompts return empty;
- empty frame lists return empty;
- weak or empty frames return empty.

Example:

```text
frame:
  marker=count, target=students, value=30

prompt:
  how many students?

answer:
  number of students is 30
```

## New Comparison API

```julia
QuantityComparisonRecord
compare_quantity_strategies(frames::Vector{QuantityFrame},
                            generate_func::Function,
                            prompt::AbstractString)

compare_quantity_strategies(mem::QuantityFrameMemory,
                            generate_func::Function,
                            prompt::AbstractString)
```

This records the independent quantity answer beside any caller-supplied
generation function. It is for measurement and development only.

## New Lightweight Memory

```julia
mutable struct QuantityFrameMemory
    frames::Vector{QuantityFrame}
    source_metadata::Vector{Dict{String,Any}}
end
```

Learning APIs:

```julia
learn_quantity_frames_from_text!(mem::QuantityFrameMemory,
                                 text::AbstractString,
                                 source::Dict{String,Any}=Dict{String,Any}()) -> Int

train_quantity_frames_from_texts!(mem::QuantityFrameMemory,
                                  texts,
                                  metadata=nothing;
                                  max_items::Int=50_000) -> Int
```

This memory is separate from `IstinbatAttentionMemory`, so quantity evidence can
be developed without changing the existing relation and question-answer paths.

Persistence APIs:

```julia
quantity_memory_to_dict(mem::QuantityFrameMemory) -> Dict
save_quantity_memory(mem::QuantityFrameMemory, path::AbstractString) -> String
load_quantity_memory(path::AbstractString) -> QuantityFrameMemory
has_quantity_records(mem::QuantityFrameMemory) -> Bool
```

## Gated Strategy

New strategy:

```text
QuantityFrameStrategy
```

Environment gate:

```text
MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY=1
```

Default behavior is unchanged. When enabled, the strategy is inserted before the
general `AqlStrategy`/`RelationStrategy` path. It reads only
`_LEARNED_QUANTITY_MEMORY`, so trained generation is unaffected unless a caller
explicitly supplies quantity memory.

Live probe:

```text
models/mirnan/scripts/quantity_frame_strategy_live_probe.jl
models/mirnan/scripts/quantity_trained_question_probe.jl
```

Observed behavior:

```text
controlled quantity
  gate off: generic fallback terms
  gate on:  عدد الطلاب هو 30.

yes/no guard
  gate off == gate on

non-quantity guard
  gate off == gate on
```

Trained live probe after rebuilding `quantity_memory.json`:

```text
QuantityFrameStrategy trained live probe
quantity_frames: 277

CASE: trained count
PROMPT: كم عدد جمع القلة يدل على؟
-- gate on --
يدل جمع القلة على من ثلاثة إلى عشرة.

CASE: trained comparison
PROMPT: أيهما أكثر خمس نجمات؟
-- gate on --
خمس نجمات أكثر من ثلاث نقاط.

CASE: trained quantifier scope
PROMPT: ما نطاق واحد منها يحتاج إلى الآخر؟
-- gate on --
النطاق: كل واحد منها يحتاج إلى الآخر.

CASE: definition guard
PROMPT: ما معنى جمع القلة؟
-- gate on --
(empty)

CASE: yes/no guard
PROMPT: هل عدد الطلاب 30؟
-- gate on --
نعم، عدد الطلاب 30.
```

This confirms that the quantity gate answers explicit quantity/count prompts
while remaining silent for definition prompts and leaving yes/no behavior to
the normal yes/no path.

By default the trained probe skips `gate off` generation to keep the smoke check
fast and stable. To compare against the ungated generator, set:

```text
MIRNAN_QUANTITY_TRAINED_PROBE_GATE_OFF=1
```

## Training And Loading

Training now builds and saves:

```text
models/mirnan/model/quantity_memory.json
```

The build step runs after `al_istinbat`:

```text
5.0g.1 building quantity frame memory...
```

`MirnanGenerator` now loads this file when present:

```text
loading quantity_memory...
```

If the file is missing or empty, `_LEARNED_QUANTITY_MEMORY` remains `nothing`,
so default generation behavior is unchanged.

## Files

- `models/mirnan/src/physics/engines/al_istinbat.jl`
- `models/mirnan/src/physics/groups/arabic_group.jl`
- `models/mirnan/src/physics/engines/strategies/base.jl`
- `models/mirnan/src/physics/engines/strategies/quantity_frame_strategy.jl`
- `models/mirnan/src/physics/engines/generator.jl`
- `models/mirnan/train.jl`
- `models/mirnan/test/test_relation_frame.jl`
- `models/mirnan/scripts/quantity_frame_strategy_live_probe.jl`

## Verification

```text
test_relation_frame.jl
  quantity_answer: count and measure | 7/7
  quantity_answer: comparison and vague quantity | 6/6
  quantity_answer: quantifier scope phrasing | 1/1
  quantity_answer: guards | 3/3
  compare_quantity_strategies: returns QuantityComparisonRecord | 9/9
  compare_quantity_strategies: guards | 5/5
  QuantityFrameMemory: learn and answer | 8/8
  QuantityFrameMemory: train and compare | 5/5
  QuantityFrameMemory: save and load | 8/8
  QuantityFrameMemory: generator autoload | 3/3
  QuantityFrameStrategy | 5/5

quantity_frame_strategy_live_probe.jl
  controlled quantity gate works
  yes/no guard unchanged
  non-quantity guard unchanged

quantity_trained_question_probe.jl
  trained count gate works
  trained comparison gate works
  trained quantifier-scope gate works
  definition guard unchanged
  yes/no guard unchanged

test_al_istinbat.jl
  al_istinbat inference attention memory | 29/29

train.jl
  parse check | ok
```

## Current Follow-up

The trained quantity gate is now validated. The remaining refinement is quality
work: improve phrasing for comparison and scope answers, then add representative
trained quantity prompts to the regular post-training smoke checks.

## Post-Training Smoke Check

The lightweight post-training check now has one entry point:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\post_training_smoke.jl
```

It runs:

- `semantic_scene_purpose_bridge_probe.jl`
- `trained_quantity_memory_probe.jl`

Defaults are intentionally conservative:

- `MIRNAN_BRIDGE_PROBE_LIMIT=200`
- `MIRNAN_BRIDGE_SCENE_ONLY_LIMIT=60`
- `MIRNAN_BRIDGE_PROBE_SECONDS=45`
- `MIRNAN_QUANTITY_TRAINED_PROBE_GATE_OFF=0`
- heavy generator-gate probes are skipped by default

The values can still be overridden from PowerShell before running the smoke
check.

To include the heavier generator-backed gate probe:

```powershell
$env:MIRNAN_POST_SMOKE_HEAVY_GATES="1"
```

This adds:

- `quantity_trained_question_probe.jl`
