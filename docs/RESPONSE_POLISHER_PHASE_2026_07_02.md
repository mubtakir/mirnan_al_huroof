# Response Polisher Phase - 2026-07-02

This note records the first conservative final-wording pass for Mirnan.

## Purpose

Mirnan now has several rational and sensory answer layers: relation frames,
quantity frames, semantic scenes, scene-purpose composites, and traditional
generator paths. The next weakness is often not the reasoning itself, but the
surface form of the answer:

- repeated adjacent words,
- uneven spacing,
- mixed Arabic/English punctuation,
- long word runs without a sentence boundary.

The response polisher is a final surface pass. It does not infer new facts and
does not choose a strategy. It only cleans the text after a strategy has already
produced an answer.

## Files

- `src/physics/engines/strategies/response_polisher.jl`
- `src/physics/engines/strategies/finisher.jl`
- `src/physics/engines/generator.jl`
- `test/test_response_polisher.jl`
- `scripts/response_polisher_live_probe.jl`
- `scripts/trained_response_polisher_probe.jl`
- `scripts/post_training_smoke.jl`

## Gate

The pass is off by default and can be enabled with:

```powershell
$env:MIRNAN_ENABLE_RESPONSE_POLISHER="1"
```

Default behavior remains unchanged:

```powershell
$env:MIRNAN_ENABLE_RESPONSE_POLISHER="0"
```

## Current Capabilities

When enabled, the polisher:

- normalizes whitespace,
- removes immediate repeated words,
- fixes light punctuation spacing,
- converts comma/semicolon to Arabic comma/semicolon when Arabic text is present,
- adds a sentence-ending period when missing,
- lightly groups very long unpunctuated word runs.

It is intentionally conservative. It does not rewrite facts, add new claims, or
replace the output with a template.

## Phase 1.1 - Type-Aware Diagnostics

The polisher now exposes a small diagnostic profile through
`response_polish_profile(prompt, text)`.

The profile reports:

- `kind`: conservative answer type guess (`purpose`, `conditional`,
  `temporal`, `spatial`, `state`, `quantity`, `scene`, `definition`, or
  `generic`),
- `language`: `arabic` or `latin`,
- `has_repetition`: whether adjacent repeated words would be collapsed,
- `run_on`: whether the answer looks like a long unpunctuated word run,
- `preserved`: whether the answer should not be touched.

The preservation guard protects code-like and multi-line memory outputs such as
Majnon code-memory records, fenced code, and hash-tagged excerpts. This matters
because those outputs are structured evidence, not prose to be polished.

This phase is diagnostic only. It prepares the ground for future type-specific
surface wording without changing strategy selection or adding new content.

## Phase 1.2 - Minimal Type-Specific Surface Style

The first type-aware surface rule is intentionally tiny:

- conditional answers now separate the condition from its result with an Arabic
  semicolon before `يترتب على ذلك`,
- scene-purpose composites now separate the sensory scene from the purpose
  clause with an Arabic semicolon (`؛ ومن جهة الغاية:`),
- state answers now separate the event from the state phrase with an Arabic
  semicolon before `وكان على حال`,
- English quantity answers collapse accidental repeated copula text such as
  `is is` to `is`.

These rules only change punctuation or immediate repetition. They do not infer,
replace, summarize, or reorder the answer content.

## Tests

Verified with:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_response_polisher.jl
```

Result:

```text
response polisher conservative final wording | 26 / 26 passed
```

Also verified that the broader relation/strategy test still passes:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\test\test_relation_frame.jl
```

## Live Probe

The live probe compares the same representative answers with the response
polisher gate off and on:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\response_polisher_live_probe.jl
```

It covers repeated words, conditional answers, scene-purpose composites, state
answers, English quantity wording, and preservation of code-like memory output.
The probe is included in `post_training_smoke.jl`, so `run_train.bat` now checks
it after training.

## Trained-Memory Probe

The trained-memory probe reads post-training memory files directly, without
loading the full generator matrices:

```powershell
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\trained_response_polisher_probe.jl
```

It currently checks quantity, conditional, and scene-purpose answers from the
trained memories. The default limit is 3 cases and can be changed with:

```powershell
$env:MIRNAN_TRAINED_POLISHER_PROBE_LIMIT="6"
```

This probe is included only in the heavy post-training smoke path:

```powershell
$env:MIRNAN_POST_SMOKE_HEAVY_GATES="1"
```
