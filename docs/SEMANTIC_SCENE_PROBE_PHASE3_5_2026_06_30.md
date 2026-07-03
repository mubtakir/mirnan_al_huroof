# SemanticScene Probe - Phase 3.5

This note documents the developer probe for Mirnan semantic imagination.

## Scope

Phase 3.5 adds a command-line diagnostic probe. It does not call `generate!`, does not add a strategy, and does not affect runtime answers.

## Script

- `models/mirnan/scripts/semantic_scene_probe.jl`

## Usage

Run with built-in examples:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_probe.jl
```

Run with a custom text file:

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"
& "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_probe.jl path\to\examples.txt
```

Each non-empty line in the file is used as a probe prompt. The whole file is also used to build a temporary semantic-calculus memory and scene memory.

## Output

For each prompt, the probe prints:

- `AGREEMENT`: `aligned`, `partial`, `divergent`, `scene_only`, `calculus_only`, or `none`
- `OVERLAP`
- scene confidence
- calculus guidance confidence
- selected scene summary
- scene effect terms
- semantic-calculus guidance terms

## Purpose

This is the inspection step before giving the semantic imagination layer any answer-generation role. It helps developers see whether the sensory scene layer and the Clifford semantic-calculus layer agree on the expected effect of an event.

## Probe Calibration

The scene selector now requires real lexical overlap before accepting a scene. Scene confidence alone is not enough. This prevents unrelated high-confidence scenes from being selected for prompts about another event.

The built-in probe examples build each scene from a pair-local semantic-calculus memory, then compare against the shared calculus memory. This keeps the sensory scene effects from being polluted by unrelated training pairs.
