# SemanticScene Purpose Bridge Probe - Phase 9.1

This phase adds a live diagnostic probe for the bridge between:

- `SemanticSceneMemory`
- `al_hisban_al_dalali`
- `IstinbatAttentionMemory`

## Script

```text
models/mirnan/scripts/semantic_scene_purpose_bridge_probe.jl
```

The script loads the trained model memories directly from:

```text
models/mirnan/model/semantic_scenes.json
models/mirnan/model/al_hisban_al_dalali.json
models/mirnan/model/al_istinbat.json
```

It does not construct a full `MirnanGenerator`, so it is lighter than a generation probe.

If the trained `al_istinbat.json` contains no purpose records, the probe prints:

```text
NOTE: trained al_istinbat has no purpose records
```

In that case, trained `cooperative` and `purpose_only` cases cannot appear until `al_istinbat` is rebuilt or the model is retrained after the `RelationFrame` learning changes.

## Output

The probe prints:

- number of semantic scenes
- number of istinbat records
- number of purpose records
- examples for:
  - `cooperative`
  - `scene_only`
  - `purpose_only`
  - `none`

For each case it prints:

- prompt
- agreement
- scene/purpose presence
- confidence values
- scene summary
- scene answer
- purpose answer

## Display Quality

The probe now skips low-quality display examples:

- empty purpose answers
- purpose answers missing a clear "purpose of what" side
- overly long purpose answers
- overly long prompts
- overly long scene summaries
- scene summaries with noisy terms such as "no/question/answer" or their Arabic equivalents
- more than `MIRNAN_BRIDGE_PROBE_LIMIT` scanned purpose records for `purpose_only`

For `cooperative`, the probe now scans from both sides:

- semantic scenes -> generated why prompts
- purpose records -> generated why prompts

This does not change the trained memory. It only makes the diagnostic output easier to read.

## Controlled Sanity Case

If no trained `cooperative` example is found in the scan window, the script prints a controlled sanity case:

```text
Khalid hit the ball with bat in yard before sunset.
Khalid hit the ball لكي يتحرك ball.
```

This confirms the bridge itself still works even if the trained memories do not yet contain a naturally matching pair.

## Command

```powershell
cd "C:\Users\allmy\Desktop\aaa\basil\majnon"; & "C:\Users\allmy\AppData\Local\Programs\Julia-1.12.6\bin\julia.exe" --project=models\mirnan models\mirnan\scripts\semantic_scene_purpose_bridge_probe.jl
```

Optional scan limit:

```powershell
$env:MIRNAN_BRIDGE_PROBE_LIMIT="500"
```

## Status

This is a diagnostic script only. It does not affect generation, strategy ordering, or any answer route.
