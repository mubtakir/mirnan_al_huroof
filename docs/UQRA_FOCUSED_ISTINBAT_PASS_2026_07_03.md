# Uqra Focused Istinbat Pass - 2026-07-03

## Summary

`data/uqra` is now treated as a structured relation-seed corpus for
`al_istinbat`, not only as ordinary training text.

The full training pipeline still trains on the complete corpus, but it now adds
a focused line-level pass over `models/mirnan/data/uqra` while building
`al_istinbat`. This preserves paragraph-level training for the main model while
ensuring short linguistic-key examples are learned as clean RelationFrame
records.

## Why

The trained response probe showed that fixed prompts such as:

- `متى سافر الطالب؟`
- `أين جلس الطفل؟`
- `كيف دخل الطفل؟`

could miss the clean `uqra` examples and fall back to strange trained-memory
records. The cause was not the response polisher; it was that structured
relation examples needed a focused path into the istinbat memory.

## Changes

- `train.jl`
  - Adds an `al_istinbat` focused pass over `data/uqra` with `granularity=:line`.
  - Marks sources with `source_dir="uqra"` and
    `knowledge_type="structured_relation_seed"`.

- `scripts/rebuild_istinbat_memory.jl`
  - New lightweight script that updates an existing trained
    `model/al_istinbat.json` from `data/uqra` without full retraining.

- `run_train.bat`
  - Now runs the focused `al_istinbat` refresh after full training and before
    evolution/scene/quantity post-build steps.

- `al_istinbat.jl`
  - RelationFrame extraction now cuts the right side at the next relation
    marker, so `قبل الفجر لكي يصل` becomes temporal `قبل الفجر` plus a separate
    purpose frame.
  - Marker matching now respects word boundaries, preventing English short
    markers such as `in` from matching inside words like `understanding`.
  - Purpose selection now uses a purpose-specific selector before composing
    scene/purpose answers, so a competing temporal frame such as
    `عندما دفع اللاعب الحجر...` cannot hide the purpose frame
    `دفع اللاعب الحجر لكي...`.

## Verification

Focused rebuild smoke:

- `متى سافر الطالب؟` -> `كان سفر الطالب قبل الفجر.`
- `أين جلس الطفل؟` -> `كان مكان جلوس الطفل في الحديقة.`
- `كيف دخل الطفل؟` -> clean state answer from `uqra`.

Trained response polisher probe:

- quantity: stable
- conditional: clean conditional answer
- temporal: fixed prompt answered from clean `uqra`
- spatial: fixed prompt answered from clean `uqra`
- state: fixed prompt answered from clean `uqra`

Tests:

- `test_relation_frame.jl` passed.
- `post_training_smoke.jl` passed:
  - response polisher
  - semantic scene / purpose bridge
  - quantity memory

Trained response polisher probe now shows a direct `scene-purpose` answer for:

- `لماذا دفع اللاعب الحجر؟`
