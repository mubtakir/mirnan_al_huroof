# Toy Corpus Curation - 2026-06-29

This note records the first conservative cleanup of `models/mirnan/data/toy_corpus`.

## Goal

Reduce training noise and corpus imbalance without deleting source material.

Mirnan currently benefits most from compact Arabic conceptual, relational, and dialogue-oriented text. Very large generic files, code corpora, and imported clean text can dominate the local physical fields during training and make the model less responsive to the intended semantic relations.

## Result

Before curation:

- `toy_corpus`: about 18.5 MB
- Main dominant sections: `aaa2_corpus`, `bbb1_clean_txt`, `code_corpus`, `julia_corpus`

After curation:

- `toy_corpus`: about 5.3 MB
- Remaining sections:
  - `aaa0_mafahem`: 336 files, about 3.86 MB
  - `aaa2_corpus`: 42 files, about 1.02 MB
  - `bbb0_a`: 9 files, about 0.30 MB
  - `aaa3_ulum`: 2 files, about 0.11 MB

## What Was Moved

Nothing was deleted. Excluded material was moved to:

`models/mirnan/data_quarantine/toy_corpus_review_2026_06_29`

Moved groups:

- `code_corpus`: code/programming material, separated from natural-language training.
- `julia_corpus`: Julia-specific code/text material, separated from natural-language training.
- `bbb1_clean_txt`: broad generic text, useful later only after selective review.
- `aaa0_mafahem/english_code_corpus.txt`: English/code file inside the conceptual section.
- Oversized files from `aaa2_corpus`: files larger than about 200 KB, plus Quran-sized long files, except focused Mirnan/dialogue/filament corpus files.

The machine-readable move list is stored in:

`models/mirnan/data_quarantine/toy_corpus_review_2026_06_29/manifest.json`

## Why This Helps Mirnan

The cleanup makes the training distribution less dominated by a few large sources. This should help:

- preserve the weight of relation examples such as knowledge, justice, mercy, peace, fear, trust, and ignorance;
- reduce accidental memorization of generic imported text;
- keep code language from leaking into ordinary Arabic answers;
- make paragraph-level training more stable because the largest long-line sources no longer dominate segmentation.

## Recommended Next Step

Run a fresh paragraph-level training pass, then test the dialogue and relation probes again. If the model becomes too narrow, selectively restore reviewed files from quarantine instead of restoring entire folders.
