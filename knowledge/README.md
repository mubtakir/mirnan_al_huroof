# Mirnan Persistent Knowledge

This directory stores cumulative knowledge that should survive model retraining.

Training may clean `models/mirnan/model`, but it must not delete this directory.
Files here are intended for human review, manual additions, and reuse across
future training runs.

Current files:

- `semantic_equivalence.json`: persistent semantic-neighbor and synonym memory
  used by `al_muradif`.
- `istinbat_attention.json`: persistent inference-attention cues used by
  `al_istinbat` to focus relation words and the terms that follow them.
- `definitions.json`: reserved for persistent reviewed definitions.
- `semantic_relation_facts.json`: structured semantic relation facts imported
  into `al_nisba` and `al_istinbat`; records must not contain ready `answer` or
  full-sentence `evidence` fields.
- `causal_relations.json`: reserved for persistent causal facts.
- `opposites.json`: reserved for future learned/manual polarity and opposition.
- `dialogue_facts.json`: reserved for stable dialogue speech acts.
- `manual_overrides.json`: reserved for carefully reviewed corrections.

The runtime model may receive merged copies of these memories under
`models/mirnan/model`, but this directory is the durable source.
