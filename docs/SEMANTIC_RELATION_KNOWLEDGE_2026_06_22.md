# Semantic Relation Facts

Date: 2026-06-22

This note documents the cleanup of the old semantic relation answer memory.

## Current Rule

Mirnan must not keep reusable semantic answers as ready text. The old
`semantic_relation_answers.json` file was removed because it stored complete
answers. The replacement is:

```text
models/mirnan/knowledge/semantic_relation_facts.json
```

This file stores structured relation facts only. It must not contain `answer`
or `evidence` fields.

## How It Is Used

During `models/mirnan/train.jl`, each structured fact is imported directly into:

- `al_nisba`, as a relation between concepts.
- `al_istinbat`, as an attention/inference field with polarity and focus terms.

The generator does not read this file as an answer source, and the training
step does not preserve a full sentence as a reusable reply.

## Allowed Record Shape

Good:

```json
{
  "id": "fact_power_peace_insufficient",
  "relation_type": "need",
  "subject": "سلام",
  "object": "علم",
  "marker": "يحتاج",
  "polarity": 1,
  "terms": ["سلام", "قوة", "يكفي", "حفظ", "علم", "عدل", "حقوق"],
  "contrast_terms": ["قوة وحدها"],
  "focus_terms": ["علم الحقوق", "إقامة العدل", "قوة منضبطة"]
}
```

Not allowed:

```json
{
  "answer": "جواب جاهز لسؤال بعينه."
}
```

Also not allowed:

```json
{
  "evidence": "جملة كاملة يمكن أن تعود كما هي في جواب المستخدم."
}
```

## Boundary

The only intentionally fixed-answer knowledge file is the social convention
list for dialogue greetings and customary replies. Semantic knowledge must be
stored as structured facts, relations, definitions, examples, or fields, not as
direct answer templates.
