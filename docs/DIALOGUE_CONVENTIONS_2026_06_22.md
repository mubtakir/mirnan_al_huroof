# Dialogue Conventions Memory

Date: 2026-06-22

This note documents the narrow persistent memory for conventional dialogue replies in Mirnan.

## Purpose

Some utterances are social conventions rather than open reasoning tasks. Examples:

- `السلام عليكم` -> `وعليكم السلام ورحمة الله.`
- `صباح الخير` -> `صباح النور.`
- `شكرا` -> `عفواً.`

These replies are treated as reviewed speech conventions. They are not open-domain answer templates, and they must not be used for definitions, causal reasoning, comparison, opinion, mathematics, code, or semantic explanation.

## Honest Boundary Statement

This is the only intentionally fixed/template-like list in Mirnan.

The reason is deliberate and explicit: a small class of social replies is conventional by nature. Humans usually answer greetings, thanks, farewell formulas, and short well-being phrases by inherited social custom, not by performing a new act of semantic reasoning each time.

Therefore this layer is accepted as a narrow etiquette memory, not as a substitute for understanding. It must remain small, reviewed, and limited to conventional speech acts. Any answer that requires meaning, causality, comparison, definition, opinion, calculation, code generation, or semantic inference must remain outside this file and pass through Mirnan's learned or physical layers.

## Location

Reviewed conventions live in:

```text
models/mirnan/knowledge/dialogue_facts.json
```

This file is outside `models/mirnan/model`, so normal retraining does not delete it.

## Runtime Loading

During generator startup:

1. Mirnan loads the trained model and runtime learning.
2. Then it merges `knowledge/dialogue_facts.json` into `al_aql.speech_acts`.

This order matters because runtime learning restores `al_aql` from disk and may clear previous speech acts. Persistent dialogue conventions are therefore loaded after runtime learning.

## Boundary

The layer is intentionally narrow:

- greetings
- farewell formulas
- thanks
- short well-being questions
- common supplications

It must not contain explanatory answers such as “why is knowledge light?” or “what is justice?”. Those belong to `al_ta3rif`, `al_istinbat`, `al_nisba`, semantic relations, or the physical generator.

## Adding New Items

Add entries as `speech_acts` with these fields:

```json
{
  "speaker": "user",
  "act_type": "تحية",
  "content": "السلام عليكم",
  "responder": "mirnan",
  "response_act": "رد_تحية",
  "response_content": "وعليكم السلام ورحمة الله.",
  "confidence": 0.99,
  "evidence": "persistent_dialogue_convention"
}
```

Keep each item short and conventional. If a reply requires reasoning, do not put it here.

## Guard Test

The guard test is:

```text
models/mirnan/test/test_dialogue_salam_and_reported_speech.jl
```

It checks that:

- greetings can use persistent dialogue conventions.
- `كيف الصحة؟` can use the well-being convention.
- `ما معنى السلام؟` is not treated as a greeting.
