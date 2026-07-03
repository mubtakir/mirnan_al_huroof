# Mirnan Understanding Policy and Roadmap

This document records the working direction for Mirnan after the 2026-06-29 discussion.

## Core Vision

Mirnan must not become a store of ready-made knowledge answers.

The goal is to make Mirnan discover the hidden mechanisms inside language: question keys, causal keys, purpose keys, evidence keys, definition keys, negation keys, and relation direction. Understanding is not repetition of a pattern. Understanding begins when the system recognizes the function behind the pattern.

In symbolic form:

- `P ... Y` is not only a memorized sequence.
- `P` becomes a question/check operator.
- `Y` becomes a truth/confirmation value.
- `N` becomes a negation/rejection value.

In Arabic:

- `هل` is a verification operator.
- `لماذا` asks for cause, purpose, or evidence.
- `كيف` asks for path, mechanism, or process.
- `يؤدي إلى` is a directional causal operator.
- `بسبب` is a causal operator whose direction may be reversed in surface form.
- `لكي` and `حتى` are purpose operators.
- `يدل على` is an evidence operator.
- `هو` may be a definition or identity operator.
- `ليس` and `لا` are negation operators.

Mirnan should answer from these mechanisms, not from stored answer templates.

## Allowed Ready Responses

Only near social memory may remain as ready responses.

These are not considered intelligence. They are conversational reflexes and etiquette:

- Greeting replies: `السلام عليكم` -> `وعليكم السلام`.
- Courtesy replies: `شكرا` -> `عفوا`.
- Simple social questions: `كيف حالك؟` -> `الحمد لله`.
- Identity question: `ما اسمك؟` -> `أنا مرنان`.
- Simple morning/evening courtesy.

This layer should stay small, explicit, and isolated from knowledge answering.

## Disallowed Ready Knowledge Answers

The following should not be implemented as fixed answer lists:

- Answers to `هل`.
- Answers to `لماذا`.
- Answers to `كيف`.
- Directional relation answers.
- Difference/comparison answers.
- Conceptual definitions.
- Cause/result explanations.
- Purpose/means explanations.
- Evidence/proof explanations.

If such answers exist in code, they are temporary scaffolding. They should either be removed or converted into mechanisms.

## Mechanism Targets

Each question path should be routed to a mechanism:

- `هل`: truth, nisba, negation, contradiction, and relation direction.
- `لماذا`: cause, purpose, evidence, or explanatory source.
- `كيف`: process, path, sequence, or operational mechanism.
- `ما العلاقة`: relation type and direction.
- `ما الفرق`: comparison across attributes, function, effect, and field.
- `ما هو`: definition, identity, class, or essence.

The generator may phrase the answer freely, but the decision should come from extracted or resonant structure.

## Roadmap After Training

1. Turn `models/mirnan/hiwar.txt` into official tests under `test/`.
2. Make those tests semantic, not exact-string tests.
3. Protect the main repaired paths:
   - `العلم يزيد الفهم` must not become negative.
   - `السلام` inside a relation question must not be treated as a greeting.
   - cause and result must not be inverted.
   - `لماذا` must produce cause, purpose, or evidence.
   - `كيف` must produce a mechanism or path.
4. Audit all ready-answer lists in the codebase.
5. Classify each list:
   - keep if it is near social memory;
   - remove or convert if it answers knowledge, causality, relation, or reasoning.
6. Build or strengthen explicit operator extraction:
   - causation;
   - purpose;
   - condition;
   - evidence;
   - definition;
   - negation;
   - generalization;
   - contrast/exception.
7. Re-run dialogue and relation probes after every cleanup.

## Development Rule

When in doubt, prefer a mechanism over a template.

A ready answer is acceptable only when a human would also answer reflexively from etiquette rather than reasoning.
