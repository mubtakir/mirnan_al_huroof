# Mirnan Question Probe Response Fixes - 2026-06-20

## Purpose

This note records the response-routing fixes made after the comprehensive
50-question probe. The goal was not to add new physics or change global
weights, but to keep explicit math and code requests inside their specialized
channels before they are taken by general text, conditional, or mechanism
generation.

## Changes

- `al_hisab` now solves more small verified problems directly:
  - square area, such as "احسب مساحة مربع طول ضلعه 5."
  - square root, such as "ما هو جذر 16؟"
  - remainder word problems, such as "إذا كان لديك 10 تفاحات وأكلت 3، كم تبقى؟"
- `al_code` now handles common direct code requests:
  - loops that print numbers.
  - simple conditions.
  - simple variable definitions, including Julia.
  - a function returning the larger of two values.
- `generator.jl` now gives explicit code and math prompts priority before
  general response planning.
- Code intent detection was tightened so a word like "العدالة" is not mistaken
  for the independent code word "دالة".
- The comprehensive question probe now judges code as code structure, not as
  Arabic prose that needs a prompt anchor.

## Probe Result

Latest report:

`models/mirnan/reports/question_probe_2026-06-20_011332.md`

Summary:

- Full success: 34/50.
- Acceptable: 3/50.
- Failed: 13/50.
- Acceptable or better: 37/50, or 74.0%.

Category results:

- Code: 5/5.
- Math: 6/6.
- Causal: 7/7.
- Mechanism/how: 7/7.
- Descriptive: 6/6 acceptable or better, but several answers remain thin.
- Definition: 4/8.
- Dialogue: 2/6.
- Opinion/evaluation: 0/5.

## Remaining Weak Areas

The next work should focus on:

- richer definition memory and better fallback for unknown definitions.
- dialogue intent and natural conversational replies.
- opinion/evaluation responses built from definition plus simple assessment.
- descriptive quality, especially when the answer is only a noun phrase.

These should be handled as routing, memory, and response-quality work before
any new physics mechanism is added.
