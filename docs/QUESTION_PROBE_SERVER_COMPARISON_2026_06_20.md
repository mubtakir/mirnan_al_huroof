# Mirnan Question Probe Server Comparison - 2026-06-20

## Purpose

This document records the full 50-question probe after restarting Mirnan with
the latest routing and UI fixes. It compares the current behavior with the
strict baseline probe that exposed failures in math and code routing.

## Reports Compared

- Baseline strict report:
  `models/mirnan/reports/question_probe_2026-06-20_005734.md`
- Current report:
  `models/mirnan/reports/question_probe_2026-06-20_013336.md`

The current probe loads the same trained model and generation code used by the
server. Manual checks through the running web server confirmed that the server
is also using the new math/code routing and the UI code-block rendering fix.

## Overall Result

| Metric | Baseline | Current | Change |
|---|---:|---:|---:|
| Full success | 27/50 | 34/50 | +7 |
| Acceptable | 3/50 | 3/50 | 0 |
| Failed | 20/50 | 13/50 | -7 |
| Acceptable or better | 30/50 (60.0%) | 37/50 (74.0%) | +14.0 pts |

## Category Comparison

| Category | Baseline | Current | Change |
|---|---:|---:|---:|
| Code | 1/5 | 5/5 | +4 |
| Math | 3/6 | 6/6 | +3 |
| Causal | 7/7 | 7/7 | 0 |
| Mechanism/how | 7/7 | 7/7 | 0 |
| Descriptive | 6/6 | 6/6 | 0 |
| Definition | 4/8 | 4/8 | 0 |
| Dialogue | 2/6 | 2/6 | 0 |
| Opinion/evaluation | 0/5 | 0/5 | 0 |

## What Improved

### Math

Math now routes to `al_hisab` before general conditional or text generation.
The following formerly weak cases now work:

- square root: "ما هو جذر 16؟"
- word remainder: "إذا كان لديك 10 تفاحات وأكلت 3، كم تبقى؟"
- square area: "احسب مساحة مربع طول ضلعه 5."

The current answer style is:

- result sentence.
- short solving steps.
- verification line.
- explicit confirmation.

This is accurate and safe, but should be monitored. For very simple user
questions, a shorter answer mode may be desirable later.

### Code

Code now routes to `al_code` before general "كيف" or "إذا" planning. The probe
now passes all code prompts:

- Python function adding two numbers.
- Python loop printing 1 to 10.
- Python condition for number greater than 10.
- Julia variable definition.
- function returning the larger of two values.

The UI now renders code blocks as LTR `pre` blocks and preserves indentation.

The generated code remains intentionally short. This is correct for code-only
answers, but future work may add optional explanation around code when the user
asks for teaching or reasoning.

## What Did Not Improve Yet

### Definition

Definition remains 4/8. This points to `al_ta3rif` coverage and fallback
behavior, not math/code routing.

### Dialogue

Dialogue remains 2/6. Conversational prompts such as "هل تحب التعلم؟" still
need a dedicated conversational response route or richer dialogue memory.

### Opinion / Evaluation

Opinion remains 0/5. Mirnan still lacks a stable evaluation response strategy:
definition plus balanced judgment plus reason.

## Current Monitoring Notes

- Math is now reliable in the probe, but its verified-step format may be too
  verbose for every context.
- Code is now reliable in the probe, but terse. It may need an optional
  explanatory wrapper later.
- Descriptive answers pass structurally, yet several remain thin.
- Definition, dialogue, and opinion are the next genuine weak areas.

## Next Recommended Work

1. Improve definition fallback without adding new physics.
2. Add a small dialogue-intent response route for greetings, yes/no questions,
   personal/conversational prompts, and polite replies.
3. Add an opinion/evaluation route based on definition plus simple assessment.
4. Re-run the 50-question probe after each focused change.

The current server state is the best measured state so far: 74.0% acceptable or
better on the strict 50-question probe.
