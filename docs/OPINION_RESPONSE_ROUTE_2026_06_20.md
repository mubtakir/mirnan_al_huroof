# Mirnan Opinion Response Route - 2026-06-20

## Purpose

This change adds a small opinion/evaluation route to `IntentResponsePlanner`.
It is not a new physics mechanism and it does not change model weights. Its job
is to catch explicit evaluative prompts before free generation fails them.

## What It Detects

The route detects prompts such as:

- `ما رأيك في ...`
- `ما قولك في ...`
- `هل ... مفيد؟`
- `هل ... واجب؟`
- `أهمية ...`

It avoids conversational prompts such as `هل تحب التعلم؟`; those remain part of
the dialogue problem and should be handled separately.

## Response Shape

The answer uses a compact evaluative structure:

1. topic extraction.
2. simple positive evaluation.
3. reason.
4. balancing condition.

Example:

```text
أرى أن موضوع المعرفة نافع إذا ارتبط بـالفهم، لأنه يساعد على توسيع الفهم.
لكنه يحتاج إلى التجربة حتى لا يبقى حكماً عاماً بلا ميزان.
```

This is a semantic evaluation, not a simulated personality.

## Probe Result

Latest report:

`models/mirnan/reports/question_probe_2026-06-20_014626.md`

Summary:

- Full success: 39/50.
- Acceptable: 3/50.
- Failed: 8/50.
- Acceptable or better: 42/50, or 84.0%.

Opinion category:

- Before: 0/5.
- After: 5/5.

Overall progress from the strict baseline:

- Baseline acceptable or better: 30/50, or 60.0%.
- Current acceptable or better: 42/50, or 84.0%.
- Net gain: +24.0 points.

## Remaining Work

The main weak areas are now:

- dialogue: 2/6.
- definition: 4/8.
- descriptive quality: structurally acceptable but often thin.

The next focused improvement should be dialogue intent and conversational
answers, followed by richer definition handling.
