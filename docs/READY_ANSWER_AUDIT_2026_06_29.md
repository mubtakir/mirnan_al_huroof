# Ready Answer Audit - 2026-06-29

This audit tracks the removal of knowledge answer templates from Mirnan.

## Policy

Ready answers are allowed only for near social memory:

- greetings;
- thanks;
- simple courtesy;
- simple identity such as "what is your name?"

Knowledge answers must come from mechanisms:

- nisba/truth for `هل`;
- causation, purpose, or evidence for `لماذا`;
- mechanism/path for `كيف`;
- relation direction for `ما العلاقة`;
- comparison for `ما الفرق`;
- definition memory for `ما هو`.

## First Finding

The first high-risk area is:

`src/physics/engines/strategies/yesno_relations.jl`

It contained canonical relation answer paths for specific conceptual pairs such as:

- knowledge and understanding;
- justice and peace;
- mercy and trust;
- peace and fear;
- reversed cause/result questions.

These paths were useful as temporary guards, but they are knowledge templates. They must not bypass the mechanism policy.

## Change Made

`MIRNAN_STRICT_NO_TEMPLATES` is now enabled by default.

The canonical yes/no and explanatory relation answer functions now return nothing when strict mode is active:

- `_canonical_yesno_relation_answer`
- `_canonical_explanatory_relation_answer`

This means the normal path must rely on learned relation memory, inference attention, semantic evidence, and other mechanisms. The old canonical paths remain available only as an explicit debugging fallback if the environment disables strict mode:

`MIRNAN_STRICT_NO_TEMPLATES=0`

## Protection

The following tests protect this direction:

- `test/test_question_paths_from_hiwar.jl`
- `test/test_strict_no_templates.jl`

`test_question_paths_from_hiwar.jl` is the acceptance test after training.

`test_strict_no_templates.jl` ensures knowledge templates stay gated when strict mode is active.

## Next Audit Targets

Continue reviewing these files:

- `yesno_relations.jl`: remaining literal fallback answers in contradiction and polarity paths.
- `definition_strategy.jl`: conceptual glosses that may be fixed templates.
- `difference_and_gate.jl`: difference answers and concept glosses.
- `evidence_relations.jl`: relation memory answers that may still phrase fixed answers.
- `dialogue_strategy.jl`: keep only social memory here.

The next cleanup should remove or mechanize one group at a time, with `hiwar` tests run after each group.
