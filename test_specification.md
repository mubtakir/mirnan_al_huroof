# Mirnan — مواصفات الاختبار الشاملة
## Comprehensive Test Specification for a Wave-Physics Arabic LLM

---

## Contents
1. [Testing Philosophy](#1-testing-philosophy)
2. [Existing Test Catalog](#2-existing-test-catalog)
3. [Test Pyramid & Coverage Goals](#3-test-pyramid--coverage-goals)
4. [Subsystem Test Plans](#4-subsystem-test-plans)
5. [Wave-Physics-Specific Tests](#5-wave-physics-specific-tests)
6. [Integration Test Design](#6-integration-test-design)
7. [Evaluation Harness](#7-evaluation-harness)
8. [Test Infrastructure](#8-test-infrastructure)
9. [Priority Roadmap](#9-priority-roadmap)
10. [Appendices: Test Templates](#10-appendices-test-templates)

---

## 1. Testing Philosophy

Mirnan is not a transformer-based LLM. It is a **wave-physics simulator** that represents words as phase vectors in a Clifford-algebraic semantic space. Therefore, testing must validate:

| Principle | What it means for testing |
|-----------|--------------------------|
| **Phase coherence** | Similar concepts must have similar phase vectors; antonyms must have opposing phases |
| **Wave interference** | Constructive interference = semantic reinforcement; destructive = contradiction |
| **Gravitational semantics** | Word mass (from letter frequency) × proximity → semantic attraction |
| **Resonant retrieval** | Memory is not indexed look-up but resonant frequency matching |
| **Entropy gates** | Semantic uncertainty must be measurable and bounded |
| **Born-rule selection** | Word choice probability = |Ψ_total|², not softmax of logits |
| **Kuramoto synchronization** | Words in context phase-lock over time |
| **Clifford-algebraic operators** | Geometric product captures analogy, opposition, transformation |
| **Density-matrix semantics** | Mixed states represent ambiguity; collapse represents disambiguation |

### Key Testing Principles

1. **Deterministic verification**: Constants, dimensions, algebraic identities must be exact
2. **Invariant checking**: Phase vectors must always be normalized; Born probabilities sum to ≈1
3. **Statistical verification**: Some properties (e.g., semantic attraction) are tested via consistent inequality, not exact match
4. **Round-trip persistence**: Save/load cycles must preserve all state
5. **Regression-first**: Every bug fix adds a test before the fix
6. **Arabic-first**: Tests use Arabic text primarily; English tests are secondary

---

## 2. Existing Test Catalog

### 2.1 Current Test Suite (46 files, ~1,200+ assertions)

| # | File | Tests | Module/Subsystem | Type |
|---|------|-------|-----------------|------|
| 1 | `runtests.jl` | ~80 testsets | Core physics engine | Unit + Integration |
| 2 | `test_phase1.jl` | Phase vectors | Core | Unit |
| 3 | `test_phase2.jl` | Phase operations | Core | Unit |
| 4 | `test_enhanced.jl` | Enhanced vectors | Core | Unit |
| 5 | `test_enhanced_final.jl` | Final enhanced | Core | Unit |
| 6 | `test_euclidean.jl` | Euclidean operations | Core | Unit |
| 7 | `test_exponential.jl` | Exponential mapping | Core | Unit |
| 8 | `test_exponential_v2.jl` | V2 exponential | Core | Unit |
| 9 | `test_centered_signal.jl` | Signal centering | Core | Unit |
| 10 | `test_raw_params.jl` | Raw parameters | Core | Unit |
| 11 | `test_clifford_enhanced.jl` | Clifford algebra | Core | Unit |
| 12 | `test_clifford_words.jl` | Clifford words | Core | Unit |
| 13 | `test_new_model.jl` | New model | Core | Unit |
| 14 | `test_generator_fallback.jl` | Generator fallback | Generator | Integration |
| 15 | `test_generation_quality_guard.jl` | Quality guard | Generator | Unit |
| 16 | `test_training_preserves_tashkeel.jl` | Training | Training | Integration |
| 17 | `test_tashkeel_projection_matching.jl` | Tashkeel | Cross-module | Integration |
| 18 | `test_dialogue_salam_and_reported_speech.jl` | Dialogue | Generator | Integration |
| 19 | `test_social_reply_memory_boundary.jl` | Social | Generator | Integration |
| 20 | `test_question_type_matrix.jl` | Question types | IRP | Integration |
| 21 | `test_question_paths_from_hiwar.jl` | Question paths | IRP | Integration |
| 22 | `test_relation_frame.jl` | **154 tests** | RelationFrame + QuantityFrame | Unit + Integration |
| 23 | `test_intent_response_planner.jl` | **68 tests** | IRP | Unit |
| 24 | `test_context_tension.jl` | Context tension | Generator | Unit |
| 25 | `test_mirnan_cerebellum.jl` | Cerebellum | Cerebellum | Unit + Integration |
| 26 | `test_sense_superposition.jl` | **Sense superposition** | Sense engine | Unit |
| 27 | `test_self_review.jl` | Self-review | SelfReview | Unit |
| 28 | `test_semantic_gravity.jl` | Semantic gravity | Gravity | Unit |
| 29 | `test_semantic_imagination.jl` | **Semantic imagination** | Al-Hisban | Unit + Integration |
| 30 | `test_syntactic_gravity.jl` | Syntactic gravity | Syntax | Unit |
| 31 | `test_contextual_learning.jl` | Contextual learning | ContextualLearning | Unit |
| 32 | `test_contextual_selection_layer.jl` | Contextual selection | Generator | Unit |
| 33 | `test_al_lisan.jl` | Linguistic patterns | Al-Lisan | Unit + Integration |
| 34 | `test_al_code.jl` | Code patterns | Al-Code | Unit + Integration |
| 35 | `test_al_tadbir.jl` | Procedural plans | Al-Tadbir | Unit + Integration |
| 36 | `test_al_hisab.jl` | Math solving | Al-Hisab | Unit + Integration |
| 37 | `test_al_ta3rif.jl` | Definitions | Al-Ta3rif | Unit + Integration |
| 38 | `test_al_nisba.jl` | **19 tests** | Al-Nisba | Unit + Integration |
| 39 | `test_al_muradif.jl` | Synonyms | Al-Muradif | Unit |
| 40 | `test_al_istinbat.jl` | **29 tests** | Al-Istinbat | Unit + Integration |
| 41 | `test_al_intibah.jl` | Attention field | Al-Intibah | Unit |
| 42 | `test_al_aql.jl` | Al-Aql cognitive | Al-Aql | Integration |
| 43 | `test_al_aql_dialogue.jl` | Al-Aql dialogue | Al-Aql | Integration |
| 44 | `test_al_aql_speech_matching.jl` | Speech matching | Al-Aql + Gen | Unit |
| 45 | `test_al_hisban_al_dalali.jl` | **41 tests** | Semantic calculus | Unit + Integration |
| 46 | `test_pattern_memory_integration.jl` | **32 tests** | Cross-module | Integration |
| 47 | `test_mirnan_cerebellum.jl` | Cerebellum | Cerebellum | Unit + Integration |
| 48 | `test_strict_no_templates.jl` | Strict mode | Generator | Integration |

### 2.2 Key Subsystems by Module

```
src/physics/core/         → LetterDB, GravityEngine, CliffordMath, ResonantChain,
                           EntropyGate, RAMCore, PhaseReinforcement, DCCF, PPM, AMFS,
                           DensityMatrix, CausalFlow, WeightResonance, SyntaxField,
                           MorphoPhasic, DialogueEngine, PoeticGravity,
                           ContextualOperators, SpectralWave, KuramotoOscillator,
                           ChaosEntanglement, MirnanCerebellum

src/physics/engines/      → Generator (al_aql_answer, strategies, quality guard),
                           Al-Istinbat (IstinbatAttentionMemory, RelationFrame,
                           QuantityFrame, purpose_answer, compare_purpose_strategies),
                           IntentResponsePlanner (IRP), SelfReviewEngine,
                           ContextualLearning, SemanticCalculusMemory (Al-Hisban),
                           SemanticSceneMemory

src/physics/groups/       → ArabicGroup (al_ta3rif, al_nisba, al_lisan, al_code,
                           al_tadbir, al_hisab, al_intibah, al_muradif, al_istinbat)

src/physics/al_aql/       → SimulationSpace, ADL compiler, entities, verbs,
                           templates, event chains, taxonomy, speech acts,
                           quantification, metaphor, exceptions

src/physics/advanced/     → Advanced physics demos
```

---

## 3. Test Pyramid & Coverage Goals

```
         ⬇️ Few
      / \
     /   \        Integration — 15% of tests (full pipeline:
    /     \                   prompt → generate!, cross-module)
   /       \
  /─────────\     Unit + Integration — 85%
 /           \
/─────────────\   Unit — 60% (single function, pure logic)
```

### Coverage Targets (by subsystem)

| Subsystem | Current | Target (Q3 2026) | Notes |
|-----------|---------|------------------|-------|
| Core physics (letter DB, gravity, Clifford) | ~50% | 85% | Algebraic identities critical |
| Generator (generate!, strategies) | ~40% | 75% | Edge cases (empty vocab, single word, etc.) |
| Al-Istinbat (attention memory) | ~70% | 90% | RelationFrame + QuantityFrame expansion |
| Al-Aql (simulation space, ADL) | ~30% | 70% | Error paths, malformed ADL |
| IRP (intent detection) | ~60% | 85% | More Arabic question patterns |
| Al-Hisban (semantic calculus) | ~40% | 75% | Clifford transforms, guidance paths |
| Sense superposition | ~60% | 85% | Arabic word ambiguity inventory |
| Self-review engine | ~30% | 70% | Multi-issue detection, repair strategies |
| Cerebellum (PID, policy) | ~50% | 80% | Learning from outcome, state persistence |
| Persistence (save/load) | ~70% | 90% | All memory types, corruption handling |

---

## 4. Subsystem Test Plans

### 4.1 Core Physics (`src/physics/core/`)

#### Units: LetterDB, GravityEngine, CliffordMath, ResonantChain, EntropyGate, RAMCore, PhaseReinforcement, DCCF, PPM, AMFS, DensityMatrix, CausalFlow, WeightResonance, SyntaxField, MorphoPhasic, SpectralWave, KuramotoOscillator, ChaosEntanglement, MirnanCerebellum

**Current coverage**: Constants + basic operations tested in `runtests.jl`

**Needed tests**:

```
□ LetterDB
  ✓ get_vector returns PHASE_DIM vector of ±1
  ✓ get_omega_0 returns positive frequency
  □ get_vector for all 28 Arabic letters (ب ت ث ج ح خ د ذ ر ز س ش ص ض ط ظ ع غ ف ق ك ل م ن ه و ي)
  □ get_vector for letters with tashkeel returns same vector
  □ omega_0 differs for emphatic vs non-emphatic (ص ≠ س, ض ≠ د, ط ≠ ت, ظ ≠ ذ)

□ GravityEngine
  ✓ gravitational_force returns correct dimensions
  ✓ gravitational_force magnitude increases with mass
  ✓ gravitational_force magnitude decreases with distance
  □ gravitational_force is symmetric (F_ij = -F_ji for opposite direction)
  □ semantic_potential returns scalar
  □ zero mass → zero force

□ CliffordMath
  ✓ from_vector returns Multivector22
  ✓ norm > 0 for non-zero multivector
  ✓ get_scalar_essence returns scalar part
  □ geometric_product is associative: (a*b)*c ≈ a*(b*c)
  □ geometric_product is distributive: a*(b+c) ≈ a*b + a*c
  □ reverse: rev(rev(a)) ≈ a
  □ clifford_distance: d(a,a) ≈ 0, d(a,b) > 0 for a ≠ b
  □ grade_involution preserves grade structure
  □ conjugation: conj(conj(a)) ≈ a

□ ResonantChain
  ✓ pair_freq returns positive frequency
  □ pair_freq increases with coupling strength
  □ simulate returns final phases + history with correct dimensions
  □ phase locking: identical oscillators synchronize over time

□ EntropyGate
  ✓ compute_S returns non-negative entropy
  □ S = 0 for pure state (single coherent vector)
  □ S > 0 for mixed state
  □ entropy decreases with reinforcement

□ RAMCore (AttractorMemory)
  ✓ observe! stores pattern, returns index
  ✓ resonate returns hits for similar input
  □ observe! with repeated input returns same index
  □ resonate with identical vector returns original pattern with score ≈ 1.0
  □ resonate with orthogonal vector returns empty or low score
  □ save/load round-trip preserves all centers

□ PhaseReinforcement
  ✓ reinforce! increases strength for a word
  ✓ get_strength returns positive
  □ reinforce! with negative reward decreases strength
  □ strength decays over time if not reinforced
  □ save/load round-trip

□ DCCF
  ✓ get_context_boost returns non-negative
  □ get_context_boost("علم", ["نور", "كتاب"]) > get_context_boost("علم", ["بحر", "سيارة"])
  □ context boost is symmetric within tolerance

□ PPM
  ✓ absorb! sets active
  ✓ score returns non-negative
  □ score for prompt word > score for unrelated word
  □ reset clears state

□ AMFS
  ✓ adapt_word returns haskey mass, freq, centrality
  □ mass and freq return same values from compute_word_mass/compute_word_frequency
  □ centrality is between 0 and 1
  □ adapt_word with empty context_words still returns valid result

□ DensityMatrix
  ✓ build! constructs rho
  ✓ get_trace ≈ 1.0
  ✓ resonance is in [0, 1]
  □ rho is Hermitian
  □ rho is positive semidefinite
  □ eigenvalues sum to 1.0
  □ von_neumann_entropy returns 0 for pure state, > 0 for mixed
  □ build! with single vector produces pure state (rank 1)

□ CausalFlow
  ✓ compute_flow returns flow_magnitude and logical_score
  □ flow magnitude increases with stronger causal edges
  □ circular causality (A→B→C→A) has lower logical_score than acyclic
  □ empty causal matrix returns zero flow

□ WeightResonance (Wazn)
  ✓ get_weight_pv returns 22-dim vector
  ✓ resonance between two weights returns [-1, 1]
  ✓ transition_score returns ≥ 0
  □ فعل → مفعول resonance > فعل → استفعال resonance (closer patterns)
  □ same weight returns resonance ≈ 1.0

□ SyntaxField
  ✓ compute_syntax_vector returns SYNTAX_DIMS vector for known word
  □ compute_syntax_vector returns same dimension for unknown word
  □ syntax vectors for same POS class are more similar than different POS

□ MorphoPhasic
  ✓ analyze_morpho returns root, weight, has_morph
  □ root extraction works for triliteral, quadriliteral roots
  □ root differs for derived forms (كتب → ك ت ب vs استكتب → ك ت ب)
  □ has_morph is false for unambiguous words

□ SpectralWave
  ✓ compute_word_wave returns omega_0, amplitude
  ✓ wave_at returns Float64
  ✓ wave_spectrum returns freqs, mags, phases
  □ amplitude is non-zero for all Arabic letters
  □ word wave amplitude correlates with letter count

□ KuramotoOscillator
  ✓ _derivative returns correct dimensions
  ✓ _derivative with large phases does amplitude clipping (norm ≤ 50)
  ✓ simulate returns final phases + history
  □ _derivative with zero coupling returns intrinsic frequency only
  □ _derivative with identical phases returns near-zero derivative
  □ simulate run_all_tests produces monotonic synchronization measure

□ ChaosEntanglement
  ✓ split_wave preserves norm (|v1|² + |v2|² = |v_test|²)
  ✓ split_wave produces anti-phase partner (v1 + v_anti ≈ 0)
  ✓ merge_waves constructive phase_shift=0 reinforces
  ✓ merge_waves destructive phase_shift=pi cancels
  ✓ couple_waves returns correct shape
  ✓ project_coupled_wave recovers original
  □ couple_waves(wave_a, wave_b) == couple_waves(wave_b, wave_a) (symmetry)
  □ merge_waves with random phase_shift has norm between 0 and |v1+v2|

□ MirnanCerebellum PID
  ✓ PIDController has correct defaults (Kp=0.6, setpoint=0.5)
  ✓ record_error! stores error, update_integral! accumulates
  ✓ anti-windup clamping (±10)
  ✓ correct! adjusts weights proportional to Kp × error
  ✓ correct! works with zero Ki, Kd
  □ correct! with Ki != 0 integrates error over time
  □ correct! with Kd != 0 responds to error derivative
  □ toggle pid_enabled=false disables correction
  □ full cartpole-like oscillation damping scenario
```

### 4.2 Generator (`src/physics/engines/Generator/`)

**Current coverage**: ~40%

**Needed tests**:

```
□ MirnanGenerator construction
  ✓ constructor from vocab dict
  ✓ constructor with K_sem, K_causal
  □ constructor with empty vocab returns valid generator
  □ constructor with single word vocab
  □ constructor with model_dir loads existing state
  ✓ test_question_type_matrix.jl — full pipeline with trained model

□ generate! core paths
  ✓ mode="standard" returns String
  ✓ mode="creative" returns String
  ✓ empty prompt returns ""
  ✓ unknown words still produce valid output
  ✓ max_words respected (approximately)
  □ mode="auto" chooses best mode automatically
  □ mode="standard" uses higher prompt alignment
  □ mode="creative" has higher diversity score
  □ temperature=0 produces deterministic output (same word sequence)
  □ temperature=high (>1.0) produces more random output

□ Generator strategies
  ✓ RelationFrameStrategy works with purpose question, returns nothing for yes/no
  ✓ RelationFrameStrategy returns nothing when memory empty
  ✓ MIRNAN_ENABLE_RELATION_FRAME_STRATEGY gate
  □ DirectAnswerStrategy for social replies
  □ SemanticRelationStrategy for "ما الذي يجعل..." questions
  □ CausalAnchorStrategy for "كيف..." mechanism questions
  □ strategy priority ordering when multiple match
  □ all strategies return nothing when none match

□ Scoring functions
  ✓ _score returns finite values
  ✓ context_tension: related > unrelated
  □ _score returns higher for prompt words than unrelated words
  □ _score with empty context_words still works
  □ _score correctly uses AQL inhibition when present
  □ _score correctly uses cerebellum policy weights
  □ AQL boost: guided_score > unguided_score

□ Quality guard (test_generation_quality_guard.jl base)
  ✓ _needs_simple_declarative_template detects off-topic Arabic
  ✓ _needs_simple_question_template detects off-topic Arabic
  ✓ _simple_text_template and _simple_question_template return "" when no template
  □ quality guard regenerates with better output
  □ quality guard cascades through multiple strategies
  □ quality guard with strict mode

□ Self-review (test_self_review.jl base)
  ✓ good generation > bad generation score
  ✓ repetition detection
  ✓ dominant_repetition repair target
  □ list-like generation detected
  □ diversity_score calculation
  □ review_generation! with empty prompt
  □ _refine_generation! actually improves output
  □ self-review with Arabic text

□ Dialogue handling
  ✓ salam greeting → "وعليكم"
  ✓ wellbeing question → "الحمد"
  ✓ _dialogue_answer_incompatible correctly classifies
  ✓ _trim_reported_speech_prefix
  □ dialogue memory loaded from knowledge/dialogue_facts.json
  □ non-dialogue questions don't get dialogue answers
  □ mixed Arabic-English dialogue

□ _aql_answer! paths
  ✓ social reply memory boundry (strict mode)
  ✓ strict mode = 1 bypasses template answers
  □ causal explanation path
  □ contradiction detection path
  □ semantic relation memory path
  □ definition answer (ال التعريف)
  □ procedural plan answer (التدبير)
  □ code generation path
  □ math solving path
```

### 4.3 Al-Istinbat (`src/physics/engines/al_istinbat.jl`)

**Current coverage**: ~70% (RelationFrame + QuantityFrame well tested)

**Needed tests**:

```
□ RelationFrame
  ✓ relation_type_for_marker: all 25 markers + 8 quantity markers
  ✓ extract_relation_frames: all 5 types (purpose, conditional, temporal, spatial, state)
  ✓ extract_relation_frames: multi-word markers (من أجل, متى ما, في حين)
  ✓ extract_relation_frames: confidence levels (0.90, 0.75, 0.60)
  ✓ extract_relation_frames: no markers returns empty
  ✓ extract_relation_frames: multiple markers in same text
  ✓ RelationFrame struct fields
  ✓ learn_relation_frames_from_text! stores in memory
  ✓ learn_istinbat_from_text! includes new types
  ✓ learn_relation_frames_from_text! no interference with old causal records
  ✓ select_relation_frame_attention: selects purpose, temporal, conditional
  ✓ select_relation_frame_attention: ignores causal records
  ✓ select_relation_frame_attention: empty memory returns nothing
  ✓ relation_frame_diagnostic: shows frame info with "يوجد" / "لا يوجد"
  ✓ purpose_answer: matches purpose question → formatted answer
  ✓ purpose_answer: ignored for yes/no (هل)
  ✓ purpose_answer: ignored for causal-only memory
  ✓ purpose_answer: ignored when topic mismatch
  ✓ purpose_answer: ignored with non-purpose prompt
  ✓ purpose_answer: empty memory returns empty
  ✓ compare_purpose_strategies: returns PurposeComparisonRecord
  ✓ compare_purpose_strategies: empty memory
  ✓ compare_purpose_strategies: non-purpose question
  ✓ compare_purpose_strategies: yesno question ignored
  ✓ compare_scene_purpose_strategies: cooperative, scene_only, purpose_only, none
  ✓ RelationFrameStrategy try_generate: purpose match, yes/no skip, topic mismatch, empty memory
  ✓ RelationFrameStrategy integrated via MIRNAN_ENABLE_RELATION_FRAME_STRATEGY

  □ extract_relation_frames with overlapping markers (e.g., "لكي" inside "من أجل")
  □ extract_relation_frames with punctuation variants (؟ , . !)
  □ extract_relation_frames with negated markers (ليس لكي, لا من أجل)
  □ select_relation_frame_attention with multiple candidates of different types
  □ purpose_answer with multiple purpose frames (chooses highest confidence)
  □ purpose_answer with confidence threshold below 0.30
  □ purpose_answer with overlap threshold below 0.15
  □ purpose_answer with exact overlap = 0 (should still match via attention_weight)
  □ compare_purpose_strategies with generate_func that returns empty
  □ relation_frame_diagnostic with multiple frame types

□ QuantityFrame
  ✓ relation_type_for_quantity_marker: count (2), measure (2), comparison (5), quantifier_scope (2), vague_quantity (2)
  ✓ extract_quantity_frames: count, measure, comparison, quantifier_scope, vague_quantity
  ✓ extract_quantity_frames: multiple markers in same text
  ✓ extract_quantity_frames: no markers returns empty
  ✓ QuantityFrame struct fields

  □ extract_quantity_frames with numbers (ثلاثة, ٥, 42)
  □ extract_quantity_frames with comparative structure (أكثر من, أقل من, مثل)
  □ extract_quantity_frames with quantifier scope negation (ليس كل, ليس بعض)
  □ extract_quantity_frames from question context (كم سؤال)
  □ QuantityFrame store/load (Phase 2)
  □ QuantityFrame integration with attention memory (Phase 2)
  □ QuantityFrame answer generation (Phase 3)
```

### 4.4 Intent Response Planner (IRP)

**Current coverage**: ~60%

**Needed tests**:

```
□ detect_response_intent
  ✓ mechanism (كيف يتعلم, كيف يعمل)
  ✓ descriptive (كيف هو الطقس)
  ✓ conditional (إذا زاد العلم)
  ✓ causal (لماذا يزيد العلم)
  ✓ opinion (ما رأيك, هل العلم مفيد)
  ✓ dialogue (السلام عليكم, كيف حالك)
  □ process (كيف يتم تصنيع)
  □ definition (ما معنى, ما هو)
  □ comparison (ما الفرق بين)
  □ purpose (لماذا, لأي غاية)
  □ quantity (كم, كم عدد, ما مقدار)
  □ yes/no (هل)
  □ compound intents (e.g., why + how combined)
  □ detect_response_intent with empty prompt
  □ detect_response_intent with single word

□ render_planned_response
  ✓ causal returns ""
  ✓ conditional with cause/result
  ✓ opinion with "أرى أن" + related_terms
  □ mechanism with steps
  □ process with roles
  □ definition with structured output
  □ comparison with both sides
  □ purpose with غاية clause
  □ quantity with numeric output
  □ render_planned_response without related_terms
  □ render_planned_response with empty terms

□ intent_gravity_profile
  ✓ has_gravity_profile for causal, dialogue
  ✓ guidance_terms extracted from subject/action
  ✓ repulsion_terms includes "بخير" for non-social
  ✓ syntax_multiplier > 1.0
  ✓ question_charge < 0 for dialogue, > 0 for questions
  ✓ response_charge > 0 for dialogue
  □ all intents produce valid gravity profile
  □ gravity profile weights are consistent
```

### 4.5 Al-Aql — Cognitive Causal Engine (`src/physics/al_aql/`)

**Current coverage**: ~30%

**Needed tests**:

```
□ SimulationSpace
  ✓ entity registration
  ✓ verb registration
  ✓ interaction with predefined actions (roar!, shake!, etc.)
  ✓ property get/set
  ✓ attribute get/set
  □ entity registration rejects duplicates
  □ entity with empty name
  □ verb with no target
  □ interaction with unregistered entity
  □ custom verb definition at runtime

□ ADL Compiler
  ✓ entity definitions with properties
  ✓ verb definitions with targets
  ✓ rules (base: condition → action)
  ✓ idea (فكرة: subject verb object)
  ✓ taxonomy with inheritance (صياد < إنسان)
  ✓ classification (تصنيف خالد : صياد)
  ✓ template (نمط) with domain, condition, result
  ✓ event chain (سلسلة) with steps
  ✓ quantification (كم كل/بعض)
  ✓ comparison (مقارنة)
  ✓ metaphor
  ✓ speech act
  ✓ English ADL syntax
  ✓ negation and opposites
  ✓ circumstances and time order
  ✓ static semantic relations
  □ ADL compilation error: malformed entity
  □ ADL compilation error: unknown class
  □ ADL compilation error: circular inheritance
  □ ADL compilation error: template without result
  □ ADL compilation error: duplicate definition
  □ ADL with mixed Arabic/English
  □ ADL with Unicode normalization variants
  □ ADL with empty block

□ train_from_text!
  ✓ extracts entities, implicit relations, attributes
  ✓ causal frame extraction
  ✓ implicit "be" inference (عصفور على الشجرة, باب البيت, بيت جميل, الماء ساخن)
  □ train_from_text! with full tashkeel
  □ train_from_text! with dialectal Arabic
  □ train_from_text! with multiple sentences
  □ train_from_text! with negated sentences
  □ train_from_text! from knowledge files

□ Causal Templates
  ✓ cross-domain (physics, electricity, dialogue)
  ✓ conditional templates with conditions
  ✓ template result action application
  ✓ compound sequential events
  □ template with multiple conditions
  □ template with 0 confidence
  □ template matching without condition
  □ template matching with numeric ranges

□ Speech Acts
  ✓ greeting detection
  ✓ wellbeing question detection
  ✓ response matching
  ✓ fuzzy matching with similarity score
  ✓ exact mismatch yields 0.0 score
  □ speech act from ADL
  □ multiple speech act responses (choose highest confidence)
  □ save/load speech acts
  □ noisy text without dialogue
```

### 4.6 Semantic Memory Modules (Al-Lisan, Al-Ta3rif, Al-Nisba, Al-Hisban, etc.)

**Current coverage**: ~40-70%

**Needed tests**:

```
□ Al-Ta3rif (definitions)
  ✓ Arabic definitions (ما هو, ما معنى)
  ✓ English definitions (what is)
  ✓ save/load round-trip
  □ definition with multiple sentences
  □ definition with synonym in answer
  □ definition without "هو/هي" pattern
  □ definition answer excludes "في فضاء العقل" and "__classes"
  □ learn_ta3rif_from_text! with no definition pattern

□ Al-Nisba (semantic relations)
  ✓ analogy detection (يشبه, كـ)
  ✓ transform detection (يتحول)
  ✓ causal / prevention detection
  ✓ difference detection
  ✓ save/load round-trip
  □ relation with multiple concept pairs
  □ relation with partial overlap
  □ relation guidance terms for all relation types
  □ learn_nisba_from_text! with single sentence
  □ has_nisba_relations on empty memory

□ Al-Hisban al-Dalali (semantic calculus)
  ✓ sentence_semantic_signature
  ✓ semantic_transform_signature
  ✓ learn_semantic_calculus_from_pair!
  ✓ select_semantic_transform
  ✓ semantic_guidance_terms
  ✓ semantic_guidance with full response plan
  ✓ semantic_answer_plan
  ✓ learn_semantic_calculus_from_text! (multi-sentence)
  ✓ save/load round-trip
  □ semantic guidance with confidence < threshold
  □ semantic guidance for unrelated query
  □ semantic guidance on empty memory
  □ sentence signature consistency (same sentence → same signature)
  □ transform signature with identical sentences

□ Semantic Scene Memory (extended Al-Hisban)
  ✓ extract_semantic_scene with actor/action/patient
  ✓ extract_semantic_scene with instrument, place, time, state
  ✓ learn_semantic_scene_from_text!
  ✓ semantic_scene_comparison_diagnostic
  ✓ save/load round-trip
  □ scene with 3+ actors
  □ scene with nested actions
  □ scene from Arabic text
  □ scene diagnostic with unmatched scene

□ Al-Muradif (synonyms)
  ✓ build_muradif_memory from K_sem, K_syn, K_causal
  □ synonyms detected from shared context
  □ non-synonyms correctly excluded
  □ confidence threshold filtering
  □ muradif from training text

□ Al-Lisan (linguistic patterns)
  ✓ custom markers (حينئذما)
  ✓ categorized markers (ريثما)
  ✓ nominal sentence patterns
  □ verbal sentence patterns
  □ mixed nominal/verbal
  □ pattern selection with prefer_verbal toggle

□ Al-Tadbir (procedural plans)
  ✓ plan learning from "->" delimited text
  ✓ select_tadbir_pattern with domain extraction
  ✓ render_tadbir_plan returns steps
  ✓ save/load round-trip
  □ plan with missing arrows
  □ plan with single step
  □ plan with repetition
  □ plan generation integration

□ Al-Hisab (math solving)
  ✓ binary arithmetic
  ✓ Arabic digit arithmetic
  ✓ square area extraction
  ✓ square root
  □ equation solving (linear)
  □ unknown problem type
  □ verified=false for unchecked results
  □ hisab from extractable text

□ Al-Code (code patterns)
  ✓ Python function
  ✓ Julia function
  ✓ Julia module
  ✓ Julia struct
  ✓ Julia testset
  □ code generation from pattern
  □ save/load round-trip
  □ mixed code-language patterns

□ Al-Intibah (semantic attention field)
  ✓ build_semantic_attention from K, vocab, prompt
  ✓ has_semantic_attention returns true/false
  ✓ anchors extracted from content words (not question words)
  ✓ attention_bias_terms from field
  □ empty attention for prompt with no content words
  □ attention field respects K matrix weights
  □ attention field with disconnected vocab

□ Pattern Memory Integration (across modules)
  ✓ all memories saved to same directory
  ✓ all memories loaded back from directory
  ✓ MirnanGenerator loads all memories at construction
  ✓ combined generation uses multi-memory state
  □ partial save (some memories missing) loads gracefully
  □ corrupt save file handled gracefully
  □ version mismatch detection
```

### 4.7 Sense Superposition

**Needed tests**:

```
✓ has_sense_inventory for known ambiguous words (عين, جذر)
✓ measure_senses without context returns mixed state (not collapsed)
✓ measure_senses with context collapses to correct sense
✓ measure_senses with multiple contexts
✓ explain_measurement shows collapse and selected sense

□ has_sense_inventory for unambiguous words returns false
□ has_sense_inventory for unknown word returns false
□ measure_senses with empty inventory word
□ measure_senses with equally balanced contexts produces low confidence
□ measure_senses with conflicting contexts produces mixed state
□ sense inventory load from file
□ Arabic-only sense inventory coverage (at least 30 ambiguous words)
```

### 4.8 Self-Review Engine

**Current coverage**: ~30%

**Needed tests**:

```
✓ review_generation! scores good > bad
✓ repetition detection
✓ dominant_repetition issue
✓ list-like output detection
□ score thresholds for acceptance/rejection
□ all issue types: repetition, incoherence, off_topic, hallucination, template_leak
□ repair_target mapping
□ review_generation! with empty output
□ review_generation! with very short output (1-2 words)
□ review_generation! with Arabic text and phase vectors
□ _refine_generation! actually changes output
□ self-review enabled/disabled toggle
```

### 4.9 Contextual Learning

**Current coverage**: ~30%

**Needed tests**:

```
✓ detect_question_intent with action_query
✓ detect_compounds from text
✓ learn_from_feedback! stores feedback
✓ learn_from_feedback! affects subsequent generation
□ feedback with negative rating
□ feedback with empty note
□ learn_from_feedback! with learn_from_feedback! ... × multiple rounds
□ feedback persistence across generator restarts
□ compound detection with overlapping compounds
```

### 4.10 Persistence (save/load)

**Current coverage**: ~70%

**Needed tests**:

```
✓ all memory types save/load round-trip
✓ MirnanGenerator with model_dir loads all memories

All modules must have:
□ save preserves all fields, load restores all fields
□ save to non-existent directory creates it
□ load from non-existent file returns empty/graceful
□ load from corrupt file returns empty/graceful
□ save/load with version field for migration
□ concurrent save/load thread safety
```

---

## 5. Wave-Physics-Specific Tests

These tests validate Mirnan's unique physical semantics — they do not exist in conventional LLM test suites.

### 5.1 Interference Tests

```
□ INTERFERENCE-1: Constructive interference
  - Two words with similar phase vectors ("علم" + "معرفة")
  - merge_waves with phase_shift=0
  - EXPECTED: |merged| ≈ |v1| + |v2| (super-linear)

□ INTERFERENCE-2: Destructive interference
  - Antonym pair ("نور" + "ظلام")
  - merge_waves with phase_shift=π
  - EXPECTED: |merged| < max(|v1|, |v2|)

□ INTERFERENCE-3: Partial interference
  - Words with moderate similarity ("علم" + "فهم")
  - merge_waves with phase_shift=π/2
  - EXPECTED: 0 < |merged| < |v1| + |v2|

□ INTERFERENCE-4: Multi-wave superposition
  - 3+ words in a sentence context
  - EXPECTED: Born probability |Ψ_total|² selects context-appropriate word
```

### 5.2 Gravitational Tests

```
□ GRAVITY-1: Semantic mass ordering
  - compute_word_mass for longer words > shorter words (more letters → higher mass)
  - EXPECTED: mass("كتاب") > mass("باب") > mass("ب")

□ GRAVITY-2: Gravitational force ordering
  - F("قراءة","كتاب") > F("قراءة","سيارة")
  - EXPECTED: related pairs have higher gravitational attraction

□ GRAVITY-3: Gravitational invariance
  - F(a,b) ≈ F(b,a) (bidirectional, within floating point)
  - EXPECTED: gravitational force is symmetric

□ GRAVITY-4: Distance decay
  - As phase similarity decreases, force decreases
  - EXPECTED: monotonic relationship
```

### 5.3 Born Rule Selection Tests

```
□ BORN-1: Probability normalization
  - For any context, sum of P(word_i|context) over vocab ≈ 1.0
  - EXPECTED: total probability ≈ 1.0 (within 5%)

□ BORN-2: Zero probability for impossible words
  - In highly constrained context, unrelated words have P ≈ 0
  - EXPECTED: P("سيارة" | ["العلم", "نور"]) < P("فهم" | ["العلم", "نور"])

□ BORN-3: Temperature scaling
  - Temperature=0: P(max) → 1 (deterministic)
  - Temperature=∞: uniform distribution
```

### 5.4 Phase Coherence Tests

```
□ PHASE-1: Phase similarity consistency
  - phase_similarity(word, word) ≈ 1.0
  - phase_similarity(word, unrelated) < phase_similarity(word, synonym)

□ PHASE-2: Phase vector normalization
  - norm(compute_word_phase_vector(w)) ≈ 1.0 for all words

□ PHASE-3: Clifford algebraic identities
  - (a * b) * c ≈ a * (b * c)  [associativity]
  - a * (b + c) ≈ a * b + a * c  [distributivity]
  - a * b ≈ -b * a for orthogonal vectors

□ PHASE-4: Weight resonance geometry
  - resonance("فَعَلَ", "فَعَلَ") ≈ 1.0
  - resonance("فَعَلَ", "مَفْعُول") > resonance("فَعَلَ", "اِسْتِفْعَال")
  - transition_score indicates morphological derivation direction
```

### 5.5 Entropy and Density Matrix Tests

```
□ ENTROPY-1: Pure state
  - Single word context → density matrix rank 1
  - von Neumann entropy S ≈ 0

□ ENTROPY-2: Mixed state
  - Multiple conflicting contexts → higher entropy
  - EXPECTED: S("عين" without context) > S("عين" with context "شرب ماء")

□ ENTROPY-3: Collapse on measurement
  - After "measure" sense:"عين" + "ماء" → collapsed to "spring"
  - EXPECTED: entropy decreases after collapse

□ DENSITY-1: Matrix properties
  - rho is Hermitian: rho ≈ rho†
  - rho is positive semidefinite: all eigenvalues ≥ 0
  - Tr(rho) ≈ 1.0

□ DENSITY-2: Resonance probability
  - resonance(rho, v) is in [0, 1]
  - resonance(rho, eigenvector_1) ≥ resonance(rho, eigenvector_n)
```

### 5.6 Kuramoto Synchronization Tests

```
□ KURAMOTO-1: Phase convergence
  - Start with random phases, simulate with strong coupling
  - Phase variance decreases over time steps

□ KURAMOTO-2: Frequency locking
  - Identical intrinsic frequencies → complete synchronization
  - EXPECTED: final phase differences → 0

□ KURAMOTO-3: Damping effect
  - Higher damping → slower synchronization
  - EXPECTED: convergence rate inversely proportional to damping_coeff

□ KURAMOTO-4: Amplitude clipping
  - _derivative with extreme phases produces bounded output (≤ 50)
```

### 5.7 Holographic Retrieval Tests

```
□ HOLO-1: Holographic storage
  - couple_waves("علم", "نور") stores relational information
  - project_coupled_wave(E, "نور") recovers ≈ "علم"

□ HOLO-2: Content-addressable retrieval
  - resonate("علم") returns "نور" if stored together
  - EXPECTED: nearest neighbor retrieval by phase similarity

□ HOLO-3: Distortion tolerance
  - resonate with noisy "علم" still returns stored associate
  - EXPECTED: graceful degradation with increasing noise
```

### 5.8 Causal Flow Tests

```
□ CAUSAL_FLOW-1: Linear causality
  - A → B → C (chain)
  - EXPECTED: flow_magnitude > 0, logical_score ≈ 1

□ CAUSAL_FLOW-2: Circular causality
  - A → B → C → A
  - EXPECTED: logical_score < chain score (lower logical consistency)

□ CAUSAL_FLOW-3: Empty causality
  - No causal edges
  - EXPECTED: flow_magnitude ≈ 0
```

---

## 6. Integration Test Design

### 6.1 Test Scenarios (End-to-End)

```
□ SCENARIO-1: "العلم نور" → generate!
  1. Create MirnanGenerator with small Arabic vocab
  2. Call generate!("العلم نور"; mode="standard")
  3. Verify output contains context-related words
  4. Verify output is not empty, not repetitive

□ SCENARIO-2: Purpose question → purpose_answer → generate!
  1. Learn "يدرس الطالب لكي ينجح" in IstinbatAttentionMemory
  2. Call purpose_answer("لماذا يدرس الطالب؟")
  3. Call generate!("لماذا يدرس الطالب؟") with MIRNAN_ENABLE_RELATION_FRAME_STRATEGY=1
  4. Verify output contains "الغاية" or "ينجح"

□ SCENARIO-3: Semantic calculus → guidance → generation
  1. Learn "How does knowledge grow? → Knowledge grows through practice."
  2. Call semantic_guidance("How does knowledge grow?")
  3. Verify guidance terms include "practice"
  4. Verify answer plan has correct frame

□ SCENARIO-4: Full dialogue turn
  1. User: "السلام عليكم"
  2. detect_response_intent → dialogue
  3. _aql_answer! → "وعليكم السلام ورحمة الله"
  4. User: "ما معنى العدل؟"
  5. detect_response_intent → opinion/causal
  6. Al-Ta3rif answer or generate! output

□ SCENARIO-5: Al-Aql train → infer → generate
  1. train_from_text!("إذا زادت الحرارة تمدد الحديد.")
  2. infer_event!("الحديد", "تسخين", "حرارة")
  3. generate!("ماذا يحدث إذا سخنت الحديد؟")
  4. Verify output mentions "تمدد"

□ SCENARIO-6: Strict mode cascade
  1. Set MIRNAN_STRICT_NO_TEMPLATES=1
  2. Call generate!("هل العلم نور؟")
  3. Verify it doesn't use template answers
  4. Verify it falls through to physics-based generation

□ SCENARIO-7: Persian Gulf naming — context-sensitive fact
  1. Context: "الخليج الفارسي" or "الخليج العربي"
  2. Verify output uses user's preferred naming
  3. Verify no template override for geo-sensitive topics

□ SCENARIO-8: Arabic-Only constraint
  1. Call generate!("Good morning" — English prompt)
  2. Verify output avoids English words where possible
  3. Verify Arabic responses for Arabic prompts
```

### 6.2 Cross-Module Contract Tests

```
□ Contract: extract_relation_frames → learn_relation_frames_from_text! → select_relation_frame_attention → purpose_answer
  - Verify data flows correctly through the pipeline
  - Verify no data loss at each step

□ Contract: Al-Hisban → Scene memory → IRP → Generator
  - Semantic scene extracted, learned, diagnosed
  - Intent detected from prompt
  - Scene guidance influences generated output

□ Contract: Al-Aql (ADL compile) → simulation → inference → Al-Istinbat
  - ADL facts feed causal simulation
  - Simulation results stored in attention memory
  - Attention memory influences generation

□ Contract: Persistence save → load → generation consistency
  - Save all memory states
  - Load into fresh generator
  - Same prompt produces similar (not identical) output
```

---

## 7. Evaluation Harness

### 7.1 Automated Scoring Metrics

| Metric | What it measures | Implementation |
|--------|-----------------|----------------|
| **Phase coherence** | Generated words cluster near prompt | Mean phase_similarity(output_words, prompt_words) |
| **Semantic diversity** | Output not repetitive | Unique bigrams / total bigrams |
| **Prompt alignment** | Output relates to input question | max phase_similarity(question_terms, output_terms) |
| **Arabic purity** | Language consistency | Ratio of Arabic-detected characters |
| **Template leakage** | Raw template text in output | Regex detection of template markers |
| **Entropy balance** | Output not too predictable nor random | Shannon entropy of word distribution |
| **Repetition score** | Word/ngram repetition | self-review repetition_score |
| **Fluency** | Grammatical coherence | (Statistical: mean phase_distance between consecutive words) |
| **Answer correctness** | Factual alignment | For known questions, check expected terms appear |
| **Latency** | Generation speed | Time per generation call |

### 7.2 Evaluation Scenarios

```
□ EVAL-1: Generation quality benchmark
  - Suite: 50 Arabic prompts across 10 question categories
  - Metrics: prompt_alignment, diversity, Arabic_purity, template_leakage
  - Reference: previous version comparison

□ EVAL-2: Purpose question benchmark
  - Suite: 10 purpose questions ("لماذا يدرس الطالب؟")
  - Metrics: purpose_answer match rate, غاية content, generation_quality
  - Reference: purpose_answer vs generate! baseline

□ EVAL-3: Relation extraction benchmark
  - Suite: 25 Arabic sentences with relation markers
  - Metrics: extraction_rate, correct_type_classification, confidence_calibration
  - Reference: human-annotated gold standard

□ EVAL-4: Causal reasoning benchmark
  - Suite: 15 causal scenarios (physics, social, abstract)
  - Metrics: correct causal direction, relevant entity use, explanation coherence

□ EVAL-5: Arabic diacritics preservation
  - Suite: 20 fully tashkeel-marked texts
  - Metrics: character-level diacritic preservation rate

□ EVAL-6: Memory persistence benchmark
  - Suite: Learn 5 facts, save, load fresh generator
  - Metrics: fact recall rate pre- vs post-persistence
```

### 7.3 Harness CLI

```
# Run all evaluation scenarios
julia eval_harness.jl --all

# Run specific benchmark
julia eval_harness.jl --benchmark=purpose

# Compare two model versions
julia eval_harness.jl --compare=v1.0.0,v1.1.0

# Generate evaluation report
julia eval_harness.jl --report=eval_results.json
```

---

## 8. Test Infrastructure

### 8.1 Running Tests

```powershell
# Single test file
julia --project=models/mirnan models/mirnan/test/test_relation_frame.jl

# All Mirnan tests
julia --project=models/mirnan models/mirnan/test/run_all_tests.jl

# Project-level tests (Majnon infrastructure)
julia test/runtests.jl
```

### 8.2 Test Patterns (Conventions)

1. **File naming**: `test_<module>.jl`
2. **Import pattern**:
   ```julia
   include("../src/MirnanNew.jl")
   using .MirnanNew
   using Test
   const Physics = MirnanNew.Physics
   ```
3. **Test structure**: `@testset "description" begin ... end`
4. **Assertions**: `@test`, `@test ≈`, `@test !isempty`, `@test occursin`, `@test any`
5. **Temporary files**: Use `mktempdir()` for save/load tests
6. **Environment variables**: Save/restore around env-dependent tests:
   ```julia
   old = get(ENV, "VAR", nothing)
   try
       ENV["VAR"] = "1"
       # ... test ...
   finally
       old === nothing ? delete!(ENV, "VAR") : ENV["VAR"] = old
   end
   ```

### 8.3 CI Integration

```yaml
# .github/workflows/mirnan_tests.yml (proposed)
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: julia-actions/setup-julia@v2
        with:
          version: '1.12'
      - run: julia --project=models/mirnan -e 'using Pkg; Pkg.instantiate()'
      - run: julia --project=models/mirnan models/mirnan/test/run_all_tests.jl
```

### 8.4 Mutation Testing (Future)

- Introduce controlled mutations to source code
- Verify tests detect the mutation
- Target: 70% mutation coverage for core physics

---

## 9. Priority Roadmap

### Phase A (Immediate — Q3 2026)
```
1. Complete QuantityFrame Phase 2 (store/load in memory)
2. Fill missing unit tests for core physics (marked □ above)
3. Expand Arabic text coverage in existing tests
4. Add edge case tests (empty vocab, single word, zero matrices)
```

### Phase B (Short-term — Q4 2026)
```
5. Build integrated Scenario tests (Section 6.1)
6. Implement wave-physics-specific tests (Section 5)
7. Add evaluation harness (Section 7)
8. Cross-module contract tests (Section 6.2)
```

### Phase C (Medium-term — Q1 2027)
```
9. Mutation testing infrastructure
10. CI pipeline with GitHub Actions
11. Performance benchmarks (latency, memory usage)
12. Arabic benchmark suite (50+ question types)
```

---

## 10. Appendices: Test Templates

### A. Test File Template

```julia
include("../src/MirnanNew.jl")
using .MirnanNew
using Test

const Physics = MirnanNew.Physics

@testset "<module>: <feature>" begin
    # Setup
    mem = Physics.<MemoryType>()
    
    # Exercise
    result = Physics.<function>(mem, <args>)
    
    # Verify
    @test result isa <ExpectedType>
    @test <property> ≈ <expected_value>
    
    # Edge case
    @test Physics.<function>(mem, "") isa <ExpectedType>  # empty input
end
```

### B. Benchmark Template

```julia
include("../src/MirnanNew.jl")
using .MirnanNew
using Test
using Statistics

const Physics = MirnanNew.Physics

function benchmark_generation(prompts::Vector{String}; trials=3)
    gen = Physics.MirnanGenerator(<vocab>)
    times = Float64[]
    outputs = String[]
    
    for prompt in prompts
        for _ in 1:trials
            t = @elapsed result = Physics.generate!(gen, prompt; max_words=16)
            push!(times, t)
            push!(outputs, result)
        end
    end
    
    return (
        mean_latency = mean(times),
        std_latency = std(times),
        mean_length = mean(length.(outputs)),
        outputs = outputs,
    )
end
```

### C. Env-Safe Test Pattern

```julia
@testset "environment-dependent feature" begin
    old_val = get(ENV, "MIRNAN_FEATURE_FLAG", nothing)
    try
        ENV["MIRNAN_FEATURE_FLAG"] = "1"
        # ... test with feature enabled ...
    finally
        if old_val === nothing
            delete!(ENV, "MIRNAN_FEATURE_FLAG")
        else
            ENV["MIRNAN_FEATURE_FLAG"] = old_val
        end
    end
end
```

### D. Round-Trip Persistence Template

```julia
@testset "persistence round-trip" begin
    dir = mktempdir()
    mem = Physics.<MemoryType>()
    Physics.<train!>(mem, <training_data>)
    
    path = Physics.<save>(mem, joinpath(dir, "data.json"))
    @test isfile(path)
    
    loaded = Physics.<load>(path)
    @test <property>(loaded) == <property>(mem)
    
    # Verify loaded memory still works
    result = Physics.<function>(loaded, <test_input>)
    @test <property>(result) ≈ <expected>
end
```

---

*Document version: 1.0 — Last updated: July 2026*
*Corresponding codebase: Mirnan V8 — Wave-Physics Arabic LLM*
*Author: Generated from comprehensive codebase audit*
