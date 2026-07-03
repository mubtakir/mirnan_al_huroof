# مرجع API — Mirnan.jl

## الوحدة الرئيسية: `Mirnan`

### الدوال المُصدّرة (Exported Functions)

---

### `MirnanGenerator`

```julia
MirnanGenerator(vocab::Dict{String,Int}, K_sem::AbstractMatrix;
                syntax_field=nothing, K_syn=nothing, K_dialogue=nothing,
                contextual_spectra=nothing, config::Dict=Dict{String,Any}()) -> MirnanGenerator
```

بناء محرك التوليد بالمحركات الفيزيائية الـ17.

**المدخلات:**
- `vocab`: قاموس (كلمة → id)
- `K_sem`: مصفوفة الاقتران الدلالي (SparseMatrixCSC)
- `syntax_field`: حقل النحو (اختياري)
- `K_syn`, `K_dialogue`: مصفوفات اقتران إضافية (اختياري)
- `config`: إعدادات إضافية (يُحمّل منها beam_width, top_k, beta, scoring_weights)

**المخرجات:** `MirnanGenerator`

**مثال:**
```julia
vocab = Dict("علم" => 1, "نور" => 2, "حياة" => 3)
K_sem = spzeros(3, 3)
K_sem[1, 2] = K_sem[2, 1] = 0.5

gen = MirnanGenerator(vocab, K_sem)
```

---

### `generate!`

```julia
generate!(gen::MirnanGenerator, prompt::String;
          mode::String="standard",
          max_words::Int=15,
          kwargs...) -> String
```

توليد نص من أمر prompt.

**المدخلات:**
- `gen`: محرك التوليد
- `prompt`: النص المُدخل (أمر)
- `mode`: نمط التوليد — `"standard"`, `"creative"`, `"dialogue"`, `"prnn"`, `"wave_coupling"`
- `max_words`: أقصى عدد كلمات في المخرج

**المخرجات:** النص المُولّد (سلسلة نصية مفصولة بمسافات)

**أمثلة:**
```julia
# توليد عادي
result = generate!(gen, "العلم نور")

# توليد إبداعي
result = generate!(gen, "في قديم الزمان"; mode="creative", max_words=20)

# توليد حواري
result = generate!(gen, "ما هو العلم؟"; mode="dialogue")
```

---

### `get_physics_report`

```julia
get_physics_report(gen::MirnanGenerator, context_words::Vector{String}) -> Dict
```

تقرير فيزيائي عن الحالة الراهنة للمحرك.

**المخرجات (Dict بالمفاتيح):**
| المفتاح | النوع | الوصف |
|---------|-------|-------|
| `mode` | String | نمط التوليد الحالي |
| `entropy` | Float64 | إنتروبيا شانون (0 = استقرار تام) |
| `S_crit` | Float64 | عتبة الانهيار الحرجة |
| `beta` | Float64 | ثابت بولتزمان العكسي |
| `temperature` | Float64 | درجة الحرارة (1/β) |
| `word_count` | Int | عدد الكلمات في السياق |
| `mass_mean` | Float64 | متوسط كتلة الكلمات |
| `mass_std` | Float64 | انحراف معياري للكتل |
| `vocab_size` | Int | حجم المعجم |
| `dccf_active` | Bool | هل الاقتران الديناميكي نشط؟ |
| `ppm_active` | Bool | هل حقل الأمر نشط؟ |
| `kb_facts` | Int | عدد الحقائق في القاعدة المعرفية |
| `ram_size` | Int | عدد الجواذب في الذاكرة |
| `reinforcement_traces` | Int | عدد مسارات التعزيز النشطة |

**مثال:**
```julia
report = get_physics_report(gen, ["علم", "نور"])
println("الإنتروبيا: ", report["entropy"])
```

---

### `compute_word_phase_vector`

```julia
compute_word_phase_vector(word::String; widen::Float64=1.0) -> Vector{Float32}
```

حساب المتجه الطوري للكلمة (9958 بُعداً).

**المدخلات:**
- `word`: الكلمة المراد تمثيلها
- `widen`: عامل توسيع التباين (> 1.0 = تباين أعلى)

**المخرجات:** `Vector{Float32}` بطول `PHASE_DIM` (9958)

**الآلية:** تطبيع → استخراج متجهات الحروف → إزاحة موضعية → جمع → تطبيع.

**مثال:**
```julia
pv = compute_word_phase_vector("سلام")           # 9958D Float32
pv_wide = compute_word_phase_vector("سلام"; widen=1.5)  # تباين أعلى
```

---

### `compute_word_mass`

```julia
compute_word_mass(word::String) -> Float64
```

كتلة الكلمة الدلالية: m = E / c² = h·f / c²

**مثال:**
```julia
m1 = compute_word_mass("الله")    # كتلة كبيرة
m2 = compute_word_mass("قلم")     # كتلة أصغر
@assert m1 > m2
```

---

### `compute_word_frequency`

```julia
compute_word_frequency(word::String) -> Float64
```

التردد الكلي للكلمة = مجموع ترددات حروفها الذاتية ω₀.

**مثال:**
```julia
f = compute_word_frequency("علم")
```

---

### `phase_similarity`

```julia
phase_similarity(v1::AbstractVector, v2::AbstractVector) -> Float64
```

التشابه الطوري (جيب التمام) بين متجهين. [-1.0, 1.0].

**مثال:**
```julia
pv1 = compute_word_phase_vector("علم")
pv2 = compute_word_phase_vector("نور")
sim = phase_similarity(pv1, pv2)  # e.g. 0.15
```

---

### `compute_extended_phase_vector`

```julia
compute_extended_phase_vector(word::String; widen::Float64=1.0) -> Vector{Float32}
```

المتجه الطوري الموسع (10000 بُعداً = 9958 + 8 + 6 + 6 + 16 + 6).

**مثال:**
```julia
epv = compute_extended_phase_vector("سلام")
# epv[1:9958]    = طور أساسي
# epv[9959:9966] = أبعاد الجذر (8)
# epv[9967:9972] = أبعاد DDE (6)
# epv[9973:9978] = أبعاد نحو (6)
# epv[9979:9994] = أبعاد دلالية (16)
# epv[9995:10000]= أبعاد قصدية (6)
```

---

### `compute_syntax_vector`

```julia
compute_syntax_vector(word::String) -> Vector{Float64}
```

المتجه النحوي 6D للكلمة. يُرجع مرتكز `noun` افتراضياً.

**المرتكزات الستة:** `verb`, `noun`, `prep`, `part`, `conj`, `kana`

**مثال:**
```julia
sv = compute_syntax_vector("علم")
```

---

### `reset!`

```julia
reset!(gen::MirnanGenerator)
```

مسح ذاكرة التخزين المؤقتة (المتجهات، الكتل) وإعادة تعيين حقل الأمر وبوابة الإنتروبيا.

---

## الوحدة: `Mirnan.Physics.Constants`

```julia
using Mirnan.Physics.Constants
```

| الثابت | القيمة | النوع |
|--------|--------|-------|
| `PLANCK_H` | 1.0 | Float64 |
| `LIGHT_SPEED_C` | 1.0 | Float64 |
| `GRAVITY_G` | 1.0 | Float64 |
| `BOLTZMANN_KB` | 0.1 | Float64 |
| `PHASE_DIM` | 9958 | Int |
| `ROOT_DIMS` | 8 | Int |
| `EXTRA_DIMS` | 6 | Int |
| `SYNTAX_DIMS` | 6 | Int |
| `SEMANTIC_DIMS` | 16 | Int |
| `PRAGMATIC_DIMS` | 6 | Int |
| `TOTAL_DIM` | 10000 | Int |
| `TOTAL_RICH_DIM` | 10013 | Int |
| `POSITION_WEIGHTS` | [3.5, 2.5, ..., 0.3] | Vector{Float64} |

---

## الوحدة: `Mirnan.Physics.LetterDB`

```julia
using Mirnan.Physics.LetterDB
```

### `LetterDatabase`

```julia
LetterDatabase(; path::Union{String,Nothing}=nothing) -> LetterDatabase
```

### `get_vector`

```julia
get_vector(db::LetterDatabase, letter::String) -> Vector{Int8}
```

متجه 9958D بقيم ±1.

### `get_omega_0`

```julia
get_omega_0(db::LetterDatabase, letter::String) -> Float64
```

التردد الذاتي: `0.5 + 2.0 * |v| / √dim`.

### `get_operator`

```julia
get_operator(db::LetterDatabase, letter::String) -> String
```

تصنيف الحرف (`"+1"`, `"-1"`, أو `"0"`).

---

## الوحدة: `Mirnan.Physics.CliffordMath`

```julia
using Mirnan.Physics.CliffordMath
```

### `Multivector22`

```julia
Multivector22(scalar::Real=0.0, vector=nothing, bivector=nothing, pseudoscalar::Real=0.0)
```

### العمليات المدعومة

| العملية | الدالة / المشغّل |
|---------|-----------------|
| الجمع | `a + b` |
| الطرح | `a - b` |
| الجداء الهندسي | `a * b` |
| القسمة | `a / b` |
| المعيار | `norm(mv)` |
| التطبيع | `normalize!(mv)` |
| المعكوس (reverse) | `reverse(mv)` |
| المعكوس الكلي | `inv(mv)` |
| الثنوية | `dual(mv)` |

### `from_vector`

```julia
from_vector(v) -> Multivector22
```

يحول متجه عادي إلى متجه كليفورد.

### `get_scalar_essence`

```julia
get_scalar_essence(mv::Multivector22) -> Float64
```

المركبة السلمية (الدرجة 0).

### `get_bivector_orientation`

```julia
get_bivector_orientation(mv::Multivector22) -> Float64
```

شدة bivector.

---

## الوحدة: `Mirnan.Physics.GravityEngine`

```julia
using Mirnan.Physics.GravityEngine
```

### `gravitational_force`

```julia
gravitational_force(m_i::Real, v_i::AbstractVector, m_j::Real, v_j::AbstractVector,
                    distance::Real) -> Vector{Float64}
```

F = G·m1·m2·(v2-v1) / (r²·|v2-v1|)

---

## الوحدة: `Mirnan.Physics.EntropyGate`

```julia
using Mirnan.Physics.EntropyGate
```

### `EntropyGate`

```julia
EntropyGate(; S_crit=2.5, k_B=0.1, beta_0=2.0) -> EntropyGate
```

### `compute_S`

```julia
compute_S(gate::EntropyGate, pv_list, target_pv) -> Float64
```

إنتروبيا شانون: `-Σ p·log(p)`.

### `evaluate`

```julia
evaluate(gate, pv_list, target_pv, k_B_cur, beta_cur) -> Tuple{Float64,Float64,Float64}
```

حساب + تصحيح = (k_B_new, beta_new, scale).

---

## الوحدة: `Mirnan.Physics.RAMCore`

```julia
using Mirnan.Physics.RAMCore
```

### `AttractorMemory`

```julia
AttractorMemory(; decay=0.01, merge_cos=0.92) -> AttractorMemory
```

### `observe!`

```julia
observe!(ram::AttractorMemory, words::Vector{String}; sigma=nothing) -> Int
```

تخزين (أو دمج) جملة كجاذب طوري. يرجع index الجاذب.

### `resonate`

```julia
resonate(ram::AttractorMemory, phi_current; top_k=3) -> Vector{Tuple{Float64,Int}}
```

استرجاع أقوى الجواذب درجة ورنيناً.

### `retrieve_context`

```julia
retrieve_context(ram, phi_current; max_words=10) -> Vector{String}
```

استرجاع كلمات السياق.

---

## الوحدة: `Mirnan.Physics.DensityMatrix`

```julia
using Mirnan.Physics.DensityMatrix
```

### `PhaseDensityMatrix`

```julia
PhaseDensityMatrix(; dim=64, decay_rate=0.8) -> PhaseDensityMatrix
```

### `build!`

```julia
build!(dm, pv_list; dims=nothing) -> Matrix{Float64}
```

ρ = Σᵢ wᵢ·|ψᵢ⟩⟨ψᵢ|

### `resonance`

```julia
resonance(dm, candidate_pv; dims=nothing) -> Float64
```

R = ⟨v| ρ |v⟩ ∈ [0, 1]

### `get_purity`

```julia
get_purity(dm) -> Float64
```

Tr(ρ²) — 1.0 = حالة نقية

### `get_spectral_entropy`

```julia
get_spectral_entropy(dm) -> Float64
```

S_vn = -Tr(ρ ln ρ) (يقيس درجة الاختلاط والتراكب الطوري الكلاسيكي في مصفوفة الكثافة)

---

## الوحدة: `Mirnan.Physics.CausalFlow`

```julia
using Mirnan.Physics.CausalFlow
```

### `CausalFlowField`

```julia
CausalFlowField(; dim=64, flow_strength=1.0) -> CausalFlowField
```

### `compute_flow`

```julia
compute_flow(cf, current_pv, context_pvs, context_ids; causal_matrix=nothing) -> Dict
```

J(pv) = Σ w·C·(target - current).

### `flow_alignment_score`

```julia
flow_alignment_score(cf, candidate_pv, current_pv, flow_vector) -> Float64
```

cos(candidate_direction, flow) ∈ [0, 1]

---

## الوحدة: `Mirnan.Physics.HolographicKB`

```julia
using Mirnan.Physics.HolographicKB
```

### `HolographicKnowledgeBase`

```julia
HolographicKnowledgeBase(; data_dir="data") -> HolographicKnowledgeBase
```

### `store_fact!`

```julia
store_fact!(kb, subj_pv, obj_pv, rel_type; subj_word="", obj_word="") -> Bool
```

تخزين حقيقة. يرجع `true` إذا نجح التخزين.

### `query`

```julia
query(kb, query_pv; rel_type=nothing, top_k=10, sharpening=3.0, cutoff=0.0) -> Vector{Tuple{Float64,String,String}}
```

استعلام التراكب الطوري التناظري. يرجع `[(confidence, obj_word, rel_type), ...]`.

### `query_by_word`

```julia
query_by_word(kb, query_word; rel_type=nothing, top_k=10) -> Vector{Tuple{Float64,String,String}}
```

استعلام مباشر بالكلمة (فاعل → مفعول).

### `reconstruct_vector`

```julia
reconstruct_vector(kb, query_pv; rel_type=nothing) -> Union{Vector{Float64},Nothing}
```

إعادة بناء متجه المفعول من استعلام الفاعل.

---

## الوحدة: `Mirnan.Physics.PhaseReinforcement`

```julia
using Mirnan.Physics.PhaseReinforcement
```

### `PhaseReinforcer`

```julia
PhaseReinforcer(; lr=0.15, decay_rate=0.005, max_traces=200) -> PhaseReinforcer
```

### `reinforce!`

```julia
reinforce!(pr, word, pv; reward=1.0)
```

تعزيز هيبياني: `trace += lr · reward · (new_pv - trace)`.

### `apply`

```julia
apply(pr, word, pv) -> Vector{Float64}
```

مزج المتجه الأصلي مع المعزَّز.

---

## الوحدة: `Mirnan.Physics.PPM`

```julia
using Mirnan.Physics.PPM
```

### `PromptField`

```julia
PromptField(; decay_rate=0.1, strength=1.0, max_examples=10) -> PromptField
```

### `absorb!`

```julia
absorb!(pf, prompt::String; examples=nothing)
```

امتصاص الأمر كحقل خارجي.

### `score`

```julia
score(pf, w_pv) -> Float64
```

cos(w_pv, field) ∈ [0, 1]

### `step!`

```julia
step!(pf)
```

اضمحلال الحقل بنسبة decay_rate.

---

## الوحدة: `Mirnan.Physics.AMFS`

```julia
using Mirnan.Physics.AMFS
```

### `adapt_word`

```julia
adapt_word(word::String; context_words=nothing, context_pvs=nothing) -> Dict
```

ترجع `Dict("mass" => ..., "freq" => ..., "phase_shift" => ..., "centrality" => ...)`.

---

## الوحدة: `Mirnan.Physics.DCCF`

```julia
using Mirnan.Physics.DCCF
```

### `DynamicCCF`

```julia
DynamicCCF(; decay_rate=0.5, mass_threshold=0.3) -> DynamicCCF
```

### `get_context_boost`

```julia
get_context_boost(dccf, word, context_words) -> Float64
```

درجة الاقتران الديناميكي مع السياق.

### `build_coupling`

```julia
build_coupling(dccf, context_words) -> Tuple{Matrix{Float64},Dict}
```

بناء مصفوفة اقتران n×n من السياق.

---

## الوحدة: `Mirnan.Physics.PRNNCore`

النواة الفيزيائية للشبكة العصبية الرنينية الطورية:

```julia
using Mirnan.Physics.PRNNCore
```

### `PRNNState`

```julia
PRNNState(N::Int) -> PRNNState
```

بناء حالة المذبذبات لنظام ستوارت-لانداو بطول البعد الطوري `N` (عادة 10,000).

* الحقول:
  * `z`: متجهات الأطوار والسعات المركبة `Vector{ComplexF64}`.
  * `a`: التعب العصبي المحلي `Vector{Float64}`.
  * `omega`: الترددات الذاتية لمنع التزامن العشوائي المطلق `Vector{Float64}`.

### `LowRankPRNN`

```julia
LowRankPRNN(transitions::Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}, beta::Float64) -> LowRankPRNN
```

بنية الاقتران السببي منخفض الرتبة الحامل للانتقالات الموزونة وقوة الربط `beta`.

### `simulate_stuart_landau!`

```julia
simulate_stuart_landau!(state::PRNNState, model::LowRankPRNN; 
                        mu=1.0, g_inh=0.5, gamma=2.0, tau_a=1.5, steps=40, dt=0.02)
```

محاكاة الحركة غير الخطية وتكامل المذبذبات تحت تأثير معادلة ستوارت-لانداو والتثبيط والتعب.

### `bind_phase`

```julia
bind_phase(v_a, v_b) -> Vector
```

الربط الهولوغرافي الطوري (الضرب النقطي المركب): $\mathbf{v}_a \odot \mathbf{v}_b$.

### `unbind_phase`

```julia
unbind_phase(v_bound, v_context) -> Vector
```

فك الربط الهولوغرافي الطوري (الضرب في المرافق المركب): $\mathbf{v}_{\text{bound}} \odot \mathbf{v}_{\text{context}}^*$.

---

## الوحدة: `Mirnan.Physics.PRNNLearner`

التعلم الهيبياني الهولوغرافي وتوسيع الطور:

```julia
using Mirnan.Physics.PRNNLearner
```

### `build_dense_phase_vector`

```julia
build_dense_phase_vector(word::String, N::Int) -> Vector{ComplexF64}
```

التوسيع الهندسي للطور: يحول المتجه الدلالي الحقيقي المتفرق لمتجه طوري كثيف على دائرة الوحدة المركبة $\exp(i v \sqrt{N}\pi)$.

### `train_hebbian_transitions!`

```julia
train_hebbian_transitions!(vocab::Dict{String,Int}, id2word::Dict{Int,String}, 
                            corpus_sentences::Vector{Vector{Int32}}, active_vocab::Vector{String}, 
                            prompt_ids::Vector{Int32}, beta::Float64, N::Int,
                            base_vectors::Dict{String, Vector{ComplexF64}}=Dict(); window::Int=5) -> Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Float64}}
```

استخراج الانتقالات السببية الهيبية وتكرارها من النوافذ المحلية للمحفز في جمل الكوربوس.

---

## الوحدة: `Mirnan.Physics.PRNNGenerator`

التوليد الطوري المستقل (النموذج الفيزيائي المستقل):

```julia
using Mirnan.Physics.PRNNGenerator
```

### `PRNNSession`

```julia
PRNNSession(vocab, id2word, corpus_sentences, seed_words; beta=3.0, N=TOTAL_DIM, window=5, max_sentences=100, base_cache=Dict()) -> PRNNSession
```

إنشاء جلسة مستقلة للـ PRNN تحسب معجم الأطوار والروابط الهيبية للبدء في التوليد.

### `prnn_generate_standalone`

```julia
prnn_generate_standalone(session::PRNNSession, prompt_tokens::Vector{String}; max_words=15, kwargs...) -> String
```

توليد النص بالانزلاق الحتمي في فضاء الطور المركب انطلاقاً من حالة الجلسة ودون استخدام محرك التوليد المركزي.
