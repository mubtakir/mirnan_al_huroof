# أمثلة الاستخدام — Examples

## الاستخدام الأساسي عبر CLI

### توليد نص
```bash
julia --project=. cli.jl "العلم نور"
# ↳ العلم نور الحياة
```

### توليد إبداعي
```bash
julia --project=. cli.jl --mode creative "في قديم الزمان"
# ↳ في قديم الزمان كان هناك عالم كبير مملوء بالأسرار
```

### وضع تفاعلي
```bash
julia --project=. cli.jl -i
> العلم نور
  ↳ العلم نور الحياة
> /mode creative
الوضع = creative
> في أعماق الظلام
  ↳ في أعماق الظلام حيث تسكن النجوم
> exit
```

---

## الاستخدام البرمجي (Julia REPL)

### إعداد مولد أساسي

```julia
push!(LOAD_PATH, "src")
using Mirnan

# بناء معجم صغير للاختبار
vocab = Dict(
    "ال" => 1, "في" => 2, "علم" => 3, "نور" => 4,
    "حياة" => 5, "كبير" => 6, "سماء" => 7, "أرض" => 8,
    "ماء" => 9, "الله" => 10, "سلام" => 11, "عالم" => 12,
    "كتاب" => 13, "قلب" => 14, "حق" => 15,
)

# بناء مصفوفة اقتران K بسيطة
K_sem = spzeros(15, 15)
K_sem[3, 4] = 0.5; K_sem[4, 3] = 0.5   # علم ↔ نور
K_sem[3, 5] = 0.4; K_sem[5, 3] = 0.4   # علم ↔ حياة
K_sem[4, 12] = 0.3; K_sem[12, 4] = 0.3  # نور ↔ عالم
K_sem[7, 8] = 0.5; K_sem[8, 7] = 0.5   # سماء ↔ أرض
K_sem[5, 9] = 0.4; K_sem[9, 5] = 0.4   # حياة ↔ ماء

gen = MirnanGenerator(vocab, K_sem)

# توليد
result = generate!(gen, "العلم نور")
println(result)  # "حياة عالم كتاب"
```

### استخدام التقرير الفيزيائي

```julia
# تقرير بعد التوليد
report = get_physics_report(gen, ["علم", "نور"])
println("الإنتروبيا: $(round(report["entropy"]; digits=3))")
println("متوسط الكتلة: $(round(report["mass_mean"]; digits=3))")
println("حجم المعجم: $(report["vocab_size"])")
```

### استخدام المحركات بشكل منفرد

```julia
using Mirnan.Physics.WordPhysics
using Mirnan.Physics.GravityEngine
using Mirnan.Physics.ResonantChain
using Mirnan.Physics.LetterDB
using Mirnan.Physics.CliffordMath

# فيزياء الكلمة
pv1 = compute_word_phase_vector("علم")     # 9958D
pv2 = compute_word_phase_vector("نور")     # 9958D
mass1 = compute_word_mass("علم")            # ~ الكتلة
freq1 = compute_word_frequency("علم")       # ~ التردد

# تشابه طوري
sim = phase_similarity(pv1, pv2)            # cosine ∈ [-1, 1]

# جاذبية دلالية
force = gravitational_force(mass1, pv1, compute_word_mass("نور"), pv2, 1.0)

# سلسلة رنينية LC
chain = ResonantChainRLC()
f = pair_freq(chain, mass1, compute_word_mass("نور"), pv1, pv2)

# قاعدة بيانات الحروف
db = LetterDatabase()
v_ain = LetterDB.get_vector(db, "ع")
omega_ain = LetterDB.get_omega_0(db, "ع")

# جبر كليفورد
mv1 = from_vector(Float64.(pv1[1:22]))
mv2 = from_vector(Float64.(pv2[1:22]))
product = mv1 * mv2              # الجداء الهندسي
scalar = get_scalar_essence(product)
```

### استخدام الذاكرة الهولوغرافية

```julia
using Mirnan.Physics.HolographicKB

kb = HolographicKnowledgeBase()

# تخزين حقائق
kb_pv(s) = compute_extended_phase_vector(s)
store_fact!(kb, kb_pv("باريس"), kb_pv("فرنسا"), "CAPITAL_OF";
            subj_word="باريس", obj_word="فرنسا")
store_fact!(kb, kb_pv("ثلج"), kb_pv("ماء"), "BECOMES_WHEN";
            subj_word="ثلج", obj_word="ماء")

# استعلام
results = query(kb, kb_pv("باريس"); top_k=5)
for (conf, word, rel) in results
    println("$word ($rel): $(round(conf; digits=3))")
end

# إعادة بناء متجه
reconstructed = reconstruct_vector(kb, kb_pv("باريس"))
```

### استخدام مصفوفة الكثافة الطورية المركبة

```julia
using Mirnan.Physics.DensityMatrix

dm = PhaseDensityMatrix(; dim=10000)

# بناء ρ من السياق
pvs = [Float64.(compute_extended_phase_vector(w)) for w in ["علم", "نور", "حياة"]]
build!(dm, pvs)

# رنين مرشح
candidate_pv = Float64.(compute_extended_phase_vector("كتاب"))
res = resonance(dm, candidate_pv)              # ∈ [0, 1]

# خصائص الحالة
trace = get_trace(dm)                          # ≈ 1.0
purity = get_purity(dm)                        # < 1.0 (حالة مختلطة)
entropy_vn = get_spectral_entropy(dm)      # S_vn
```

### استخدام التدفق السببي

```julia
using Mirnan.Physics.CausalFlow

cf = CausalFlowField(; dim=10000)

# بناء مصفوفة سببية
causal_K = spzeros(10, 10)
causal_K[1, 2] = 0.8   # A → B
causal_K[2, 3] = 0.6   # B → C
causal_K[3, 4] = 0.4   # C → D

# حساب التدفق
current_pv = compute_extended_phase_vector("A")
context_pvs = [compute_extended_phase_vector(w) for w in ["A", "B", "C"]]
flow = compute_flow(cf, current_pv, context_pvs, [1, 2, 3];
                    causal_matrix=causal_K)

# توافق المرشح مع التدفق
candidate_pv = compute_extended_phase_vector("D")
align = flow_alignment_score(cf, candidate_pv, current_pv, flow["flow_vector"])

# استدلال تعددي
strength = compute_transitive_flow(cf, [1, 2, 3, 4], causal_K)
println("قوة السلسلة السببية: $(strength["flow_strength"])")
```

---

## التدريب من الصفر

### تحضير ملف الكوربس

أنشئ `data/corpus.txt`:
```
العلم نور والجهل ظلام
السماء صافية والأرض خضراء
الحياة جميلة والعالم كبير
الله خالق كل شيء
الكتاب مفيد والعلم نور
الماء سر الحياة والأرض خضراء
السلام عليكم ورحمة الله
القلب الكبير يعرف الحب
```

### تشغيل التدريب
```bash
julia --project=. train.jl

# أو مع ملف كوربس مخصص
julia --project=. train.jl --corpus data/my_corpus.txt
```

**مخرجات التدريب:**
```
مرنان V8 — تدريب فيزيائي
════════════════════════════════════════
1. بناء المعجم...
  ✓ 32 كلمة فريدة
2. بناء مصفوفات الاقتران...
  ✓ K_sem: 86 اقتران دلالي
  ✓ K_syn: 42 اقتران نحوي
3. حفظ النموذج...
  ✓ معجم: 32 كلمة
4. اختبار التوليد...
  ↳ العلم نور → حياة
  ✓ الإنتروبيا: 0.089
  ✓ β: 2.0
════════════════════════════════════════
اكتمل التدريب ✓
```

---

## اختبار المكونات

### تشغيل كل الاختبارات
```bash
julia --project=. test/runtests.jl
```

**النتيجة المتوقعة:**
```
Mirnan V8 — Physics Engine: ...
  Constants: ...
    ✓ PLANCK_H == 1.0
    ✓ GRAVITY_G == 1.0
    ...
  Letter Database: ...
  Word Physics: ...
  Phase Similarity: ...
  Gravity: ...
  Clifford Math: ...
  Resonant Chain: ...
  Entropy Gate: ...
  RAM Core: ...
  Phase Reinforcement: ...
  DCCF: ...
  PPM: ...
  AMFS: ...
  Density Matrix: ...
  Causal Flow: ...
  Generator: ...
  Generator Edge Cases: ...
```

---

## استخدام متقدم

### تحميل قاعدة معرفية من JSON

```julia
using Mirnan.Physics.HolographicKB
using JSON

kb = HolographicKnowledgeBase()
# تحميل حقائق منسقة
facts_data = JSON.parsefile("data/facts_expanded.json")
for fact in facts_data["facts"]
    subj = fact["subject"]
    obj = fact["object"]
    rel = fact["relation"]
    
    subj_pv = compute_extended_phase_vector(subj)
    obj_pv = compute_extended_phase_vector(obj)
    
    store_fact!(kb, subj_pv, obj_pv, rel;
               subj_word=subj, obj_word=obj)
end
println("تم تحميل $(kb.fact_count) حقيقة")
```

### تخصيص أوزان التسجيل

```julia
gen = MirnanGenerator(vocab, K_sem)

# تعديل الأوزان (قبل normalize)
gen.scoring_weights["align"] = 7.0       # رفع وزن التوافق الطوري
gen.scoring_weights["diversity"] = 3.0   # رفع وزن التنوع
gen.scoring_weights["kb_knowledge"] = 5.0  # رفع وزن المعرفة

# إعادة تطبيع
total = sum(values(gen.scoring_weights))
for k in keys(gen.scoring_weights)
    gen.scoring_weights[k] *= 0.95 / total
end
```

### مراقبة حالة التعزيز الطوري

```julia
using Mirnan.Physics.PhaseReinforcement

pr = PhaseReinforcer()

# بعد جلسة توليد
for (word, strength) in pr.strengths
    if strength > 0.5
        println("$word: $(round(strength; digits=2)) — أثر قوي")
    end
end
```

---

## تثبيت الحزم (مرة واحدة)

```bash
cd mirnan_julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

هذا يُثبت:
- `JSON` — قراءة/كتابة JSON
- `YAML` — تحميل config.yaml
- `NPZ` — قراءة/كتابة .npz
- `StaticArrays` — مصفوفات صغيرة ثابتة الحجم
