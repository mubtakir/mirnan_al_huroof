# حوكمة المعرفة في al_aql

هذه الطبقة تجعل `al_aql` أكثر من مخزن علاقات. صار يملك ذاكرة حاكمة تميز بين:

- `curated_rules`: قواعد موثوقة لها أولوية عليا.
- `proposed_rules`: فرضيات مرشحة للمراجعة.
- `rejected_rules`: قواعد رُفضت ولا ينبغي أن تعود بقوة لمجرد أنها ظهرت في البيانات.
- `rejection_lessons`: دروس عامة من أسباب الرفض، مثل قلب السببية أو الخلط بين التلازم والسببية.
- `corpus_annotations`: وسوم نقدية على الجمل لا تغيّر النص الخام، لكنها توجه إعادة الوزن في التدريب اللاحق.

## الدستور والكبح

الدستور الصريح يبنى عبر:

```julia
curate_rule!(space, "حديد مقيد", "ينتج", "تمدد";
             polarity=-1, confidence=1.0, tags=["تمدد"])
```

إذا كانت `polarity=-1` فهذه ليست نصيحة إيجابية، بل كبح عقلي. عند التوليد يبني `MirnanGenerator` قاموس `aql_inhibition` من هذه القواعد، ثم يدخل في `_score` كموجة هدّامة عبر وزن:

```text
aql_inhibition
```

أي أن `al_aql` صار يستطيع أن يقول: هذا المسار ممنوع أو ضعيف، لا أنه يرفع مسارا آخر فقط.

## الاقتراح والرفض

القواعد المرشحة تسجل عبر:

```julia
propose_rule!(space, "الدخان", "يسبب", "النار";
              confidence=0.9, tags=["causal"])
```

قبل قبول الاقتراح مبدئيا، يمر على `score_rule_proposal`. هذه الدالة تفحص `rejection_lessons`. فإذا كان هناك درس يقول مثلا إن نمطاً معيناً معرض لقلب السببية، تخفض الثقة أو ترفض القاعدة تلقائيا.

رفض قاعدة:

```julia
reject_rule!(space, "rule_id";
             reason="causal_direction_error",
             lesson_pattern="يسبب",
             penalty_tags=["causal"])
```

هذا لا يحفظ الرفض فقط، بل ينشئ درساً يمكن تطبيقه على اقتراحات لاحقة.

## المراجعة النقدية للذاكرة

`critical_corpus_pass!` لا تعدل corpus الخام. هي تنشئ طبقة وسوم:

```julia
critical_corpus_pass!(space, [
    "إذا زادت الحرارة فإن الحديد يتمدد إلا إذا كان مقيداً",
    "جيش الليل زحف",
])
```

الوسوم الحالية تشمل:

- `causal`
- `correlation`
- `metaphoric`
- `requires_condition`
- `uncertain`
- `lesson:<reason>`

كل وسم يحمل أثرا مقترحا على:

- `effect_on_k_sem`
- `effect_on_k_causal`
- `effect_on_al_aql`

وتوجد دالتان مساعدتان:

```julia
annotation_weight(space, "runtime:1"; matrix=:causal)
aql_memory_audit(space)
```

الأولى تعطي مضاعف وزن لجملة معينة عند إعادة بناء مصفوفة، والثانية تعطي تقريرا مختصرا عن الذاكرة النقدية والقواعد والدروس.

## الحفظ والتحميل

تحفظ هذه الطبقة داخل:

```text
model/runtime_learning/runtime_learning.json
```

مع بقية تعلم التشغيل:

```julia
save_runtime_learning!(gen)
load_runtime_learning!(gen)
```

وهذا يعني أن الدستور والرفض والدروس ووسوم corpus تعود عند إنشاء مولد جديد.

## صيغة ADL

أضيفت صيغ عملية للغة `al_aql`:

```text
curated { subject: حديد مقيد; predicate: ينتج; object: تمدد; polarity: -1; tags: تمدد }
proposed { subject: الدخان; predicate: يسبب; object: النار; confidence: 0.9; tags: causal }
reject { id: smoke_causes_fire; reason: causal_direction_error; pattern: يسبب; tags: causal }
lesson { reason: causal_direction_error; pattern: يسبب; tags: causal; confidence: 1.0 }
annotation { sentence_id: s1; span: جيش الليل زحف; tag: metaphoric; effect_on_k_sem: 0.65 }
critical_corpus إذا زادت الحرارة فإن الحديد يتمدد إلا إذا كان مقيداً
```

هذه الصيغ تجعل الحوكمة قابلة للتعلم اليدوي، لا مجرد دوال داخلية.
