"""
CodePhaseEngine — محرك البرمجة الفيزيائي.

يعامل المفاهيم البرمجية كمتجهات طورية في فضاء 10000D.
كل مفهوم (for, if, list, function...) = متجه طوري فريد.
الانتقالات البرمجية = تحويلات هندسية (T = v_next / v_current).

المبادئ:
  - ليس Markov chain إحصائياً — فيزياء بحتة
  - v(for) في نفس فضاء v(حلقة) — فهم ثنائي اللغة
  - الكود = جداء كليفورد للمفاهيم: for ∘ list ∘ append
  - التوثيق العربي ↔ الكود: اقترانات K في نفس الفضاء
"""
module CodePhaseEngineModule

using LinearAlgebra, SparseArrays, Statistics
using ..Constants: TOTAL_DIM, PHASE_DIM
using ..WordPhysics: compute_word_phase_vector, compute_extended_phase_vector
using ..CliffordMath: Multivector22, from_vector

export CodePhaseEngine, CodeConcept, CodePattern,
       encode_concept, decode_to_code, learn_code_patterns!,
       generate_code_physics, score_code_candidate

# ═══════════ مفاهيم برمجية أساسية ═══════════

const _CODE_CONCEPTS = Dict{String,Vector{String}}(
    # هياكل التحكم
    "loop"     => ["for", "while", "حلقة", "طالما", "لكل"],
    "condition"=> ["if", "else", "elif", "اذا", "وإلا", "شرط"],
    "function" => ["def", "function", "دالة", "اقتران", "تعريف"],
    "class"    => ["class", "struct", "صنف", "كلاس", "هيكل"],
    "return"   => ["return", "yield", "ارجع", "عاد", "نتيجة"],
    # أنواع البيانات
    "list"     => ["list", "array", "قائمة", "مصفوفة", "متسلسلة"],
    "dict"     => ["dict", "map", "قاموس", "خريطة", "مفتاح"],
    "string"   => ["string", "str", "نص", "سلسلة", "حرفي"],
    "number"   => ["int", "float", "number", "رقم", "عدد", "صحيح", "عشري"],
    "boolean"  => ["bool", "true", "false", "منطقي", "صح", "خطأ"],
    # عمليات
    "print"    => ["print", "println", "اطبع", "اكتب", "عرض", "أظهر"],
    "input"    => ["input", "read", "ادخل", "اقرأ", "استقبل"],
    "sort"     => ["sort", "sorted", "رتب", "فرز", "ترتيب"],
    "search"   => ["find", "search", "ابحث", "بحث", "جد"],
    "append"   => ["append", "add", "push", "اضف", "الحق", "أضف"],
    "remove"   => ["remove", "delete", "pop", "احذف", "امسح", "أزل"],
    # متقدم
    "import"   => ["import", "from", "require", "استورد", "ضمن", "استدع"],
    "variable" => ["var", "let", "متغير", "عرف", "عين"],
    "error"    => ["try", "catch", "except", "حاول", "امسك", "استثناء"],
)

"""
    CodeConcept

مفهوم برمجي واحد = متجه طوري + كلمات مفتاحية (عربي + إنجليزي).
"""
struct CodeConcept
    name::String               # اسم المفهوم (loop, condition, ...)
    keywords::Vector{String}   # الكلمات المرتبطة (عربي + إنجليزي)
    pv::Vector{Float64}        # المتجه الطوري للمفهوم
end

"""
    CodePattern

نمط برمجي = سلسلة انتقالات بين مفاهيم (مثل: loop → list → append).
"""
struct CodePattern
    concepts::Vector{String}    # تسلسل المفاهيم
    template::String            # قالب الكود المولَّد
    count::Int                  # عدد مرات الظهور في التدريب
end

"""
    CodePhaseEngine

محرك البرمجة الفيزيائي.
"""
mutable struct CodePhaseEngine
    concepts::Dict{String,CodeConcept}   # اسم المفهوم → المفهوم
    word_to_concept::Dict{String,String} # كلمة (عربي/إنجليزي) → اسم المفهوم
    patterns::Vector{CodePattern}        # أنماط برمجية متعلمة
    K_code_sem::Any                      # مصفوفة اقتران: كود ↔ لغة طبيعية
    active::Bool
end

function CodePhaseEngine()
    eng = CodePhaseEngine(Dict{String,CodeConcept}(),
                           Dict{String,String}(),
                           CodePattern[],
                           nothing, true)
    _init_concepts!(eng)
    return eng
end

"""
    _init_concepts!(engine)

يهيئ المفاهيم البرمجية الأساسية بمتجهاتها الطورية.
"""
function _init_concepts!(eng::CodePhaseEngine)
    for (name, keywords) in _CODE_CONCEPTS
        # المتجه = متوسط متجهات الكلمات المفتاحية
        pvs = Vector{Float64}[]
        for kw in keywords
            try
                push!(pvs, Float64.(compute_word_phase_vector(kw)))
            catch e
                @warn "Failed to compute phase vector for '$kw': $e"
            end
        end
        if isempty(pvs)
            pv = zeros(Float64, PHASE_DIM)
        else
            pv = mean(pvs)
            nrm = norm(pv)
            nrm > 1e-10 && (pv ./= nrm)
        end
        concept = CodeConcept(name, keywords, pv)
        eng.concepts[name] = concept
        for kw in keywords
            eng.word_to_concept[kw] = name
        end
    end
end

"""
    encode_concept(engine, word::String) -> (String, Vector{Float64})

يحول كلمة (عربية أو إنجليزية) إلى مفهوم برمجي + متجهه.
مثال: "حلقة" → ("loop", v(loop))
      "for"   → ("loop", v(loop))
"""
function encode_concept(eng::CodePhaseEngine, word::AbstractString)
    w_str = String(word)
    concept_name = get(eng.word_to_concept, lowercase(w_str), nothing)
    if concept_name !== nothing && haskey(eng.concepts, concept_name)
        return concept_name, eng.concepts[concept_name].pv
    end
    # ليست كلمة مفتاحية معروفة — عاملها كمعرّف عام
    try
        return "identifier", Float64.(compute_word_phase_vector(w_str))
    catch e
        @debug "CodePhaseEngine: failed to compute PV for '$w_str': $e"
        return "unknown", zeros(Float64, PHASE_DIM)
    end
end

"""
    decode_to_code(engine, concepts::Vector{String}; lang::String="python") -> String

يحول سلسلة مفاهيم برمجية إلى كود فعلي.
مثال: ["function","loop","print"] → "def f():\n    for i in range(10):\n        print(i)"
"""
function decode_to_code(eng::CodePhaseEngine, concepts::Vector{String}; lang::String="python")
    isempty(concepts) && return "# no concepts"

    # ابحث عن أقرب قالب مطابق
    best_pattern = nothing
    best_match = 0
    for pat in eng.patterns
        match_count = 0
        for c in concepts
            c in pat.concepts && (match_count += 1)
        end
        if match_count > best_match
            best_match = match_count
            best_pattern = pat
        end
    end

    if best_pattern !== nothing && best_match >= length(concepts) * 0.6
        return best_pattern.template
    end

    # توليد من المفاهيم مباشرة
    return _concepts_to_code(concepts; lang=lang)
end

"""
    _concepts_to_code(concepts; lang) -> String

يحول سلسلة مفاهيم إلى كود بايثون/جوليا.
"""
function _concepts_to_code(concepts::Vector{String}; lang::String="python")
    lines = String[]
    indent = 0
    used_vars = String[]

    for c in concepts
        if lang == "python"
            line, indent_change = _python_stub(c, indent, used_vars)
        else
            line, indent_change = _julia_stub(c, indent, used_vars)
        end
        indent += indent_change
        indent = max(0, indent)
        indent > 0 && push!(lines, repeat("    ", indent) * line)
        indent == 0 && push!(lines, line)
    end

    return join(lines, "\n")
end

function _python_stub(concept::String, indent::Int, used_vars::Vector{String})
    if concept == "function"
        return ("def solve():", 1)
    elseif concept == "class"
        return ("class Solution:", 1)
    elseif concept == "loop"
        push!(used_vars, "i")
        return ("for i in range(len(data)):", 1)
    elseif concept == "condition"
        return ("if True:", 1)
    elseif concept == "print"
        var = isempty(used_vars) ? "result" : used_vars[end]
        return ("print($var)", -1)
    elseif concept == "return"
        var = isempty(used_vars) ? "result" : used_vars[end]
        return ("return $var", -1)
    elseif concept == "list"
        push!(used_vars, "data")
        return ("data = []", 0)
    elseif concept == "append"
        var = length(used_vars) >= 2 ? used_vars[end-1] : "data"
        val = isempty(used_vars) ? "x" : used_vars[end]
        return ("$var.append($val)", 0)
    elseif concept == "sort"
        var = isempty(used_vars) ? "data" : used_vars[end]
        return ("$var.sort()", 0)
    elseif concept == "variable"
        push!(used_vars, "x")
        return ("x = 0", 0)
    elseif concept == "input"
        push!(used_vars, "user_input")
        return ("user_input = input()", 0)
    elseif concept == "import"
        return ("import math", 0)
    else
        return ("pass", 0)
    end
end

function _julia_stub(concept::String, indent::Int, used_vars::Vector{String})
    if concept == "function"
        return ("function solve()", 1)
    elseif concept == "loop"
        push!(used_vars, "i")
        return ("for i in 1:length(data)", 1)
    elseif concept == "condition"
        return ("if true", 1)
    elseif concept == "print"
        return ("println(result)", -1)
    elseif concept == "return"
        return ("return result", -1)
    elseif concept == "list"
        push!(used_vars, "data")
        return ("data = []", 0)
    elseif concept == "append"
        return ("push!(data, x)", 0)
    elseif concept == "variable"
        push!(used_vars, "x")
        return ("x = 0", 0)
    else
        return ("# $concept", 0)
    end
end

"""
    learn_code_patterns!(engine, code_snippets::Vector{String})

يتعلم أنماطاً برمجية من أمثلة أكواد.
يحلل الكود إلى سلسلة مفاهيم ← يخزن القالب.
"""
function learn_code_patterns!(eng::CodePhaseEngine, code_snippets::Vector{String})
    for snippet in code_snippets
        concepts = _extract_concepts_from_code(snippet)
        if length(concepts) >= 2
            # ابحث عن قالب مماثل أو أنشئ جديداً
            found = false
            for pat in eng.patterns
                if length(intersect(concepts, pat.concepts)) >= length(concepts) * 0.7
                    pat.count += 1
                    found = true
                    break
                end
            end
            if !found
                push!(eng.patterns, CodePattern(concepts, snippet, 1))
            end
        end
    end
end

"""
    _extract_concepts_from_code(code::String) -> Vector{String}

يستخرج سلسلة المفاهيم البرمجية من كود خام.
"""
function _extract_concepts_from_code(code::String)
    concepts = String[]
    for line in split(code, "\n")
        stripped = strip(line)
        isempty(stripped) && continue
        for (concept_name, keywords) in _CODE_CONCEPTS
            for kw in keywords
                if occursin(Regex("\\b$kw\\b"), stripped)
                    push!(concepts, concept_name)
                    break
                end
            end
        end
    end
    return unique(concepts)
end

"""
    generate_code_physics(engine, prompt::String; lang::String="python") -> String

يولد كوداً من وصف عربي باستخدام الفيزياء:
1. يحول الكلمات العربية إلى مفاهيم
2. يجد أفضل سلسلة مفاهيم للهدف
3. يفك السلسلة إلى كود
"""
function generate_code_physics(eng::CodePhaseEngine, prompt::AbstractString; lang::AbstractString="python")
    prompt_words = split(prompt)

    # استخرج المفاهيم من الأمر
    concepts = String[]
    for w in prompt_words
        cname, _ = encode_concept(eng, w)
        if cname != "unknown" && cname != "identifier"
            push!(concepts, cname)
        end
    end

    unique!(concepts)

    # أضف مفاهيم السياق الضرورية
    if !in("function", concepts) && !in("class", concepts)
        insert!(concepts, 1, "function")
    end

    return decode_to_code(eng, concepts; lang=String(lang))
end

"""
    score_code_candidate(engine, concept_pv, candidate_pv) -> Float64

يقيم مرشحاً برمجياً: مدى توافقه الطوري مع المفهوم الحالي.
"""
function score_code_candidate(eng::CodePhaseEngine, concept_pv::AbstractVector, candidate_pv::AbstractVector)
    d = min(length(concept_pv), length(candidate_pv))
    n1 = norm(view(concept_pv, 1:d))
    n2 = norm(view(candidate_pv, 1:d))
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    return max(0.0, dot(view(concept_pv, 1:d), view(candidate_pv, 1:d)) / (n1 * n2))
end

end # module CodePhaseEngineModule
