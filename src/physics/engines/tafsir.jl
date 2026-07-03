"""Tafsir — تفسير (شرح المتجهات والأبعاد)."""
module Tafsir
export explain_word, explain_dimension

const DIM_MEANINGS = Dict(
    "concentration" => "التركيز: شدة انحصار المعنى", "temperature" => "الحرارة: درجة النشاط الدلالي",
    "density" => "الكثافة: ثقل المعنى", "motion_linear" => "الحركة الخطية: اتجاهية المعنى",
    "mass" => "الكتلة: وزن الكلمة الدلالي", "charge" => "الشحنة: قطبية المعنى",
    "hardness_solid" => "الصلابة: جمود المعنى", "penetration" => "الاختراق: نفاذ المعنى",
)

function explain_word(word::String, pv_fn)
    pv = pv_fn(word)
    d = min(22, length(pv))
    top_idx = sortperm(abs.(pv[1:d]); rev=true)[1:3]
    parts = String[]
    for i in top_idx
        name = get(DIM_NAMES, "", i <= length(DIM_NAMES) ? DIM_NAMES[i] : "بعد_$i")
        meaning = get(DIM_MEANINGS, name, name)
        val = round(pv[i]; digits=3)
        push!(parts, "$name($val)")
    end
    return "كلمة '$word': الأبعاد المهيمنة " * join(parts, ", ")
end

function explain_dimension(dim_name::String)
    return get(DIM_MEANINGS, dim_name, "بُعد فيزيائي")
end
end
