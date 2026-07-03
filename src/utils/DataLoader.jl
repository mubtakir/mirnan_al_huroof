"""
DataLoader.jl - تحميل البيانات
"""

module DataLoader

export load_root_db, load_dictionary, save_analysis, load_analysis

"""
    load_root_db() -> Dict{String, Any}
    تحميل قاعدة بيانات الجذور
"""
function load_root_db()
    roots = Dict{String, Any}()
    
    # جذور أساسية
    basic_roots = [
        ("كتب", "يكتب", "فعل", "أساسي", ["كاتب", "كتابة", "كتاب", "مكتوب"]),
        ("قرأ", "يقرأ", "فعل", "أساسي", ["قارئ", "قراءة", "كتاب"]),
        ("علم", "يعلم", "فعل", "معرفة", ["عالم", "علم", "معرفة"]),
        ("فهم", "يفهم", "فعل", "معرفة", ["فاهم", "فهم", "مفهوم"]),
        ("درّس", "يدرس", "فعل", "تعليم", ["مدرّس", "دراسة", "مدرسة"]),
        ("تعلّم", "يتعلم", "فعل", "تعليم", ["طالب", "تعلم", "تعليم"]),
        ("بحث", "يبحث", "فعل", "بحث", ["باحث", "بحث", "دراسة"]),
        ("قال", "يقول", "فعل", "اتصال", ["قول", "مقول", "مقال"]),
        ("جاء", "يجيء", "فعل", "حركة", ["مجيء", "قادم"]),
        ("ذهب", "يذهب", "فعل", "حركة", ["ذهاب", "ذاهب"]),
        ("كبر", "يكبر", "فعل", "حجم", ["كبير", "كبرى"]),
        ("صغر", "يصغر", "فعل", "حجم", ["صغير", "صغرى"]),
        ("جمال", "يجمّل", "فعل", "جمالي", ["جميل", "جمال"]),
        ("شمس", "تشمس", "فعل", "فلك", ["شمس", "شمسي"]),
        ("قمر", "يقمر", "فعل", "فلك", ["قمر", "قمري"]),
        ("نجم", "ينجم", "فعل", "فلك", ["نجم", "نجمي"]),
        ("سماء", "يسما", "فعل", "فضاء", ["سماء", "سماوي"]),
        ("أرض", "أرض", "فعل", "أرض", ["أرضي", "أرضية"]),
        ("ماء", "تماء", "فعل", "ماء", ["مائي", "مياه"]),
        ("نار", "tonner", "فعل", "نار", ["ناري", "حريق"]),
        ("مجتمع", "يجمّع", "فعل", "اجتماعي", ["اجتماعي", "مجتمع"]),
        ("ثقافة", "يثقّف", "فعل", "ثقافي", ["ثقافي", "ثقافة"]),
        ("تاريخ", "يتأرّخ", "فعل", "تاريخي", ["تاريخي", "تاريخ"]),
        ("فن", "يفنّن", "فعل", "فني", ["فني", "فنان"]),
        ("أدب", "يؤدّب", "فعل", "أدبي", ["أديب", "أدبي"]),
        ("لغة", "يلغّي", "فعل", "لغوي", ["لغوي", "لغة"]),
        ("دين", "يدّين", "فعل", "ديني", ["ديني", "دين"]),
        ("قانون", "يقنّن", "فعل", "قانوني", ["قانوني", "قانون"]),
        ("سياسة", "يساس", "فعل", "سياسي", ["سياسي", "سياسة"]),
        ("اقتصاد", "يconomy", "فعل", "اقتصادي", ["اقتصادي", "اقتصاد"]),
    ]
    
    for (root, meaning, type, category, derivatives) in basic_roots
        roots[root] = Dict(
            "root" => root,
            "meaning" => meaning,
            "type" => type,
            "category" => category,
            "derivatives" => derivatives,
            "frequency" => 50
        )
    end
    
    return roots
end

"""
    load_dictionary() -> Dict{String, Any}
    تحميل القاموس الأساسي
"""
function load_dictionary()
    return Dict(
        "كتب" => Dict("meaning" => "يكتب", "type" => "فعل", "root" => "كتب"),
        "كتاب" => Dict("meaning" => "كتاب", "type" => "اسم", "root" => "كتب"),
        "مدرسة" => Dict("meaning" => "مدرسة", "type" => "اسم", "root" => "درّس"),
        "طالب" => Dict("meaning" => "طالب", "type" => "اسم", "root" => "طلّب"),
        "معلم" => Dict("meaning" => "معلم", "type" => "اسم", "root" => "علّم"),
        "جميل" => Dict("meaning" => "جميل", "type" => "صفة", "root" => "جمال"),
        "كبير" => Dict("meaning" => "كبير", "type" => "صفة", "root" => "كبر"),
        "صغير" => Dict("meaning" => "صغير", "type" => "صفة", "root" => "صغر"),
    )
end

"""
    save_analysis(filename, analysis)
    حفظ نتائج التحليل
"""
function save_analysis(filename::String, analysis::Dict)
    open(filename, "w") do f
        JSON.print(f, analysis, 2)
    end
end

"""
    load_analysis(filename) -> Dict
    تحميل نتائج التحليل
"""
function load_analysis(filename::String)
    if !isfile(filename)
        return Dict()
    end
    return JSON.parsefile(filename)
end

"""
    get_root_info(roots, word) -> Union{Dict, Nothing}
    الحصول على معلومات الجذر لكلمة
"""
function get_root_info(roots::Dict, word::AbstractString)
    # البحث المباشر
    if haskey(roots, word)
        return roots[word]
    end
    
    # البحث في المشتقات
    for (root, info) in roots
        if word in info["derivatives"]
            return info
        end
    end
    
    return nothing
end

"""
    search_by_category(roots, category) -> Vector{Dict}
    البحث بالتصنيف
"""
function search_by_category(roots::Dict, category::AbstractString)
    results = Dict[]
    for (root, info) in roots
        if info["category"] == category
            push!(results, info)
        end
    end
    return results
end

"""
    get_all_roots() -> Vector{String}
    الحصول على جميع الجذور
"""
function get_all_roots(roots::Dict)
    return collect(keys(roots))
end

"""
    get_statistics(roots) -> Dict{String, Int}
    إحصائيات قاعدة البيانات
"""
function get_statistics(roots::Dict)
    stats = Dict{String, Int}()
    for (root, info) in roots
        cat = info["category"]
        stats[cat] = get(stats, cat, 0) + 1
    end
    return stats
end

end # module DataLoader
