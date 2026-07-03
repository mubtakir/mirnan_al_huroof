"""
حقل الجذر — RootField.

أداة للباحث اللغوي والأديب والنحوي.
يكتب المستخدم كلمة واحدة، فيرد النظام بكل كلمات الجذر نفسه من المعجم،
مرتبة حسب قوة التناغم الصرفي (تشابه المتجه الطوري).

مثال: "نور" → "نوار، منار، منير، أنور، نيران، نورا، تنوير، منورة..."
"""
module RootFieldModule

using LinearAlgebra
using ..WordPhysics: _extract_root_light, compute_word_phase_vector
using ..Constants: PHASE_DIM

export find_root_family, root_field_report, ROOT_FIELD_FORMATS

const ROOT_FIELD_FORMATS = ["list", "detailed", "poetic"]

"""
    find_root_family(word::String, vocab::Dict{String,Int};
                     max_results::Int=30, min_similarity::Float64=0.05) -> Vector{Tuple{String,Float64,Vector{Char}}}

يبحث في المعجم عن كل الكلمات التي تشترك في الجذر مع الكلمة المعطاة.
يرجع قائمة (كلمة, درجة التشابه الطوري, حروف الجذر) مرتبة تنازلياً.
"""
function find_root_family(word::String, vocab::Dict{String,Int};
                           max_results::Int=30, min_similarity::Float64=0.05)
    # استخراج جذر الكلمة المدخلة
    root_chars = try
        _extract_root_light(word)
    catch e
        @debug "Root field: extract root failed for '$word': $e"
        Char[]
    end

    isempty(root_chars) && return Tuple{String,Float64,Vector{Char}}[]

    root_str = join(root_chars)

    # احصل على المتجه الطوري للكلمة المدخلة
    word_pv = try
        Float64.(compute_word_phase_vector(word))
    catch e
        @debug "Root field: PV computation failed for '$word': $e"
        zeros(Float64, PHASE_DIM)
    end
    word_norm = norm(word_pv)

    # افحص كل كلمة في المعجم
    results = Tuple{String,Float64,Vector{Char}}[]
    for (candidate, _) in vocab
        candidate == word && continue

        # استخراج جذر المرشح
        cand_root = try
            _extract_root_light(candidate)
        catch e
            @debug "Root field: extract root failed for '$candidate': $e"
            Char[]
        end
        isempty(cand_root) && continue

        cand_root_str = join(cand_root)

        # حساب التشابه الطوري (يعكس قرب الكلمتين في الفضاء)
        cand_pv = try
            Float64.(compute_word_phase_vector(candidate))
        catch e
            @debug "Root field: PV computation failed for '$candidate': $e"
            zeros(Float64, PHASE_DIM)
        end
        cand_norm = norm(cand_pv)

        sim = 0.0
        if word_norm > 1e-10 && cand_norm > 1e-10
            sim = max(0.0, dot(word_pv, cand_pv) / (word_norm * cand_norm))
        end

        # فلترة: إما نفس الجذر، أو تشابه عالٍ
        if root_str == cand_root_str || sim >= min_similarity
            push!(results, (candidate, sim, cand_root))
        end
    end

    # ترتيب: الكلمات ذات نفس الجذر أولاً، ثم حسب التشابه
    sort!(results; by=x -> (join(x[3]) == root_str ? 1.0 : 0.5) + x[2], rev=true)
    return results[1:min(max_results, end)]
end

"""
    root_field_report(word::String, vocab::Dict{String,Int};
                      format::String="detailed", max_results::Int=30) -> String

يولد تقريراً عن حقل الجذر لكلمة. يدعم 3 صيغ:

- "list":     قائمة كلمات فقط
- "detailed": مع الجذر والتشابه
- "poetic":   صيغة أدبية للشعراء
"""
function root_field_report(word::String, vocab::Dict{String,Int};
                            format::String="detailed", max_results::Int=30)
    results = find_root_family(word, vocab; max_results=max_results)
    isempty(results) && return "لم أجد كلمات من جذر [$word] في المعجم."

    root_of_input = try join(_extract_root_light(word)) catch e; @debug "Root field: extract root failed for report: $e"; "" end

    # فصل الكلمات: نفس الجذر بالضبط vs جذور قريبة
    same_root = filter(r -> join(r[3]) == root_of_input, results)
    related = filter(r -> join(r[3]) != root_of_input, results)

    if format == "list"
        words = [r[1] for r in same_root]
        related_words = [r[1] for r in related]
        out = "حقل [$word] (الجذر: $root_of_input):\n"
        isempty(words) || (out *= join(words, "، ") * "\n")
        isempty(related_words) || (out *= "كلمات قريبة: " * join(related_words, "، "))
        return out

    elseif format == "poetic"
        words = [r[1] for r in same_root]
        isempty(words) && return "ما من كلمة تجاور [$word] في جذرها."
        lines = String[
            "يا سائلاً عن جذر [$word] وجوارها",
            "هذي الحقول تناغمت أزهارها",
            "",
        ]
        if length(words) >= 6
            push!(lines, join(words[1:div(length(words), 2)], "  —  "))
            push!(lines, "")
            push!(lines, join(words[div(length(words), 2)+1:end], "  —  "))
        else
            push!(lines, join(words, "  •  "))
        end
        push!(lines, "")
        push!(lines, "الجذر: $root_of_input  |  الكلمات: $(length(words))")
        return join(lines, "\n")

    else  # detailed (افتراضي)
        lines = String["═══ حقل الجذر: [$word] ═══",
                       "الجذر الثلاثي/الرباعي: $root_of_input",
                       "عدد الكلمات في الحقل: $(length(same_root))",
                       ""]
        if !isempty(same_root)
            push!(lines, "─ كلمات من نفس الجذر ─")
            for (i, (w, sim, root)) in enumerate(same_root)
                sim_pct = round(sim * 100; digits=1)
                bar = repeat("█", ceil(Int, sim * 20))
                push!(lines, "  $(i). $w  [$sim_pct%] $bar")
            end
        end
        if !isempty(related)
            push!(lines, "")
            push!(lines, "─ كلمات من جذور قريبة ─")
            for (i, (w, sim, root)) in enumerate(related)
                sim_pct = round(sim * 100; digits=1)
                push!(lines, "  $(i). $w  [$sim_pct%]  (جذر: $(join(root)))")
            end
        end
        return join(lines, "\n")
    end
end

end # module RootFieldModule
