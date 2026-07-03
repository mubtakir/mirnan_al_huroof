#!/usr/bin/env julia
# 📊 تحليل الكلمات لتوازن بيانات تدريب مرنان V8
# يقوم بقراءة وتحليل ملفات الكوربس وموازنتها متطابقاً مع منطق train.jl

using Printf

# ═══════════ تطبيع النص العربي — متطابق مع train.jl ═══════════
const _AR_NORMALIZE_MAP = Dict(
    'آ'=>'ا','أ'=>'ا','إ'=>'ا','ؤ'=>'و','ئ'=>'ي','ة'=>'ه','ى'=>'ي',
)
const _AR_TATWEEL = 0x0640

function _normalize_arabic_text(text::AbstractString)
    result = Char[]
    for ch in text
        if haskey(_AR_NORMALIZE_MAP, ch)
            push!(result, _AR_NORMALIZE_MAP[ch])
        elseif Int(ch) == _AR_TATWEEL
            continue
        else
            push!(result, ch)
        end
    end
    return String(result)
end

function _is_meaningful_word(w::AbstractString)
    n = length(w)
    n < 2 && return false
    all(c -> isdigit(c) || c in Set(['.',',','-','/','\\','(',')','[',']','{','}',':',';','"','\'','«','»']), w) && return false
    occursin(r"^\d+$", w) && return false
    occursin(r"[\[\]]", w) && return false
    return true
end

const _PUNCT_EDGE = Set(['.', ',', '،', ':', '؛', '?', '؟', '!', ')', '(', '"', '\'',
                          '«', '»', '-', '…', '\u200F', '\u200E', '\u00AD'])

function _strip_punct_boundary(word::AbstractString)
    chars = collect(word)
    while !isempty(chars) && !isempty(chars) && first(chars) in _PUNCT_EDGE
        popfirst!(chars)
    end
    while !isempty(chars) && last(chars) in _PUNCT_EDGE
        pop!(chars)
    end
    return String(chars)
end

function _strip_dialogue_labels(text::AbstractString)
    return replace(text, r"^\s*\[[^\]]+\]:?\s*"m => "")
end

function strip_code_blocks(text::AbstractString)
    cleaned = replace(text, r"```[\s\S]*?```" => "")
    cleaned = replace(cleaned, r"`[^`\n]+`" => "")
    return cleaned
end

function should_skip_path(path::AbstractString)
    parts = splitpath(path)
    return any(p -> p in ("rules", ".git", ".DS_Store", "cleaned", "model"), parts)
end

format_comma(n) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+(?![0-9]))" => ",")

function analyze_text_files(root_path::String)
    # word => Dict("total_count" => 0, "files" => Dict(relative_path => count))
    word_stats = Dict{String, Dict{String, Any}}()
    files_info = Vector{Dict{String, Any}}()
    
    total_files = 0
    total_words = 0
    
    println("="^80)
    println("📊 بدء تحليل الملفات النصية بلغة جوليا لنموذج مرنان...")
    println("="^80)
    
    for (root, dirs, files) in walkdir(root_path)
        should_skip_path(root) && continue
        for file in files
            ext = lowercase(splitext(file)[2])
            ext in (".txt", ".md", ".csv", ".tsv") || continue
            
            file_path = joinpath(root, file)
            relative_path = relpath(file_path, root_path)
            
            try
                content = read(file_path, String)
                content = _normalize_arabic_text(content)
                content = _strip_dialogue_labels(content)
                content = strip_code_blocks(content)
                
                # Extract words based on format
                raw_words = String[]
                if ext == ".csv" || ext == ".tsv"
                    sep = ext == ".csv" ? "," : "\t"
                    for line in eachline(IOBuffer(content))
                        isempty(strip(line)) && continue
                        parts = split(line, sep)
                        txt = strip(length(parts) >= 2 ? parts[end] : parts[1], ['"', '\''])
                        for w in split(txt)
                            push!(raw_words, String(w))
                        end
                    end
                else
                    for w in split(content)
                        push!(raw_words, String(w))
                    end
                end
                
                # Filter and normalize words
                file_words = String[]
                for w in raw_words
                    cleaned_w = _strip_punct_boundary(strip(w))
                    if _is_meaningful_word(cleaned_w)
                        push!(file_words, cleaned_w)
                    end
                end
                
                # Count frequencies in this file
                word_counts = Dict{String, Int}()
                for w in file_words
                    word_counts[w] = get(word_counts, w, 0) + 1
                end
                
                # Update global stats
                for (word, count) in word_counts
                    if !haskey(word_stats, word)
                        word_stats[word] = Dict{String, Any}(
                            "total_count" => 0,
                            "files" => Dict{String, Int}()
                        )
                    end
                    word_stats[word]["total_count"] += count
                    word_stats[word]["files"][relative_path] = count
                end
                
                push!(files_info, Dict{String, Any}(
                    "path" => relative_path,
                    "total_words" => length(file_words),
                    "unique_words" => length(word_counts)
                ))
                
                total_files += 1
                total_words += length(file_words)
                
                println("✅ تم تحليل: $relative_path | كلمات: $(format_comma(length(file_words))) | كلمات فريدة: $(format_comma(length(word_counts)))")
                
            catch e
                println("❌ خطأ في قراءة الملف $relative_path: $e")
            end
        end
    end
    
    return word_stats, files_info, total_files, total_words
end

function print_detailed_report(top_words, files_info, total_files, total_words)
    println("\n" * "="^80)
    println("📈 تقرير تحليل الكلمات الأكثر تكراراً (طبعة جوليا)")
    println("="^80)
    println("📁 عدد الملفات المحللة: $total_files")
    println("📝 إجمالي الكلمات: $(format_comma(total_words))")
    println("="^80)
    
    println("\n🔝 أعلى 30 كلمة تكراراً:\n")
    println(rpad("الترتيب", 8) * rpad("الكلمة", 25) * rpad("التكرار الكلي", 15) * rpad("عدد الملفات", 15) * rpad("نسبة الانتشار", 15))
    println("-"^80)
    
    for i in 1:min(30, length(top_words))
        word, stats = top_words[i]
        num_files = length(stats["files"])
        spread_percentage = total_files > 0 ? (num_files / total_files * 100) : 0.0
        
        row_str = rpad(string(i), 8) * 
                  rpad(word, 25) * 
                  rpad(format_comma(stats["total_count"]), 15) * 
                  rpad(string(num_files), 15) * 
                  @sprintf("%5.1f%%", spread_percentage)
        println(row_str)
    end
    
    # تفاصيل أكثر الكلمات انتشاراً في الملفات
    println("\n" * "="^80)
    println("📋 تحليل انتشار الكلمات في الملفات:")
    println("="^80)
    
    for i in 1:min(10, length(top_words))
        word, stats = top_words[i]
        println("\n$i. كلمة '$word' - إجمالي التكرار: $(format_comma(stats["total_count"]))")
        println("   توجد في $(length(stats["files"])) ملف:")
        
        # sort files by count descending
        sorted_files = sort(collect(stats["files"]), by=x->x[2], rev=true)
        for j in 1:min(5, length(sorted_files))
            file_path, count = sorted_files[j]
            println("   • $file_path: $count مرة")
        end
        
        if length(sorted_files) > 5
            println("   • ... و $(length(sorted_files) - 5) ملف آخر")
        end
    end
    
    # إحصائيات عامة عن الملفات
    println("\n" * "="^80)
    println("📊 إحصائيات الملفات:")
    println("="^80)
    
    if !isempty(files_info)
        sorted_by_size = sort(files_info, by=x->x["total_words"], rev=true)
        println("\nأكبر 5 ملفات (عدد الكلمات):")
        for i in 1:min(5, length(sorted_by_size))
            f_info = sorted_by_size[i]
            println("$i. $(f_info["path"]): $(format_comma(f_info["total_words"])) كلمة")
        end
        
        avg_words = total_words / total_files
        @printf("\n📊 متوسط عدد الكلمات في الملف: %.0f كلمة\n", avg_words)
        
        word_counts = [f["total_words"] for f in files_info]
        println("📊 أقل ملف: $(format_comma(minimum(word_counts))) كلمة")
        println("📊 أكبر ملف: $(format_comma(maximum(word_counts))) كلمة")
    end
end

function save_report_to_file(top_words, files_info, total_files, total_words, output_path)
    open(output_path, "w") do io
        println(io, "="^80)
        println(io, "تقرير تحليل الكلمات لتوازن بيانات التدريب (طبعة جوليا)")
        println(io, "="^80 * "\n")
        
        println(io, "عدد الملفات المحللة: $total_files")
        println(io, "إجمالي الكلمات: $(format_comma(total_words))\n")
        
        println(io, "أعلى 50 كلمة تكراراً:")
        println(io, "-"^80)
        println(io, rpad("الترتيب", 8) * rpad("الكلمة", 25) * rpad("التكرار", 15) * rpad("عدد الملفات", 15))
        println(io, "-"^80)
        
        for i in 1:length(top_words)
            word, stats = top_words[i]
            println(io, rpad(string(i), 8) * rpad(word, 25) * rpad(format_comma(stats["total_count"]), 15) * rpad(string(length(stats["files"])), 15))
        end
        
        println(io, "\n\nتفاصيل الكلمات الأكثر انتشاراً:")
        println(io, "="^80)
        
        for i in 1:min(20, length(top_words))
            word, stats = top_words[i]
            println(io, "\nكلمة: $word (التكرار: $(format_comma(stats["total_count"])))")
            sorted_files = sort(collect(stats["files"]), by=x->x[2], rev=true)
            for j in 1:min(10, length(sorted_files))
                file_path, count = sorted_files[j]
                println(io, "  • $file_path: $count")
            end
        end
        
        println(io, "\n\nإحصائيات الملفات:")
        println(io, "="^80)
        sorted_by_size = sort(files_info, by=x->x["total_words"], rev=true)
        
        println(io, "\nجميع الملفات مرتبة حسب الحجم:")
        for i in 1:length(sorted_by_size)
            f_info = sorted_by_size[i]
            println(io, "$i. $(f_info["path"]): $(format_comma(f_info["total_words"])) كلمة")
        end
    end
end

function main()
    root_path = joinpath(@__DIR__, "data")
    if !isdir(root_path)
        println("❌ المسار غير موجود: $root_path")
        return
    end
    
    word_stats, files_info, total_files, total_words = analyze_text_files(root_path)
    
    if total_files == 0
        println("❌ لم يتم العثور على ملفات نصية في المسار المحدد.")
        return
    end
    
    # Sort words by total count descending
    top_words = sort(collect(word_stats), by=x->x[2]["total_count"], rev=true)
    
    print_detailed_report(top_words, files_info, total_files, total_words)
    
    report_path = joinpath(dirname(root_path), "word_analysis_report.txt")
    save_report_to_file(top_words, files_info, total_files, total_words, report_path)
    
    println("\n" * "="^80)
    println("💾 تم حفظ التقرير المفصل في: $report_path")
    println("="^80)
    
    # توصيات لموازنة البيانات
    println("\n🎯 توصيات لموازنة البيانات:")
    println("-"^40)
    
    problematic_words = Tuple{String, Dict{String, Any}}[]
    for (word, stats) in top_words
        if length(stats["files"]) == 1 && stats["total_count"] > 100
            push!(problematic_words, (word, stats))
        end
    end
    
    if !isempty(problematic_words)
        println("\n⚠️ كلمات تظهر بكثرة في ملف واحد فقط (قد تسبب عدم توازن):")
        for i in 1:min(5, length(problematic_words))
            word, stats = problematic_words[i]
            file_name = first(keys(stats["files"]))
            println("   • '$word' تكررت $(stats["total_count"]) مرة في $file_name")
        end
    end
    
    println("\n💡 اقتراحات:")
    println("   1. تأكد من توزيع الكلمات الشائعة بشكل متوازن عبر الملفات")
    println("   2. انتبه للملفات الكبيرة جداً التي قد تهيمن على التدريب")
    println("   3. فكر في تقسيم الملفات الكبيرة إلى أجزاء أصغر متوازنة")
end

main()
