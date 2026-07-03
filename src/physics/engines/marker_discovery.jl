"""
محرك الاكتشاف الذاتي للمفاتيح — Marker Discovery Engine.
يقوم بتحليل نصوص الكوربس التدريبي واستخلاص الكلمات الوظيفية والمفاتيح اللوجستية
(أدوات الاستنطاق، النفي، والسببية) تلقائياً بناءً على إنتروبيا السياق ونظرية المعلومات.
"""
module MarkerDiscoveryModule

using JSON, Statistics

export DiscoveredMarker, discover_markers, save_discovered_markers, load_discovered_markers

struct DiscoveredMarker
    word::String
    category::String    # "question", "negation", "causal", "connector"
    entropy::Float64
    frequency::Int
    confidence::Float64
end

"""
    _tokenize_with_boundaries(text::AbstractString) -> Vector{String}

تقسيم النص إلى كلمات مع الحفاظ على علامات الترقيم الهيكلية كرموز منفصلة.
"""
function _tokenize_with_boundaries(text::AbstractString)::Vector{String}
    # تنظيف وتجهيز النص
    cleaned = replace(String(text), r"([؟\.\!\?،؛:])" => s" \1 ")
    raw_tokens = split(cleaned)
    tokens = String[]
    for t in raw_tokens
        t_clean = strip(t)
        if !isempty(t_clean)
            push!(tokens, String(t_clean))
        end
    end
    return tokens
end

"""
    discover_markers(texts::Vector{String}, vocab::Dict{String,Int}, id2word::Dict{Int,String}) -> Dict{String, DiscoveredMarker}

تحليل نصوص الكوربس واكتشاف المفاتيح الإستراتيجية تلقائياً باستخدام إنتروبيا السياق.
"""
function discover_markers(texts::Vector{String}, vocab::Dict{String,Int}, id2word::Dict{Int,String})
    # 1. تجميع الإحصائيات وسياق الجوار لكل كلمة في المعجم
    left_contexts = Dict{String, Dict{String, Int}}()
    right_contexts = Dict{String, Dict{String, Int}}()
    word_freqs = Dict{String, Int}()
    
    # لتصنيف أدوات الاستفهام: تتبع أول كلمة في الجمل التي تنتهي بـ "؟"
    first_word_in_question_counts = Dict{String, Int}()
    total_sentence_starts = Dict{String, Int}()
    
    # لتصنيف أدوات السببية: تتبع الكلمات التي تفصل بين علامات الترقيم (الفاصلة أو الفاصلة المنقوطة)
    clause_connect_counts = Dict{String, Int}()

    for paragraph in texts
        tokens = _tokenize_with_boundaries(paragraph)
        N = length(tokens)
        N == 0 && continue
        
        # تقسيم الفقرة إلى جمل بناءً على النقاط وعلامات الاستفهام
        sentence_start_idx = 1
        for i in 1:N
            token = tokens[i]
            
            # تحديث تكرار الكلمة
            if !haskey(vocab, token) && i < N && tokens[i] != "؟"
                # لا نحسب علامات الترقيم كعلامات دلالية في التكرار الرئيسي
            end
            
            if token in (".", "؟", "!", "?")
                # نهاية جملة
                if sentence_start_idx < i
                    start_tok = tokens[sentence_start_idx]
                    total_sentence_starts[start_tok] = get(total_sentence_starts, start_tok, 0) + 1
                    if token == "؟" || token == "?"
                        first_word_in_question_counts[start_tok] = get(first_word_in_question_counts, start_tok, 0) + 1
                    end
                end
                sentence_start_idx = i + 1
            end
            
            # رصد الجيران لحساب الإنتروبيا
            if haskey(vocab, token)
                word_freqs[token] = get(word_freqs, token, 0) + 1
                
                # الجار الأيسر
                if i > 1
                    left_tok = tokens[i-1]
                    if !haskey(left_contexts, token)
                        left_contexts[token] = Dict{String, Int}()
                    end
                    left_contexts[token][left_tok] = get(left_contexts[token], left_tok, 0) + 1
                end
                
                # الجار الأيمن
                if i < N
                    right_tok = tokens[i+1]
                    if !haskey(right_contexts, token)
                        right_contexts[token] = Dict{String, Int}()
                    end
                    right_contexts[token][right_tok] = get(right_contexts[token], right_tok, 0) + 1
                end
                
                # رصد روابط الجمل الفرعية
                if i > 1 && i < N && tokens[i-1] in ("،", "؛", ",")
                    clause_connect_counts[token] = get(clause_connect_counts, token, 0) + 1
                end
            end
        end
    end
    
    # 2. حساب إنتروبيا السياق لكل كلمة
    discovered = Dict{String, DiscoveredMarker}()
    
    for (word, freq) in word_freqs
        freq < 3 && continue # تصفية الكلمات النادرة لضمان موثوقية الإحصاء
        
        # إنتروبيا السياق الأيسر
        h_l = 0.0
        if haskey(left_contexts, word)
            counts = left_contexts[word]
            total = sum(values(counts))
            if total > 0
                for c in values(counts)
                    p = c / total
                    h_l -= p * log2(p)
                end
            end
        end
        
        # إنتروبيا السياق الأيمن
        h_r = 0.0
        if haskey(right_contexts, word)
            counts = right_contexts[word]
            total = sum(values(counts))
            if total > 0
                for c in values(counts)
                    p = c / total
                    h_r -= p * log2(p)
                end
            end
        end
        
        avg_entropy = (h_l + h_r) / 2.0
        
        # الكلمة الوظيفية (Marker) تتميز بإنتروبيا سياق عالية وتكرار معقول
        # في الكوربس الصغير، نعتبر الإنتروبيا > 0.8 مؤشراً جيداً
        if avg_entropy >= 0.8
            # 3. تحديد الصنف (Category) واليقين (Confidence)
            category = "connector"
            confidence = 0.5
            
            # أ. فحص علامة الاستفهام
            q_starts = get(first_word_in_question_counts, word, 0)
            total_starts = get(total_sentence_starts, word, 0)
            if q_starts >= 2 && (q_starts / max(1, total_starts)) >= 0.35
                category = "question"
                confidence = q_starts / max(1, total_starts)
            # ب. فحص النفي (كلمات هيكلية معروفة أو مقترنة بالنفي)
            elseif word in ("لا", "ليس", "ليست", "غير", "not", "no", "never")
                category = "negation"
                confidence = 0.95
            # ج. فحص السببية
            elseif word in ("لأن", "لان", "بسبب", "لذلك", "إذ", "اذ", "because", "therefore")
                category = "causal"
                confidence = 0.90
            elseif get(clause_connect_counts, word, 0) >= 2
                # الكلمة تعمل كرابط جمل سببية أو تعليلية
                category = "causal"
                confidence = 0.70
            end
            
            discovered[word] = DiscoveredMarker(word, category, avg_entropy, freq, confidence)
        end
    end
    
    return discovered
end

"""
    save_discovered_markers(file_path::AbstractString, markers::Dict{String, DiscoveredMarker})

حفظ العلامات المكتشفة ذاتياً إلى ملف JSON.
"""
function save_discovered_markers(file_path::AbstractString, markers::Dict{String, DiscoveredMarker})
    data = Dict{String, Any}()
    for (w, m) in markers
        data[w] = Dict(
            "category" => m.category,
            "entropy" => m.entropy,
            "frequency" => m.frequency,
            "confidence" => m.confidence
        )
    end
    open(file_path, "w") do io
        JSON.print(io, data, 4)
    end
end

"""
    load_discovered_markers(file_path::AbstractString) -> Dict{String, String}

تحميل العلامات المكتشفة ذاتياً كقاموس مبسط (كلمة -> صنف).
"""
function load_discovered_markers(file_path::AbstractString)::Dict{String, String}
    res = Dict{String, String}()
    if isfile(file_path)
        try
            data = JSON.parsefile(file_path)
            if data isa AbstractDict
                for (w, details) in data
                    if details isa AbstractDict && haskey(details, "category")
                        res[w] = String(details["category"])
                    end
                end
            end
        catch e
            @warn "Failed to load discovered markers from $file_path: $e"
        end
    end
    return res
end

end # module MarkerDiscoveryModule
