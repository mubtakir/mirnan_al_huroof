"""
Integration.jl - الربط المتكامل لجميع مكونات مرنان

التدفق: نص → معالجة → تقطيع → تضمين دلالي → تحليل نحوي → فيزياء → استجابة
"""

module Integration

using LinearAlgebra
using Statistics
using ..Preprocessing
using ..Semantics
using ..Grammar

"""
    WordAnalysis - تحليل كلمة واحدة
"""
struct WordAnalysis
    word::String
    root::String
    semantic_vector::Vector{Float64}
    semantic_similarity_to_context::Float64
    syntactic_role::String
    morphemes::Vector{String}
    frequency::Float64
end

"""
    SentenceAnalysis - تحليل جملة واحدة
"""
struct SentenceAnalysis
    sentence::String
    words::Vector{String}
    sentence_type::String
    syntactic_roles::Vector{String}
    semantic_coherence::Float64
    grammatical_correctness::Float64
end

"""
    SemanticSummary - ملخص دلالي
"""
struct SemanticSummary
    total_words::Int
    unique_words::Int
    avg_semantic_density::Float64
    root_distribution::Dict{String, Int}
    semantic_fields::Vector{String}
end

"""
    GrammarSummary - ملخص نحوي
"""
struct GrammarSummary
    total_sentences::Int
    sentence_types::Dict{String, Int}
    avg_sentence_length::Float64
    syntactic_complexity::Float64
    grammatical_accuracy::Float64
end

"""
    PhysicsSummary - ملخص فيزيائي
"""
struct PhysicsSummary
    total_energy::Float64
    avg_mass::Float64
    phase_coherence::Float64
    semantic_gravity::Float64
    syntactic_oscillation::Float64
end

"""
    PipelineResult - نتيجة التحليل الكامل
"""
struct PipelineResult
    original_text::String
    normalized_text::String
    words::Vector{String}
    sentences::Vector{String}
    word_analyses::Vector{WordAnalysis}
    sentence_analyses::Vector{SentenceAnalysis}
    semantic_summary::SemanticSummary
    grammar_summary::GrammarSummary
    physics_summary::PhysicsSummary
    response::String
end

"""
    MirnanPipeline - خط الأنابيب الرئيسي
"""
struct MirnanPipeline
    preprocessor::TextPreprocessor
    semantic_analyzer::SemanticAnalyzer
    grammar_analyzer::GrammarAnalyzer
end

export MirnanPipeline, PipelineResult, WordAnalysis, SentenceAnalysis,
       SemanticSummary, GrammarSummary, PhysicsSummary,
       analyze_text, analyze_word, analyze_sentence_full,
       generate_response, get_analysis_report

function MirnanPipeline()
    return MirnanPipeline(
        TextPreprocessor(),
        SemanticAnalyzer(),
        GrammarAnalyzer()
    )
end

"""
    analyze_text(pipeline, text) -> PipelineResult
    تحليل نص كامل عبر خط الأنابيب
"""
function analyze_text(pipeline::MirnanPipeline, text::AbstractString)
    # 1. المعالجة المسبقة
    normalized = preprocess_text(text; preprocessor = pipeline.preprocessor)
    
    # 2. استخراج الكلمات والجمل
    words = extract_words(normalized)
    sentences = extract_sentences(normalized)
    
    # 3. تحليل كل كلمة
    word_analyses = WordAnalysis[]
    for word in words
        wa = analyze_word(pipeline, word, words)
        push!(word_analyses, wa)
    end
    
    # 4. تحليل كل جملة
    sentence_analyses = SentenceAnalysis[]
    for sentence in sentences
        sa = analyze_sentence_full(pipeline, sentence, words)
        push!(sentence_analyses, sa)
    end
    
    # 5. الملخص الدلالي
    semantic_summary = compute_semantic_summary(word_analyses)
    
    # 6. الملخص النحوي
    grammar_summary = compute_grammar_summary(sentence_analyses)
    
    # 7. الملخص الفيزيائي
    physics_summary = compute_physics_summary(word_analyses)
    
    # 8. توليد الاستجابة
    response = generate_response(pipeline, word_analyses, sentence_analyses)
    
    return PipelineResult(
        text,
        normalized,
        words,
        sentences,
        word_analyses,
        sentence_analyses,
        semantic_summary,
        grammar_summary,
        physics_summary,
        response
    )
end

"""
    analyze_word(pipeline, word, context) -> WordAnalysis
    تحليل كلمة واحدة
"""
function analyze_word(pipeline::MirnanPipeline, word::AbstractString, context::Vector{String})
    # التضمين الدلالي
    embedding = embed_word(pipeline.semantic_analyzer, word)
    
    # استخراج الجذر
    root_info = extract_root(pipeline.semantic_analyzer, word)
    
    # التشابه مع السياق
    context_similarity = 0.0
    if !isempty(context)
        similarities = Float64[]
        for ctx_word in context
            if ctx_word != word
                ctx_embedding = embed_word(pipeline.semantic_analyzer, ctx_word)
                sim = compute_semantic_similarity(embedding, ctx_embedding)
                push!(similarities, sim)
            end
        end
        if !isempty(similarities)
            context_similarity = sum(similarities) / length(similarities)
        end
    end
    
    # التحليل النحوي
    sentence_analysis = analyze_sentence(pipeline.grammar_analyzer, word)
    
    # استخراج المورفيات (مبسط)
    morphemes = extract_morphemes(word)
    
    # التكرار (مبسط)
    frequency = estimate_frequency(word)
    
    # استخراج الجذر
    root = extract_root(pipeline.semantic_analyzer, word)
    
    return WordAnalysis(
        word,
        root,
        embedding.semantic_vector,
        context_similarity,
        string(sentence_analysis.sentence_type),
        morphemes,
        frequency
    )
end

"""
    analyze_sentence_full(pipeline, sentence, context_words) -> SentenceAnalysis
    تحليل جملة كاملة
"""
function analyze_sentence_full(pipeline::MirnanPipeline, sentence::AbstractString, context_words::Vector{String})
    # تحليل الجملة
    grammar_result = analyze_sentence(pipeline.grammar_analyzer, sentence)
    
    # تحليل الكلمات في الجملة
    words_in_sentence = extract_words(sentence)
    
    # حساب التماسك الدلالي
    semantic_coherence = compute_semantic_coherence(pipeline, words_in_sentence)
    
    # حساب الصحة النحوية (مبسط)
    grammatical_correctness = estimate_grammatical_correctness(grammar_result)
    
    # استخراج الأدوار النحوية
    roles = String[]
    if grammar_result.subject !== nothing
        push!(roles, "Subject")
    end
    if grammar_result.verb !== nothing
        push!(roles, "Verb")
    end
    if grammar_result.object !== nothing
        push!(roles, "Object")
    end
    
    return SentenceAnalysis(
        sentence,
        words_in_sentence,
        string(grammar_result.sentence_type),
        roles,
        semantic_coherence,
        grammatical_correctness
    )
end

"""
    extract_morphemes(word) -> Vector{String}
    استخراج المورفيات (مبسط)
"""
function extract_morphemes(word::AbstractString)
    morphemes = String[]
    chars = collect(word)
    
    # البادئات
    prefixes = ["ال", "ب", "ك", "ف", "ل", "و", "ي", "ت", "ن"]
    for prefix in prefixes
        prefix_chars = collect(prefix)
        if length(chars) > length(prefix_chars) && chars[1:length(prefix_chars)] == prefix_chars
            push!(morphemes, prefix)
            chars = chars[length(prefix_chars)+1:end]
            break
        end
    end
    
    # اللواحق
    suffixes = ["ون", "ين", "ات", "ان", "ية", "ي", "ة", "ه", "هم", "هن", "ك", "نا"]
    for suffix in sort(suffixes; by=length, rev=true)
        suffix_chars = collect(suffix)
        if length(chars) > length(suffix_chars) && chars[end-length(suffix_chars)+1:end] == suffix_chars
            push!(morphemes, suffix)
            chars = chars[1:end-length(suffix_chars)]
            break
        end
    end
    
    # الجذر
    if !isempty(chars)
        push!(morphemes, String(chars))
    end
    
    return morphemes
end

"""
    estimate_frequency(word) -> Float64
    تقدير تكرار الكلمة (مبسط)
"""
function estimate_frequency(word::AbstractString)
    # تقدير مبسط بناءً على طول الكلمة والشعبية
    freq = 1.0 / (1.0 + length(word) * 0.1)
    
    # كلمات شائعة
    common_words = ["ال", "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك"]
    if word in common_words
        freq *= 5.0
    end
    
    return freq
end

"""
    compute_semantic_coherence(pipeline, words) -> Float64
    حساب التماسك الدلالي بين الكلمات
"""
function compute_semantic_coherence(pipeline::MirnanPipeline, words::Vector{String})
    if length(words) < 2
        return 1.0
    end
    
    similarities = Float64[]
    for i in 1:length(words)
        for j in i+1:length(words)
            emb_i = embed_word(pipeline.semantic_analyzer, words[i])
            emb_j = embed_word(pipeline.semantic_analyzer, words[j])
            sim = compute_semantic_similarity(emb_i, emb_j)
            push!(similarities, sim)
        end
    end
    
    return isempty(similarities) ? 0.0 : sum(similarities) / length(similarities)
end

"""
    estimate_grammatical_correctness(grammar_result) -> Float64
    تقدير الصحة النحوية
"""
function estimate_grammatical_correctness(grammar_result)
    # تقدير مبسط
    score = 0.5
    
    # إذا كان النوع معروفاً
    if grammar_result.sentence_type != :UNKNOWN
        score += 0.2
    end
    
    # إذا كان هناك موضوع أو فعل
    if grammar_result.subject !== nothing || grammar_result.verb !== nothing
        score += 0.1
    end
    
    # إذا كانت الجملة لها بناء واضح
    if length(grammar_result.words) >= 2
        score += 0.1
    end
    
    return min(score, 1.0)
end

"""
    compute_semantic_summary(word_analyses) -> SemanticSummary
    حساب الملخص الدلالي
"""
function compute_semantic_summary(word_analyses::Vector{WordAnalysis})
    total = length(word_analyses)
    unique_words = length(Set([wa.word for wa in word_analyses]))
    
    # متوسط الكثافة الدلالية
    densities = [norm(wa.semantic_vector) for wa in word_analyses]
    avg_density = isempty(densities) ? 0.0 : sum(densities) / length(densities)
    
    # توزيع الجذور
    root_dist = Dict{String, Int}()
    for wa in word_analyses
        root = wa.root
        root_dist[root] = get(root_dist, root, 0) + 1
    end
    
    # الحقول الدلالية (مبسط)
    semantic_fields = identify_semantic_fields(word_analyses)
    
    return SemanticSummary(
        total,
        unique_words,
        avg_density,
        root_dist,
        semantic_fields
    )
end

"""
    identify_semantic_fields(word_analyses) -> Vector{String}
    تحديد الحقول الدلالية
"""
function identify_semantic_fields(word_analyses::Vector{WordAnalysis})
    fields = String[]
    
    # كلمات متعلقة بالعلم والمعرفة
    knowledge_words = ["علم", "معرفة", "فهم", "تعلم", "دراسة", "بحث", "اكتشاف"]
    
    # كلمات متعلقة بالطبيعة
    nature_words = ["طبيعة", "سماء", "أرض", "ماء", "نار", "هواء", "شمس", "قمر"]
    
    # كلمات متعلقة بالمجتمع
    society_words = ["مجتمع", "ثقافة", "تاريخ", "فن", "أدب", "لغة"]
    
    for wa in word_analyses
        if wa.word in knowledge_words && !("المعرفة" in fields)
            push!(fields, "المعرفة")
        elseif wa.word in nature_words && !("الطبيعة" in fields)
            push!(fields, "الطبيعة")
        elseif wa.word in society_words && !("المجتمع" in fields)
            push!(fields, "المجتمع")
        end
    end
    
    return fields
end

"""
    compute_grammar_summary(sentence_analyses) -> GrammarSummary
    حساب الملخص النحوي
"""
function compute_grammar_summary(sentence_analyses::Vector{SentenceAnalysis})
    total = length(sentence_analyses)
    
    # أنواع الجمل
    type_dist = Dict{String, Int}()
    for sa in sentence_analyses
        type_dist[sa.sentence_type] = get(type_dist, sa.sentence_type, 0) + 1
    end
    
    # متوسط طول الجملة
    lengths = [length(sa.words) for sa in sentence_analyses]
    avg_length = isempty(lengths) ? 0.0 : sum(lengths) / length(lengths)
    
    # التعقيد النحوي (مبسط)
    complexity = 0.0
    for sa in sentence_analyses
        complexity += length(sa.syntactic_roles)
    end
    avg_complexity = total > 0 ? complexity / total : 0.0
    
    # الدقة النحوية
    accuracies = [sa.grammatical_correctness for sa in sentence_analyses]
    avg_accuracy = isempty(accuracies) ? 0.0 : sum(accuracies) / length(accuracies)
    
    return GrammarSummary(
        total,
        type_dist,
        avg_length,
        avg_complexity,
        avg_accuracy
    )
end

"""
    compute_physics_summary(word_analyses) -> PhysicsSummary
    حساب الملخص الفيزيائي
"""
function compute_physics_summary(word_analyses::Vector{WordAnalysis})
    # الطاقة الكلية
    total_energy = 0.0
    masses = Float64[]
    coherences = Float64[]
    
    for wa in word_analyses
        # تقدير الكتلة والطاقة
        vec_norm = norm(wa.semantic_vector)
        energy = vec_norm^2
        mass = vec_norm
        
        total_energy += energy
        push!(masses, mass)
        
        # تماسك الطور
        push!(coherences, wa.semantic_similarity_to_context)
    end
    
    avg_mass = isempty(masses) ? 0.0 : sum(masses) / length(masses)
    avg_coherence = isempty(coherences) ? 0.0 : sum(coherences) / length(coherences)
    
    # الجاذبية الدلالية (تقريب)
    semantic_gravity = total_energy / (length(word_analyses) + 1)
    
    # التذبذب النحوي (تقريب)
    syntactic_oscillation = avg_coherence * 2.0
    
    return PhysicsSummary(
        total_energy,
        avg_mass,
        avg_coherence,
        semantic_gravity,
        syntactic_oscillation
    )
end

"""
    generate_response(pipeline, word_analyses, sentence_analyses) -> String
    توليد استجابة مبنية على التحليل
"""
function generate_response(pipeline::MirnanPipeline, word_analyses::Vector{WordAnalysis}, sentence_analyses::Vector{SentenceAnalysis})
    if isempty(word_analyses)
        return "لا توجد كلمات للتحليل."
    end
    
    # تحليل الكلمات الرئيسية
    key_words = sort(word_analyses; by=wa -> norm(wa.semantic_vector), rev=true)
    top_words = key_words[1:min(3, length(key_words))]
    
    # بناء الاستجابة
    response_parts = String[]
    
    # معلومات عامة
    push!(response_parts, "تم تحليل $(length(word_analyses)) كلمة في $(length(sentence_analyses)) جملة.")
    
    # الكلمات الرئيسية
    if !isempty(top_words)
        key_word_str = join([wa.word for wa in top_words], "، ")
        push!(response_parts, "الكلمات الرئيسية: $key_word_str")
    end
    
    # معلومات دلالية
    if !isempty(word_analyses)
        avg_sim = mean([wa.semantic_similarity_to_context for wa in word_analyses])
        push!(response_parts, "متوسط التشابه الدلالي: $(round(avg_sim * 100, digits=1))%")
    end
    
    # معلومات نحوية
    if !isempty(sentence_analyses)
        types = Set([sa.sentence_type for sa in sentence_analyses])
        push!(response_parts, "أنواع الجمل: $(join(types, "، "))")
    end
    
    return join(response_parts, "\n")
end

"""
    get_analysis_report(result::PipelineResult) -> String
    تقرير تحليلي مفصل
"""
function get_analysis_report(result::PipelineResult)
    report = IOBuffer()
    
    println(report, "═══════════════════════════════════════════")
    println(report, "          تقرير التحليل الشامل")
    println(report, "═══════════════════════════════════════════")
    println(report, "")
    println(report, "النص الأصلي: ", result.original_text)
    println(report, "النص المُعالَج: ", result.normalized_text)
    println(report, "")
    
    println(report, "─── الإحصائيات العامة ───")
    println(report, "عدد الكلمات: ", length(result.words))
    println(report, "عدد الجمل: ", length(result.sentences))
    println(report, "الكلمات الفريدة: ", result.semantic_summary.unique_words)
    println(report, "")
    
    println(report, "─── الملخص الدلالي ───")
    println(report, "متوسط الكثافة الدلالية: ", round(result.semantic_summary.avg_semantic_density, digits=3))
    if !isempty(result.semantic_summary.semantic_fields)
        println(report, "الحقول الدلالية: ", join(result.semantic_summary.semantic_fields, "، "))
    end
    println(report, "")
    
    println(report, "─── الملخص النحوي ───")
    println(report, "أنواع الجمل: ", result.grammar_summary.sentence_types)
    println(report, "متوسط طول الجملة: ", round(result.grammar_summary.avg_sentence_length, digits=1))
    println(report, "الدقة النحوية: ", round(result.grammar_summary.grammatical_accuracy * 100, digits=1), "%")
    println(report, "")
    
    println(report, "─── الملخص الفيزيائي ───")
    println(report, "الطاقة الكلية: ", round(result.physics_summary.total_energy, digits=3))
    println(report, "متوسط الكتلة: ", round(result.physics_summary.avg_mass, digits=3))
    println(report, "تماسك الطور: ", round(result.physics_summary.phase_coherence, digits=3))
    println(report, "")
    
    println(report, "─── الاستجابة ───")
    println(report, result.response)
    
    return String(take!(report))
end

end # module Integration
