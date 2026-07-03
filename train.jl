#!/usr/bin/env julia
# تدريب كامل لمرنان — يبني ويحفظ المعجم ومصفوفات K
# يدعم قراءة الكوربس من مجلدات مرتبة أبجدياً — كل ملف = وثيقة كاملة
using Pkg; Pkg.activate(@__DIR__)
include(joinpath(@__DIR__, "src", "MirnanNew.jl"))
using .MirnanNew
using JSON, SparseArrays, LinearAlgebra, Logging
using Dates

const PROJECT_DIR = @__DIR__
const DATA_DIR = joinpath(PROJECT_DIR, "data")
const MODEL_DIR = joinpath(PROJECT_DIR, "model")
const KNOWLEDGE_DIR = joinpath(PROJECT_DIR, "knowledge")
const MAJNON_ROOT_DIR = dirname(dirname(PROJECT_DIR))
mkpath(DATA_DIR); mkpath(MODEL_DIR); mkpath(KNOWLEDGE_DIR)

function _safe_rm(path::AbstractString; recursive::Bool=false, retries::Int=5)
    ispath(path) || return true
    last_error = nothing
    for attempt in 1:retries
        try
            rm(path; force=true, recursive=recursive, allow_delayed_delete=true)
            return true
        catch e
            last_error = e
            sleep(0.15 * attempt)
        end
    end
    @warn "Could not remove path; it may be locked by another process" path exception=last_error
    return false
end

function _clean_model_dir!(model_dir::AbstractString)
    skipped = String[]
    if isdir(model_dir)
        for f in readdir(model_dir; join=true)
            _safe_rm(f; recursive=isdir(f)) || push!(skipped, f)
        end
    end
    mkpath(model_dir)
    return skipped
end

# Training text preservation:
# Do not strip Arabic diacritics, tanwin, hamza forms, ta marbuta, alif
# maqsura, or tatweel. These are part of the observed language and must enter
# the vocabulary and co-occurrence matrices exactly as written.

function _strip_diacritics(text::String)
    return text
end

function _normalize_arabic_text(text::String)
    return text
end

function _is_meaningful_word(w::AbstractString)
    n = length(w)
    n < 2 && return false
    all(c -> isdigit(c) || c in Set(['.',',','-','/','\\','(',')','[',']','{','}',':',';','"','\'','«','»']), w) && return false
    occursin(r"^\d+$", w) && return false
    occursin(r"[\[\]]", w) && return false
    return true
end

function _is_arabic_text(text::AbstractString; threshold::Float64=0.3)::Bool
    arabic_count = 0
    total_count = 0
    for ch in text
        if isletter(ch)
            total_count += 1
            if '\u0600' <= ch <= '\u06FF' || '\u0750' <= ch <= '\u077F' ||
               '\uFB50' <= ch <= '\uFDFF' || '\uFE70' <= ch <= '\uFEFF'
                arabic_count += 1
            end
        end
    end
    total_count == 0 && return false
    return (arabic_count / total_count) >= threshold
end

const _CODE_PATH_HINTS = (
    "code_corpus", ".py", ".jl", ".js", ".ts", ".java", ".cpp", ".c", ".h",
    ".rs", ".go", ".rb", ".php", ".cs", ".scala", ".sql", ".sh", ".ps1",
)

const _CODE_LINE_PATTERNS = (
    r"^\s*(def|class|import|from|return|yield|async|await|for|while|if|elif|else|try|except|with)\b",
    r"^\s*(function|module|using|export|struct|mutable struct|abstract type|const|let|begin|end)\b",
    r"^\s*(var|let|const|public|private|static|class|interface|package|namespace)\b",
)

function _looks_like_code_path(path::AbstractString)
    lower_path = lowercase(String(path))
    return any(h -> occursin(h, lower_path), _CODE_PATH_HINTS)
end

function _env_flag(name::String, default::Bool=false)
    raw = lowercase(strip(get(ENV, name, default ? "1" : "0")))
    raw in ("1", "true", "yes", "on", "y") && return true
    raw in ("0", "false", "no", "off", "n") && return false
    return default
end

_include_code_experience_text() = _env_flag("MIRNAN_INCLUDE_CODE_EXPERIENCE_TEXT", false)

function _as_string_vec(value)
    value isa AbstractVector && return String[strip(String(v)) for v in value if !isempty(strip(String(v)))]
    value === nothing && return String[]
    s = strip(String(value))
    return isempty(s) ? String[] : String[s]
end

function _semantic_fact_terms(item::AbstractDict)
    terms = String[]
    append!(terms, _as_string_vec(get(item, "subject", nothing)))
    append!(terms, _as_string_vec(get(item, "object", nothing)))
    append!(terms, _as_string_vec(get(item, "terms", Any[])))
    append!(terms, _as_string_vec(get(item, "focus_terms", Any[])))
    seen = Set{String}()
    out = String[]
    for term in terms
        term in seen && continue
        push!(seen, term)
        push!(out, term)
    end
    return out
end

function _semantic_fact_marker(item::AbstractDict)
    marker = strip(String(get(item, "marker", "")))
    !isempty(marker) && return marker
    relation_type = strip(String(get(item, "relation_type", "")))
    defaults = Dict(
        "analogy" => "يشبه",
        "causal" => "لأن",
        "transform" => "يتحول",
        "need" => "يحتاج",
        "prevention" => "يمنع",
        "difference" => "بين",
        "contradiction" => "ليس",
        "negation" => "ليس",
    )
    return get(defaults, relation_type, relation_type)
end

function _load_semantic_relation_facts(path::AbstractString)
    isfile(String(path)) || return Dict{String,Any}[]
    out = Dict{String,Any}[]
    try
        data = JSON.parsefile(String(path))
        for item in get(data, "records", Any[])
            item isa AbstractDict || continue
            relation_type = strip(String(get(item, "relation_type", "")))
            isempty(relation_type) && continue
            terms = _semantic_fact_terms(item)
            length(terms) < 2 && continue
            record = Dict{String,Any}(String(k) => v for (k, v) in item)
            record["terms"] = terms
            record["marker"] = _semantic_fact_marker(record)
            record["polarity"] = Int(get(record, "polarity", 1)) < 0 ? -1 : 1
            push!(out, record)
        end
    catch e
        @warn "failed to load semantic relation facts" path=String(path) error=e
    end
    return out
end

function _is_majnon_code_seed_doc(content::AbstractString, source_path::AbstractString)
    path = replace(String(source_path), '\\' => '/')
    occursin("majnon_code_corpus", path) && return true
    occursin("[majnon-code-sha1:", String(content)) && return true
    return false
end

function _looks_like_code_content(content::AbstractString)
    hits = 0
    checked = 0
    for raw_line in split(String(content), '\n')
        line = strip(raw_line)
        isempty(line) && continue
        checked += 1
        any(p -> occursin(p, line), _CODE_LINE_PATTERNS) && (hits += 1)
        occursin(r"[{}();=]", line) && occursin(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", line) && (hits += 1)
        checked >= 200 && break
    end
    return hits >= 3
end

function _collect_code_blocks!(code_blocks::Vector{String}, content::String, source_path::String)
    fenced = MirnanNew.Physics.MathBridgeModule.extract_code_blocks(content)
    append!(code_blocks, fenced)
    if _is_majnon_code_seed_doc(content, source_path) && !isempty(fenced)
        return code_blocks
    end

    if _looks_like_code_path(source_path) || _looks_like_code_content(content)
        stripped = strip(content)
        isempty(stripped) || push!(code_blocks, stripped)
    end
    return code_blocks
end

# علامات الترقيم التي تُنزع من أطراف الكلمة (بداية ونهاية)
const _PUNCT_EDGE = Set(['.', ',', '،', ':', '؛', '?', '؟', '!', ')', '(', '"', '\'',
                          '«', '»', '-', '…', '\u200F', '\u200E', '\u00AD'])

"""
    _strip_punct_boundary(word::AbstractString) -> String

ينزع علامات الترقيم من بداية ونهاية الكلمة فقط (وليس من وسطها).
مثال: \"نور،\" → \"نور\"، \"(فهم)\" → \"فهم\"، \"اين.\" → \"اين\"
"""
function _strip_punct_boundary(word::AbstractString)
    chars = collect(word)
    while !isempty(chars) && first(chars) in _PUNCT_EDGE
        popfirst!(chars)
    end
    while !isempty(chars) && last(chars) in _PUNCT_EDGE
        pop!(chars)
    end
    return String(chars)
end

"""
    _strip_dialogue_labels(text::String) -> String

يحذف علامات المتحدثين في الحوارات مثل:
  `[صديق 1]: `, `[الأم]: `, `[الذات الناقدة]: `
ويُبقي فقط على النص الحواري الفعلي.
"""
function _strip_dialogue_labels(text::String)
    # Remove leading dialogue tags: [Speaker Name] or [Speaker Name]: or [Speaker Name Number]:
    return replace(text, r"^\s*\[[^\]]+\]:?\s*"m => "")
end

# ═══════════ قراءة الكوربس ═══════════

"""
    load_corpus_from_dirs(root_dir; granularity=:document) -> Vector{String}

يقرأ كل الملفات النصية من المجلدات الفرعية مرتبة أبجدياً.

granularity:
  :document  — كل ملف = وثيقة واحدة متصلة (تحافظ على السرد والسياق)
  :paragraph — كل فقرة (مفصولة بسطر فارغ) = وثيقة
  :line      — كل سطر = وثيقة (للبيانات المهيكلة فقط)
"""
function load_corpus_from_dirs(root_dir::String; granularity::Symbol=:document,
                               code_blocks::Union{Nothing,Vector{String}}=nothing)
    texts = String[]
    if !isdir(root_dir)
        @warn "مجلد الكوربس غير موجود: $root_dir"
        return texts
    end
    _read_dir_recursive!(texts, root_dir, basename(root_dir), 0;
                         granularity=granularity, code_blocks=code_blocks)
    return texts
end

function load_corpus_from_single(file_path::String; granularity::Symbol=:document,
                                 code_blocks::Union{Nothing,Vector{String}}=nothing,
                                 metadata::Union{Nothing,Vector{Dict{String,Any}}}=nothing)
    texts = String[]
    if !isfile(file_path)
        @warn "ملف الكوربس غير موجود: $file_path"
        return texts
    end
    raw_content = read(file_path, String)
    code_blocks !== nothing && _collect_code_blocks!(code_blocks, raw_content, file_path)
    if !_include_code_experience_text() && _is_majnon_code_seed_doc(raw_content, file_path)
        println("  📄 ملف خبرة كود | وُجه إلى al_code فقط: $(basename(file_path))")
        return texts
    end
    content = _normalize_arabic_text(raw_content)
    content = _strip_dialogue_labels(content)

    fname = basename(file_path)
    if granularity == :document
        doc = join([strip(l) for l in split(content, r"\n") if !isempty(strip(l))], " ")
        if !isempty(doc)
            push!(texts, doc)
            if metadata !== nothing
                push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => 1))
            end
        end
    elseif granularity == :paragraph
        paras = split(content, r"\n\s*\n")
        idx = 1
        for p in paras
            para = join([strip(l) for l in split(p, r"\n") if !isempty(strip(l))], " ")
            if !isempty(para)
                push!(texts, para)
                if metadata !== nothing
                    push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => idx))
                end
                idx += 1
            end
        end
    else
        idx = 1
        for line in eachline(IOBuffer(content))
            txt = strip(line)
            if !isempty(txt)
                push!(texts, txt)
                if metadata !== nothing
                    push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => idx))
                end
                idx += 1
            end
        end
    end
    println("  📄 ملف واحد | وثائق: $(length(texts))")
    return texts
end

function _agent_training_corpus_candidates()
    candidates = String[]
    if haskey(ENV, "MAJNON_MIRNAN_TRAINING_CORPUS")
        push!(candidates, ENV["MAJNON_MIRNAN_TRAINING_CORPUS"])
    end
    append!(candidates, String[
        joinpath(MAJNON_ROOT_DIR, ".agent_workspace", ".agent", "mirnan", "training_corpus.txt"),
        joinpath(MAJNON_ROOT_DIR, ".agent", "mirnan", "training_corpus.txt"),
        joinpath(PROJECT_DIR, "basil_agent", "agent_workspace", ".agent", "mirnan", "training_corpus.txt"),
    ])

    unique_paths = String[]
    seen = Set{String}()
    for path in candidates
        isempty(strip(path)) && continue
        normalized = try realpath(path) catch; abspath(normpath(path)) end
        normalized in seen && continue
        push!(seen, normalized)
        push!(unique_paths, path)
    end
    unique_paths
end

function load_agent_experience_corpus(; granularity::Symbol=:document,
                                      code_blocks::Union{Nothing,Vector{String}}=nothing,
                                      metadata::Union{Nothing,Vector{Dict{String,Any}}}=nothing)
    texts = String[]
    for path in _agent_training_corpus_candidates()
        if isfile(path) && filesize(path) > 0
            println("  -> reading Majnon experience corpus: $path")
            append!(texts, load_corpus_from_single(path;
                                                   granularity=granularity,
                                                   code_blocks=code_blocks,
                                                   metadata=metadata))
        end
    end
    texts
end

function _read_dir_recursive!(texts::Vector{String}, dir_path::String, label::String, n_files::Int;
                                granularity::Symbol=:document,
                                code_blocks::Union{Nothing,Vector{String}}=nothing,
                                metadata::Union{Nothing,Vector{Dict{String,Any}}}=nothing)
    entries = sort(readdir(dir_path; join=false))
    for entry_name in entries
        entry_path = joinpath(dir_path, entry_name)
        if isdir(entry_path)
            _read_dir_recursive!(texts, entry_path, label, n_files;
                                 granularity=granularity, code_blocks=code_blocks,
                                 metadata=metadata)
            continue
        end
        !isfile(entry_path) && continue
        ext = lowercase(splitext(entry_name)[2])
        ext in [".txt", ".md", ".csv", ".tsv", ""] || continue
        try
            raw_content = read(entry_path, String)
            code_blocks !== nothing && _collect_code_blocks!(code_blocks, raw_content, entry_path)
            if !_include_code_experience_text() && _is_majnon_code_seed_doc(raw_content, entry_path)
                continue
            end
            # تجاهل الملفات غير العربية (إنجليزي، ألماني، إلخ)
            if !_is_arabic_text(raw_content; threshold=0.3)
                continue
            end
            content = _normalize_arabic_text(raw_content)
            content = _strip_dialogue_labels(content)
            isempty(strip(content)) && continue
            
            fname = entry_name
            if ext == ".csv" || ext == ".tsv"
                sep = ext == ".csv" ? "," : "\t"
                idx = 1
                for line in eachline(IOBuffer(content))
                    isempty(strip(line)) && continue
                    parts = split(line, sep)
                    txt = strip(length(parts) >= 2 ? parts[end] : parts[1], ['"', '\''])
                    if !isempty(txt)
                        push!(texts, txt)
                        if metadata !== nothing
                            push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => idx))
                        end
                        idx += 1
                    end
                end
            elseif granularity == :document
                doc = join([strip(l) for l in split(content, r"\n") if !isempty(strip(l))], " ")
                if !isempty(doc)
                    push!(texts, doc)
                    if metadata !== nothing
                        push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => 1))
                    end
                end
            elseif granularity == :paragraph
                idx = 1
                for p in split(content, r"\n\s*\n")
                    para = join([strip(l) for l in split(p, r"\n") if !isempty(strip(l))], " ")
                    if !isempty(para)
                        push!(texts, para)
                        if metadata !== nothing
                            push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => idx))
                        end
                        idx += 1
                    end
                end
            else # :line
                idx = 1
                for line in eachline(IOBuffer(content))
                    txt = strip(line)
                    if !isempty(txt)
                        push!(texts, txt)
                        if metadata !== nothing
                            push!(metadata, Dict{String,Any}("file_name" => fname, "paragraph_index" => idx))
                        end
                        idx += 1
                    end
                end
            end
        catch e
            @warn "خطأ في قراءة: $entry_path — $e"
        end
    end
end

function load_all_corpus(; granularity::Symbol=:document,
                         code_blocks::Union{Nothing,Vector{String}}=nothing)
    texts = String[]
    metadata = Dict{String,Any}[]

    # ─── 1) مجلدات فرعية مباشرة داخل data/ (دائماً، مرتبة أبجدياً) ───
    if isdir(DATA_DIR)
        # نجمع كل المجلدات الفرعية الموجودة في data/ مباشرة
        all_entries = sort(readdir(DATA_DIR; join=false))
        subdirs = filter(e -> isdir(joinpath(DATA_DIR, e)), all_entries)
        # استثناء المجلدات الداخلية الخاصة
        subdirs = filter(e -> e ∉ ["rules", ".git", ".DS_Store"], subdirs)

        if !isempty(subdirs)
            println("  → قراءة المجلدات الفرعية من: $DATA_DIR")
            println("  → ترتيب المجلدات أبجدياً: $(join(subdirs, " ← "))")
            for dir_name in subdirs
                dir_path = joinpath(DATA_DIR, dir_name)
                n_before = length(texts)
                n_files = 0
                _read_dir_recursive!(texts, dir_path, dir_name, n_files;
                                     granularity=granularity, code_blocks=code_blocks,
                                     metadata=metadata)
                n_added = length(texts) - n_before
                println("    📁 [$dir_name] → $n_added وثيقة")
            end
        end
    end

    # ─── 2) ملف corpus.txt الواحد (يُضاف دائماً إذا وُجد) ───
    corpus_file = joinpath(DATA_DIR, "corpus.txt")
    if isfile(corpus_file)
        println("  → قراءة الملف: corpus.txt")
        append!(texts, load_corpus_from_single(corpus_file;
                                               granularity=granularity,
                                               code_blocks=code_blocks,
                                               metadata=metadata))
    end

    # ─── 3) أي ملفات .txt أخرى مباشرة في data/ (مرتبة أبجدياً) ───
    if isdir(DATA_DIR)
        data_files = sort(filter(f -> lowercase(splitext(f)[2]) == ".txt" && isfile(joinpath(DATA_DIR, f)),
                                 readdir(DATA_DIR; join=false)))
        for f in data_files
            f == "corpus.txt" && continue
            println("  → قراءة: $f")
            append!(texts, load_corpus_from_single(joinpath(DATA_DIR, f);
                                                   granularity=granularity,
                                                   code_blocks=code_blocks,
                                                   metadata=metadata))
        end
    end

    # ─── 4) مسار مخصص عبر وسيطة سطر الأوامر — مع تجنب التكرار ───
    skip_next = false
    already_read = Set{String}()
    # سجّل المجلدات التي قُرئت في القسم 1
    if isdir(DATA_DIR)
        for e in readdir(DATA_DIR; join=false)
            ep = joinpath(DATA_DIR, e)
            isdir(ep) && push!(already_read, realpath(ep))
        end
        push!(already_read, realpath(DATA_DIR))
    end
    for arg in ARGS
        if skip_next; skip_next = false; continue; end
        if arg in ("--data", "--level", "--threads"); skip_next = true; continue; end
        startswith(arg, "--") && continue
        rarg = try realpath(arg) catch; arg; end
        rarg in already_read && continue
        if isdir(arg)
            println("  → قراءة من مسار مخصص (مجلد): $arg")
            append!(texts, load_corpus_from_dirs(arg;
                                                 granularity=granularity,
                                                 code_blocks=code_blocks,
                                                 metadata=metadata))
            push!(already_read, rarg)
        elseif isfile(arg) && lowercase(splitext(arg)[2]) == ".txt"
            println("  → قراءة من مسار مخصص (ملف): $arg")
            append!(texts, load_corpus_from_single(arg;
                                                   granularity=granularity,
                                                   code_blocks=code_blocks,
                                                   metadata=metadata))
            push!(already_read, rarg)
        end
    end

    # ─── احتياطي: نصوص افتراضية إذا لا يوجد شيء ───
    agent_texts = load_agent_experience_corpus(; granularity=granularity, code_blocks=code_blocks, metadata=metadata)
    if !isempty(agent_texts)
        append!(texts, agent_texts)
        println("  -> Majnon experience documents added: $(length(agent_texts))")
    end

    if isempty(texts)
        println("  ⚠ لا كوربس — استخدام نصوص افتراضية")
        texts = [
            "العلم نور والجهل ظلام والسماء صافية والأرض خضراء والحياة جميلة والعالم كبير",
            "الله خالق كل شيء والكتاب مفيد والعلم نور والماء سر الحياة والأرض خضراء",
            "السلام عليكم ورحمة الله وبركاته القلب الكبير يعرف الحب والطريق الحق",
            "كان هناك رجل عالم يسعى دائما نحو الأفضل يعمل في النهار ويقرأ في الليل",
            "فقال الحكيم إن الصبر مفتاح الفرج ومن يسعى يجد ومن يطلب يحقق الأمل",
            "إن الإنسان لا يتعلم من الخطأ بل من التأمل في الخطأ والحكمة ضالة المؤمن",
        ]
        for idx in 1:length(texts)
            push!(metadata, Dict{String,Any}("file_name" => "fallback_corpus.txt", "paragraph_index" => idx))
        end
    end

    return texts, metadata
end

function _aql_candidate_sentences(texts::Vector{String}; max_sentences::Int=50_000)
    selected = String[]
    causal_markers = ("إذا", "اذا", "لو", "فإن", "فان", "ينتج", "نتيجة",
                      "يسبب", "تسبب", "يؤدي", "تؤدي", "يرفع", "ترفع",
                      "يزيد", "تزيد", "ينقص", "تنقص", "يقلل", "تقلل",
                      "تجذب", "يتجاذب", "تتنافر", "يتنافر", "اضافة", "إضافة", "إضافه")
    effect_markers = ("درجة", "درجه", "غليان", "حرارة", "حراره", "ضغط",
                      "كثافة", "كثافه", "سرعة", "سرعه", "خوف", "استقرار")

    for text in texts
        for raw_sentence in split(text, r"[\n؟?؛;]+|(?<!\d)[.!]+|[.!]+(?!\d)")
            sentence = strip(raw_sentence)
            length(sentence) >= 8 || continue
            any(m -> occursin(m, sentence), causal_markers) || continue
            any(m -> occursin(m, sentence), effect_markers) || continue
            push!(selected, sentence)
            length(selected) >= max_sentences && return unique(selected)
        end
    end
    return unique(selected)
end

# ═══════════ بناء المعجم والمصفوفات ═══════════

function _aql_dialogue_texts(texts::Vector{String}; max_texts::Int=20_000)
    selected = String[]
    marker_re = r"(?:سؤال|س)\s*[:：].*(?:جواب|ج)\s*[:：]|(?:جواب|ج)\s*[:：].*(?:سؤال|س)\s*[:：]"
    dialogue_terms = ("السلام عليكم", "وعليكم السلام", "كيف حالك", "مرحبا", "أهلا", "اهلا")
    for text in texts
        value = strip(text)
        length(value) >= 6 || continue
        short_dialogue = length(value) <= 500
        has_marker = occursin(marker_re, value) || occursin('\t', value)
        has_dialogue_terms = short_dialogue && count(term -> occursin(term, value), dialogue_terms) >= 2
        has_question_answer = short_dialogue && (occursin("؟", value) || occursin("?", value)) &&
                              any(term -> occursin(term, value), ("نعم", "لا", "أنا", "انا", "هو", "هي"))
        (has_marker || has_dialogue_terms || has_question_answer) || continue
        push!(selected, value)
        length(selected) >= max_texts && return unique(selected)
    end
    return unique(selected)
end

function build_vocab(texts::Vector{String}; min_count::Union{Int,Nothing}=nothing, max_vocab::Int=200_000)
    word_counts = Dict{String,Int}()
    total_words = 0
    for text in texts
        for word in split(text)
            w = _strip_punct_boundary(strip(word))
            _is_meaningful_word(w) || continue
            word_counts[w] = get(word_counts, w, 0) + 1
            total_words += 1
        end
    end

    adaptive_min = min_count !== nothing ? min_count : 3

    candidates = String[]
    for (w, count) in word_counts
        if count >= adaptive_min
            push!(candidates, w)
        end
    end

    sort!(candidates, by = w -> word_counts[w], rev = true)
    
    if length(candidates) > max_vocab
        println("   ⚠ تقليص المعجم: $(length(candidates)) → $max_vocab")
        candidates = candidates[1:max_vocab]
    end

    bare_groups = Dict{String, Vector{String}}()
    for w in candidates
        bare = _strip_diacritics(w)
        if !haskey(bare_groups, bare)
            bare_groups[bare] = String[]
        end
        push!(bare_groups[bare], w)
    end

    vocab = Dict{String,Int}()
    next_id = 1

    for (bare, words) in bare_groups
        accepted = Tuple{String, Int, Vector{Float32}}[]
        for w in words
            pv = Float32.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w))
            duplicate_id = 0
            for (acc_w, acc_id, acc_pv) in accepted
                sim = MirnanNew.Physics.WordPhysics.phase_similarity(pv, acc_pv)
                if sim >= 0.90
                    duplicate_id = acc_id
                    break
                end
            end
            if duplicate_id > 0
                vocab[w] = duplicate_id
            else
                vocab[w] = next_id
                push!(accepted, (w, next_id, pv))
                next_id += 1
            end
        end
    end

    return vocab
end

function corpus_word_frequencies(vocab::Dict{String,Int}, texts::Vector{String})
    V = length(vocab)
    word_freqs = zeros(Float64, V)
    for text in texts
        for word in split(text)
            w = _strip_punct_boundary(strip(word))
            _is_meaningful_word(w) || continue
            id = get(vocab, w, 0)
            id > 0 && id <= V && (word_freqs[id] += 1.0)
        end
    end
    return word_freqs
end

function _env_int(name::String, default::Int)
    raw = get(ENV, name, string(default))
    try
        value = parse(Int, strip(raw))
        return max(0, value)
    catch
        @warn "$name غير صالح: $raw — سيتم استخدام $default"
        return default
    end
end

function _even_sample_indices(n::Int, limit::Int)
    n <= 0 && return Int[]
    limit <= 0 && return Int[]
    limit >= n && return collect(1:n)
    step = n / limit
    chosen = Int[]
    seen = Set{Int}()
    for i in 1:limit
        idx = clamp(round(Int, 1 + (i - 1) * step), 1, n)
        if !(idx in seen)
            push!(chosen, idx)
            push!(seen, idx)
        end
    end
    return chosen
end

function _top_vocab_words(vocab::Dict{String,Int}, word_freqs::Vector{Float64};
                          max_words::Int=20_000, min_freq::Float64=2.0)
    chosen = Tuple{String,Int,Float64}[]
    seen_ids = Set{Int}()
    function add_candidates!(threshold::Float64)
        for (word, id) in vocab
            (id <= 0 || id > length(word_freqs)) && continue
            id in seen_ids && continue
            freq = word_freqs[id]
            freq >= threshold || continue
            _is_meaningful_word(word) || continue
            push!(seen_ids, id)
            push!(chosen, (word, id, freq))
        end
    end
    add_candidates!(min_freq)
    length(chosen) < min(max_words, 1_000) && add_candidates!(1.0)
    sort!(chosen; by=x -> (-x[3], x[1]))
    return chosen[1:min(max_words, length(chosen))]
end

function _save_phase_evolution_report(path::String, pe, id_to_word::Dict{Int,String},
                                      observed_pairs::Int, sampled_sentences::Int,
                                      active_words::Int)
    shifted = Tuple{String,Float64}[]
    for (id, shift) in pe.shifts
        word = get(id_to_word, id, "")
        isempty(word) && continue
        push!(shifted, (word, Float64(norm(shift))))
    end
    sort!(shifted; by=x -> (-x[2], x[1]))
    report = Dict{String,Any}(
        "version" => 1,
        "mode" => "sampled",
        "sampled_sentences" => sampled_sentences,
        "observed_pairs" => observed_pairs,
        "active_words" => active_words,
        "words_shifted" => length(shifted),
        "top_shifted_words" => [
            Dict("word" => word, "shift_norm" => shift_norm)
            for (word, shift_norm) in shifted[1:min(200, length(shifted))]
        ],
    )
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, report)
    end
    return report
end

function _folded_word_phase_vector(word::String; dim::Int=27)
    raw = Float64.(MirnanNew.Physics.WordPhysics.compute_word_phase_vector(word))
    folded = zeros(Float64, dim)
    isempty(raw) && return folded
    for (i, value) in enumerate(raw)
        folded[mod1(i, dim)] += value
    end
    nrm = norm(folded)
    nrm > 1e-12 && (folded ./= nrm)
    return folded
end

function _train_phase_evolution_sample!(texts::Vector{String}, vocab::Dict{String,Int},
                                        word_freqs::Vector{Float64};
                                        max_sentences::Int=8_000,
                                        max_words::Int=20_000,
                                        max_distance::Int=2)
    pe = MirnanNew.Physics.PhaseEvolution(dim=27)
    top = _top_vocab_words(vocab, word_freqs; max_words=max_words, min_freq=2.0)
    active_ids = Set(id for (_, id, _) in top)
    phase_cache = Dict{String,Vector{Float64}}()
    phase_fn = function(w::String)
        get!(phase_cache, w) do
            _folded_word_phase_vector(w; dim=27)
        end
    end

    observed_pairs = 0
    sampled_sentences = 0
    for text in texts
        sampled_sentences >= max_sentences && break
        for line in split(text, r"[.!\n؟?؛;]+")
            sampled_sentences >= max_sentences && break
            trimmed = strip(line)
            isempty(trimmed) && continue
            words = String[_strip_punct_boundary(strip(w)) for w in split(trimmed)]
            filter!(w -> haskey(vocab, w) && (vocab[w] in active_ids), words)
            length(words) >= 2 || continue
            sampled_sentences += 1
            for i in 1:length(words)
                for j in (i+1):min(length(words), i + max_distance)
                    w1, w2 = words[i], words[j]
                    MirnanNew.Physics.observe_cooccurrence!(
                        pe, phase_fn(w1), phase_fn(w2);
                        w1_id=vocab[w1], w2_id=vocab[w2], distance=j-i)
                    observed_pairs += 1
                end
            end
        end
    end
    return pe, Dict(
        "observed_pairs" => observed_pairs,
        "sampled_sentences" => sampled_sentences,
        "active_words" => length(active_ids),
    )
end

function _train_twistor_sample!(vocab::Dict{String,Int}, word_freqs::Vector{Float64};
                                max_words::Int=20_000,
                                max_pairs::Int=8_000)
    twistor = MirnanNew.Physics.MorphoTwistor.TwistorEngine()
    top = _top_vocab_words(vocab, word_freqs; max_words=max_words, min_freq=2.0)
    words = sort(String[word for (word, _, _) in top])
    pairs = MirnanNew.Physics.MorphoTwistor.find_morphological_pairs(words)
    if length(pairs) > max_pairs
        pairs = pairs[1:max_pairs]
    end
    phase_cache = Dict{String,Vector{Float64}}()
    phase_fn = function(w::String)
        get!(phase_cache, w) do
            _folded_word_phase_vector(w; dim=27)
        end
    end
    MirnanNew.Physics.MorphoTwistor.learn_patterns!(
        twistor, pairs, phase_fn; sim_threshold=0.68, min_cluster_size=2)
    return twistor, Dict(
        "sampled_words" => length(words),
        "pairs" => length(pairs),
        "patterns" => length(twistor.patterns),
    )
end

function save_word_balance_report(path::String, vocab::Dict{String,Int},
                                  word_freqs::Vector{Float64},
                                  weights::Vector{Float64},
                                  meta::Dict{String,Any})
    id_to_word = Dict{Int,String}()
    for (word, id) in vocab
        id > 0 && id <= length(weights) || continue
        if !haskey(id_to_word, id) || length(word) < length(id_to_word[id])
            id_to_word[id] = word
        end
    end

    active_ids = [id for id in 1:length(weights) if id <= length(word_freqs) && word_freqs[id] > 0]
    boosted_ids = sort(active_ids; by=id -> weights[id], rev=true)[1:min(25, length(active_ids))]
    reduced_ids = sort(active_ids; by=id -> weights[id])[1:min(25, length(active_ids))]

    function entry_for(id)
        return Dict{String,Any}(
            "id" => id,
            "word" => get(id_to_word, id, string(id)),
            "count" => word_freqs[id],
            "weight" => weights[id],
        )
    end

    weights_by_word = Dict{String,Float64}()
    counts_by_word = Dict{String,Float64}()
    for (word, id) in vocab
        id > 0 && id <= length(weights) && id <= length(word_freqs) || continue
        weights_by_word[word] = round(weights[id]; digits=6)
        counts_by_word[word] = word_freqs[id]
    end

    data = Dict{String,Any}(
        "description" => "Hidden training-only balance weights. They are not corpus text and are not used directly by generation.",
        "meta" => meta,
        "top_boosted" => [entry_for(id) for id in boosted_ids],
        "top_reduced" => [entry_for(id) for id in reduced_ids],
        "weights_by_word" => weights_by_word,
        "counts_by_word" => counts_by_word,
    )
    open(path, "w") do io
        JSON.print(io, data)
    end
    return path
end

function build_K(vocab::Dict{String,Int}, texts::Vector{String}; window::Int=15, causal::Bool=false,
                  subsample_threshold::Float64=1e-5, particle_penalty::Float64=0.1,
                  use_ppmi::Bool=true, apply_svd::Bool=false, svd_dim::Int=300,
                  word_freqs::Union{Nothing,Vector{Float64}}=nothing,
                  balance_weights::Union{Nothing,Vector{Float64}}=nothing)
    V = length(vocab)
    
    # المرحلة صفر: حساب التكرارات الخام لكل كلمة
    if word_freqs === nothing
        word_freqs = corpus_word_frequencies(vocab, texts)
    elseif length(word_freqs) < V
        error("word_freqs length ($(length(word_freqs))) is smaller than vocab size ($V)")
    else
        word_freqs = Float64.(word_freqs)
    end
    total_tokens = max(sum(word_freqs), 1.0)

    active_balance = nothing
    if balance_weights !== nothing
        length(balance_weights) < V && error("balance_weights length ($(length(balance_weights))) is smaller than vocab size ($V)")
        active_balance = Float64.(balance_weights)
    end
    
    # ═══ المرحلة الأولى: subsampling الكلمات الشائعة جداً ═══
    # صيغة word2vec: P(discard) = 1 - sqrt(t / freq)
    # حيث t = subsample_threshold * total_tokens
    t = subsample_threshold * total_tokens
    keep_prob = ones(Float64, V)
    for id in 1:V
        freq = word_freqs[id] / total_tokens
        if freq > subsample_threshold
            keep_prob[id] = sqrt(subsample_threshold / freq)
        end
    end
    
    # تحديد الكلمات التي تعتبر حروف معاني
    is_particle_id = zeros(Bool, V)
    for (word, id) in vocab
        if id > 0 && id <= V
            is_particle_id[id] = MirnanNew.Physics.Particles.is_particle(word)
        end
    end
    
    # ═══ المرحلة الثانية: بناء مصفوفة التزامن مع subsampling ═══
    total_words_est = 0
    for text in texts
        total_words_est += length(split(text))
    end
    est_coords = min(total_words_est * window * 2, 2_000_000_000)
    max_entries = 200_000_000  # حد أقصى 200M إدخال لمراقبة الذاكرة
    
    I_coords = Int64[]
    J_coords = Int64[]
    V_coords = Float64[]
    sizehint!(I_coords, min(est_coords, 500_000_000))
    sizehint!(J_coords, min(est_coords, 500_000_000))
    sizehint!(V_coords, min(est_coords, 500_000_000))
    
    for text in texts
        words = String[_strip_punct_boundary(strip(w)) for w in split(text) if _is_meaningful_word(_strip_punct_boundary(strip(w)))]
        ids_raw = Int64[get(vocab, w, -1) for w in words]
        
        # تطبيق subsampling: تجاهل عشوائي للكلمات فائقة التكرار
        ids = Int64[]
        for id in ids_raw
            id <= 0 && continue
            if keep_prob[id] >= 1.0 || rand() < keep_prob[id]
                push!(ids, id)
            end
        end
        
        for i in 1:length(ids)
            id_i = ids[i]
            id_i > V && continue
            start_j = causal ? i + 1 : max(1, i - window)
            for j in start_j:min(length(ids), i+window)
                j == i && continue
                id_j = ids[j]
                id_j > V && continue
                balance = MirnanNew.Physics.TrainingBalanceModule.pair_balance_weight(active_balance, id_i, id_j)
                push!(I_coords, id_i)
                push!(J_coords, id_j)
                push!(V_coords, balance / abs(j - i))
            end
            if length(I_coords) >= max_entries
                break
            end
        end
        if length(I_coords) >= max_entries
            break
        end
    end
    
    if isempty(I_coords)
        return spzeros(V, V)
    end
    
    # بناء مصفوفة التكرار الخام
    C = sparse(I_coords, J_coords, V_coords, V, V)
    
    # ═══ المرحلة الثالثة: تحويل إلى PPMI (Positive PMI) ═══
    # PPMI: max(PMI, 0) — يزيل الارتباطات السلبية الضوضائية
    # أفضل من PMI البسيط لأنه لا يسمح بقيم سلبية (تميل لتكون ضوضاء)
    epsilon_smooth = 0.5  # تمهيد توزيعي يمنع انعدام السياقات
    
    # حساب عامل العقوبة التصاعدي لكل حرف بناءً على تردده
    max_freq = max(maximum(word_freqs), 1.0)
    min_freq_particle_penalty = particle_penalty
    max_freq_particle_penalty = particle_penalty * 0.15  # أقوى بـ 6.7x للحروف فائقة التكرار
    
    for j in 1:V
        for idx in C.colptr[j]:(C.colptr[j+1]-1)
            i = C.rowval[idx]
            c_val = C.nzval[idx]
            
            pw_i = (word_freqs[i] + epsilon_smooth) / (total_tokens + epsilon_smooth * V)
            pw_j = (word_freqs[j] + epsilon_smooth) / (total_tokens + epsilon_smooth * V)
            expected = pw_i * pw_j * total_tokens
            
            # PPMI: Positive Pointwise Mutual Information
            # PMI = log2(P(x,y) / (P(x) * P(y)))
            # PPMI = max(PMI, 0)
            pmi = log((c_val + 1.0) / (expected + 1.0))
            val = use_ppmi ? max(pmi, 0.0) : pmi
            
            # عقوبة حروف تصاعدية: الحروف فائقة التكرار تعاقب أشد
            pi = is_particle_id[i]
            pj = is_particle_id[j]
            if pi || pj
                # التردد النسبي للحرف (أعلى الترددين)
                rel_freq = max(word_freqs[i], word_freqs[j]) / max_freq
                # عامل العقوبة يتدرج من particle_penalty إلى max_freq_particle_penalty
                scaled_penalty = max_freq_particle_penalty + (1.0 - rel_freq) * (particle_penalty - max_freq_particle_penalty)
                if pi && pj
                    val *= scaled_penalty^2
                else
                    val *= scaled_penalty
                end
            end
            
            C.nzval[idx] = val
        end
    end
    
    # ═══ المرحلة الرابعة: SVD (اختياري) ═══
    # SVD: تقليل الأبعاد مع الحفاظ على الهيكل الدلالي
    # يعمل كـ LSA (Latent Semantic Analysis) — يكشف المعاني الضمنية
    if apply_svd && V > svd_dim
        println("   📐 تطبيق SVD (أبعاد: $V → $svd_dim)...")
        try
            # تحويل المصفوفة الكثيفة للعمل مع SVD
            C_dense = Matrix(C)
            F = svd(C_dense)
            # الاحتفاظ بأول svd_dim مكونات فقط
            U = F.U[:, 1:svd_dim]
            S = Diagonal(F.S[1:svd_dim])
            # إعادة بناء المصفوفة: C ≈ U * S * V'
            C_reconstructed = U * S * F.Vt[1:svd_dim, :]
            C = sparse(C_reconstructed)
            # ضمان عدم وجود قيم سالبة بعد SVD
            C.nzval .= max.(C.nzval, 0.0)
            println("   ✓ SVD مكتمل — الكثافة: $(nnz(C)) / $(length(C.nzval))")
        catch e
            println("   ⚠ فشل SVD: $e — استخدام المصفوفة الأصلية")
        end
    end
    
    return C
end

function extract_knowledge_sentences(knowledge_dir::String)
    sentences = String[]
    # 1. definitions.json
    def_path = joinpath(knowledge_dir, "definitions.json")
    if isfile(def_path)
        try
            defs = JSON.parsefile(def_path)
            for (word, item) in defs
                if item isa AbstractDict
                    push!(sentences, get(item, "definition", ""))
                    for ex in get(item, "examples", [])
                        push!(sentences, ex)
                    end
                end
            end
        catch e
            @warn "Failed to extract from definitions.json: $e"
        end
    end
    
    # 2. istinbat_attention.json
    ist_path = joinpath(knowledge_dir, "istinbat_attention.json")
    if isfile(ist_path)
        try
            ist = JSON.parsefile(ist_path)
            for rec in get(ist, "records", [])
                for ex in get(rec, "examples", [])
                    push!(sentences, ex)
                end
            end
        catch e
            @warn "Failed to extract from istinbat_attention.json: $e"
        end
    end

    # 3. dialogue_facts.json
    fact_path = joinpath(knowledge_dir, "dialogue_facts.json")
    if isfile(fact_path)
        try
            facts = JSON.parsefile(fact_path)
            for act in get(facts, "speech_acts", [])
                push!(sentences, get(act, "prompt", ""))
                push!(sentences, get(act, "response", ""))
            end
        catch e
            @warn "Failed to extract from dialogue_facts.json: $e"
        end
    end

    filter!(!isempty, sentences)
    return unique!(sentences)
end

function save_model(vocab, K_sem, K_syn, K_dial, K_causal=nothing, texts=nothing; out_dir=MODEL_DIR)
    mkpath(out_dir)
    open(joinpath(out_dir, "vocab.json"), "w") do io; JSON.print(io, vocab); end

    for (name, K) in [("K_sem", K_sem), ("K_syn", K_syn), ("K_dialogue", K_dial), ("K_causal", K_causal)]
        if K !== nothing
            open(joinpath(out_dir, "$(name).dat"), "w") do io
                write(io, "SPARSE_CSC\n")
                write(io, Int32(size(K,1)), Int32(size(K,2)), Int32(length(K.nzval)))
                write(io, Int32.(K.colptr))
                write(io, Int32.(rowvals(K)))
                write(io, Float64.(nonzeros(K)))
            end
        end
    end
    
    if texts !== nothing
        # حفظ الكوربس كجمل من معرفات الكلمات للـ PRNN
        corpus_sentences = Vector{Int32}[]
        for text in texts
            # نقسم المستند إلى جمل أو فقرات فرعية إذا كان طويلاً
            lines = split(text, r"[.!\n؟]")
            for line in lines
                trimmed = strip(line)
                isempty(trimmed) && continue
                words = String[_strip_punct_boundary(strip(w)) for w in split(trimmed) if length(strip(w)) >= 2]
                ids = Int32[get(vocab, w, -1) for w in words]; filter!(x -> x > 0, ids)
                if length(ids) >= 2
                    push!(corpus_sentences, ids)
                end
            end
        end
        open(joinpath(out_dir, "corpus_sentences.dat"), "w") do io
            write(io, Int32(length(corpus_sentences)))
            for s in corpus_sentences
                write(io, Int32(length(s)))
                write(io, s)
            end
        end
    end
    return true
end

function load_model(in_dir=MODEL_DIR)
    vf = joinpath(in_dir, "vocab.json")
    if !isfile(vf); return nothing; end
    vocab = Dict{String,Int}(k => Int(v) for (k,v) in JSON.parsefile(vf))

    function load_sparse(path)
        isfile(path) || return spzeros(length(vocab), length(vocab))
        open(path, "r") do io
            header = readline(io); @assert header == "SPARSE_CSC"
            m = read(io, Int32); n = read(io, Int32); nnz = read(io, Int32)
            colptr = read!(io, Vector{Int32}(undef, n+1))
            rows = read!(io, Vector{Int32}(undef, nnz))
            vals = read!(io, Vector{Float64}(undef, nnz))
            return SparseMatrixCSC(Int(m), Int(n), Vector{Int}(colptr), Vector{Int}(rows), vals)
        end
    end

    K_sem = load_sparse(joinpath(in_dir, "K_sem.dat"))
    K_syn = load_sparse(joinpath(in_dir, "K_syn.dat"))
    K_dial = load_sparse(joinpath(in_dir, "K_dialogue.dat"))
    K_causal = load_sparse(joinpath(in_dir, "K_causal.dat"))

    return Dict("vocab" => vocab, "K_sem" => K_sem, "K_syn" => K_syn, "K_dial" => K_dial, "K_causal" => K_causal)
end

# ═══════════ الرئيسية ═══════════

function _parse_segment_level(value::AbstractString, fallback::Symbol=:paragraph)
    normalized = lowercase(strip(String(value)))
    startswith(normalized, ":") && (normalized = normalized[2:end])
    normalized = replace(normalized, "-" => "_")
    normalized in ("paragraph", "para", "فقرة", "فقره") && return :paragraph
    normalized in ("document", "doc", "file", "ملف", "وثيقة", "وثيقه") && return :document
    normalized in ("line", "row", "سطر") && return :line
    @warn "MIRNAN_SEGMENT_LEVEL غير معروف: $value — سيتم استخدام $fallback"
    return fallback
end

function main()
    start_time = now()
    println("╔══════════════════════════════════════════════╗")
    println("║        مرنان V8 — تدريب فيزيائي             ║")
    println("╚══════════════════════════════════════════════╝")
    println()
    println("⏱  وقت بداية التدريب: ", Dates.format(start_time, "yyyy-mm-dd HH:MM:SS"))
    println()

    # تحديد مستوى التقسيم
    granularity = _parse_segment_level(get(ENV, "MIRNAN_SEGMENT_LEVEL", "paragraph"), :paragraph)
    use_preprocessing = false
    preprocess_source = ""
    use_word_balance = true
    i = 1
    while i <= length(ARGS)
        arg = ARGS[i]
        if arg == "--paragraph"; granularity = :paragraph; end
        if arg == "--line"; granularity = :line; end
        if arg == "--document"; granularity = :document; end
        if arg == "--level" && i < length(ARGS)
            granularity = _parse_segment_level(ARGS[i + 1], granularity)
            i += 1
        elseif startswith(arg, "--level=")
            granularity = _parse_segment_level(replace(arg, "--level=" => ""), granularity)
        end
        if arg == "--balance"; use_word_balance = true; end
        if arg == "--no-balance"; use_word_balance = false; end
        if arg == "--preprocess"; use_preprocessing = true; end
        if startswith(arg, "--preprocess=")
            use_preprocessing = true
            preprocess_source = replace(arg, "--preprocess=" => "")
        end
        i += 1
    end

    # ═══ خط المعالجة المسبقة (اختياري) ═══
    if use_preprocessing && !isempty(preprocess_source)
        println("🧹 تشغيل خط المعالجة المسبقة على: $preprocess_source")
        config = MirnanNew.Physics.PipelineConfig(;
            target_lang="arabic",
            strip_diacritics=false,
            granularity="sentence",
            verbose=true,
        )
        if isdir(preprocess_source)
            cleaned, pstats = MirnanNew.Physics.run_pipeline_directory(
                preprocess_source; config=config,
                save_dir=joinpath(DATA_DIR, "cleaned"))
            println("   ✓ خط المعالجة: $(pstats.input_texts) → $(pstats.final_texts) جملة نظيفة")
        elseif isfile(preprocess_source)
            cleaned, pstats = MirnanNew.Physics.run_pipeline_file(
                preprocess_source; config=config)
            println("   ✓ خط المعالجة: $(pstats.input_texts) → $(pstats.final_texts) جملة نظيفة")
        end
        println()
    end
    println("📋 مستوى التقسيم: $granularity")
    println("   :document  — كل ملف = وثيقة واحدة (يحافظ على السرد)")
    println("   :paragraph — كل فقرة = وثيقة (يحافظ على الفقرة)")
    println("   :line      — كل سطر = وثيقة (للبيانات المهيكلة فقط)")
    println()

    # ═══ حذف كل مخرجات التدريبات السابقة ═══
    println("🗑  تنظيف كامل لمجلد النموذج...")
    skipped_cleanup = _clean_model_dir!(MODEL_DIR)
    remaining_count = isdir(MODEL_DIR) ? length(readdir(MODEL_DIR)) : 0
    if isempty(skipped_cleanup)
        println("   ✓ تم حذف كل ملفات التدريب السابق ($(remaining_count) متبقي)")
    else
        println("   ⚠ تعذر حذف $(length(skipped_cleanup)) ملف/مجلد مقفول؛ سيستمر التدريب ويعاد توليد المخرجات المتاحة.")
        for path in skipped_cleanup[1:min(end, 5)]
            println("     - $(basename(path))")
        end
    end

    println("📂 قراءة الكوربس...")
    code_blocks = String[]
    raw_texts, raw_metadata = load_all_corpus(granularity=granularity, code_blocks=code_blocks)

    if isempty(raw_texts)
        println("❌ خطأ: لا توجد بيانات كوربس! ضع الملفات في data/corpus/ أو مرر المسار كوسيطة.")
        return
    end

    # ═══ فصل كتل الكود عن النص الطبيعي ═══
    # الكود البرمجي لا يخضع لنفس سياسة الجمل العادية — يُفصل في K_code مستقلة
    clean_texts = String[]
    clean_metadata = Dict{String,Any}[]
    for (idx_raw, t) in enumerate(raw_texts)
        blocks = MirnanNew.Physics.MathBridgeModule.extract_code_blocks(t)
        append!(code_blocks, blocks)
        cleaned = MirnanNew.Physics.MathBridgeModule.strip_code_blocks(t)
        cleaned = MirnanNew.Physics.CodeEngineModule.strip_inline_code(cleaned)
        if !isempty(strip(cleaned))
            push!(clean_texts, cleaned)
            push!(clean_metadata, raw_metadata[idx_raw])
        end
    end
    texts = clean_texts
    metadata = clean_metadata
    unique!(code_blocks)
    aql_training_sentences = _aql_candidate_sentences(texts)
    aql_dialogue_texts = _aql_dialogue_texts(texts)
    aql_training_texts = unique(vcat(aql_training_sentences, aql_dialogue_texts))

    total_words = sum(length(split(t)) for t in texts)
    total_chars = sum(length(t) for t in texts)
    println("   إجمالي الوثائق: $(length(texts)) | الكلمات: $total_words | الأحرف: $total_chars")
    if !isempty(code_blocks)
        println("   🖥  كتل كود مستخرجة: $(length(code_blocks)) — ستبنى K_code منفصلة عنها")
    end
    if !isempty(aql_training_texts)
        println("   🧠 جمل سببية مرشحة لـ al_aql: $(length(aql_training_sentences))")
    end
    println()

    println("1️⃣  بناء المعجم...")
    if !isempty(aql_dialogue_texts)
        println("   al_aql dialogue candidate texts: $(length(aql_dialogue_texts))")
    end
    vocab = build_vocab(texts; max_vocab=200_000)
    println("   ✓ $(length(vocab)) كلمة فريدة")
    println()

    println("1️⃣.1 حساب ميزان التكرار الوهمي...")
    word_freqs = corpus_word_frequencies(vocab, texts)
    word_balance_weights = nothing
    if use_word_balance
        balance_config = MirnanNew.Physics.TrainingBalanceModule.WordBalanceConfig()
        weights, balance_meta = MirnanNew.Physics.TrainingBalanceModule.build_word_balance_weights(
            word_freqs; config=balance_config)
        word_balance_weights = weights
        report_path = save_word_balance_report(
            joinpath(MODEL_DIR, "word_balance_weights.json"),
            vocab, word_freqs, weights, balance_meta)
        println("   ✓ موازنة ناعمة مفعلة: مرفوعة $(balance_meta["boosted"]) | مخفضة $(balance_meta["reduced"]) | محايدة $(balance_meta["neutral"])")
        println("   ✓ مجال الوزن: $(round(balance_meta["min_observed_weight"]; digits=3)) → $(round(balance_meta["max_observed_weight"]; digits=3)) | الهدف: $(round(balance_meta["target_count"]; digits=2))")
        println("   ✓ حفظ تقرير الميزان: $(basename(report_path))")
    else
        println("   ⏭ ميزان التكرار معطل (--no-balance)")
    end
    println()

    # ═══ بناء مصفوفات الاقتران الدلالي ═══
    # PPMI (Positive PMI): يزيل الارتباطات السلبية الضوضائية
    # نافذة 15: تتبع سياق أوسع للكلمات البعيدة
    println("2️⃣  بناء K_sem (نافذة 15 — PPMI + subsample 5e-4 + ميزان تكرار + عقوبة حروف 0.1)...")
    K_sem = build_K(vocab, texts; window=15, subsample_threshold=5e-4, particle_penalty=0.1,
                    use_ppmi=true, word_freqs=word_freqs, balance_weights=word_balance_weights)
    println("   ✓ $(length(K_sem.nzval)) اقتران غير صفري")
    println()

    println("3️⃣  بناء K_syn (نافذة 3 — نحوي اتجاهي — PPMI + subsample 3e-4 + ميزان تكرار + عقوبة 0.2)...")
    K_syn = build_K(vocab, texts; window=3, causal=true, subsample_threshold=3e-4, particle_penalty=0.2,
                    use_ppmi=true, word_freqs=word_freqs, balance_weights=word_balance_weights)
    println("   ✓ $(length(K_syn.nzval)) اقتران غير صفري")
    println()

    println("4️⃣  بناء K_dialogue (نافذة 2 — حواري اتجاهي — PPMI + subsample 3e-4 + ميزان تكرار + عقوبة 0.2)...")
    K_dial = build_K(vocab, texts; window=2, causal=true, subsample_threshold=3e-4, particle_penalty=0.2,
                     use_ppmi=true, word_freqs=word_freqs, balance_weights=word_balance_weights)
    println("   ✓ $(length(K_dial.nzval)) اقتران غير صفري")
    println()

    println("4️⃣.0 بناء K_causal (نافذة 15 — سببي اتجاهي PPMI + subsample 5e-4 + ميزان تكرار + عقوبة 0.1)...")
    K_causal = build_K(vocab, texts; window=15, causal=true, subsample_threshold=5e-4, particle_penalty=0.1,
                       use_ppmi=true, word_freqs=word_freqs, balance_weights=word_balance_weights)
    println("   ✓ $(length(K_causal.nzval)) اقتران غير صفري")
    println()

    # ═══ تدريب اقتران الموجة الحاملة (carrier coupling) ═══
    println("4️⃣.1 اقتران الموجة الحاملة (اسم ←→ فعل/صفة)...")
    id2word = Dict{Int,String}(v => k for (k, v) in vocab)
    morpho = MirnanNew.Physics.MorphoPhasic.MorphoPhasicEngine()
    pv_cache = Dict{String,Vector{Float64}}()
    pv_fn = function(w)
        if !haskey(pv_cache, w)
            pv_cache[w] = Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w))
        end
        return pv_cache[w]
    end
    mass_fn = w -> MirnanNew.Physics.WordPhysics.compute_word_mass(w)

    cwe = MirnanNew.Physics.CarrierWave.CarrierWaveEngine()
    carrier_sentences = Vector{String}[]
    for t in texts[1:min(length(texts), 3000)]
        lines = split(t, r"[.!\n؟]")
        for line in lines
            trimmed = strip(line)
            isempty(trimmed) && continue
            words = String[_strip_punct_boundary(strip(w)) for w in split(trimmed) if length(strip(w)) >= 2]
            filter!(w -> haskey(vocab, w), words)
            if length(words) >= 2
                push!(carrier_sentences, words)
            end
        end
    end
    try
        MirnanNew.Physics.CarrierWave.train_carrier_coupling!(
            cwe, carrier_sentences, pv_cache, pv_fn;
            max_sentences=3000, eta=0.1)
        println("   ✓ اكتمل اقتران الموجة الحاملة")
    catch e
        println("   ⚠ اقتران الموجة الحاملة: $e")
    end
    println()

    # ═══ تطور المتجهات الطورية — تدريب مخفف بعينة ذكية بدل التخطي الكامل ═══
    println("4️⃣.2 تطور المتجهات الطورية — عينة ذكية مخففة...")
    phase_sample_sentences = _env_int("MIRNAN_PHASE_SAMPLE_SENTENCES", 8_000)
    phase_sample_words = _env_int("MIRNAN_PHASE_SAMPLE_WORDS", 20_000)
    phase_max_distance = max(1, _env_int("MIRNAN_PHASE_MAX_DISTANCE", 2))
    try
        pe, phase_stats = _train_phase_evolution_sample!(
            texts, vocab, word_freqs;
            max_sentences=phase_sample_sentences,
            max_words=phase_sample_words,
            max_distance=phase_max_distance)
        phase_report = _save_phase_evolution_report(
            joinpath(MODEL_DIR, "phase_evolution_sample.json"),
            pe, id2word,
            Int(phase_stats["observed_pairs"]),
            Int(phase_stats["sampled_sentences"]),
            Int(phase_stats["active_words"]))
        println("   ✓ عينة الجمل: $(phase_stats["sampled_sentences"]) | الأزواج: $(phase_stats["observed_pairs"]) | الكلمات النشطة: $(phase_stats["active_words"])")
        println("   ✓ حفظ تقرير التطور الطوري: phase_evolution_sample.json ($(phase_report["words_shifted"]) كلمة منزاحة)")
    catch e
        println("   ⚠ تطور المتجهات الطورية المخفف: $e")
    end
    println()

    # ═══ MorphoTwistor — عينة كلمات عالية التكرار بدل البحث الكامل في كل المعجم ═══
    println("4️⃣.3 تعلم قوالب الالتواء الصرفي — عينة كلمات عالية التكرار...")
    twistor = MirnanNew.Physics.MorphoTwistor.TwistorEngine()
    try
        twistor_words = _env_int("MIRNAN_TWISTOR_SAMPLE_WORDS", 20_000)
        twistor_pairs = _env_int("MIRNAN_TWISTOR_MAX_PAIRS", 8_000)
        twistor, twistor_stats = _train_twistor_sample!(
            vocab, word_freqs;
            max_words=twistor_words,
            max_pairs=twistor_pairs)
        println("   ✓ كلمات العينة: $(twistor_stats["sampled_words"]) | الأزواج: $(twistor_stats["pairs"]) | القوالب: $(twistor_stats["patterns"])")
    catch e
        println("   ⚠ تعلم قوالب الالتواء الصرفي المخفف: $e")
    end
    println()


    # ═══ حفظ قوالب الالتواء ═══
    try
        twistor_path = joinpath(MODEL_DIR, "twistor_patterns.json")
        twistor_data = Dict{String,Any}(
            "n_patterns" => length(twistor.patterns),
            "patterns" => [Dict(
                "label" => p.label,
                "count" => p.count,
                "examples" => [Dict("w1"=>w1, "w2"=>w2) for (w1,w2) in p.examples],
                "center_magnitude" => p.center.magnitude,
                "center_rotation" => p.center.rotation,
                "center_scale" => p.center.scale,
            ) for p in twistor.patterns]
        )
        open(twistor_path, "w") do io
            JSON.print(io, twistor_data)
        end
        println("   ✓ حفظ قوالب الالتواء إلى twistor_patterns.json ($(length(twistor.patterns)) قالب)")
    catch e
        println("   ⚠ حفظ القوالب: $e")
    end
    println()

    println("5️⃣  حفظ النموذج...")
    println("5.0 building al_lisan sentence-pattern memory...")
    try
        lisan = MirnanNew.Physics.LinguisticPatternMemory()
        learned_lisan = MirnanNew.Physics.train_lisan_from_texts!(
            lisan, texts; max_sentences=50_000)
        lisan_path = MirnanNew.Physics.save_lisan(lisan, joinpath(MODEL_DIR, "al_lisan.json"))
        println("   ✓ al_lisan learned $(learned_lisan) sentences and $(length(lisan.patterns)) patterns")
        println("   ✓ saved tongue memory: $(basename(lisan_path))")
    catch e
        println("   ⚠ al_lisan memory: $e")
    end
    println("5.0b building al_tadbir procedural-plan memory...")
    try
        tadbir = MirnanNew.Physics.TadbirMemory()
        learned_tadbir = MirnanNew.Physics.train_tadbir_from_texts!(
            tadbir, texts; max_plans=50_000)
        tadbir_path = MirnanNew.Physics.save_tadbir(tadbir, joinpath(MODEL_DIR, "al_tadbir.json"))
        println("   ✓ al_tadbir learned $(learned_tadbir) plan blocks and $(length(tadbir.patterns)) patterns")
        println("   ✓ saved procedural memory: $(basename(tadbir_path))")
    catch e
        println("   ⚠ al_tadbir memory: $e")
    end
    println("5.0c building al_hisab verified-math memory...")
    try
        hisab = MirnanNew.Physics.HisabMemory()
        learned_hisab = MirnanNew.Physics.train_hisab_from_texts!(
            hisab, texts; max_problems=50_000)
        hisab_path = MirnanNew.Physics.save_hisab(hisab, joinpath(MODEL_DIR, "al_hisab.json"))
        println("   ✓ al_hisab learned $(learned_hisab) math problems and $(length(hisab.patterns)) patterns")
        println("   ✓ saved verified math memory: $(basename(hisab_path))")
    catch e
        println("   ⚠ al_hisab memory: $e")
    end
    println("5.0d building al_ta3rif general definition-relation memory...")
    try
        ta3rif = MirnanNew.Physics.Ta3rifMemory()
        learned_ta3rif = MirnanNew.Physics.train_ta3rif_from_texts!(
            ta3rif, texts, metadata; max_items=50_000)
        persistent_ta3rif_path = joinpath(KNOWLEDGE_DIR, "definitions.json")
        persistent_ta3rif = MirnanNew.Physics.load_ta3rif(persistent_ta3rif_path)
        MirnanNew.Physics.merge_ta3rif!(ta3rif, persistent_ta3rif)
        ta3rif_path = MirnanNew.Physics.save_ta3rif(ta3rif, joinpath(MODEL_DIR, "al_ta3rif.json"))
        println("   ✓ al_ta3rif learned/merged $(learned_ta3rif) definition/relation facts and $(length(ta3rif.records)) subjects")
        println("   ✓ persistent definition subjects: $(length(persistent_ta3rif.records))")
        println("   ✓ saved definition memory: $(basename(ta3rif_path))")
    catch e
        println("   ⚠ al_ta3rif memory: $e")
    end
    println("5.0e building al_nisba learned semantic-relation memory...")
    try
        nisba = MirnanNew.Physics.NisbaMemory()
        semantic_relation_facts = _load_semantic_relation_facts(
            joinpath(KNOWLEDGE_DIR, "semantic_relation_facts.json"))
        learned_nisba = MirnanNew.Physics.train_nisba_from_texts!(
            nisba, texts, metadata; max_items=50_000)
        learned_nisba_facts = 0
        for (idx, fact) in enumerate(semantic_relation_facts)
            source = Dict{String,Any}(
                "file_name" => "semantic_relation_facts.json",
                "paragraph_index" => idx,
                "knowledge_type" => "structured_relation_fact")
            learned_nisba_facts += MirnanNew.Physics.learn_nisba_fact!(
                nisba,
                String(fact["relation_type"]),
                String.(fact["terms"]);
                markers=String[_semantic_fact_marker(fact)],
                polarity=Int(get(fact, "polarity", 1)),
                source=source)
        end
        nisba_path = MirnanNew.Physics.save_nisba(nisba, joinpath(MODEL_DIR, "al_nisba.json"))
        println("   ✓ al_nisba learned $(learned_nisba) semantic relations + $(learned_nisba_facts) persistent relation facts and $(length(nisba.relations)) relation records")
        println("   ✓ saved relation memory: $(basename(nisba_path))")
    catch e
        println("   ⚠ al_nisba memory: $e")
    end
    println("5.0f building al_muradif learned semantic-equivalence memory...")
    try
        muradif_words = _env_int("MIRNAN_MURADIF_MAX_WORDS", 5000)
        muradif_neighbors = _env_int("MIRNAN_MURADIF_TOP_NEIGHBORS", 24)
        muradif_candidates = _env_int("MIRNAN_MURADIF_MAX_CANDIDATES", 6)
        persistent_muradif_path = joinpath(KNOWLEDGE_DIR, "semantic_equivalence.json")
        persistent_muradif = MirnanNew.Physics.load_muradif(persistent_muradif_path)
        muradif = MirnanNew.Physics.build_muradif_memory(
            vocab, K_sem;
            K_syn=K_syn,
            K_causal=K_causal,
            max_words=muradif_words,
            top_neighbors=muradif_neighbors,
            max_candidates=muradif_candidates)
        MirnanNew.Physics.merge_muradif!(muradif, persistent_muradif)
        muradif_path = MirnanNew.Physics.save_muradif(
            muradif, joinpath(MODEL_DIR, "semantic_equivalence.json"))
        println("   ✓ al_muradif learned/merged $(length(muradif.entries)) semantic-equivalence entries")
        println("   ✓ persistent semantic equivalence entries: $(length(persistent_muradif.entries))")
        println("   ✓ saved semantic equivalence memory: $(basename(muradif_path))")
    catch e
        println("   ⚠ al_muradif memory: $e")
    end
    println("5.0g building al_istinbat learned inference-attention memory...")
    try
        persistent_istinbat_path = joinpath(KNOWLEDGE_DIR, "istinbat_attention.json")
        persistent_istinbat = MirnanNew.Physics.load_istinbat(persistent_istinbat_path)
        istinbat = MirnanNew.Physics.IstinbatAttentionMemory()
        learned_istinbat = MirnanNew.Physics.train_istinbat_from_texts!(
            istinbat, texts, metadata; max_items=50_000)
        learned_istinbat_uqra = 0
        uqra_dir = joinpath(DATA_DIR, "uqra")
        if isdir(uqra_dir)
            uqra_texts = String[]
            uqra_metadata = Dict{String,Any}[]
            _read_dir_recursive!(uqra_texts, uqra_dir, "uqra", 0;
                                 granularity=:line, metadata=uqra_metadata)
            for src in uqra_metadata
                src["source_dir"] = "uqra"
                src["knowledge_type"] = "structured_relation_seed"
            end
            learned_istinbat_uqra = MirnanNew.Physics.train_istinbat_from_texts!(
                istinbat, uqra_texts, uqra_metadata; max_items=20_000)
            println("   ✓ al_istinbat uqra focused pass: $(learned_istinbat_uqra) observations from $(length(uqra_texts)) lines")
        end
        learned_persistent_istinbat_examples = 0
        for rec in values(persistent_istinbat.records)
            for (ex_idx, ex) in enumerate(rec.examples)
                source = ex_idx <= length(rec.source_metadata) ?
                    copy(rec.source_metadata[ex_idx]) : Dict{String,Any}()
                source["knowledge_type"] = "persistent_istinbat_example"
                source["record_id"] = rec.record_id
                learned_persistent_istinbat_examples +=
                    MirnanNew.Physics.learn_opposition_from_text!(istinbat, ex, source)
                learned_persistent_istinbat_examples +=
                    MirnanNew.Physics.learn_direct_negation_from_text!(istinbat, ex, source)
            end
        end
        semantic_relation_facts = _load_semantic_relation_facts(
            joinpath(KNOWLEDGE_DIR, "semantic_relation_facts.json"))
        learned_istinbat_facts = 0
        for (idx, fact) in enumerate(semantic_relation_facts)
            source = Dict{String,Any}(
                "file_name" => "semantic_relation_facts.json",
                "paragraph_index" => idx,
                "knowledge_type" => "structured_relation_fact")
            before_terms = _as_string_vec(get(fact, "subject", Any[]))
            isempty(before_terms) && (before_terms = String.(fact["terms"])[1:1])
            after_terms = _as_string_vec(get(fact, "object", Any[]))
            append!(after_terms, _as_string_vec(get(fact, "target_terms", Any[])))
            append!(after_terms, _as_string_vec(get(fact, "context_terms", Any[])))
            append!(after_terms, _as_string_vec(get(fact, "contrast_terms", Any[])))
            isempty(after_terms) && length(fact["terms"]) > 1 && (after_terms = String.(fact["terms"])[2:end])
            focus_terms = String.(fact["terms"])
            learned_istinbat_facts += MirnanNew.Physics.learn_istinbat_fact!(
                istinbat,
                String(fact["relation_type"]);
                marker=_semantic_fact_marker(fact),
                before_terms=before_terms,
                after_terms=after_terms,
                focus_terms=focus_terms,
                polarity=Int(get(fact, "polarity", 1)),
                source=source)
        end
        MirnanNew.Physics.merge_istinbat!(istinbat, persistent_istinbat)
        
        # ═══ الاكتشاف الذاتي للمفاتيح ═══
        println("   🔍 تشغيل محرك الاكتشاف الذاتي للمفاتيح (Marker Discovery)...")
        discovered = MirnanNew.Physics.discover_markers(texts, vocab, id2word)
        for (word, dm) in discovered
            istinbat.discovered_markers[word] = dm.category
            istinbat.discovered_confidences[word] = dm.confidence
        end
        println("   🔍 تم اكتشاف $(length(discovered)) علامات بنجاح ودمجها في ذاكرة الاستنباط.")

        istinbat_path = MirnanNew.Physics.save_istinbat(
            istinbat, joinpath(MODEL_DIR, "al_istinbat.json"))
        println("   ✓ al_istinbat learned/merged $(learned_istinbat) inference-attention observations + $(learned_istinbat_uqra) uqra focused observations + $(learned_istinbat_facts) persistent relation facts + $(learned_persistent_istinbat_examples) persistent example observations and $(length(istinbat.records)) records")
        println("   ✓ persistent istinbat attention records: $(length(persistent_istinbat.records))")
        println("   ✓ saved inference attention memory: $(basename(istinbat_path))")
    catch e
        println("   ⚠ al_istinbat memory: $e")
    end
    println("5.0g.1 building quantity frame memory...")
    try
        quantity_mem = MirnanNew.Physics.QuantityFrameMemory()
        learned_quantity = MirnanNew.Physics.train_quantity_frames_from_texts!(
            quantity_mem, texts, metadata; max_items=50_000)
        quantity_path = MirnanNew.Physics.save_quantity_memory(
            quantity_mem, joinpath(MODEL_DIR, "quantity_memory.json"))
        println("   ✓ quantity memory learned $(learned_quantity) quantity frames")
        println("   ✓ saved quantity memory: $(basename(quantity_path))")
    catch e
        println("   ⚠ quantity memory: $e")
    end
    println("5.0h building al_hisban_al_dalali Clifford semantic-calculus memory...")
    try
        hisban = MirnanNew.Physics.SemanticCalculusMemory()
        learned_hisban = MirnanNew.Physics.train_semantic_calculus_from_texts!(
            hisban, texts; max_pairs=50_000)
        hisban_path = MirnanNew.Physics.save_semantic_calculus(
            hisban, joinpath(MODEL_DIR, "al_hisban_al_dalali.json"))
        println("   ✓ al_hisban learned $(learned_hisban) semantic transitions and $(length(hisban.records)) relation records")
        println("   ✓ saved semantic calculus memory: $(basename(hisban_path))")
        println("5.0i building semantic scene imagination memory...")
        try
            scene_mem = MirnanNew.Physics.SemanticSceneMemory(max_scenes=50_000)
            learned_scenes = MirnanNew.Physics.train_semantic_scenes_from_texts!(
                scene_mem, hisban, texts; max_items=50_000)
            scene_path = MirnanNew.Physics.save_semantic_scenes(
                scene_mem, joinpath(MODEL_DIR, "semantic_scenes.json"))
            println("   ✓ semantic scenes learned $(learned_scenes) event scenes")
            println("   ✓ saved semantic scene memory: $(basename(scene_path))")
        catch scene_error
            println("   ⚠ semantic scene memory: $scene_error")
        end
    catch e
        println("   ⚠ al_hisban_al_dalali memory: $e")
    end
    # ═══ حساب متجهات مراكز جاذبية الفقرات (Paragraph Centroids) ═══
    println("🎯 حساب متجهات مراكز جاذبية الفقرات...")
    try
        paragraph_centroids = Dict{String,Any}[]
        centroid_limit = _env_int("MIRNAN_PARAGRAPH_CENTROID_LIMIT", 5_000)
        centroid_indices = _even_sample_indices(length(texts), centroid_limit)
        centroid_index_set = Set(centroid_indices)
        println("   عينة مراكز الفقرات: $(length(centroid_indices)) من $(length(texts)) وثيقة")
        for (j, para) in enumerate(texts)
            (j in centroid_index_set) || continue
            words = String[_strip_punct_boundary(strip(w)) for w in split(para) if length(strip(w)) >= 2]
            isempty(words) && continue
            
            dim = 0
            centroid_vec = Float64[]
            sum_mass = 0.0
            for w in words
                pv = try
                    Float64.(MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w))
                catch
                    nothing
                end
                pv === nothing && continue
                if dim == 0
                    dim = length(pv)
                    centroid_vec = zeros(Float64, dim)
                end
                
                g_w = try
                    Float64(MirnanNew.Physics.WordPhysics.compute_word_mass(w))
                catch
                    1.0
                end
                
                centroid_vec .+= g_w .* pv
                sum_mass += g_w
            end
            
            if sum_mass > 0.0
                centroid_vec ./= sum_mass
                f_name = (j <= length(metadata)) ? metadata[j]["file_name"] : "unknown.txt"
                p_idx = (j <= length(metadata)) ? metadata[j]["paragraph_index"] : j
                push!(paragraph_centroids, Dict{String,Any}(
                    "file_name" => f_name,
                    "paragraph_index" => p_idx,
                    "centroid" => Float32.(centroid_vec)
                ))
            end
            if length(paragraph_centroids) > 0 && length(paragraph_centroids) % 1000 == 0
                println("     مراكز محفوظة مؤقتاً: $(length(paragraph_centroids))")
                flush(stdout)
            end
        end
        centroid_path = joinpath(MODEL_DIR, "paragraph_centroids.json")
        open(centroid_path, "w") do io
            JSON.print(io, paragraph_centroids)
        end
        println("   ✓ تم حفظ متجهات مراكز الفقرات إلى $(basename(centroid_path)) ($(length(paragraph_centroids)) فقرة)")
    catch e
        println("   ⚠ فشل حساب مراكز الفقرات: $e")
    end
    println()

    # ═══ بناء قاعدة معرفة RAPG (RAPGKnowledgeBase) ═══
    println("🎯 بناء قاعدة المعرفة RAPG (RAPGKnowledgeBase)...")
    try
        db_path = joinpath(MODEL_DIR, "rapg_kb.db")
        # إزالة قاعدة البيانات القديمة إن وجدت للبدء من جديد
        isfile(db_path) && _safe_rm(db_path)
        MirnanNew.init_rapg_db!(db_path)
        rapg_batch = Tuple{String,Vector{Float32},String}[]
        rapg_stored = 0

        function flush_rapg_batch!()
            isempty(rapg_batch) && return
            rapg_stored += MirnanNew.Physics.RAPGModule.store_passages!(db_path, rapg_batch)
            empty!(rapg_batch)
            if rapg_stored > 0 && rapg_stored % 1000 == 0
                println("     RAPG passages stored: $(rapg_stored)")
                flush(stdout)
            end
        end

        function queue_rapg_passage!(content::AbstractString, source::AbstractString)
            text = strip(content)
            isempty(text) && return
            vec = MirnanNew.Physics.RAPGModule.get_mirnan_vector(text, w -> MirnanNew.Physics.WordPhysics.compute_extended_phase_vector(w))
            push!(rapg_batch, (String(text), vec, String(source)))
            length(rapg_batch) >= 250 && flush_rapg_batch!()
        end
        
        # 1. definitions.json
        def_path = joinpath(KNOWLEDGE_DIR, "definitions.json")
        if isfile(def_path)
            def_data = JSON.parsefile(def_path)
            for rec in get(def_data, "records", [])
                subject = get(rec, "subject", "")
                defs = get(rec, "definitions", Dict())
                for (d, _) in defs
                    sentence = "$subject هو $d"
                    queue_rapg_passage!(sentence, "definitions")
                end
                for ex in get(rec, "examples", [])
                    queue_rapg_passage!(ex, "definitions")
                end
            end
        end
        
        # 2. semantic_relation_facts.json
        rel_path = joinpath(KNOWLEDGE_DIR, "semantic_relation_facts.json")
        if isfile(rel_path)
            rel_data = JSON.parsefile(rel_path)
            for rec in get(rel_data, "records", [])
                sub = get(rec, "subject", "")
                obj = get(rec, "object", "")
                marker = get(rec, "marker", "")
                if !isempty(sub) && !isempty(obj)
                    sentence = "$sub $marker $obj"
                    queue_rapg_passage!(sentence, "semantic_relation_facts")
                end
            end
        end
        
        # 3. istinbat_attention.json
        ist_path = joinpath(KNOWLEDGE_DIR, "istinbat_attention.json")
        if isfile(ist_path)
            ist_data = JSON.parsefile(ist_path)
            for rec in get(ist_data, "records", [])
                for ex in get(rec, "examples", [])
                    queue_rapg_passage!(ex, "istinbat_attention")
                end
            end
        end
        
        # 4. نصوص فقرات التدريب
        rapg_training_limit = _env_int("MIRNAN_RAPG_TRAINING_LIMIT", 12_000)
        rapg_training_indices = _even_sample_indices(length(texts), rapg_training_limit)
        rapg_training_set = Set(rapg_training_indices)
        println("   RAPG training-corpus sample: $(length(rapg_training_indices)) من $(length(texts)) وثيقة")
        for (idx, para) in enumerate(texts)
            (idx in rapg_training_set) || continue
            trimmed = strip(para)
            isempty(trimmed) && continue
            queue_rapg_passage!(trimmed, "training_corpus")
        end
        flush_rapg_batch!()
        
        kb_test = MirnanNew.load_rapg_kb(db_path)
        println("   ✓ تم بناء وحفظ قاعدة معرفة RAPG بنجاح بنسبة $(length(kb_test.passages)) فقرة معرفية.")
    catch e
        println("   ⚠ فشل بناء قاعدة المعرفة RAPG: $e")
    end
    println()


    # ═══ الأبعاد الموضوعية للحروف (Letter Topic Embeddings) ═══
    println("🎯 تدريب الأبعاد الموضوعية للحروف (ذاكرة الحرف الموضوعية)...")
    try
        # Extract facts/concepts sentences and append them to texts to enrich letter embeddings
        k_sentences = extract_knowledge_sentences(KNOWLEDGE_DIR)
        println("   ℹ تم استخراج $(length(k_sentences)) جملة معرفية من فضاء الحقائق لدمجها في أبعاد الحروف.")
        enriched_texts = vcat(texts, k_sentences)
        MirnanNew.Physics.WordPhysics.train_letter_topic_embeddings!(enriched_texts, vocab, MODEL_DIR; min_occurrences=10)
    catch e
        println("   ⚠ فشل تدريب الأبعاد الموضوعية للحروف: $e")
    end
    println()

    save_model(vocab, K_sem, K_syn, K_dial, K_causal, texts)
    println("   ✓ vocab.json + K_sem.dat + K_syn.dat + K_dialogue.dat + K_causal.dat + corpus_sentences.dat")
    println()

    # ═══ بناء K_code منفصلة للكود البرمجي ═══
    if !isempty(code_blocks)
        println("🖥  بناء K_code (مصفوفة اقتران منفصلة للكود)...")
        code_vocab, K_code_mat, _ = MirnanNew.Physics.CodeEngineModule.build_K_code(code_blocks)
        println("   ✓ K_code: $(code_vocab.next-1) رمز برمجي | $(length(K_code_mat.nzval)) اقتران")
        MirnanNew.Physics.CodeEngineModule.save_code_model(
            MirnanNew.Physics.CodeEngineModule.CodeEngine(code_vocab=code_vocab, K_code=K_code_mat, cpv=MirnanNew.Physics.CodeEngineModule.CodePhaseVector()),
            MODEL_DIR)
        println("   ✓ code_vocab.json + K_code.dat")
        try
            code_memory = MirnanNew.Physics.CodePatternMemory()
            learned_code_patterns = MirnanNew.Physics.train_code_patterns_from_texts!(
                code_memory, code_blocks; max_blocks=50_000)
            code_memory_path = MirnanNew.Physics.save_al_code(
                code_memory, joinpath(MODEL_DIR, "al_code.json"))
            println("   ✓ al_code learned $(learned_code_patterns) code blocks and $(length(code_memory.patterns)) patterns")
            println("   ✓ saved code pattern memory: $(basename(code_memory_path))")
        catch e
            println("   ⚠ al_code memory: $e")
        end
        println()
    end

    println("6️⃣  اختبار التوليد...")
    gen = MirnanNew.Physics.Generator.MirnanGenerator(vocab, K_sem; K_syn=K_syn,
                                                       K_causal=K_causal,
                                                       top_k=min(100, length(vocab)),
                                                       model_dir=MODEL_DIR)

    if !isempty(aql_training_sentences)
        println("5️⃣.1 بناء ذاكرة al_aql السببية من النصوص...")
        for i in 1:500:length(aql_training_texts)
            chunk = join(aql_training_texts[i:min(i + 499, end)], "\n")
            MirnanNew.Physics.AlAql.train_from_text!(gen.aql_space, chunk)
        end
        saved_runtime = MirnanNew.Physics.Generator.save_runtime_learning!(gen)
        println("   al_aql speech acts: $(length(gen.aql_space.speech_acts))")
        println("   ✓ حفظ معرفة al_aql: $(length(gen.aql_space.templates)) نمط، $(length(gen.aql_space.dynamic_verbs)) فعل، $(length(gen.aql_space.semantic_relations)) علاقة")
        println("   ✓ $saved_runtime")
        println()
    end

    test_prompts = ["العلم نور", "السماء صافية"]
    for prompt in test_prompts
        result = try
            MirnanNew.Physics.Generator.generate!(gen, prompt; max_words=5)
        catch e
            println("   ⚠ خطأ في التوليد: $e")
            ""
        end
        println("   ↳ \"$prompt\" → $result")
        flush(stdout)
    end
    println()

    end_time = now()
    elapsed = end_time - start_time
    total_seconds = Dates.value(elapsed) / 1000
    hours = floor(Int, total_seconds / 3600)
    minutes = floor(Int, (total_seconds % 3600) / 60)
    seconds = round(Int, total_seconds % 60)
    println("⏱  وقت انتهاء التدريب: ", Dates.format(end_time, "yyyy-mm-dd HH:MM:SS"))
    println("⏱  المدة الإجمالية: $hours ساعة، $minutes دقيقة، $seconds ثانية")
    println()

    println("╔══════════════════════════════════════════════╗")
    println("║           اكتمل التدريب بنجاح ✓             ║")
    println("╚══════════════════════════════════════════════╝")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
