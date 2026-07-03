"""
RAPG (Retrieval-Augmented Physical Generation) — التوليد الفيزيائي المسترد.
Reuses the SQLite tables format from basil_agent for maximum compatibility.
"""
module RAPGModule

using SQLite
using JSON
using LinearAlgebra
using ..Constants
using ..WordPhysics: compute_extended_phase_vector

export RAPGKnowledgeBase, init_rapg_db!, store_passage!, store_passages!, search_passages, retrieve, retrieve_by_category, get_mirnan_vector, load_rapg_kb

mutable struct RAPGKnowledgeBase
    passages::Vector{String}
    embeddings::Matrix{Float32} # 10000 × N
    sources::Vector{String}
    db_path::String
end

"""
    init_rapg_db!(db_path)

تهيئة قاعدة بيانات SQLite وجداولها المتوافقة مع عميل Basil.
"""
function init_rapg_db!(db_path::AbstractString)
    conn = SQLite.DB(db_path)
    try
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS embeddings (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                source   TEXT,
                content  TEXT,
                embedding TEXT,
                timestamp REAL
            )
        """)
        SQLite.execute(conn, "CREATE INDEX IF NOT EXISTS idx_emb_source ON embeddings(source)")
    catch e
        @error "Failed to initialize RAPG database: $e"
    end
end

"""
    store_passage!(db_path, content, vector, source; session_id)

تخزين فقرة معرفية في قاعدة البيانات مع متجهها الممتد.
"""
function store_passage!(db_path::AbstractString, content::AbstractString, vector::Vector{Float32}, source::AbstractString; session_id::AbstractString="GLOBAL")::Bool
    if isempty(strip(content))
        return false
    end
    conn = SQLite.DB(db_path)
    try
        embedding_json = JSON.json(vector)
        stmt = SQLite.Stmt(conn, "INSERT INTO embeddings (session_id, source, content, embedding, timestamp) VALUES (?, ?, ?, ?, ?)")
        SQLite.execute(stmt, (session_id, source, content, embedding_json, time()))
        return true
    catch e
        @error "RAPG Store failed: $e"
        return false
    end
end

"""
    store_passages!(db_path, records; session_id)

Store many RAPG records using one SQLite connection and one transaction.
Each record is `(content, vector, source)`.
"""
function store_passages!(db_path::AbstractString,
                         records::AbstractVector{<:Tuple{<:AbstractString,Vector{Float32},<:AbstractString}};
                         session_id::AbstractString="GLOBAL")::Int
    isempty(records) && return 0
    conn = SQLite.DB(db_path)
    stored = 0
    try
        SQLite.execute(conn, "PRAGMA synchronous=NORMAL")
        SQLite.execute(conn, "BEGIN TRANSACTION")
        stmt = SQLite.Stmt(conn, "INSERT INTO embeddings (session_id, source, content, embedding, timestamp) VALUES (?, ?, ?, ?, ?)")
        now = time()
        for (content, vector, source) in records
            isempty(strip(content)) && continue
            embedding_json = JSON.json(vector)
            SQLite.execute(stmt, (session_id, String(source), String(content), embedding_json, now))
            stored += 1
        end
        SQLite.execute(conn, "COMMIT")
        return stored
    catch e
        try
            SQLite.execute(conn, "ROLLBACK")
        catch
        end
        @error "RAPG batch store failed: $e"
        return stored
    end
end

"""
    get_mirnan_vector(text, pv_fn) -> Vector{Float32}

تحويل نص كامل إلى متجه طوري ممتد عبر متوسط متجهات كلماته.
"""
function get_mirnan_vector(text::AbstractString, pv_fn)::Vector{Float32}
    words = split(lowercase(text))
    # تصفية الكلمات الفارغة وغير المفيدة
    valid_words = [String(w) for w in words if length(filter(c -> isletter(c) || isnumeric(c) || '\u0600' <= c <= '\u06FF', String(w))) >= 2]
    if isempty(valid_words)
        return zeros(Float32, Constants.TOTAL_DIM)
    end
    
    vec_sum = zeros(Float32, Constants.TOTAL_DIM)
    for w in valid_words
        try
            pv = pv_fn(w)
            if length(pv) == Constants.TOTAL_DIM
                vec_sum .+= Float32.(pv)
            end
        catch e
            # fallback
        end
    end
    
    nrm = norm(vec_sum)
    if nrm > 1e-10
        vec_sum ./= nrm
    end
    return vec_sum
end

"""
    load_rapg_kb(db_path) -> RAPGKnowledgeBase

تحميل نصوص ومتجهات قاعدة بيانات SQLite بالكامل للذاكرة للبحث السريع.
"""
function load_rapg_kb(db_path::AbstractString)::RAPGKnowledgeBase
    passages = String[]
    embeddings_list = Vector{Float32}[]
    sources = String[]
    
    if isfile(db_path)
        try
            conn = SQLite.DB(db_path)
            table_check = SQLite.DBInterface.execute(conn, "SELECT name FROM sqlite_master WHERE type='table' AND name='embeddings'") |> collect
            if !isempty(table_check)
                rows = SQLite.DBInterface.execute(conn, "SELECT content, embedding, source FROM embeddings")
                for r in rows
                    content = r[1]
                    emb_str = r[2]
                    source = r[3]
                    
                    if ismissing(content) || ismissing(emb_str)
                        continue
                    end
                    
                    emb_vec = try
                        parsed = JSON.parse(emb_str)
                        if parsed isa AbstractVector
                            Vector{Float32}(parsed)
                        elseif parsed isa AbstractDict
                            # Fallback in case of old TF-IDF dict format
                            zeros(Float32, Constants.TOTAL_DIM)
                        else
                            zeros(Float32, Constants.TOTAL_DIM)
                        end
                    catch
                        zeros(Float32, Constants.TOTAL_DIM)
                    end
                    
                    if length(emb_vec) == Constants.TOTAL_DIM
                        push!(passages, String(content))
                        push!(embeddings_list, emb_vec)
                        push!(sources, ismissing(source) ? "general" : String(source))
                    end
                end
            end
        catch e
            @warn "Failed to load RAPGKnowledgeBase from $db_path: $e"
        end
    end
    
    N = length(passages)
    embeddings_mat = zeros(Float32, Constants.TOTAL_DIM, N)
    for i in 1:N
        embeddings_mat[:, i] .= embeddings_list[i]
    end
    
    return RAPGKnowledgeBase(passages, embeddings_mat, sources, db_path)
end

"""
    retrieve(kb, query, k, pv_fn) -> (passages, pvs, similarities)

استرجاع أعلى k فقرات مطابقة للاستعلام باستخدام تشابه جيب التمام في الذاكرة.
"""
function retrieve(kb::RAPGKnowledgeBase, query::AbstractString, k::Int, pv_fn)
    N = length(kb.passages)
    if N == 0
        return String[], Vector{Float32}[], Float64[]
    end
    
    query_vec = get_mirnan_vector(query, pv_fn)
    q_norm = norm(query_vec)
    if q_norm < 1e-5
        return String[], Vector{Float32}[], Float64[]
    end
    
    similarities = zeros(Float64, N)
    for i in 1:N
        col = kb.embeddings[:, i]
        cn = norm(col)
        if cn > 1e-5
            similarities[i] = dot(query_vec, col) / cn
        else
            similarities[i] = 0.0
        end
    end
    
    idx = sortperm(similarities, rev=true)
    top_k = min(k, N)
    top_idx = idx[1:top_k]
    
    # فلترة النتائج التي تحقق حداً أدنى من التشابه
    valid_idx = [i for i in top_idx if similarities[i] >= 0.10]
    
    ret_passages = kb.passages[valid_idx]
    ret_pvs = [kb.embeddings[:, i] for i in valid_idx]
    ret_sims = similarities[valid_idx]
    
    return ret_passages, ret_pvs, ret_sims
end

"""
    retrieve_by_category(kb, query, category, k, pv_fn) -> (passages, pvs, similarities)

استرجاع أعلى k فقرات مطابقة للاستعلام باستخدام تشابه جيب التمام، مع الفلترة حسب المصدر (Source/Category).
"""
function retrieve_by_category(kb::RAPGKnowledgeBase, query::AbstractString, category::AbstractString, k::Int, pv_fn)
    target_source = if category == "question"
        "definitions"
    elseif category == "causal"
        "istinbat_attention"
    elseif category == "negation"
        "semantic_relation_facts"
    elseif category == "connector"
        "training_corpus"
    else
        "all"
    end

    N = length(kb.passages)
    if N == 0
        return String[], Vector{Float32}[], Float64[]
    end

    query_vec = get_mirnan_vector(query, pv_fn)
    q_norm = norm(query_vec)
    if q_norm < 1e-5
        return String[], Vector{Float32}[], Float64[]
    end

    matched_indices = Int[]
    for i in 1:N
        if target_source == "all" || kb.sources[i] == target_source
            push!(matched_indices, i)
        end
    end

    if isempty(matched_indices)
        matched_indices = collect(1:N)
    end

    similarities = Float64[]
    for i in matched_indices
        col = kb.embeddings[:, i]
        cn = norm(col)
        if cn > 1e-5
            push!(similarities, dot(query_vec, col) / cn)
        else
            push!(similarities, 0.0)
        end
    end

    idx = sortperm(similarities, rev=true)
    top_k = min(k, length(matched_indices))
    top_idx = idx[1:top_k]

    valid_top_idx = [i for i in top_idx if similarities[i] >= 0.10]

    final_indices = matched_indices[valid_top_idx]
    ret_passages = kb.passages[final_indices]
    ret_pvs = [kb.embeddings[:, i] for i in final_indices]
    ret_sims = similarities[valid_top_idx]

    return ret_passages, ret_pvs, ret_sims
end

end # module RAPGModule
