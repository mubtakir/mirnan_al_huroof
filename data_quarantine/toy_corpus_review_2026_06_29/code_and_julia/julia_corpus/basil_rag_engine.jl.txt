module RAGEngine

using SQLite
using JSON3
using Dates
using LinearAlgebra
using Mirnan

export RAGManager, init_rag_db!, store_memory!, search_memories, semantic_recall, get_project_context, auto_index_observation!

struct RAGManager
    db_path::String
    session_id::String
end

function init_rag_db!(rag::RAGManager)
    conn = SQLite.DB(rag.db_path)
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
        @error "Failed to initialize RAG database: $e"
    end
end

function _get_mirnan_vector(text::String)::Vector{Float32}
    words = split(lowercase(text))
    # Filter empty or non-alphabetic/numeric words
    valid_words = [String(w) for w in words if length(filter(c -> isletter(c) || isnumeric(c), String(w))) >= 2]
    if isempty(valid_words)
        return zeros(Float32, 10000)
    end
    
    vec_sum = zeros(Float32, 10000)
    for w in valid_words
        try
            pv = Mirnan.compute_extended_phase_vector(w)
            vec_sum .+= pv
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

function _vector_similarity(v1::Vector{Float32}, v2::Vector{Float32})::Float64
    n1 = norm(v1)
    n2 = norm(v2)
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    return max(0.0, dot(v1, v2) / (n1 * n2))
end

function store_memory!(rag::RAGManager, content::String; source::String="general")::Bool
    if isempty(strip(content))
        return false
    end
    
    conn = SQLite.DB(rag.db_path)
    try
        vec = _get_mirnan_vector(content)
        embedding_json = JSON3.write(vec)
        
        stmt = SQLite.Stmt(conn, "INSERT INTO embeddings (session_id, source, content, embedding, timestamp) VALUES (?, ?, ?, ?, ?)")
        SQLite.execute(stmt, (rag.session_id, source, content, embedding_json, time()))
        return true
    catch e
        @error "RAG Store failed: $e"
        return false
    end
end

function search_memories(rag::RAGManager, query::String; top_k::Int=5, source_filter::Union{String, Nothing}=nothing, min_similarity::Float64=0.15)
    conn = SQLite.DB(rag.db_path)
    results = []
    
    try
        query_vec = _get_mirnan_vector(query)
        if norm(query_vec) < 1e-5
            return []
        end
        
        cursor = if source_filter !== nothing
            SQLite.DBInterface.execute(conn, "SELECT id, session_id, source, content, embedding, timestamp FROM embeddings WHERE source = ? ORDER BY timestamp DESC LIMIT 500", (source_filter,))
        else
            SQLite.DBInterface.execute(conn, "SELECT id, session_id, source, content, embedding, timestamp FROM embeddings ORDER BY timestamp DESC LIMIT 500")
        end
        
        scored = []
        for row in cursor
            emb_json = row.embedding
            emb_vec = try
                JSON3.read(emb_json, Vector{Float32})
            catch
                # Fallback in case of old TF-IDF dict records
                zeros(Float32, 10000)
            end
            
            sim = _vector_similarity(query_vec, emb_vec)
            if sim >= min_similarity
                push!(scored, (sim, Dict{Symbol, Any}(
                    :id => row.id,
                    :source => row.source,
                    :content => row.content,
                    :similarity => round(sim, digits=3),
                    :session_id => row.session_id,
                    :timestamp => row.timestamp
                )))
            end
        end
        
        sort!(scored, by = x -> x[1], rev=true)
        return [item for (_, item) in first(scored, top_k)]
    catch e
        @error "RAG search failed: $e"
        return []
    end
end

function semantic_recall(rag::RAGManager, query::String; top_k::Int=5)::String
    results = search_memories(rag, query, top_k=top_k)
    if isempty(results)
        return "[RAG] No semantic memories found for: '$(first(query, 80))'"
    end
    
    lines = ["[RAG RECALL: '$(first(query, 60))'] — Found $(length(results)) relevant memories:\n"]
    for (i, r) in enumerate(results)
        preview = replace(first(r[:content], 250), "\n" => " ")
        push!(lines, "  $i. [$(r[:source])] (sim=$(r[:similarity])) $preview")
    end
    return join(lines, "\n")
end

function get_project_context(rag::RAGManager)::String
    all_sources = ["architecture", "project", "error_solved", "code"]
    context_parts = String[]
    conn = SQLite.DB(rag.db_path)
    
    for source in all_sources
        try
            cursor = SQLite.DBInterface.execute(conn, "SELECT content FROM embeddings WHERE source = ? ORDER BY timestamp DESC LIMIT 5", (source,))
            rows = collect(cursor)
            if !isempty(rows)
                entries = join(["  - $(first(r.content, 200))" for r in rows], "\n")
                push!(context_parts, "[$(uppercase(source))]\n$entries")
            end
        catch e
            @debug "Failed fetching RAG source $source: $e"
        end
    end
    
    if isempty(context_parts)
        return "[RAG] No project context accumulated yet."
    end
    
    return "### Sovereign RAG Context (Cross-Session Knowledge)\n" * join(context_parts, "\n\n")
end

function auto_index_observation!(rag::RAGManager, observation::String, tool_name::String, success::Bool)
    important_tools = Set(["write_file", "patch_file", "runtime_run_code", "runtime_run_tests", "runtime_install", "browser_get_text"])
    if tool_name in important_tools && length(observation) > 50
        source = occursin("[ERROR]", observation) && success ? "error_solved" : "observation"
        store_memory!(rag, first(observation, 800), source=source)
    end
end

end # module RAGEngine
