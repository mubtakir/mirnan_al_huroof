module Memory

using SQLite
using Dates

export SovereignDatabase, init_db!, add_episode!, add_or_update_knowledge!, query_memories, get_recent_episodes!

struct SovereignDatabase
    db_path::String
end

function init_db!(db::SovereignDatabase)
    db_dir = dirname(db.db_path)
    if !isempty(db_dir)
        mkpath(db_dir)
    end
    conn = SQLite.DB(db.db_path)
    try
        # Episodes table (Episodic Memory)
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                summary TEXT,
                msg_count INTEGER,
                tags TEXT
            )
        """)
        # Knowledge table (Semantic Memory)
        SQLite.execute(conn, """
            CREATE TABLE IF NOT EXISTS knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                error_signature TEXT UNIQUE,
                fix_content TEXT,
                success_count INTEGER DEFAULT 1,
                last_updated TEXT
            )
        """)
    catch e
        @error "Failed to initialize SQLite database: $e"
    end
end

function add_episode!(db::SovereignDatabase, summary::String, msg_count::Int, tags::String="")
    conn = SQLite.DB(db.db_path)
    try
        stmt = SQLite.Stmt(conn, "INSERT INTO episodes (timestamp, summary, msg_count, tags) VALUES (?, ?, ?, ?)")
        SQLite.execute(stmt, (Dates.format(now(), "yyyy-mm-ddTHH:MM:SS"), summary, msg_count, tags))
    catch e
        @warn "Failed to add episode: $e"
    end
end

function add_or_update_knowledge!(db::SovereignDatabase, signature::String, fix::String)
    conn = SQLite.DB(db.db_path)
    try
        stmt = SQLite.Stmt(conn, """
            INSERT INTO knowledge (error_signature, fix_content, last_updated)
            VALUES (?, ?, ?)
            ON CONFLICT(error_signature) DO UPDATE SET
                fix_content = excluded.fix_content,
                success_count = success_count + 1,
                last_updated = excluded.last_updated
        """)
        SQLite.execute(stmt, (signature, fix, Dates.format(now(), "yyyy-mm-ddTHH:MM:SS")))
    catch e
        @warn "Failed to add/update knowledge: $e"
    end
end

function query_memories(db::SovereignDatabase, keywords::Vector{String}; limit::Int=5)
    if isempty(keywords)
        return []
    end
    conn = SQLite.DB(db.db_path)
    results_map = Dict{String, Dict{Symbol, Any}}()
    
    try
        # Search episodes
        for kw in keywords
            kw_pattern = "%$kw%"
            cursor = SQLite.DBInterface.execute(conn, "SELECT id, timestamp, summary FROM episodes WHERE summary LIKE ?", (kw_pattern,))
            for row in cursor
                id_str = "ep_$(row.id)"
                if !haskey(results_map, id_str)
                    results_map[id_str] = Dict{Symbol, Any}(
                        :score => 0.0,
                        :type => "episode",
                        :ts => row.timestamp,
                        :content => row.summary
                    )
                end
                results_map[id_str][:score] += 1.0
            end
        end

        # Search knowledge
        for kw in keywords
            kw_pattern = "%$kw%"
            cursor = SQLite.DBInterface.execute(conn, "SELECT id, error_signature, fix_content FROM knowledge WHERE error_signature LIKE ? OR fix_content LIKE ?", (kw_pattern, kw_pattern))
            for row in cursor
                id_str = "kn_$(row.id)"
                if !haskey(results_map, id_str)
                    results_map[id_str] = Dict{Symbol, Any}(
                        :score => 0.0,
                        :type => "knowledge",
                        :sig => row.error_signature,
                        :content => row.fix_content
                    )
                end
                results_map[id_str][:score] += 2.0
            end
        end
    catch e
        @error "Querying database failed: $e"
    end

    # Sort and return
    sorted_results = sort(collect(values(results_map)), by = x -> (x[:score], get(x, :ts, "")), rev=true)
    return first(sorted_results, limit)
end

function get_recent_episodes!(db::SovereignDatabase; limit::Int=5)
    conn = SQLite.DB(db.db_path)
    results = []
    try
        cursor = SQLite.DBInterface.execute(conn, "SELECT timestamp, summary FROM episodes ORDER BY timestamp DESC LIMIT ?", (limit,))
        for row in cursor
            push!(results, Dict{Symbol, Any}(
                :type => "episode",
                :ts => row.timestamp,
                :content => row.summary
            ))
        end
    catch e
        @debug "Failed to get recent episodes: $e"
    end
    return results
end

end # module Memory
