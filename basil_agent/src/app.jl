module App

using HTTP
using JSON3
using Dates
using ..Constants
using ..Agent
using ..Planner

export start_server

global _agent = nothing

function get_agent()
    global _agent
    if _agent === nothing
        _agent = MajnoonAgent(Constants.WORKSPACE_ROOT)
    end
    return _agent
end

function handle_route(req::HTTP.Request)
    uri = HTTP.URI(req.target)
    path = uri.path
    method = req.method
    
    @info "🌐 [HTTP] $method $path"
    
    cors_headers = [
        "Access-Control-Allow-Origin" => "*",
        "Access-Control-Allow-Methods" => "POST, GET, OPTIONS",
        "Access-Control-Allow-Headers" => "Content-Type, X-Request-ID"
    ]
    
    if method == "OPTIONS"
        return HTTP.Response(200, cors_headers, "")
    end
    
    if path == "/"
        html_path = joinpath(dirname(@__DIR__), "templates", "index.html")
        if isfile(html_path)
            body = read(html_path, String)
            return HTTP.Response(200, [cors_headers..., "Content-Type" => "text/html; charset=utf-8"], body)
        else
            return HTTP.Response(404, cors_headers, "Welcome to BasilAgent.jl (index.html not found)")
        end
        
    elseif startswith(path, "/static/")
        static_file = strip(path[length("/static/")+1:end], '/')
        if occursin("..", static_file)
            return HTTP.Response(403, cors_headers, "Forbidden")
        end
        file_path = joinpath(dirname(@__DIR__), "static", static_file)
        if isfile(file_path)
            content_type = endswith(static_file, ".css") ? "text/css; charset=utf-8" : 
                           endswith(static_file, ".js") ? "application/javascript; charset=utf-8" :
                           "text/plain"
            body = read(file_path, String)
            return HTTP.Response(200, [cors_headers..., "Content-Type" => content_type], body)
        else
            return HTTP.Response(404, cors_headers, "Static file not found")
        end

    elseif path == "/dashboard" && method == "GET"
        agent = get_agent()
        body = JSON3.write(Dict(
            "status" => "success",
            "project_name" => "mirnan_julia",
            "stats" => Dict("files" => 80, "dirs" => 12, "loc" => 9000),
            "languages" => Dict("Julia" => 100),
            "agent_version" => Constants.AGENT_VERSION,
            "device" => Constants.LLM_PROVIDER
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)

    elseif path == "/git/status" && method == "GET"
        body = JSON3.write(Dict(
            "status" => "available",
            "branch" => "main",
            "info" => "Git is operational (inside mirnan_julia)"
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)

    elseif path == "/notifications/stream" && method == "GET"
        headers = [
            cors_headers...,
            "Content-Type" => "text/event-stream",
            "Cache-Control" => "no-cache",
            "Connection" => "keep-alive"
        ]
        return HTTP.Response(200, headers, Channel() do chan
            initial_msg = Dict("level" => "info", "title" => "الوكيل جاهز", "body" => "متصل بنجاح بلغة جوليا", "ts" => "")
            put!(chan, "data: " * JSON3.write(initial_msg) * "\n\n")
            while true
                sleep(20)
                put!(chan, ": heartbeat\n\n")
            end
        end)

        
    elseif path == "/reset" && method == "POST"
        agent = get_agent()
        empty!(agent.conversation_history)
        agent.current_task_tree = nothing
        agent.total_tool_calls = 0
        agent.loop_restarts = 0
        agent.replan_count = 0
        empty!(agent.files_modified)
        
        body = JSON3.write(Dict("status" => "success", "message" => "$(Constants.AGENT_NAME) Sovereign Kernel Reset Complete"))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)
        
    elseif path == "/diagnostics" && method == "GET"
        agent = get_agent()
        body = JSON3.write(Dict(
            "version" => Constants.AGENT_VERSION,
            "device" => Constants.LLM_PROVIDER,
            "stats" => Dict(
                "tool_calls" => agent.total_tool_calls,
                "restarts" => agent.loop_restarts,
                "replans" => agent.replan_count
            ),
            "health" => Dict(
                "status" => "Excellent"
            )
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)
        
    elseif path == "/api/taskboard" && method == "GET"
        agent = get_agent()
        tree = agent.current_task_tree
        phases_data = []
        
        next_task = get_next_task(agent.planner, tree)
        next_id = next_task !== nothing ? next_task["id"] : nothing
        
        if tree !== nothing && haskey(tree, "phases")
            for phase in tree["phases"]
                tasks_data = []
                for task in get(phase, "subtasks", [])
                    tid = String(task["id"])
                    status = if tid in agent.planner.completed_tasks
                        "completed"
                    elseif tid == next_id
                        "in_progress"
                    else
                        "pending"
                    end
                    
                    push!(tasks_data, Dict(
                        "id" => tid,
                        "description" => get(task, "description", "")[1:min(120, end)],
                        "status" => status,
                        "effort" => get(task, "estimated_effort", "medium"),
                        "validation" => get(task, "validation", "")[1:min(80, end)]
                    ))
                end
                push!(phases_data, Dict(
                    "name" => get(phase, "phase_name", "Phase"),
                    "summary" => get(phase, "summary", "")[1:min(100, end)],
                    "tasks" => tasks_data
                ))
            end
        end
        
        prog = get_progress_summary(agent.planner, tree)
        
        body = JSON3.write(Dict(
            "status" => "ok",
            "goal" => tree !== nothing ? get(tree, "goal", "No active task") : "No active task",
            "global_summary" => tree !== nothing ? get(tree, "global_summary", "") : "",
            "phases" => phases_data,
            "progress" => Dict(
                "percent" => round(get(prog, :percent, 0.0), digits=1),
                "completed" => get(prog, :completed, 0),
                "total" => get(prog, :total, 0),
                "eta_seconds" => round(get(prog, :eta_seconds, 0.0))
            ),
            "session" => Dict(
                "tool_calls" => agent.total_tool_calls,
                "files_modified" => collect(agent.files_modified),
                "loop_restarts" => agent.loop_restarts,
                "replans" => agent.replan_count,
                "provider" => Constants.LLM_PROVIDER
            )
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)
        
    elseif path == "/worldstate" && method == "GET"
        agent = get_agent()
        ws = agent.planner.world_state
        
        body = JSON3.write(Dict(
            "status" => "ok",
            "world_state" => Dict(
                "step" => ws.step_count,
                "facts" => ws.facts,
                "architecture_decisions" => ws.architecture_decisions,
                "detected_issues" => ws.detected_issues
            )
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)
        
    elseif path == "/cmd/center" && method == "GET"
        agent = get_agent()
        files = String[]
        try
            for (root, dirs, fs) in walkdir(agent.workspace_root)
                # Ignore hidden or build folders
                if occursin(".git", root) || occursin(".julia", root) || occursin(".memory", root) || occursin("agent_workspace", root)
                    continue
                end
                for f in fs
                    rel = relpath(joinpath(root, f), agent.workspace_root)
                    if !occursin(".git", rel) && !occursin(".julia", rel) && !occursin(".memory", rel) && !occursin("Manifest.toml", rel) && !occursin("agent_workspace", rel)
                        push!(files, replace(rel, '\\' => '/'))
                    end
                    if length(files) >= 50
                        break
                    end
                end
                if length(files) >= 50
                    break
                end
            end
        catch e
            @warn "Failed to walk directory for /cmd/center: $e"
        end
        
        debug_history = [
            Dict("ts" => Dates.format(Dates.now(), "HH:MM:SS"), "type" => "SYSTEM", "msg" => "Sovereign Kernel Active"),
            Dict("ts" => Dates.format(Dates.now(), "HH:MM:SS"), "type" => "INFO", "msg" => "Julia Engine $(VERSION) Port 5000")
        ]
        
        body = JSON3.write(Dict(
            "status" => "success",
            "files" => files,
            "debug_history" => debug_history
        ))
        return HTTP.Response(200, [cors_headers..., "Content-Type" => "application/json"], body)

    elseif path == "/chat" && method == "POST"
        payload = try
            JSON3.read(req.body)
        catch
            return HTTP.Response(400, cors_headers, "Invalid JSON body")
        end
        user_message = get(payload, :message, "")
        if isempty(user_message)
            return HTTP.Response(400, cors_headers, "Empty message parameter")
        end
        
        agent = get_agent()
        
        headers = [
            cors_headers...,
            "Content-Type" => "text/event-stream",
            "Cache-Control" => "no-cache",
            "Connection" => "keep-alive"
        ]
        
        return HTTP.Response(200, headers, Channel() do chan
            initial_msg = Dict("type" => "system", "content" => "--- [Sovereign Canal Locked: v$(Constants.AGENT_VERSION) Julia Bridge] ---")
            put!(chan, "data: " * JSON3.write(initial_msg) * "\n\n")
            
            try
                stream_run(agent, user_message) do event
                    put!(chan, "data: " * JSON3.write(event) * "\n\n")
                end
            catch e
                err_msg = Dict("type" => "error", "content" => "Kernel Panic: $e")
                put!(chan, "data: " * JSON3.write(err_msg) * "\n\n")
            end
        end)
    end
    
    return HTTP.Response(404, cors_headers, "Not Found")
end

function start_server(host::String="127.0.0.1", port::Int=5000)
    @info "🚀 Starting Sovereign Web Server on $host:$port..."
    HTTP.serve(handle_route, host, port)
end

end # module App
