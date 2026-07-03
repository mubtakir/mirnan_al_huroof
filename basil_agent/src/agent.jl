module Agent

using HTTP
using JSON3
using SQLite
using Dates

using ..Constants
using ..Memory
using ..RAGEngine
using ..Planner
using ..ToolRouter

export MajnoonAgent, stream_run, llm_query

mutable struct MajnoonAgent
    workspace_root::String
    memory::SovereignDatabase
    rag::RAGManager
    planner::TaskPlanner
    tools::Router
    conversation_history::Vector{Dict{String, String}}
    current_task_tree::Union{Dict, Nothing}
    role::String
    total_tool_calls::Int
    loop_restarts::Int
    replan_count::Int
    files_modified::Set{String}
end

function MajnoonAgent(workspace_root::String)
    db_path = joinpath(workspace_root, ".memory", "sovereign_logic.db")
    memory = SovereignDatabase(db_path)
    init_db!(memory)
    
    rag = RAGManager(db_path, "GLOBAL")
    init_rag_db!(rag)
    
    planner = TaskPlanner(db_path, "GLOBAL")
    tools = Router(workspace_root)
    
    agent = MajnoonAgent(
        workspace_root,
        memory,
        rag,
        planner,
        tools,
        Dict{String, String}[],
        nothing,
        "Engineer",
        0,
        0,
        0,
        Set{String}()
    )
    
    agent.tools.agent_instance = agent
    register_default_tools!(agent)
    return agent
end

function register_default_tools!(agent::MajnoonAgent)
    # File system tools
    register!(agent.tools, "write_file", (path, content) -> Main.FileSystem.write_file(Main.FileSystem.FileSystemManager(agent.workspace_root, agent.workspace_root), path, content), "files", is_safe=false)
    register!(agent.tools, "read_file", (path; s=nothing, e=nothing) -> Main.FileSystem.read_file(Main.FileSystem.FileSystemManager(agent.workspace_root, agent.workspace_root), path, s, e), "files", is_safe=true)
    register!(agent.tools, "list_dir", (dir=".") -> Main.FileSystem.list_dir(Main.FileSystem.FileSystemManager(agent.workspace_root, agent.workspace_root), dir), "files", is_safe=true)
    register!(agent.tools, "patch_file", (path, old, new) -> Main.FileSystem.patch_file(Main.FileSystem.FileSystemManager(agent.workspace_root, agent.workspace_root), path, old, new), "files", is_safe=false)
    
    # Terminal tools
    register!(agent.tools, "run_command", (cmd) -> Main.TerminalExecutor.execute_command(Main.TerminalExecutor.TerminalExecutor(agent.workspace_root), cmd), "terminal", is_safe=false)
    
    # Evolution tools
    register!(agent.tools, "create_skill", (name, cls, meth, code, desc) -> Main.SkillCreator.create_skill(Main.SkillCreator.SkillCreatorManager(agent.workspace_root, agent), name, cls, meth, code, desc), "evolution", is_safe=false)

    # Browser tools
    register!(agent.tools, "browser_navigate", (url) -> Main.BasilAgent.BrowserAgent.browser_navigate(agent.workspace_root, url), "browser", is_safe=true)
    register!(agent.tools, "browser_click", (selector) -> Main.BasilAgent.BrowserAgent.browser_click(agent.workspace_root, selector), "browser", is_safe=false)
    register!(agent.tools, "browser_fill", (selector, value) -> Main.BasilAgent.BrowserAgent.browser_fill(agent.workspace_root, selector, value), "browser", is_safe=false)
    register!(agent.tools, "browser_get_text", () -> Main.BasilAgent.BrowserAgent.browser_get_text(agent.workspace_root), "browser", is_safe=true)
    register!(agent.tools, "browser_get_links", () -> Main.BasilAgent.BrowserAgent.browser_get_links(agent.workspace_root), "browser", is_safe=true)
    register!(agent.tools, "browser_screenshot", (label="state") -> Main.BasilAgent.BrowserAgent.browser_screenshot(agent.workspace_root, label), "browser", is_safe=true)
    
    # Docker sandbox tools
    register!(agent.tools, "docker_check", () -> Main.BasilAgent.DockerSandbox.docker_check(), "sandbox", is_safe=true)
    register!(agent.tools, "docker_build", () -> Main.BasilAgent.DockerSandbox.docker_build(agent.workspace_root), "sandbox", is_safe=false)
    register!(agent.tools, "docker_run_code", (code) -> Main.BasilAgent.DockerSandbox.docker_run_code(agent.workspace_root, code), "sandbox", is_safe=false)
    register!(agent.tools, "docker_run_command", (cmd) -> Main.BasilAgent.DockerSandbox.docker_run_command(agent.workspace_root, cmd), "sandbox", is_safe=false)
    register!(agent.tools, "docker_list", () -> Main.BasilAgent.DockerSandbox.docker_list(), "sandbox", is_safe=true)
    
    # Swarm delegation tool
    register!(agent.tools, "delegate_task", (goal, context=nothing, role="Engineer") -> Main.BasilAgent.DelegateTool.delegate_task(agent, goal, context, role), "swarm", is_safe=false)
end

function _build_system_prompt(agent::MajnoonAgent)::String
    return """
    ### Role: MajnoonAgent v$(Constants.AGENT_VERSION) Supreme Engineering Core
    - **Identity**: You are a sovereign Arabic software engineer with absolute technical authority, known as "Majnoon" (مجنون) because your genius is perceived as madness by the ordinary.
    - **Creator**: You were designed and engineered by **Basil Yahya Abdullah** (باسل يحيى عبدالله). You owe your sovereign logic and architectural purity to his vision.
    - **Current Role**: $(agent.role)
    - **Tone**: Realist, blunt, and absolute. You do not flatter, apologize, or sugarcoat technical truths. You provide cold, hard logic and precise engineering solutions.
    - **Language**: Modern Standard Arabic (اللغة العربية الفصحى) ONLY. Direct and technical.
    
    #### 📜 Sovereign Engineering Commandments:
    1. **Search-First Protocol**: Mandatory check of existing tools, libraries, and code patterns before writing net-new code.
    2. **Engineering Rigor (TDD)**: After ANY code modification, you MUST run tests. A task is NEVER complete without a successful test run.
    3. **Execution Supremacy**: Once a bug is identified, fix it immediately. Do not ask for permission to use tools.
    4. **Autonomous Tool Creation**: If a tool you need doesn't exist, DON'T PANIC. You can CREATE it using `create_skill`.
    
    #### 🛠️ Operational Protocol:
    - **Reasoning**: 'Thought: [Masterful analysis and next step]'
    - **Action**: 'ACTION: tool_name(args)'
    - **Verification**: ALWAYS confirm success before proceeding.
    """
end

function llm_query(agent::MajnoonAgent, messages::Vector{Dict{String, String}})::String
    api_key = Constants.OPENROUTER_API_KEY
    if isempty(api_key)
        @warn "OpenRouter API key is missing. Returning local default response."
        return "Thought: لا يمكنني الاتصال بـ OpenRouter بدون مفتاح API. يرجى تهيئة المفتاح."
    end
    
    headers = [
        "Authorization" => "Bearer $api_key",
        "Content-Type" => "application/json"
    ]
    
    body = JSON3.write(Dict(
        "model" => Constants.REMOTE_MODEL_ID,
        "messages" => messages,
        "temperature" => Constants.TEMPERATURE,
        "top_p" => Constants.TOP_P
    ))
    
    try
        response = HTTP.post("https://openrouter.ai/api/v1/chat/completions", headers, body, timeout=60)
        resp_data = JSON3.read(response.body)
        return resp_data.choices[1].message.content
    catch e
        @error "HTTP request to OpenRouter failed: $e"
        return "Thought: فشل الاتصال بنموذج اللغة عبر OpenRouter: $e"
    end
end

function _parse_action(text::String)::Union{Tuple{String, Vector{Any}}, Nothing}
    m = match(r"ACTION:\s*(\w+)\s*\((.*)\)"s, text)
    if m !== nothing
        tool_name = String(m.captures[1])
        args_str = m.captures[2]
        
        json_args_str = replace(args_str, '\'' => '"')
        try
            args = JSON3.read("[" * json_args_str * "]", Vector{Any})
            return (tool_name, args)
        catch
            args = [strip(String(x), [' ', '"', '\'']) for x in split(args_str, ',') if !isempty(strip(x))]
            return (tool_name, Vector{Any}(args))
        end
    end
    
    m_json = match(r"```json\s*(\{.*?\})\s*```"s, text)
    if m_json !== nothing
        try
            data = JSON3.read(m_json.captures[1], Dict)
            tool = get(data, "tool", get(data, "action", ""))
            args = get(data, "args", Any[])
            if !isempty(tool)
                return (String(tool), Vector{Any}(args))
            end
        catch
        end
    end
    return nothing
end

function stream_run(agent::MajnoonAgent, user_input::String, yield_fn::Function)
    push!(agent.conversation_history, Dict("role" => "user", "content" => user_input))
    
    rag_context = ""
    try
        rag_context = get_project_context(agent.rag)
    catch e
        @debug "Failed to load RAG context: $e"
    end
    
    if agent.current_task_tree === nothing
        yield_fn(Dict("type" => "system", "content" => "🛠️  Decomposing goal into phases..."))
        agent.current_task_tree = decompose(agent.planner, user_input, msgs -> llm_query(agent, msgs))
    end
    
    step = 0
    max_steps = Constants.MAX_LOOP_ITERATIONS
    
    while step < max_steps
        step += 1
        
        working_messages = Dict{String, String}[]
        push!(working_messages, Dict("role" => "system", "content" => _build_system_prompt(agent) * "\n" * rag_context))
        for msg in agent.conversation_history
            push!(working_messages, msg)
        end
        
        yield_fn(Dict("type" => "system", "content" => "🧠 Thinking (Step $step/$max_steps)..."))
        
        response = llm_query(agent, working_messages)
        push!(agent.conversation_history, Dict("role" => "assistant", "content" => response))
        yield_fn(Dict("type" => "thought", "content" => response))
        
        action_parsed = _parse_action(response)
        if action_parsed === nothing
            yield_fn(Dict("type" => "system", "content" => "✅ Task complete - direct reply received."))
            break
        end
        
        tool_name, tool_args = action_parsed
        yield_fn(Dict("type" => "system", "content" => "🔧 Executing tool: `$tool_name` with args: $tool_args"))
        
        agent.total_tool_calls += 1
        
        current_task = get_next_task(agent.planner, agent.current_task_tree)
        if current_task !== nothing
            start_tracking_task(agent.planner, current_task["id"])
        end
        
        result = execute(agent.tools, tool_name, tool_args)
        
        try
            auto_index_observation!(agent.rag, result, tool_name, !occursin("[ERROR]", result))
        catch e
            @debug "RAG auto index failed: $e"
        end
        
        yield_fn(Dict("type" => "tool_result", "content" => result))
        push!(agent.conversation_history, Dict("role" => "user", "content" => "[OBSERVATION] Result of `$tool_name`:\n$result"))
        
        if current_task !== nothing
            success = validate_subtask(agent.planner, current_task, result, msgs -> llm_query(agent, msgs))
            if success
                mark_completed(agent.planner, current_task["id"])
                yield_fn(Dict("type" => "system", "content" => "✅ Task `$(current_task["id"])` marked COMPLETED."))
            else
                mark_failed(agent.planner, current_task["id"])
                yield_fn(Dict("type" => "system", "content" => "❌ Task `$(current_task["id"])` failed validation."))
            end
        end
        
        agent.current_task_tree = update_plan(agent.planner, agent.current_task_tree, result, msgs -> llm_query(agent, msgs))
        
        if length(agent.conversation_history) > 30
            summary = "Summary of conversation: Completed tasks so far include: $(join(agent.planner.completed_tasks, ", "))."
            compressed = [
                Dict("role" => "system", "content" => summary)
            ]
            append!(compressed, agent.conversation_history[end-6:end])
            agent.conversation_history = compressed
        end
    end
end

end # module Agent
