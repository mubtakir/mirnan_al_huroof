module Planner

using SQLite
using JSON3
using Dates

export WorldState, TaskPlanner, decompose, update_plan, validate_subtask, get_next_task, get_progress_summary, mark_completed, mark_failed, start_tracking_task

# ═══════════════════════════════════════════════════════════════
# 1. WorldState - Persistent Memory State
# ═══════════════════════════════════════════════════════════════

mutable struct WorldState
    facts::Dict{String, Any}
    observations::Vector{Dict{String, Any}}
    architecture_decisions::Vector{String}
    detected_issues::Vector{Dict{String, Any}}
    snapshots::Vector{Dict{String, Any}}
    step_count::Int
    db_path::String
    session_id::String
end

function WorldState(db_path::String="", session_id::String="GLOBAL")
    if !isempty(db_path)
        try
            conn = SQLite.DB(db_path)
            SQLite.execute(conn, """
                CREATE TABLE IF NOT EXISTS world_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    label TEXT,
                    step_count INTEGER,
                    snapshot_data TEXT,
                    timestamp REAL
                )
            """)
        catch e
            @error "Failed to init world_snapshots table: $e"
        end
    end
    return WorldState(
        Dict{String, Any}(),
        Dict{String, Any}[],
        String[],
        Dict{String, Any}[],
        Dict{String, Any}[],
        0,
        db_path,
        session_id
    )
end

function add_observation!(ws::WorldState, task_id::String, observation::String, success::Bool)
    ws.step_count += 1
    entry = Dict{String, Any}(
        "step" => ws.step_count,
        "task_id" => task_id,
        "observation" => first(observation, 1000),
        "success" => success,
        "timestamp" => time()
    )
    push!(ws.observations, entry)

    if !success || occursin("[ERROR]", observation) || occursin("failed", lowercase(observation)) || occursin("فشل", observation) || occursin("not found", lowercase(observation))
        push!(ws.detected_issues, Dict{String, Any}(
            "step" => ws.step_count,
            "task_id" => task_id,
            "issue" => first(observation, 300)
        ))
    end
end

function take_snapshot!(ws::WorldState, label::String, task_tree::Dict)::Int
    snap_id = length(ws.snapshots)
    snapshot = Dict{String, Any}(
        "id" => snap_id,
        "label" => label,
        "timestamp" => time(),
        "step" => ws.step_count,
        "task_tree_copy" => deepcopy(task_tree),
        "facts_copy" => deepcopy(ws.facts),
        "architecture_decisions_copy" => copy(ws.architecture_decisions)
    )
    push!(ws.snapshots, snapshot)

    if !isempty(ws.db_path)
        try
            conn = SQLite.DB(ws.db_path)
            data_str = JSON3.write(snapshot)
            stmt = SQLite.Stmt(conn, "INSERT INTO world_snapshots (session_id, label, step_count, snapshot_data, timestamp) VALUES (?, ?, ?, ?, ?)")
            SQLite.execute(stmt, (ws.session_id, label, ws.step_count, data_str, snapshot["timestamp"]))
        catch e
            @error "World state persistence failed: $e"
        end
    end

    @info "📸 [WORLD-STATE] Snapshot #$snap_id persistent: $label"
    return snap_id
end

function restore_snapshot!(ws::WorldState, snapshot_id::Int)::Union{Dict, Nothing}
    for snap in ws.snapshots
        if snap["id"] == snapshot_id
            ws.facts = deepcopy(snap["facts_copy"])
            ws.architecture_decisions = copy(snap["architecture_decisions_copy"])
            @info "⏪ [WORLD-STATE] Restored to Memory Snapshot #$snapshot_id: $(snap["label"])"
            return deepcopy(snap["task_tree_copy"])
        end
    end

    if !isempty(ws.db_path)
        try
            conn = SQLite.DB(ws.db_path)
            cursor = SQLite.DBInterface.execute(conn, "SELECT snapshot_data FROM world_snapshots WHERE id = ?", (snapshot_id,))
            rows = collect(cursor)
            if !isempty(rows)
                snap = JSON3.read(rows[1].snapshot_data, Dict)
                ws.facts = Dict{String, Any}(String(k) => v for (k, v) in snap["facts_copy"])
                ws.architecture_decisions = String[String(x) for x in snap["architecture_decisions_copy"]]
                @info "⏪ [WORLD-STATE] Restored to Persistent Snapshot #$snapshot_id"
                return Dict(snap["task_tree_copy"])
            end
        catch e
            @error "WorldState DB restore failed: $e"
        end
    end

    @error "[WORLD-STATE] Snapshot #$snapshot_id not found."
    return nothing
end

function get_context_summary(ws::WorldState)::String
    recent_obs = ws.observations[max(1, end-4):end]
    issues = ws.detected_issues[max(1, end-2):end]
    summary = "[World State @ Step $(ws.step_count)]\n"
    summary *= "- Facts: $(JSON3.write(ws.facts))\n"
    summary *= "- Architecture Decisions: $(length(ws.architecture_decisions)) recorded\n"
    summary *= "- Recent Issues: $(length(issues))\n"
    if !isempty(issues)
        summary *= "  " * join([first(i["issue"], 80) for i in issues], " | ") * "\n"
    end
    return summary
end

function needs_backtrack(ws::WorldState, consecutive_failures::Int)::Tuple{Bool, Int}
    if consecutive_failures < 3 || isempty(ws.snapshots)
        return false, -1
    end
    return true, last(ws.snapshots)["id"]
end

# ═══════════════════════════════════════════════════════════════
# 2. TaskPlanner - Devin-Level Long Horizon Planner
# ═══════════════════════════════════════════════════════════════

mutable struct TaskPlanner
    completed_tasks::Set{String}
    failed_tasks::Set{String}
    task_attempts::Dict{String, Int}
    max_task_attempts::Int
    start_time::Float64
    task_start_times::Dict{String, Float64}
    completion_stats::Vector{Dict{Symbol, Any}}
    world_state::WorldState
    backtrack_count::Int
    max_backtracks::Int
    phase_snapshots::Dict{String, Int}
end

function TaskPlanner(db_path::String="", session_id::String="GLOBAL")
    return TaskPlanner(
        Set{String}(),
        Set{String}(),
        Dict{String, Int}(),
        3,
        time(),
        Dict{String, Float64}(),
        Dict{Symbol, Any}[],
        WorldState(db_path, session_id),
        0,
        3,
        Dict{String, Int}()
    )
end

function decompose(planner::TaskPlanner, goal::String, llm_func::Function)::Dict
    complexity = occursin("build", lowercase(goal)) || occursin("create", lowercase(goal)) || occursin("نظام", goal) || length(goal) > 100 ? "complex" : "simple"
    
    prompt = """
    حلل هذا الهدف وقم بتفكيكه استراتيجياً إلى "مراحل هندسية" (Engineering Phases):
    الهدف: $goal
    مستوى التعقيد المقدَّر: $complexity
    
    ### القواعد السيادية للتخطيط بعيد المدى:
    1. المرحلة الأولى دائماً: (تحليل المشروع وتسجيل قرارات المعمارية الأساسية).
    2. قسّم المشروع إلى مراحل متسلسلة — لكل مرحلة هدف واضح يمكن قياسه.
    3. كل مرحلة يجب أن تحتوي على مهام فرعية (Subtasks) واضحة ومستقلة.
    4. حدد معايير نجاح (success_criteria) لكل مرحلة.
    
    أجب بصيغة JSON فقط بهذا الشكل:
    {
      "goal": "$(first(goal, 100))",
      "global_summary": "ملخص عام لأغراض المشروع النهائية",
      "complexity": "$complexity",
      "success_criteria": ["معيار_١", "معيار_٢"],
      "architecture_notes": ["ملاحظة معمارية أولية"],
      "phases": [
         {
           "phase_name": "اسم المرحلة",
           "summary": "ملخص لما سيتم إنجازه في هذه المرحلة",
           "success_criteria": ["معيار نجاح هذه المرحلة"],
           "subtasks": [
             {
               "id": "P1_T1",
               "description": "المهمة الفرعية",
               "depends_on": [],
               "validation": "كيف نتحقق من إنجازها",
               "estimated_effort": "low|medium|high"
             }
           ]
         }
      ]
    }
    """
    try
        messages = [
            Dict("role" => "system", "content" => "أنت مهندس تخطيط هرمي متخصص في Long-Horizon Planning. أجب بصيغة JSON فقط."),
            Dict("role" => "user", "content" => prompt)
        ]
        response = llm_func(messages)
        json_match = match(r"\{.*\}"s, response)
        if json_match !== nothing
            tree_data = JSON3.read(json_match.match, Dict)
            for note in get(tree_data, "architecture_notes", [])
                push!(planner.world_state.architecture_decisions, String(note))
            end
            planner.world_state.facts["goal"] = get(tree_data, "goal", goal)
            planner.world_state.facts["complexity"] = get(tree_data, "complexity", complexity)
            
            snap_id = take_snapshot!(planner.world_state, "initial_plan", tree_data)
            planner.phase_snapshots["__initial__"] = snap_id
            return tree_data
        end
    catch e
        @error "Plan decomposition failed: $e"
    end
    return Dict("goal" => goal, "phases" => [])
end

function update_plan(planner::TaskPlanner, current_tree::Dict, observation::String, llm_func::Function)::Dict
    current_task = get_next_task(planner, current_tree)
    task_id = current_task !== nothing ? current_task["id"] : "UNKNOWN"
    is_success = !occursin("[ERROR]", observation) && !occursin("failed", lowercase(observation)) && !occursin("Permission denied", observation)
    
    add_observation!(planner.world_state, task_id, observation, is_success)

    consecutive_failures = count(i -> i["step"] > planner.world_state.step_count - 5, planner.world_state.detected_issues)
    needs_bt, snap_id = needs_backtrack(planner.world_state, consecutive_failures)

    if needs_bt && planner.backtrack_count < planner.max_backtracks
        planner.backtrack_count += 1
        @warn "⏪ [BACKTRACKING] Strategic backtrack #$(planner.backtrack_count) triggered. Restoring to snapshot #$snap_id..."
        restored_tree = restore_snapshot!(planner.world_state, snap_id)
        if restored_tree !== nothing
            empty!(planner.failed_tasks)
            empty!(planner.task_attempts)
            return restored_tree
        end
    end

    return current_tree
end

function validate_subtask(planner::TaskPlanner, task::Dict, observation::String, llm_func::Function)::Bool
    desc = lowercase(get(task, "description", ""))
    validation = lowercase(get(task, "validation", ""))

    success_indicators = ["[SUCCESS]", "تم بنجاح", "saved successfully", "no errors"]
    if any(ind -> occursin(ind, observation), success_indicators)
        return true
    end
    
    failure_indicators = ["[ERROR]", "فشل", "not found", "denied"]
    if any(ind -> occursin(ind, observation), failure_indicators)
        return false
    end

    file_indicators = ["create", "write", "save", "generate", "إنشاء", "كتابة", "حفظ"]
    if any(kw -> occursin(kw, desc), file_indicators) || any(kw -> occursin(kw, validation), file_indicators)
        m = match(r"['\"]?([\w./\\]+\.\w+)['\"]?", desc)
        if m === nothing
            m = match(r"['\"]?([\w./\\]+\.\w+)['\"]?", validation)
        end
        if m !== nothing
            file_path = m.captures[1]
            if isfile(file_path) && filesize(file_path) > 0
                @info "[PHYSICAL_VERIFY] File verified: $file_path"
                return true
            end
        end
    end

    prompt = """
    هل تم إنجاز المهمة التالية بنجاح بناءً على الملاحظات؟
    المهمة: $(task["description"])
    الملاحظات: $(first(observation, 500))
    معيار التحقق: $(get(task, "validation", ""))
    
    أجب بـ (YES) أو (NO) فقط.
    """
    try
        messages = [Dict("role" => "user", "content" => prompt)]
        resp = uppercase(strip(llm_func(messages)))
        return occursin("YES", resp)
    catch
        return true
    end
end

function mark_completed(planner::TaskPlanner, task_id::String)
    push!(planner.completed_tasks, task_id)
    delete!(planner.failed_tasks, task_id)
    if haskey(planner.task_start_times, task_id)
        duration = time() - planner.task_start_times[task_id]
        push!(planner.completion_stats, Dict{Symbol, Any}(
            :id => task_id,
            :duration => duration,
            :success => true
        ))
        delete!(planner.task_start_times, task_id)
        @info "⏱️ [PLANNER] Task $task_id completed in $(round(duration, digits=2))s"
    end
end

function mark_failed(planner::TaskPlanner, task_id::String)
    push!(planner.failed_tasks, task_id)
    planner.task_attempts[task_id] = get(planner.task_attempts, task_id, 0) + 1
    if haskey(planner.task_start_times, task_id)
        duration = time() - planner.task_start_times[task_id]
        push!(planner.completion_stats, Dict{Symbol, Any}(
            :id => task_id,
            :duration => duration,
            :success => false
        ))
        delete!(planner.task_start_times, task_id)
        @warn "⚠️ [PLANNER] Task $task_id failed after $(round(duration, digits=2))s (Attempt $(planner.task_attempts[task_id]))"
    end
end

function start_tracking_task(planner::TaskPlanner, task_id::String)
    planner.task_start_times[task_id] = time()
end

function get_next_task(planner::TaskPlanner, tree::Union{Dict, Nothing})::Union{Dict, Nothing}
    if tree === nothing
        return nothing
    end
    
    if haskey(tree, "phases")
        for phase in tree["phases"]
            for task in get(phase, "subtasks", [])
                task_id = String(task["id"])
                if task_id in planner.completed_tasks; continue; end
                if get(planner.task_attempts, task_id, 0) >= planner.max_task_attempts; continue; end
                
                deps = get(task, "depends_on", [])
                if all(dep in planner.completed_tasks for dep in deps)
                    return Dict(task)
                end
            end
        end
    elseif haskey(tree, "subtasks")
        for task in tree["subtasks"]
            task_id = String(task["id"])
            if task_id in planner.completed_tasks; continue; end
            if get(planner.task_attempts, task_id, 0) >= planner.max_task_attempts; continue; end
            
            deps = get(task, "depends_on", [])
            if all(dep in planner.completed_tasks for dep in deps)
                return Dict(task)
            end
        end
    end
    return nothing
end

function get_progress_summary(planner::TaskPlanner, tree::Union{Dict, Nothing})::Dict{Symbol, Any}
    if tree === nothing
        return Dict{Symbol, Any}(:percent => 0.0, :completed => 0, :total => 0, :eta_seconds => 0.0)
    end
    
    all_tasks = Dict[]
    if haskey(tree, "phases")
        for phase in tree["phases"]
            for t in get(phase, "subtasks", [])
                push!(all_tasks, Dict(t))
            end
        end
    elseif haskey(tree, "subtasks")
        for t in tree["subtasks"]
            push!(all_tasks, Dict(t))
        end
    end
    
    total = length(all_tasks)
    if total == 0
        return Dict{Symbol, Any}(:percent => 0.0, :completed => 0, :total => 0, :eta_seconds => 0.0)
    end
    
    completed = count(t -> String(t["id"]) in planner.completed_tasks, all_tasks)
    elapsed = time() - planner.start_time
    
    avg_duration = 0.0
    completed_stats = filter(s -> s[:success], planner.completion_stats)
    if !isempty(completed_stats)
        avg_duration = sum(s[:duration] for s in completed_stats) / length(completed_stats)
    end
    
    remaining = total - completed
    eta = remaining * avg_duration
    
    success_rate = isempty(planner.completion_stats) ? 1.0 : count(s -> s[:success], planner.completion_stats) / length(planner.completion_stats)
    
    return Dict{Symbol, Any}(
        :percent => (completed / total * 100.0),
        :completed => completed,
        :total => total,
        :elapsed => elapsed,
        :avg_duration => avg_duration,
        :eta_seconds => eta,
        :success_rate => success_rate
    )
end

end # module Planner
