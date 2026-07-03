"""
SIO — Synthetic Intelligence Orchestrator.
الذكاء التوليفي: فكرة ← GoalParser ← PhasePlanner ← Executor ← SelfMonitor ← Integrator ← منتج
يدعم: نص عادي + رياضيات + برمجة
"""
module SIO
using Statistics
using ..Physics.Generator: generate!
using ..Physics.MathBridgeModule: MathBridge, detect_mode, is_math_expression, evaluate_math, is_code_prompt
using ..Physics.SymbolicMathEngineModule: SymbolicMathEngine, solve_arithmetic, learn_math_pattern!,
                                           generate_math_explanation, encode_operation
using ..Physics.CodeEngineModule: CodeEngine, generate_code, compile_check, tokenize_code, validate_syntax
using ..Physics.CodePhaseEngineModule: CodePhaseEngine, generate_code_physics, learn_code_patterns!

export SIOOrchestrator, GoalParser, PhasePlanner, PhaseExecutor, SelfMonitor, Integrator,
       synthesize!

# ═══ GoalParser ═══
mutable struct GoalParser; gen::Any; end
GoalParser(;gen=nothing)=GoalParser(gen)

function parse(gp::GoalParser, goal::String)
    phases=Dict{String,Any}[]
    goal_l=lowercase(goal)
    mode = detect_mode(goal)

    if mode == "math"
        push!(phases,Dict("name"=>"math_solve","type"=>"math","prompt"=>goal,"mode"=>"math"))
    elseif mode == "code"
        if occursin("بحث",goal)||occursin("تقرير",goal)||occursin("search",goal_l)||occursin("report",goal_l)
            push!(phases,Dict("name"=>"plan","type"=>"structure","prompt"=>"خطط لهيكل $goal"))
            push!(phases,Dict("name"=>"code","type"=>"code","prompt"=>goal,"mode"=>"code"))
            push!(phases,Dict("name"=>"review","type"=>"quality","prompt"=>"راجع $goal"))
        else
            push!(phases,Dict("name"=>"code","type"=>"code","prompt"=>goal,"mode"=>"code"))
        end
    elseif occursin("بحث",goal)||occursin("تقرير",goal)||occursin("مقاله",goal)||occursin("search",goal_l)||occursin("report",goal_l)
        push!(phases,Dict("name"=>"plan","type"=>"structure","prompt"=>"خطط لهيكل $goal"))
        push!(phases,Dict("name"=>"research","type"=>"knowledge","prompt"=>"ابحث عن $goal"))
        push!(phases,Dict("name"=>"write","type"=>"generation","prompt"=>goal))
        push!(phases,Dict("name"=>"review","type"=>"quality","prompt"=>"راجع $goal"))
    elseif occursin("قصه",goal)||occursin("قصيدة",goal)||occursin("story",goal_l)||occursin("poem",goal_l)
        push!(phases,Dict("name"=>"concept","type"=>"structure","prompt"=>"اختر فكرة لـ $goal"))
        push!(phases,Dict("name"=>"create","type"=>"generation","prompt"=>goal))
        push!(phases,Dict("name"=>"polish","type"=>"quality","prompt"=>"حسّن $goal"))
    else
        push!(phases,Dict("name"=>"generate","type"=>"generation","prompt"=>goal))
    end
    return phases
end

# ═══ PhasePlanner ═══
mutable struct PhasePlanner
    phases::Vector{Dict{String,Any}}
    completed::Set{String}
    failed::Dict{String,Int}
    current::Int
    gen::Any
end
PhasePlanner(;gen=nothing)=PhasePlanner(Dict[],Set(),Dict(),1,gen)

function set_plan!(pp::PhasePlanner, phases)
    pp.phases=phases; pp.current=1; pp.completed=Set(); pp.failed=Dict()
end

current_phase(pp::PhasePlanner)=pp.current>length(pp.phases) ? nothing : pp.phases[pp.current]
is_complete(pp::PhasePlanner)=pp.current>length(pp.phases)
function next_phase!(pp::PhasePlanner); pp.current+=1; end
function mark_completed!(pp::PhasePlanner, name::String, output=nothing); push!(pp.completed,name); end
function mark_failed!(pp::PhasePlanner, name::String, diagnosis="")
    pp.failed[name]=get(pp.failed,name,0)+1
    return Dict("retry"=>pp.failed[name]<=3)
end
progress(pp::PhasePlanner)=length(pp.phases)>0 ? (pp.current-1)/length(pp.phases) : 0.0

function get_plan_summary(pp::PhasePlanner)
    return Dict("total"=>length(pp.phases),"completed"=>length(pp.completed),
                "current"=>pp.current,"failed"=>pp.failed)
end

# ═══ PhaseExecutor ═══
mutable struct PhaseExecutor
    gen::Any
    math_engine::SymbolicMathEngine
    code_engine::CodeEngine
    code_phase_engine::CodePhaseEngine
end
function PhaseExecutor(;gen=nothing)
    math_eng = gen !== nothing ? gen.symbolic_math : SymbolicMathEngine()
    code_eng = gen !== nothing ? gen.code_engine : CodeEngine()
    code_phase_eng = gen !== nothing ? gen.code_phase : CodePhaseEngine()
    PhaseExecutor(gen, math_eng, code_eng, code_phase_eng)
end

function execute(pe::PhaseExecutor, phase::Dict, planner::PhasePlanner, max_retries=5)
    name=phase["name"]; prompt=phase["prompt"]
    mode = get(phase, "mode", "text")

    for attempt in 1:max_retries
        result=""; success=false
        try
            if mode == "math"
                result = _execute_math(pe, prompt)
                success = !isempty(result)
            elseif mode == "code"
                result = _execute_code(pe, prompt)
                success = !isempty(result) && length(result)>3
            else
                if pe.gen !== nothing
                    result = generate!(pe.gen, prompt; mode="standard", max_words=30)
                end
                success=!isempty(result) && length(result)>5
            end
        catch e
            result="Error: $e"
        end
        if success
            return Dict("success"=>true,"output"=>result,"iterations"=>attempt,"diagnosis"=>"")
        end
    end
    return Dict("success"=>false,"output"=>"","iterations"=>max_retries,"diagnosis"=>"استنفدت المحاولات")
end

function _execute_math(pe::PhaseExecutor, prompt::String)
    # محاولة حساب مباشر أولاً
    mb = MathBridge()
    direct = evaluate_math(mb, prompt)
    if direct !== nothing
        return "الناتج: $direct"
    end

    # محاولة عبر SymbolicMathEngine
    try
        # استخراج الأعداد والعملية
        words = split(prompt)
        numbers = Int[]
        op_word = ""
        for w in words
            w_clean = replace(w, r"[^\d٠-٩]" => "")
            if !isempty(w_clean)
                # تحويل الأرقام العربية
                ar_digits = Dict('٠'=>'0','١'=>'1','٢'=>'2','٣'=>'3','٤'=>'4','٥'=>'5','٦'=>'6','٧'=>'7','٨'=>'8','٩'=>'9')
                en_clean = map(c -> get(ar_digits, c, c), w_clean)
                try push!(numbers, parse(Int, en_clean)); catch e; @warn "SIO: failed to parse math number: $e"; end
            else
                op_word = string(w)
            end
        end

        if length(numbers) >= 2
            result, confidence = solve_arithmetic(pe.math_engine, numbers[1], op_word, numbers[2])
            if confidence > 0.1
                explanation = generate_math_explanation(pe.math_engine, numbers[1], op_word, numbers[2], result)
                return explanation
            end
        end
    catch e
        @warn "SIO: execute_math failed: $e"
    end

    return ""
end

function _execute_code(pe::PhaseExecutor, prompt::String)
    lower_prompt = lowercase(prompt)
    is_algo = occursin("fibonacci", lower_prompt) || occursin("فيبوناتشي", lower_prompt) ||
              occursin("factorial", lower_prompt) || occursin("مضروب", lower_prompt) ||
              occursin("sort", lower_prompt) || occursin("ترتيب", lower_prompt) || occursin("فرز", lower_prompt) ||
              occursin("prime", lower_prompt) || occursin("أولي", lower_prompt) || occursin("اولي", lower_prompt)

    if !is_algo
        # محاولة عبر CodePhaseEngine أولاً (فيزيائي)
        try
            result = generate_code_physics(pe.code_phase_engine, prompt; lang="python")
            if !isempty(result) && result != "# no concepts" && !occursin("def solve():", result) && !occursin("function solve()", result)
                ok, _ = compile_check(result)
                if ok
                    return result
                end
            end
        catch e
            @warn "SIO: execute_code (physics) failed: $e"
        end
    end

    # محاولة عبر CodeEngine (قوالب)
    try
        result = generate_code(pe.code_engine, prompt; max_tokens=30)
        if !isempty(result)
            return result
        end
    catch e
        @warn "SIO: execute_code (template) failed: $e"
    end

    # Fallback to physics if template failed but not yet tried
    if is_algo
        try
            result = generate_code_physics(pe.code_phase_engine, prompt; lang="python")
            if !isempty(result) && result != "# no concepts"
                ok, _ = compile_check(result)
                if ok
                    return result
                end
            end
        catch e
            @warn "SIO: execute_code (physics fallback) failed: $e"
        end
    end

    return ""
end

# ═══ SelfMonitor ═══
mutable struct SelfMonitor
    gen::Any
    coherence_history::Vector{Float64}
    alerts::Vector{String}
end
SelfMonitor(;gen=nothing)=SelfMonitor(gen,Float64[],String[])

function validate(sm::SelfMonitor, output::String, phase_name::String, planner::PhasePlanner)
    if isempty(output)||length(output)<3
        push!(sm.alerts,"مرحلة $phase_name: مخرج قصير جداً")
        return Dict("accepted"=>false,"alerts"=>sm.alerts,"coherence"=>0.0)
    end
    words=split(output)
    coherence=min(1.0,length(words)/20*0.7+0.3)
    push!(sm.coherence_history,coherence)
    accepted=coherence>0.3
    !accepted && push!(sm.alerts,"مرحلة $phase_name: تماسك منخفض")
    return Dict("accepted"=>accepted,"alerts"=>sm.alerts,"coherence"=>coherence)
end

function get_status(sm::SelfMonitor)
    return Dict("coherence_mean"=>isempty(sm.coherence_history) ? 0.0 : mean(sm.coherence_history),
                "alerts"=>sm.alerts)
end

# ═══ Integrator ═══
struct Integrator end

function integrate(::Integrator, phase_outputs::Vector{Dict}, goal::String)
    if isempty(phase_outputs)
        return Dict("deliverable"=>"","summary"=>"لا مخرجات","structure"=>[])
    end
    deliverable=join([p["output"] for p in phase_outputs],"\n\n")
    structure=[p["phase"] for p in phase_outputs]
    summary="# $(length(phase_outputs)) مراحل مكتملة: " * join(structure,", ")
    return Dict("deliverable"=>deliverable,"summary"=>summary,"structure"=>structure)
end

# ═══ SIOOrchestrator ═══
mutable struct SIOOrchestrator
    gen::Any
    parser::GoalParser
    planner::PhasePlanner
    executor::PhaseExecutor
    monitor::SelfMonitor
    integrator::Integrator
    current_goal::String
    phase_outputs::Vector{Dict}
    start_time::Float64
    status::String
end

function SIOOrchestrator(gen=nothing)
    SIOOrchestrator(gen, GoalParser(gen=gen), PhasePlanner(gen=gen),
                    PhaseExecutor(gen=gen), SelfMonitor(gen=gen), Integrator(),
                    "", Dict[], 0.0, "idle")
end

function synthesize!(sio::SIOOrchestrator, goal::String; max_retries=5)
    sio.current_goal=goal
    sio.phase_outputs=Dict[]
    sio.start_time=time()
    sio.status="parsing"

    phases=parse(sio.parser, goal)
    set_plan!(sio.planner, phases)
    sio.status="executing"

    failed=String[]
    while !is_complete(sio.planner)
        phase=current_phase(sio.planner)
        phase===nothing && break
        name=phase["name"]
        sio.status="executing:$name"

        result=execute(sio.executor, phase, sio.planner, max_retries)
        if result["success"]
            validated=validate(sio.monitor, result["output"], name, sio.planner)
            if validated["accepted"]
                mark_completed!(sio.planner, name, result["output"])
                push!(sio.phase_outputs, Dict(
                    "phase"=>name,
                    "output"=>result["output"],
                    "iterations"=>result["iterations"],
                    "coherence"=>validated["coherence"]
                ))
                next_phase!(sio.planner)
            else
                correction=mark_failed!(sio.planner, name)
                if !correction["retry"]
                    push!(failed, name)
                    next_phase!(sio.planner)
                end
            end
        else
            correction=mark_failed!(sio.planner, name, result["diagnosis"])
            if !correction["retry"]
                push!(failed, name)
                next_phase!(sio.planner)
            end
        end
    end

    sio.status="integrating"
    integrated=integrate(sio.integrator, sio.phase_outputs, goal)
    sio.status="complete"

    return Dict(
        "goal"=>goal,
        "deliverable"=>integrated["deliverable"],
        "summary"=>integrated["summary"],
        "structure"=>integrated["structure"],
        "phases_completed"=>[p["phase"] for p in sio.phase_outputs],
        "phases_failed"=>failed,
        "total_phases"=>length(phases),
        "phase_details"=>[Dict(
            "name"=>p["phase"],
            "iterations"=>p["iterations"],
            "coherence"=>get(p,"coherence",0.0)
        ) for p in sio.phase_outputs],
        "plan"=>get_plan_summary(sio.planner),
        "time_elapsed"=>round(time()-sio.start_time,digits=2)
    )
end

function reset!(sio::SIOOrchestrator)
    sio.current_goal=""
    sio.phase_outputs=Dict[]
    sio.status="idle"
    set_plan!(sio.planner, Dict[])
    sio.monitor.coherence_history=Float64[]
    sio.monitor.alerts=String[]
end

end # module SIO
