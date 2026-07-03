"""
al_tadbir - procedural plan pattern memory for Mirnan.

This layer remembers action paths: how work tends to be ordered. It is separate
from al_lisan (wording), al_code (program structure), and al_aql (logic/causality).
"""
module AlTadbir

using JSON

export TadbirPatternRecord, TadbirMemory,
       learn_tadbir_from_text!, train_tadbir_from_texts!,
       select_tadbir_pattern, preferred_tadbir_slot_values,
       render_tadbir_plan,
       save_tadbir, load_tadbir, tadbir_to_dict,
       has_tadbir_patterns

const AL_TADBIR_VERSION = 1

mutable struct TadbirPatternRecord
    domain::String
    shape::String
    roles::Vector{String}
    count::Int
    examples::Vector{String}
    slots::Dict{String,Dict{String,Int}}
end

mutable struct TadbirMemory
    patterns::Dict{String,TadbirPatternRecord}
    max_examples::Int
    max_slot_values::Int
end

function TadbirMemory(; max_examples::Int=5, max_slot_values::Int=80)
    return TadbirMemory(Dict{String,TadbirPatternRecord}(), max_examples, max_slot_values)
end

const STEP_PREFIX_RE = r"^\s*(?:[-*•]|\d+[\.\)]|[أابجدهوزحطي]\))\s*"
const AR_PLAN_WORDS = Set([
    "\u062e\u0637\u0629", "\u062e\u0637\u0647", "\u062e\u0637\u0648\u0627\u062a",
    "\u0627\u0641\u062d\u0635", "\u0627\u0642\u0631\u0623", "\u0634\u062e\u0635",
    "\u0639\u062f\u0644", "\u0646\u0641\u0630", "\u0627\u062e\u062a\u0628\u0631",
    "\u062a\u062d\u0642\u0642", "\u0642\u0627\u0631\u0646", "\u0627\u0644\u0646\u062a\u064a\u062c\u0629",
])

function _clean_step(s::AbstractString)
    x = replace(strip(String(s)), STEP_PREFIX_RE => "")
    x = replace(x, r"\s+" => " ")
    return strip(x)
end

function _split_plan_steps(text::AbstractString)
    s = String(text)
    if occursin("->", s)
        return String[_clean_step(p) for p in split(s, "->") if !isempty(_clean_step(p))]
    end
    steps = String[]
    for raw in split(s, '\n')
        line = _clean_step(raw)
        isempty(line) && continue
        if occursin(STEP_PREFIX_RE, raw) || occursin(":", raw) || length(steps) > 0
            push!(steps, line)
        end
    end
    return steps
end

function _has_arabic(s::AbstractString)
    return occursin(r"[\u0600-\u06FF]", String(s))
end

function _words(s::AbstractString)
    return [lowercase(m.match) for m in eachmatch(r"[\p{L}\p{N}_]+", String(s))]
end

function _step_role(step::AbstractString)
    words = Set(_words(step))
    raw = lowercase(String(step))
    if any(w -> w in words, ["test", "verify", "check", "validate"]) ||
       any(w -> occursin(w, raw), ["\u0627\u062e\u062a\u0628\u0631", "\u062a\u062d\u0642\u0642", "\u062a\u062b\u0628\u062a"])
        return "verify"
    elseif any(w -> w in words, ["read", "inspect", "observe", "scan", "look", "review"]) ||
       any(w -> occursin(w, raw), ["\u0627\u0642\u0631", "\u0627\u0641\u062d\u0635", "\u0631\u0627\u062c\u0639"])
        return "observe"
    elseif any(w -> w in words, ["diagnose", "isolate", "trace", "identify", "analyze"]) ||
           any(w -> occursin(w, raw), ["\u0634\u062e\u0635", "\u062d\u062f\u062f", "\u062d\u0644\u0644"])
        return "diagnose"
    elseif any(w -> w in words, ["design", "choose", "plan", "outline"]) ||
           any(w -> occursin(w, raw), ["\u0635\u0645\u0645", "\u0627\u062e\u062a\u0631", "\u062e\u0637\u0637"])
        return "design"
    elseif any(w -> w in words, ["implement", "patch", "edit", "write", "build", "run"]) ||
           any(w -> occursin(w, raw), ["\u0646\u0641\u0630", "\u0639\u062f\u0644", "\u0627\u0643\u062a\u0628", "\u0627\u0628\u0646"])
        return "execute"
    elseif any(w -> w in words, ["summarize", "report", "explain", "deliver"]) ||
           any(w -> occursin(w, raw), ["\u0644\u062e\u0635", "\u0642\u0631\u0631", "\u0627\u0634\u0631\u062d", "\u0633\u0644\u0645"])
        return "report"
    elseif any(w -> w in words, ["compare", "rank", "evaluate"]) ||
           any(w -> occursin(w, raw), ["\u0642\u0627\u0631\u0646", "\u0642\u064a\u0645"])
        return "compare"
    end
    return "step"
end

function _shape_label(role::String)
    role == "observe" && return "OBSERVE"
    role == "diagnose" && return "DIAGNOSE"
    role == "design" && return "DESIGN"
    role == "execute" && return "EXECUTE"
    role == "verify" && return "VERIFY"
    role == "report" && return "REPORT"
    role == "compare" && return "COMPARE"
    return "STEP"
end

function _domain_from_steps(steps::Vector{String})
    joined = lowercase(join(steps, " "))
    if occursin("test", joined) || occursin("bug", joined) || occursin("code", joined) ||
       occursin("\u0643\u0648\u062f", joined) || occursin("\u0627\u062e\u062a\u0628\u0627\u0631", joined)
        return "engineering"
    elseif occursin("research", joined) || occursin("source", joined) || occursin("compare", joined) ||
           occursin("\u0645\u0635\u0627\u062f\u0631", joined) || occursin("\u0642\u0627\u0631\u0646", joined)
        return "research"
    elseif occursin("math", joined) || occursin("solve", joined) ||
           occursin("\u0645\u0633\u0623\u0644\u0629", joined) || occursin("\u062d\u0644", joined)
        return "math"
    end
    return "general"
end

function _push_slot_value!(bucket::Dict{String,Int}, value::String, max_values::Int)
    isempty(strip(value)) && return
    bucket[value] = get(bucket, value, 0) + 1
    if length(bucket) > max_values
        ordered = sort(collect(bucket); by=x -> (x[2], x[1]))
        delete!(bucket, ordered[1][1])
    end
end

function _learn_steps!(mem::TadbirMemory, steps::Vector{String})
    2 <= length(steps) <= 20 || return 0
    roles = [_step_role(s) for s in steps]
    shape = join(_shape_label.(roles), " ")
    domain = _domain_from_steps(steps)
    key = string(domain, "\t", shape)
    rec = get!(mem.patterns, key) do
        TadbirPatternRecord(domain, shape, copy(roles), 0, String[], Dict{String,Dict{String,Int}}())
    end
    rec.count += 1
    example = join(steps, " -> ")
    if length(rec.examples) < mem.max_examples && !(example in rec.examples)
        push!(rec.examples, example)
    end
    for (role, step) in zip(roles, steps)
        bucket = get!(rec.slots, role, Dict{String,Int}())
        _push_slot_value!(bucket, step, mem.max_slot_values)
    end
    return 1
end

function learn_tadbir_from_text!(mem::TadbirMemory, text::AbstractString)
    steps = _split_plan_steps(text)
    return _learn_steps!(mem, steps)
end

function train_tadbir_from_texts!(mem::TadbirMemory, texts::Vector{String}; max_plans::Int=50_000)
    total = 0
    for text in texts
        total >= max_plans && break
        total += learn_tadbir_from_text!(mem, text)
    end
    return total
end

has_tadbir_patterns(mem::TadbirMemory) = !isempty(mem.patterns)

function _prompt_domain(prompt::AbstractString)
    p = lowercase(String(prompt))
    if occursin("code", p) || occursin("bug", p) || occursin("test", p) || occursin("\u0643\u0648\u062f", p)
        return "engineering"
    elseif occursin("research", p) || occursin("source", p) || occursin("compare", p) || occursin("\u0628\u062d\u062b", p)
        return "research"
    elseif occursin("math", p) || occursin("solve", p) || occursin("\u0645\u0633\u0623\u0644\u0629", p)
        return "math"
    end
    return ""
end

function _prompt_overlap(rec::TadbirPatternRecord, prompt::AbstractString)
    pts = Set(_words(prompt))
    isempty(pts) && return 0.0
    hits = 0
    total = 0
    for bucket in values(rec.slots)
        for value in keys(bucket)
            for word in _words(value)
                total += 1
                word in pts && (hits += 1)
            end
        end
    end
    total == 0 && return 0.0
    return hits / min(total, length(pts))
end

function _intent_score(rec::TadbirPatternRecord, prompt::AbstractString)
    p = lowercase(String(prompt))
    score = 0.0
    if occursin("debug", p) || occursin("bug", p) || occursin("\u062e\u0637\u0623", p)
        "diagnose" in rec.roles && (score += 1.5)
        "verify" in rec.roles && (score += 0.8)
    end
    if occursin("implement", p) || occursin("build", p) || occursin("\u0646\u0641\u0630", p)
        "execute" in rec.roles && (score += 1.2)
    end
    if occursin("research", p) || occursin("compare", p) || occursin("\u0642\u0627\u0631\u0646", p)
        "compare" in rec.roles && (score += 1.2)
        "report" in rec.roles && (score += 0.7)
    end
    if occursin("test", p) || occursin("verify", p) || occursin("\u0627\u062e\u062a\u0628\u0631", p)
        "verify" in rec.roles && (score += 1.0)
    end
    return score
end

function select_tadbir_pattern(mem::TadbirMemory, prompt::AbstractString; domain::String="")
    isempty(mem.patterns) && return nothing
    wanted = isempty(domain) ? _prompt_domain(prompt) : domain
    scored = Tuple{Float64,TadbirPatternRecord}[]
    for rec in values(mem.patterns)
        if !isempty(wanted) && rec.domain != wanted
            continue
        end
        score = log(1 + rec.count)
        score += 1.5 * _prompt_overlap(rec, prompt)
        score += _intent_score(rec, prompt)
        length(rec.roles) > 12 && (score -= 0.2)
        push!(scored, (score, rec))
    end
    isempty(scored) && return nothing
    sort!(scored; by=x -> -x[1])
    return scored[1][2]
end

function preferred_tadbir_slot_values(rec::TadbirPatternRecord, role::String; limit::Int=20)
    bucket = get(rec.slots, role, Dict{String,Int}())
    vals = sort(collect(bucket); by=x -> (-x[2], x[1]))
    isempty(vals) && return String[]
    return [v[1] for v in vals[1:min(limit, length(vals))]]
end

function render_tadbir_plan(mem::TadbirMemory, prompt::AbstractString; domain::String="")
    rec = select_tadbir_pattern(mem, prompt; domain=domain)
    rec === nothing && return String[]
    out = String[]
    for role in rec.roles
        vals = preferred_tadbir_slot_values(rec, role; limit=1)
        push!(out, isempty(vals) ? role : vals[1])
    end
    return out
end

function tadbir_to_dict(mem::TadbirMemory)
    records = sort(collect(values(mem.patterns)); by=r -> (-r.count, r.domain, r.shape))
    return Dict{String,Any}(
        "version" => AL_TADBIR_VERSION,
        "n_patterns" => length(records),
        "patterns" => [Dict{String,Any}(
            "domain" => rec.domain,
            "shape" => rec.shape,
            "roles" => rec.roles,
            "count" => rec.count,
            "examples" => rec.examples,
            "slots" => rec.slots,
        ) for rec in records],
    )
end

function save_tadbir(mem::TadbirMemory, path::String)
    mkpath(dirname(path))
    open(path, "w") do io
        JSON.print(io, tadbir_to_dict(mem))
    end
    return path
end

function load_tadbir(path::String)
    mem = TadbirMemory()
    isfile(path) || return mem
    data = JSON.parsefile(path)
    for item in get(data, "patterns", Any[])
        item isa AbstractDict || continue
        domain = String(get(item, "domain", "general"))
        shape = String(get(item, "shape", ""))
        isempty(shape) && continue
        roles = String[String(r) for r in get(item, "roles", String[])]
        examples = String[String(e) for e in get(item, "examples", String[])]
        raw_slots = get(item, "slots", Dict{String,Any}())
        slots = Dict{String,Dict{String,Int}}()
        for (role, bucket) in raw_slots
            bucket isa AbstractDict || continue
            slots[String(role)] = Dict{String,Int}(String(k) => Int(v) for (k, v) in bucket)
        end
        key = string(domain, "\t", shape)
        mem.patterns[key] = TadbirPatternRecord(
            domain, shape, roles, Int(get(item, "count", 0)), examples, slots)
    end
    return mem
end

end # module AlTadbir
