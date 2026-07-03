"""
clifford_qa_projector - Clifford QA Projector Layer (Semantic Calculus V2).

This module learns the geometric shift vector (transition) and Clifford rotors from question space
to statement/answer space using the full 10000-dimensional phase vectors,
enabling zero-shot question answering on unseen facts.
"""
module CliffordQAProjectorModule

using LinearAlgebra
using ..WordPhysics: compute_extended_phase_vector

export QAProjectorMemory, learn_qa_shift!, project_question, retrieve_answer_facts,
       CliffordRotor, construct_rotor, apply_rotor

const SIGNATURE_DIM = 10000

struct CliffordRotor
    e1::Vector{Float64}
    e2::Vector{Float64}
    theta::Float64
end

mutable struct QAProjectorMemory
    shifts::Dict{String, Vector{Float64}}
    counts::Dict{String, Int}
    rotors::Dict{String, CliffordRotor}
end

function QAProjectorMemory()
    shifts = Dict{String, Vector{Float64}}(
        "yes_no" => zeros(Float64, SIGNATURE_DIM),
        "method" => zeros(Float64, SIGNATURE_DIM),
        "reason" => zeros(Float64, SIGNATURE_DIM),
        "definition" => zeros(Float64, SIGNATURE_DIM),
        "time" => zeros(Float64, SIGNATURE_DIM),
        "place" => zeros(Float64, SIGNATURE_DIM),
        "general" => zeros(Float64, SIGNATURE_DIM)
    )
    counts = Dict{String, Int}(
        "yes_no" => 0,
        "method" => 0,
        "reason" => 0,
        "definition" => 0,
        "time" => 0,
        "place" => 0,
        "general" => 0
    )
    rotors = Dict{String, CliffordRotor}()
    for k in keys(shifts)
        rotors[k] = CliffordRotor(zeros(Float64, SIGNATURE_DIM), zeros(Float64, SIGNATURE_DIM), 0.0)
    end
    return QAProjectorMemory(shifts, counts, rotors)
end

function construct_rotor(u::Vector{Float64}, v::Vector{Float64})
    length(u) == length(v) || throw(DimensionMismatch("rotor endpoints must have the same dimension"))
    nu = norm(u)
    nv = norm(v)
    if nu < 1e-10 || nv < 1e-10
        return CliffordRotor(zeros(Float64, length(u)), zeros(Float64, length(u)), 0.0)
    end
    u_norm = u ./ nu
    v_norm = v ./ nv
    
    cos_theta = clamp(dot(u_norm, v_norm), -1.0, 1.0)
    theta = acos(cos_theta)
    
    if abs(theta) < 1e-8
        return CliffordRotor(u_norm, zeros(Float64, length(u)), 0.0)
    elseif abs(theta - pi) < 1e-8
        idx = argmin(abs.(u_norm))
        orthogonal = zeros(Float64, length(u))
        orthogonal[idx] = 1.0
        orthogonal .-= dot(orthogonal, u_norm) .* u_norm
        n_orth = norm(orthogonal)
        orthogonal ./= n_orth > 1e-10 ? n_orth : 1.0
        return CliffordRotor(u_norm, orthogonal, pi)
    end
    
    e2 = v_norm .- cos_theta .* u_norm
    n_e2 = norm(e2)
    e2 ./= n_e2 > 1e-10 ? n_e2 : 1.0
    
    return CliffordRotor(u_norm, e2, theta)
end

function apply_rotor(rotor::CliffordRotor, x::Vector{Float64})
    length(rotor.e1) == length(x) || throw(DimensionMismatch("rotor and vector must have the same dimension"))
    length(rotor.e2) == length(x) || throw(DimensionMismatch("rotor plane vectors must have the same dimension as the input"))
    norm(x) < 1e-10 && return x
    if abs(rotor.theta) < 1e-8
        return x
    end
    
    c1 = dot(x, rotor.e1)
    c2 = dot(x, rotor.e2)
    
    x_parallel = c1 .* rotor.e1 .+ c2 .* rotor.e2
    x_perp = x .- x_parallel
    
    c1_rotated = c1 * cos(rotor.theta) - c2 * sin(rotor.theta)
    c2_rotated = c1 * sin(rotor.theta) + c2 * cos(rotor.theta)
    
    x_parallel_rotated = c1_rotated .* rotor.e1 .+ c2_rotated .* rotor.e2
    
    return x_perp .+ x_parallel_rotated
end

function detect_qtype(question::AbstractString)
    s = lowercase(strip(String(question)))
    if occursin(r"^(هل|أهل|اهل|is|are|do|does|did|can|could|will|would)\b", s)
        return "yes_no"
    elseif occursin(r"^(كيف|how)\b", s)
        return "method"
    elseif occursin(r"^(لماذا|لما|why)\b", s)
        return "reason"
    elseif occursin(r"^(ما|ماذا|what)\b", s)
        return "definition"
    elseif occursin(r"^(met|when|متى)\b", s) || occursin("متى", s)
        return "time"
    elseif occursin(r"^(أين|اين|where)\b", s) || occursin("أين", s) || occursin("اين", s)
        return "place"
    end
    return "general"
end

function get_sentence_vector(text::AbstractString)
    words = String[m.match for m in eachmatch(r"[\p{L}\p{N}_]+", String(text))]
    vec = zeros(Float64, SIGNATURE_DIM)
    for w in words
        length(w) < 2 && continue
        v = Float64.(compute_extended_phase_vector(String(w)))
        vec .+= v
    end
    n = norm(vec)
    n > 1e-10 && (vec ./= n)
    return vec
end

function learn_qa_shift!(mem::QAProjectorMemory, question::AbstractString, answer::AbstractString)
    q_vec = get_sentence_vector(question)
    a_vec = get_sentence_vector(answer)
    
    delta = a_vec .- q_vec
    qtype = detect_qtype(question)
    
    # Update running average shift vector
    count = get(mem.counts, qtype, 0)
    old_shift = get(mem.shifts, qtype, zeros(Float64, SIGNATURE_DIM))
    new_shift = (old_shift .* count .+ delta) ./ (count + 1)
    nrm = norm(new_shift)
    nrm > 1e-10 && (new_shift ./= nrm)
    mem.shifts[qtype] = new_shift
    mem.counts[qtype] = count + 1
    
    # General shift
    gen_count = get(mem.counts, "general", 0)
    old_gen = get(mem.shifts, "general", zeros(Float64, SIGNATURE_DIM))
    new_gen = (old_gen .* gen_count .+ delta) ./ (gen_count + 1)
    nrm_gen = norm(new_gen)
    nrm_gen > 1e-10 && (new_gen ./= nrm_gen)
    mem.shifts["general"] = new_gen
    mem.counts["general"] = gen_count + 1
    
    # Update Clifford rotor
    r = construct_rotor(q_vec, a_vec)
    mem.rotors[qtype] = r
    mem.rotors["general"] = r
    
    return true
end

function project_question(mem::QAProjectorMemory, question::AbstractString)
    q_vec = get_sentence_vector(question)
    qtype = detect_qtype(question)
    
    rotor = get(mem.rotors, qtype, get(mem.rotors, "general", nothing))
    if rotor !== nothing && abs(rotor.theta) > 1e-8
        projected = apply_rotor(rotor, q_vec)
    else
        shift = get(mem.shifts, qtype, get(mem.shifts, "general", zeros(Float64, SIGNATURE_DIM)))
        projected = q_vec .+ shift
    end
    
    nrm = norm(projected)
    nrm > 1e-10 && (projected ./= nrm)
    return projected
end

function retrieve_answer_facts(mem::QAProjectorMemory, question::AbstractString, 
                              corpus_sentences::Vector{String}; limit::Int=3)
    proj_vec = project_question(mem, question)
    
    scored = Tuple{Float64, String}[]
    for sentence in corpus_sentences
        length(strip(sentence)) < 3 && continue
        s_vec = get_sentence_vector(sentence)
        
        denom = norm(proj_vec) * norm(s_vec)
        sim = denom > 1e-10 ? dot(proj_vec, s_vec) / denom : 0.0
        
        push!(scored, (sim, sentence))
    end
    
    sort!(scored; by=x -> -x[1])
    return scored[1:min(limit, length(scored))]
end

end # module CliffordQAProjectorModule
