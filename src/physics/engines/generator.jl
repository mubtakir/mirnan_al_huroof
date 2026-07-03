"""
Generator for MirnanNew — Wave-Based Physics Generation.
All scoring uses wave superposition + Born rule.
Engines: Kuramoto, EntropyGate, CausalFlow, Cascade, Heterodyne, ResonantChain,
         PhaseAlignment, PromptAlignment, Diversity, Gravity, Syntax,
         DensityMatrix, DCCF, PPM, RootAffinity, SurfaceAffinity, K_sem.
"""
module Generator

using LinearAlgebra, Random, SparseArrays, JSON

using ..Constants
using ..MirnanConfig: MirnanCfg, load_config, get_val, get_section
using ..WordPhysics: compute_extended_phase_vector, compute_word_mass, compute_word_frequency
using ..WordPhysics
using ..EntropyGateModule
using ..RAMCore
using ..ResonantChain
using ..Heterodyne
using ..PhaseReinforcement
using ..DensityMatrix
using ..DCCF
using ..PPM
using ..AMFS
using ..CausalFlow
using ..HolographicKB
using ..PotentialCascade
using ..SemanticComprehension
using ..SyntaxField
using ..KuramotoOscillator
using ..ContextualLearning
using ..PRNNCore
using ..PRNNLearner
using ..WaveField: WaveContribution, wave_superposition, born_rule,
    compute_phases_from_context, compute_phase_from_syntax,
    compute_phase_from_gravity, compute_phase_from_ksem,
    compute_phase_from_causal_flow, compute_phase_from_density_matrix,
    compute_phase_from_ppm, destructive_interference_ratio
using ..MathBridgeModule: MathBridge, detect_mode, evaluate_math
using ..SymbolicMathEngineModule: SymbolicMathEngine, solve_arithmetic, generate_math_explanation
using ..CodeEngineModule: CodeEngine, generate_code, compile_check, load_code_model
using ..AlCode: CodePatternMemory, load_al_code, has_code_patterns,
                generate_code_from_pattern
using ..AlTadbir: TadbirMemory, load_tadbir, has_tadbir_patterns,
                  render_tadbir_plan
using ..AlHisab: HisabMemory, load_hisab, solve_hisab, render_hisab_solution,
                 render_hisab_solution_ar
using ..AlTa3rif: Ta3rifMemory, load_ta3rif, answer_ta3rif, merge_ta3rif!
using ..AlHisbanAlDalali: SemanticCalculusMemory, load_semantic_calculus,
                          has_semantic_calculus, semantic_guidance,
                          semantic_guidance_terms, semantic_relation_movement
using ..CliffordQAProjectorModule: QAProjectorMemory, learn_qa_shift!, project_question, retrieve_answer_facts
using ..CliffordTrajectoryTrackerModule: CognitiveTrajectoryTracker, reset_tracker!, absorb_input!, absorb_word!, trajectory_alignment_score
using ..SemanticImagination: SemanticSceneMemory, load_semantic_scenes
using ..AlNisba: NisbaMemory, load_nisba, select_nisba_relation
using ..AlMuradif: MuradifMemory, load_muradif, muradif_terms
using ..AlIstinbat
using ..AlIstinbat: IstinbatAttentionRecord, IstinbatAttentionMemory,
                    load_istinbat, merge_istinbat!, select_istinbat_attention,
                    select_contradiction_attention,
                    contradiction_answer_from_attention,
                    select_causal_anchor_attention, causal_anchor_answer_from_attention,
                    terms_are_opposed, terms_are_negated, update_marker_confidence!,
                    QuantityFrameMemory, load_quantity_memory, has_quantity_records
using ..AlIntibah: SemanticAttentionField, build_semantic_attention,
                   has_semantic_attention, attention_bias_terms
using ..CodePhaseEngineModule: CodePhaseEngine, generate_code_physics
using ..LexicalOracleModule: analyze_unknown_word, coin_word_for_concept
using ..RootFieldModule: root_field_report
using ..ResponsePlanning: ResponsePlanner, ResponseArchitect, TrajectoryPlanner,
    plan!, plan_response!, plan_architect!, response_fidelity, architect_score, trajectory_score, _detect_intent
using ..IntentResponsePlanner: detect_response_intent, intent_gravity_profile,
                              render_planned_response, has_plannable_response,
                              has_gravity_profile
using ..PhysicsMetrics: compute_quality_report, k_density, transition_entropy
import ..MirnanCerebellumModule
using ..MirnanCerebellumModule: MirnanCerebellum, CerebellumPolicy,
    observe_prompt, choose_policy!, apply_cerebellum_policy!,
    learn_from_outcome!, policy_summary, cerebellum_state_dict,
    restore_cerebellum_state!, reset_cerebellum!
using ..SenseSuperpositionModule: has_sense_inventory, measure_senses,
    explain_measurement
using ..SelfReviewModule: SelfReviewEngine, review_generation!, review_summary,
    predict_review_repair, learn_review_treatment!, predict_review_treatment,
    self_review_state_dict, restore_self_review_state!, reset_self_review!
using ..AlLisan: LinguisticPatternMemory, load_lisan, has_lisan_patterns,
                 select_lisan_pattern, token_role_phase, ROLE_SYNTAX_PHASE
using ..AlAql
using ..RAPGModule: RAPGKnowledgeBase, load_rapg_kb, retrieve, retrieve_by_category
import ..GpuAcceleratorModule
using ..GpuAcceleratorModule: GpuContext, gpu_release!

export MirnanGenerator, generate!, get_physics_report, pattern_memory_summary,
       gpu_init!, reset!, learn_from_feedback!,
       save_runtime_learning!, load_runtime_learning!,
       PRNNStrategy

const RUNTIME_SYNAPTIC_DECAY = 0.98

function _apply_runtime_synaptic_decay(K::SparseMatrixCSC{T,I};
                                      factor::Real=RUNTIME_SYNAPTIC_DECAY) where {T<:Real,I<:Integer}
    f = Float64(factor)
    if !isfinite(f) || f < 0.0 || f > 1.0
        throw(ArgumentError("runtime synaptic decay factor must be finite and inside [0, 1]"))
    end
    Kd = copy(K)
    Kd.nzval .*= f
    return Kd
end

const DEFAULT_WEIGHTS = Dict{String,Float64}(
    "align" => 5.0, "gravity" => 3.5, "prompt_align" => 6.5,
    "k_sem" => 5.5,
    "resonant_chain" => 0.0, "syntax" => 5.5, "diversity" => 3.0,
    "dccf" => 0.0, "causal_flow_align" => 3.0, "density_resonance" => 0.0,
    "heterodyne" => 0.0, "oscillator" => 0.0, "kb_knowledge" => 3.0,
    "cascade" => 0.4, "ppm" => 1.7, "repulsion" => 1.5,
    "root_affinity" => 2.5, "surface_affinity" => 0.0,
    "context_tension" => 1.5,
    "aql_guidance" => 4.0,
    "aql_inhibition" => 8.0,
)

const _GUIDED_VOCAB_ID_CACHE = IdDict{Any,Dict{String,Vector{Int}}}()
const _CORPUS_SENTENCE_CACHE = IdDict{Any,Vector{Vector{Int32}}}()
const _RAW_TRAINING_SENTENCE_CACHE = IdDict{Any,Vector{String}}()
const _RAW_DIALOGUE_PAIR_CACHE = IdDict{Any,Vector{Tuple{String,String}}}()
const _LEARNED_MURADIF_MEMORY = Ref{Union{MuradifMemory,Nothing}}(nothing)
const _LEARNED_ISTINBAT_MEMORY = Ref{Union{IstinbatAttentionMemory,Nothing}}(nothing)
const _SEMANTIC_RELATION_KNOWLEDGE = Ref{Union{Vector{Any},Nothing}}(nothing)
const _LEARNED_SEMANTIC_SCENE_MEMORY = Ref{Union{SemanticSceneMemory,Nothing}}(nothing)
const _LEARNED_QUANTITY_MEMORY = Ref{Union{QuantityFrameMemory,Nothing}}(nothing)

mutable struct MirnanGenerator
    vocab::Dict{String,Int}
    id2word::Dict{Int,String}
    K_sem::Union{SparseMatrixCSC{Float64,Int32},Nothing}
    K_syn::Union{SparseMatrixCSC{Float64,Int32},Nothing}
    K_causal::Union{SparseMatrixCSC{Float64,Int32},Nothing}
    beam_width::Int
    top_k::Int
    dim::Int
    entropy::EntropyGate
    ram::AttractorMemory
    resonant_chain::ResonantChainRLC
    heterodyne::HeterodyneEngine
    reinforcement::PhaseReinforcer
    density_matrix::PhaseDensityMatrix
    dccf::DynamicCCF
    prompt_field::PromptField
    causal_flow::CausalFlowField
    kb::HolographicKnowledgeBase
    cascade::PotentialCascadeLayer
    topic::TopicDensityMatrix
    math_bridge::MathBridge
    symbolic_math::SymbolicMathEngine
    code_engine::CodeEngine
    code_patterns::CodePatternMemory
    tadbir::TadbirMemory
    hisab::HisabMemory
    ta3rif::Ta3rifMemory
    hisban::SemanticCalculusMemory
    qa_projector::QAProjectorMemory
    trajectory_tracker::CognitiveTrajectoryTracker
    semantic_scenes::SemanticSceneMemory
    nisba::NisbaMemory
    code_phase::CodePhaseEngine
    response_planner::ResponsePlanner
    trajectory::TrajectoryPlanner
    architect::ResponseArchitect
    lisan::LinguisticPatternMemory
    cerebellum::MirnanCerebellum
    self_review::SelfReviewEngine
    pv_cache::Dict{String,Vector{Float32}}
    mass_cache::Dict{String,Float64}
    syntax_cache::Dict{String,Vector{Float32}}
    scoring_weights::Dict{String,Float64}
    k_sem_config::Dict{String,Float64}
    cfg::MirnanCfg
    contextual_learning::ContextualLearningState
    aql_space::Union{AlAql.SimulationSpace,Nothing}
    aql_bias::Dict{String,Float64}
    aql_inhibition::Dict{String,Float64}
    gpu::GpuContext
    rng::MersenneTwister
    model_dir::String
    runtime_learning_dir::String
    paragraph_centroids::Vector{Dict{String,Any}}
    rapg_kb::Union{RAPGKnowledgeBase, Nothing}
    retrieved_passages::Vector{String}
    retrieved_pvs::Vector{Vector{Float32}}
    retrieved_similarities::Vector{Float64}
end


function _build_scoring_weights(cfg::MirnanCfg)
    sw = copy(DEFAULT_WEIGHTS)
    cfg_sw = get_section(cfg, "scoring_weights")
    for (k, v) in cfg_sw
        sw[k] = Float64(v)
    end
    return sw
end

function _build_k_sem_config(cfg::MirnanCfg)
    ksem = Dict{String,Float64}("strength" => 1.0, "temperature" => 1.0, "threshold" => 0.0)
    cfg_ks = get_section(cfg, "k_sem")
    for (k, v) in cfg_ks
        if haskey(ksem, k)
            ksem[k] = Float64(v)
        end
    end
    return ksem
end

function MirnanGenerator(vocab::Dict{String,Int}, K_sem=nothing;
                         K_syn=nothing, K_causal=nothing,
                         beam_width::Int=5, top_k::Int=500,
                         config_path::String="", cfg::Union{MirnanCfg,Nothing}=nothing,
                         model_dir::String="")
    if K_sem !== nothing
        @assert size(K_sem, 1) <= typemax(Int32) "K_sem size $(size(K_sem, 1)) exceeds Int32 range"
    end
    K_sem_32 = K_sem === nothing ? nothing : SparseMatrixCSC{Float64, Int32}(K_sem)
    K_syn_32 = K_syn === nothing ? nothing : SparseMatrixCSC{Float64, Int32}(K_syn)
    K_causal_32 = K_causal === nothing ? nothing : SparseMatrixCSC{Float64, Int32}(K_causal)

    id2word = Dict{Int,String}(v=>k for (k,v) in vocab)
    if cfg === nothing
        cfg = isempty(config_path) ? load_config() : load_config(config_path)
    end
    scoring_weights = _build_scoring_weights(cfg)
    ksem_cfg = _build_k_sem_config(cfg)

    gen_cfg_dccf = get_section(cfg, "dccf")
    gen_cfg_entropy = get_section(cfg, "entropy")
    gen_cfg_ppm = get_section(cfg, "prompt_field")
    gen_cfg_phase = get_section(cfg, "phase_reinforcement")
    gen_cfg_cascade = get_section(cfg, "cascade")
    gen_cfg_density = get_section(cfg, "density_matrix")
    gen_cfg_heterodyne = get_section(cfg, "heterodyne")

    dccf_mt = get(gen_cfg_dccf, "mass_threshold", 0.3)
    dccf_dr = get(gen_cfg_dccf, "decay_rate", 0.5)
    scrit = get(gen_cfg_entropy, "S_crit", 2.5)
    pf_decay = get(gen_cfg_ppm, "decay_rate", 0.1)
    pf_str = get(gen_cfg_ppm, "strength", 1.0)
    pr_lr = get(gen_cfg_phase, "learning_rate", 0.15)
    pr_decay = get(gen_cfg_phase, "decay", 0.005)
    cas_lc = get(gen_cfg_cascade, "lambda_cascade", 3.0)
    cas_gam = get(gen_cfg_cascade, "gamma", 2.0)
    cas_del = get(gen_cfg_cascade, "delta", 0.3)
    cas_plt = get(gen_cfg_cascade, "phase_lock_threshold", 0.65)
    cas_rep = get(gen_cfg_cascade, "repulsion_strength", 2.5)
    cas_fric = get(gen_cfg_cascade, "friction_decay", 0.3)
    dm_dr = get(gen_cfg_density, "decay_rate", 0.8)
    het_bw = get(gen_cfg_heterodyne, "bandwidth", 0.15)
    het_ctx = get(gen_cfg_heterodyne, "context_window", 8)

    gen_bw = get(get_section(cfg, "generation"), "beam_width", beam_width)
    gen_tk = get(get_section(cfg, "generation"), "top_k", top_k)

    model_root = isempty(model_dir) ? _default_model_dir() : model_dir
    _load_with_progress(label::String, f::Function) = begin
        println("  -> loading $label...")
        flush(stdout)
        t0 = time()
        result = f()
        println("     done $label in $(round(time() - t0; digits=1))s")
        flush(stdout)
        result
    end
    code_engine = _load_code_engine_for_generator(model_root)
    code_patterns = _load_with_progress("al_code", () -> load_al_code(joinpath(model_root, "al_code.json")))
    tadbir = _load_with_progress("al_tadbir", () -> load_tadbir(joinpath(model_root, "al_tadbir.json")))
    hisab = _load_with_progress("al_hisab", () -> load_hisab(joinpath(model_root, "al_hisab.json")))
    ta3rif = _load_with_progress("al_ta3rif", () -> load_ta3rif(joinpath(model_root, "al_ta3rif.json")))
    persistent_ta3rif_path = joinpath(dirname(model_root), "knowledge", "definitions.json")
    isfile(persistent_ta3rif_path) && merge_ta3rif!(ta3rif, load_ta3rif(persistent_ta3rif_path))
    hisban = _load_with_progress("al_hisban_al_dalali", () -> load_semantic_calculus(joinpath(model_root, "al_hisban_al_dalali.json")))
    semantic_scenes = _load_with_progress("semantic_scenes", () -> load_semantic_scenes(joinpath(model_root, "semantic_scenes.json")))
    _LEARNED_SEMANTIC_SCENE_MEMORY[] = semantic_scenes
    nisba = _load_with_progress("al_nisba", () -> load_nisba(joinpath(model_root, "al_nisba.json")))
    muradif = _load_with_progress("al_muradif", () -> load_muradif(joinpath(model_root, "semantic_equivalence.json")))
    _LEARNED_MURADIF_MEMORY[] = muradif
    istinbat = _load_with_progress("al_istinbat", () -> load_istinbat(joinpath(model_root, "al_istinbat.json")))
    _LEARNED_ISTINBAT_MEMORY[] = istinbat
    quantity_memory = _load_with_progress("quantity_memory", () -> load_quantity_memory(joinpath(model_root, "quantity_memory.json")))
    _LEARNED_QUANTITY_MEMORY[] = has_quantity_records(quantity_memory) ? quantity_memory : nothing
    lisan = _load_with_progress("al_lisan", () -> load_lisan(joinpath(model_root, "al_lisan.json")))
    _load_with_progress("letter_topic_embeddings", () -> begin
        WordPhysics.load_letter_topic_embeddings!(model_root)
    end)

    paragraph_centroids = Dict{String,Any}[]
    centroid_path = joinpath(model_root, "paragraph_centroids.json")
    if isfile(centroid_path)
        try
            paragraph_centroids = _load_with_progress("paragraph_centroids", () -> begin
                data = JSON.parsefile(centroid_path)
                if data isa AbstractVector
                    return Dict{String,Any}[Dict{String,Any}(String(k) => v for (k, v) in item) for item in data if item isa AbstractDict]
                end
                return Dict{String,Any}[]
            end)
        catch e
            @warn "Generator: failed to load paragraph centroids: $e"
        end
    end

    # ═══ تهيئة قاعدة معرفة RAPG ═══
    rag_db_path = ""
    path_a = joinpath(model_root, "basil_agent", "agent_workspace", ".memory", "sovereign_logic.db")
    path_b = joinpath(dirname(model_root), "basil_agent", "agent_workspace", ".memory", "sovereign_logic.db")
    path_c = joinpath(model_root, "rapg_kb.db")
    
    if isfile(path_a)
        rag_db_path = path_a
    elseif isfile(path_b)
        rag_db_path = path_b
    else
        rag_db_path = path_c
    end
    
    println("  -> loading RAPGKnowledgeBase from $rag_db_path...")
    flush(stdout)
    rapg_kb = load_rapg_kb(rag_db_path)

    qa_projector = QAProjectorMemory()
    for rec in values(hisban.records)
        for ex in rec.examples
            q = get(ex, "source", "")
            a = get(ex, "target", "")
            if !isempty(q) && !isempty(a)
                learn_qa_shift!(qa_projector, q, a)
            end
        end
    end

    trajectory_tracker = CognitiveTrajectoryTracker()

    println("  -> constructing generator core...")
    flush(stdout)
    t_core = time()
    gen = MirnanGenerator(
        vocab, id2word, K_sem_32, K_syn_32, K_causal_32,
        gen_bw, gen_tk, TOTAL_DIM,
        EntropyGate(; S_crit=scrit), AttractorMemory(),
        ResonantChainRLC(),
        HeterodyneEngine(; bandwidth=het_bw, context_window=het_ctx),
        PhaseReinforcer(; lr=pr_lr, decay_rate=pr_decay),
        PhaseDensityMatrix(; decay_rate=dm_dr),
        DynamicCCF(; decay_rate=dccf_dr, mass_threshold=dccf_mt),
        PromptField(; decay_rate=pf_decay, strength=pf_str),
        CausalFlowField(), HolographicKnowledgeBase(),
        PotentialCascadeLayer(; lambda_cascade=cas_lc, gamma=cas_gam, delta=cas_del,
                               phase_lock_threshold=cas_plt, repulsion_strength=cas_rep,
                               friction_decay=cas_fric),
        TopicDensityMatrix(),
        MathBridge(), SymbolicMathEngine(), code_engine, code_patterns, tadbir, hisab, ta3rif, hisban, qa_projector, trajectory_tracker, semantic_scenes, nisba, CodePhaseEngine(),
        ResponsePlanner(), TrajectoryPlanner(), ResponseArchitect(),
        lisan,
        MirnanCerebellum(), SelfReviewEngine(),
        Dict{String,Vector{Float32}}(),
        Dict{String,Float64}(),
        Dict{String,Vector{Float32}}(),
        scoring_weights, ksem_cfg, cfg, ContextualLearningState(),
        AlAql.SimulationSpace(), Dict{String,Float64}(), Dict{String,Float64}(),
        GpuContext(), MersenneTwister(42),
        model_root, joinpath(model_root, "runtime_learning"),
        paragraph_centroids,
        rapg_kb, String[], Vector{Float32}[], Float64[]
    )
    println("     done generator core in $(round(time() - t_core; digits=1))s")
    flush(stdout)

    println("  -> loading runtime learning...")
    flush(stdout)
    t_runtime = time()
    load_runtime_learning!(gen)
    println("     done runtime learning in $(round(time() - t_runtime; digits=1))s")
    flush(stdout)
    persistent_dialogue_path = joinpath(dirname(model_root), "knowledge", "dialogue_facts.json")
    isfile(persistent_dialogue_path) && _load_persistent_dialogue_facts!(gen, persistent_dialogue_path)
    return gen
end

function _default_model_dir()
    return normpath(joinpath(@__DIR__, "..", "..", "..", "model"))
end

const _PUNCT_EDGE = Set(['.', ',', '،', ':', '؛', '?', '؟', '!', ')', '(', '"', '\'',
                          '«', '»', '-', '…', '\u200F', '\u200E', '\u00AD'])

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

function _cosine_similarity(a::AbstractVector{T}, b::AbstractVector{S}) where {T<:Real, S<:Real}
    (isempty(a) || isempty(b) || length(a) != length(b)) && return 0.0
    dot_prod = 0.0
    norm_a = 0.0
    norm_b = 0.0
    @inbounds for i in 1:length(a)
        dot_prod += Float64(a[i]) * Float64(b[i])
        norm_a += Float64(a[i]) * Float64(a[i])
        norm_b += Float64(b[i]) * Float64(b[i])
    end
    (norm_a == 0.0 || norm_b == 0.0) && return 0.0
    return dot_prod / (sqrt(norm_a) * sqrt(norm_b))
end

function _query_centroid(gen::MirnanGenerator, prompt_tokens::Vector{String})
    word_weights = Dict{String,Float64}()
    for tok in prompt_tokens
        w = _strip_punct_boundary(strip(tok))
        isempty(w) && continue
        length(w) < 2 && continue
        word_weights[w] = max(get(word_weights, w, 0.0), 1.0)
        
        # Expand synonyms using Muradif
        muradif = _LEARNED_MURADIF_MEMORY[]
        if muradif !== nothing
            syns = muradif_terms(muradif, w; min_score=0.20, limit=4)
            for syn in syns
                syn_clean = _strip_punct_boundary(strip(syn))
                if length(syn_clean) >= 2 && syn_clean != w
                    word_weights[syn_clean] = max(get(word_weights, syn_clean, 0.0), 0.5)
                end
            end
        end
    end
    
    dim = 0
    centroid_vec = Float64[]
    sum_mass = 0.0
    for (w, expansion_w) in word_weights
        pv = try
            Float64.(compute_extended_phase_vector(w))
        catch
            nothing
        end
        pv === nothing && continue
        if dim == 0
            dim = length(pv)
            centroid_vec = zeros(Float64, dim)
        end
        g_w = try
            Float64(compute_word_mass(w))
        catch
            1.0
        end
        total_w = expansion_w * g_w
        centroid_vec .+= total_w .* pv
        sum_mass += total_w
    end
    if sum_mass > 0.0
        centroid_vec ./= sum_mass
    end
    return centroid_vec
end

function _get_active_paragraphs(gen::MirnanGenerator, prompt_tokens::Vector{String})
    active_paras = Dict{Tuple{String,Int},Float64}()
    q_vec = _query_centroid(gen, prompt_tokens)
    isempty(q_vec) && return active_paras
    for pc in gen.paragraph_centroids
        f_name = get(pc, "file_name", "")::String
        p_idx = get(pc, "paragraph_index", 0)::Int
        centroid = get(pc, "centroid", nothing)
        if !isempty(f_name) && p_idx > 0 && centroid isa AbstractVector
            c_vec = Float64.(centroid)
            sim = _cosine_similarity(q_vec, c_vec)
            if sim > 0.45
                active_paras[(String(f_name), Int(p_idx))] = sim
            end
        end
    end
    return active_paras
end

function _load_code_engine_for_generator(model_dir::String)
    dir = isempty(model_dir) ? _default_model_dir() : model_dir
    isfile(joinpath(dir, "code_vocab.json")) || return CodeEngine()
    try
        return load_code_model(dir)
    catch e
        @warn "Generator: failed to load K_code from $dir: $e"
        return CodeEngine()
    end
end

function _pv(gen::MirnanGenerator, word::String, current_topic::Union{Vector{Float32}, Nothing}=nothing)
    if current_topic === nothing
        get!(gen.pv_cache, word) do
            compute_extended_phase_vector(word)
        end
    else
        compute_extended_phase_vector(word; current_topic=current_topic)
    end
end

function _mass(gen::MirnanGenerator, word::String)
    get!(gen.mass_cache, word) do
        compute_word_mass(word)
    end
end

function _syn(gen::MirnanGenerator, word::String)
    get!(gen.syntax_cache, word) do
        Float32.(compute_syntax_vector(word))
    end
end

function _safe_root(word::String)
    try
        return WordPhysics._extract_root_light(word)
    catch e
        @debug "_safe_root failed for '$word': $e"
        return Char[]
    end
end

function _surface_chars(word::String)
    chars = Set{Char}()
    for c in collect(WordPhysics._normalize_letters(word))
        isspace(c) && continue
        Int(c) in keys(WordPhysics.IRAB_MAP) && continue
        push!(chars, c)
    end
    return chars
end

const GENERATION_STOPWORDS = Set([
    "الى", "إلى", "الي", "إلي", "علي", "على", "في", "من", "عن", "ان", "أن", "إن", "انها",
    "انهم", "انه", "فانها", "فإنها", "فانه", "فإنه", "لانها", "لانه",
    "الا", "إلا", "الَّا", "او", "أو", "ثم", "قد", "لقد", "سوف", "كان",
    "كانت", "يكون", "تكون", "هذا", "هذه", "ذلك", "تلك", "الذي", "التي",
    "الذين", "اللاتي", "هو", "هي", "هم", "هن", "كما", "حيث", "اذا", "إذا",
    "كل", "بعض", "غير", "اما", "أما", "حتى", "بين", "بعد", "قبل", "مع",
    "اليك", "إليك", "انا", "أنا", "وانا", "وأنا", "اني", "إني", "اريد", "أريد",
    "لك", "له", "لها", "لهم", "بها", "به", "فيه", "فيها", "هناك", "هنا",
    "استطيع", "أستطيع", "انني", "أنني", "كنت", "كنتُ", "دقيقا", "دقيقاً",
    "اكتب", "أكتب", "جملة", "وصف", "وصفا", "وصفاً",
    "the", "a", "an", "and", "or", "of", "in", "on", "to", "for", "with", "from",
    "is", "are", "was", "were", "be", "being", "been",
    "do", "does", "did", "have", "has", "had",
    "can", "could", "may", "might", "must", "shall", "should", "will", "would",
    "if", "when", "because", "although", "while", "unless", "however",
    "there", "then", "not", "no", "never",
])

const _EXTRA_GENERATION_STOPWORDS = Set([
    "\u0627\u0644\u064a\u0643", "\u0625\u0644\u064a\u0643",
    "\u0627\u0646\u0627", "\u0623\u0646\u0627",
    "\u0648\u0627\u0646\u0627", "\u0648\u0623\u0646\u0627",
    "\u0627\u0646\u064a", "\u0625\u0646\u064a",
    "\u0627\u0631\u064a\u062f", "\u0623\u0631\u064a\u062f",
    "\u0627\u0633\u062a\u0637\u064a\u0639", "\u0623\u0633\u062a\u0637\u064a\u0639",
    "\u0627\u0646\u0646\u064a", "\u0623\u0646\u0646\u064a",
    "\u0643\u0646\u062a", "\u0643\u0646\u062a\u064f",
    "\u062f\u0642\u064a\u0642\u0627", "\u062f\u0642\u064a\u0642\u0627\u064b",
    "\u0627\u0643\u062a\u0628", "\u0623\u0643\u062a\u0628",
    "\u062c\u0645\u0644\u0629", "\u0648\u0635\u0641", "\u0648\u0635\u0641\u0627",
    "\u0628\u0627\u0633\u0645\u0643", "\u064a\u062f\u064a\u0647", "\u063a\u0644\u0627\u0645",
    "\u0641\u0642\u0627\u0644", "\u0631\u0633\u0648\u0644",
])

const _QUESTION_TOOL_KEYS = Set([
    "\u0645\u0627", "\u0645\u0627\u0630\u0627", "\u0645\u0646", "\u0627\u064a\u0646", "\u0623\u064a\u0646",
    "\u0645\u062a\u0649", "\u0643\u064a\u0641", "\u0644\u0645\u0627\u0630\u0627", "\u0647\u0644",
    "what", "where", "when", "why", "how",
])

function _generation_key(word::AbstractString)
    s = lowercase(strip(String(word)))
    s = replace(s, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    return s
end

function _generation_projection_key(word::AbstractString)
    key = _generation_key(word)
    projected = replace(key, r"[\u064B-\u065F\u0670]" => "")
    if occursin('\u064B', key) && endswith(projected, "ا")
        projected = projected[begin:prevind(projected, lastindex(projected))]
    end
    return projected
end

function _generation_family_key(word::AbstractString)
    s = _generation_key(word)
    s = replace(s, r"^(?:و|ف)+" => "")
    s = replace(s, r"^(?:بال|كال|فال|وال|لل|ال)" => "")
    s = replace(s, r"^(?:ب|ك|ل)" => "")
    return s
end

const _CONCEPT_ALIAS_GROUPS = Vector{Set{String}}([
    Set(["جهل", "غفله", "غفلة", "غافل", "الغفله", "الغفلة", "الغافل"]),
    Set(["جاهل", "غافل", "الجاهل", "الغافل"]),
    Set(["علم", "تعلم", "تعليم", "معرفه", "معرفة", "العلم", "التعلم", "التعليم", "المعرفه", "المعرفة"]),
    Set(["عقل", "بصيره", "بصيرة", "العقل", "البصيره", "البصيرة"]),
    Set(["فهم", "بصيره", "بصيرة", "الفهم", "البصيره", "البصيرة"]),
    Set(["نور", "ضوء", "بصيره", "بصيرة", "النور", "الضوء", "البصيره", "البصيرة"]),
    Set(["ظلام", "ظلمه", "ظلمة", "حيره", "حيرة", "الظلام", "الظلمه", "الظلمة", "الحيره", "الحيرة"]),
    Set(["خبره", "خبرة", "تجربه", "تجربة", "الخبره", "الخبرة", "التجربه", "التجربة"]),
    Set(["قوه", "قوة", "قدره", "قدرة", "سلطه", "سلطة", "القوه", "القوة", "القدره", "القدرة", "السلطه", "السلطة"]),
    Set(["عدل", "عداله", "عدالة", "انصاف", "إنصاف", "العدل", "العداله", "العدالة", "الانصاف", "الإنصاف"]),
    Set(["ظلم", "جور", "الظلم", "الجور"]),
    Set(["نزاع", "خصومه", "خصومة", "النزاع", "الخصومه", "الخصومة"]),
    Set(["عمل", "اثر", "أثر", "واقع", "العمل", "الاثر", "الأثر", "الواقع"]),
])

function _expand_concept_aliases!(keys::Set{String})
    changed = true
    while changed
        changed = false
        for group in _CONCEPT_ALIAS_GROUPS
            if !isempty(intersect(keys, group))
                before = length(keys)
                union!(keys, group)
                changed |= length(keys) > before
            end
        end
    end
    return keys
end

function _expand_learned_muradif!(keys::Set{String})
    mem = _LEARNED_MURADIF_MEMORY[]
    mem === nothing && return keys
    additions = String[]
    for key in collect(keys)
        for term in muradif_terms(mem, key; min_score=0.20, limit=4)
            push!(additions, term)
        end
    end
    for term in additions
        isempty(term) || push!(keys, term)
    end
    return keys
end

function _base_generation_keys(word::AbstractString)
    cleaned = strip(String(word), [' ', '\t', '\n', '\r', '.', ',', '،', '؛', ':', '?', '؟', '!', '"', '\''])
    raw = _generation_key(cleaned)
    fam = _generation_family_key(cleaned)
    projected = _generation_projection_key(cleaned)
    projected_fam = _generation_family_key(projected)
    keys = Set([raw, fam, projected, projected_fam])
    push!(keys, replace(raw, r"^(?:وال|فال|بال|كال|لل|ال)" => ""))
    if length(raw) > 4
        push!(keys, replace(raw, r"^(?:و|ف)" => ""))
    end
    if length(raw) > 4 && startswith(raw, "ب")
        push!(keys, raw[nextind(raw, firstindex(raw)):end])
    end
    if length(raw) > 4 && startswith(raw, "ك")
        push!(keys, raw[nextind(raw, firstindex(raw)):end])
    end
    if length(raw) > 4 && startswith(raw, "ل")
        push!(keys, raw[nextind(raw, firstindex(raw)):end])
    end
    if length(raw) > 3 && endswith(raw, "ا")
        push!(keys, raw[begin:prevind(raw, lastindex(raw))])
    end
    if length(fam) > 4 && startswith(fam, "مت")
        push!(keys, fam[nextind(fam, firstindex(fam)):end])
    end
    if length(raw) > 4 && startswith(raw, "ين")
        push!(keys, raw[nextind(raw, firstindex(raw)):end])
    end
    if length(raw) > 4 && startswith(raw, "نا")
        push!(keys, raw[nextind(raw, firstindex(raw)):end])
    end
    filter!(!isempty, keys)
    return keys
end

function _generation_keys(word::AbstractString)
    keys = _base_generation_keys(word)
    _expand_learned_muradif!(keys)
    _expand_concept_aliases!(keys)
    return keys
end

_any_key_in(keys, values) = any(k -> k in values, keys)

function _context_is_question(context_words)
    context_words === nothing && return false
    text = join(String.(context_words), " ")
    occursin("?", text) && return true
    occursin("\u061F", text) && return true
    return any(w -> _any_key_in(_generation_keys(String(w)), _QUESTION_TOOL_KEYS), context_words)
end

_has_arabic_letter(s::AbstractString) =
    any(c -> ('\u0600' <= c <= '\u06FF') || ('\u0750' <= c <= '\u077F'), s)

_has_latin_letter(s::AbstractString) =
    any(c -> ('a' <= lowercase(c) <= 'z'), s)

function _context_allows_latin(context_words)
    context_words === nothing && return false
    return any(w -> _has_latin_letter(String(w)), context_words)
end

function _is_generation_candidate(word::AbstractString, context_words=nothing)
    s = strip(String(word))
    length(s) >= 3 || return false
    any(isletter, s) || return false
    base_keys = _base_generation_keys(s)
    _any_key_in(base_keys, GENERATION_STOPWORDS) && return false
    _any_key_in(base_keys, _EXTRA_GENERATION_STOPWORDS) && return false
    _any_key_in(base_keys, _QUESTION_TOOL_KEYS) && !_context_is_question(context_words) && return false
    occursin(r"^[\W_]+$", s) && return false
    occursin(r"[#<>{}\[\]\(\)=+*/\\|@%^~`$]", s) && return false
    digit_count = count(isdigit, s)
    letter_count = count(isletter, s)
    digit_count > 0 && digit_count >= letter_count && return false
    _has_latin_letter(s) && !_context_allows_latin(context_words) && return false
    return true
end

function _aql_candidate_bias(gen::MirnanGenerator, word::AbstractString)
    bias = clamp(get(gen.aql_bias, String(word), 0.0), 0.0, 1.0)
    if bias <= 0.0
        w = String(word)
        for (key, value) in gen.aql_bias
            if occursin(w, key) || occursin(key, w)
                bias = max(bias, min(1.0, value * 0.65))
            end
        end
    end
    return bias
end

function _aql_candidate_inhibition(gen::MirnanGenerator, word::AbstractString)
    inhibition = clamp(get(gen.aql_inhibition, String(word), 0.0), 0.0, 1.0)
    if inhibition <= 0.0
        w = String(word)
        for (key, value) in gen.aql_inhibition
            if occursin(w, key) || occursin(key, w)
                inhibition = max(inhibition, min(1.0, value * 0.75))
            end
        end
    end
    return inhibition
end

function _set_overlap(a::Set{Char}, b::Set{Char})
    (isempty(a) || isempty(b)) && return 0.0
    return length(intersect(a, b)) / length(union(a, b))
end

function _root_affinity(word::String, context_words)
    context_words === nothing && return 0.0
    w_root = Set(_safe_root(word))
    w_surface = _surface_chars(word)
    score = 0.0
    weight = 0.0
    for (age, cw) in enumerate(reverse(context_words[max(1, end-5):end]))
        isempty(strip(cw)) && continue
        recency = 1.0 / age
        c_root = Set(_safe_root(cw))
        c_surface = _surface_chars(cw)
        root_score = _set_overlap(w_root, c_root)
        surface_score = _set_overlap(w_surface, c_surface)
        score += recency * max(root_score, 0.55 * surface_score)
        weight += recency
    end
    weight < 1e-10 && return 0.0
    return clamp(score / weight, 0.0, 1.0)
end

function _surface_affinity(word::String, context_words)
    context_words === nothing && return 0.0
    w_surface = _surface_chars(word)
    isempty(w_surface) && return 0.0
    best = 0.0
    for cw in context_words[max(1, end-5):end]
        best = max(best, _set_overlap(w_surface, _surface_chars(cw)))
    end
    return clamp(best, 0.0, 1.0)
end

function _context_tension(gen::MirnanGenerator, word::String, context_words, w_pv::AbstractVector)
    context_words === nothing && return 0.0, 0.0
    isempty(context_words) && return 0.0, 0.0
    w_norm = norm(w_pv)
    w_norm < 1e-10 && return 0.0, 0.0

    amp = 0.0
    phase_re = 0.0
    phase_im = 0.0
    total_weight = 0.0
    lexical_anchor = max(_root_affinity(word, context_words),
                         0.65 * _surface_affinity(word, context_words))

    for (age, cw) in enumerate(reverse(context_words[max(1, end-6):end]))
        cpv = _pv(gen, cw)
        cn = norm(cpv)
        cn < 1e-10 && continue
        recency = 1.0 / sqrt(age)
        phase_lock = max(0.0, clamp(dot(w_pv, cpv) / (w_norm * cn), -1.0, 1.0))
        local_anchor = max(
            _set_overlap(Set(_safe_root(word)), Set(_safe_root(cw))),
            0.55 * _set_overlap(_surface_chars(word), _surface_chars(cw)),
        )
        anchor = max(lexical_anchor, local_anchor)
        local_amp = recency * max(anchor, phase_lock * (0.35 + 0.65 * anchor))
        amp += local_amp
        phase_re += local_amp * (cpv[1] / cn)
        phase_im += local_amp * (length(cpv) >= 2 ? cpv[2] / cn : 0.0)
        total_weight += recency
    end

    total_weight < 1e-10 && return 0.0, 0.0
    return clamp(amp / total_weight, 0.0, 1.0), atan(phase_im, phase_re)
end

# ═══════════════════════════════════════════════════════
# THE WAVE SCORING FUNCTION
# Every engine contributes: amplitude × exp(i × phase)
# Selection via Born rule: P = |Ψ_total|²
# ═══════════════════════════════════════════════════════

_env_on(name::String, default::String) =
    lowercase(get(ENV, name, default)) in ("1", "true", "yes", "on")

function _strict_no_templates_enabled()
    return _env_on("MIRNAN_STRICT_NO_TEMPLATES", "1")
end

function _env_int(name::String, default::Int)
    try
        return parse(Int, get(ENV, name, string(default)))
    catch
        return default
    end
end

function _env_float(name::String, default::Float64)
    try
        return parse(Float64, get(ENV, name, string(default)))
    catch
        return default
    end
end

function _context_selection_enabled()
    _env_on("MIRNAN_CONTEXT_SELECTION", "1")
end

function _compact_context_similarity(word::AbstractString, context_words::Vector{String})
    isempty(context_words) && return 0.5
    wv = try
        WordPhysics.compute_compact_phase_vector(String(word))
    catch
        return 0.5
    end
    sims = Float64[]
    for cw in context_words[max(1, end-6):end]
        isempty(strip(cw)) && continue
        cv = try
            WordPhysics.compute_compact_phase_vector(cw)
        catch
            nothing
        end
        cv === nothing && continue
        push!(sims, (WordPhysics.compact_phase_similarity(wv, cv) + 1.0) / 2.0)
    end
    isempty(sims) && return 0.5
    sort!(sims; rev=true)
    return clamp(sum(sims[1:min(3, end)]) / min(3, length(sims)), 0.0, 1.0)
end

function _k_context_support(gen::MirnanGenerator, word::AbstractString, context_ids)
    matrix = gen.K_sem
    matrix === nothing && return 0.0
    wid = get(gen.vocab, String(word), 0)
    wid <= 0 && return 0.0
    (wid > size(matrix, 1) || wid > size(matrix, 2)) && return 0.0
    best = 0.0
    for cid in context_ids[max(1, end-6):end]
        cid === nothing && continue
        cid <= 0 && continue
        cid > size(matrix, 1) && continue
        val = 0.0
        try
            val = max(abs(matrix[cid, wid]), abs(matrix[wid, cid]))
        catch
            val = 0.0
        end
        best = max(best, min(1.0, log1p(val) / 4.0))
    end
    return best
end

function _select_contextual_candidates(gen::MirnanGenerator, candidates::Vector{String},
                                       context_ids, used_set)
    unique!(candidates)
    filter!(w -> !(w in used_set), candidates)
    (!_context_selection_enabled() || length(candidates) <= 1) && return candidates

    context_words = String[]
    for cid in context_ids[max(1, end-8):end]
        cid === nothing && continue
        w = get(gen.id2word, cid, nothing)
        w !== nothing && push!(context_words, w)
    end
    isempty(context_words) && return candidates

    scored = Tuple{Float64,String}[]
    min_score = _env_float("MIRNAN_CONTEXT_SELECTION_THRESHOLD", 0.18)
    keep_n = _env_int("MIRNAN_CONTEXT_SELECTION_TOP_K", min(gen.top_k, 160))
    keep_n = max(16, min(keep_n, length(candidates)))

    for w in candidates
        k_support = _k_context_support(gen, w, context_ids)
        compact = _compact_context_similarity(w, context_words)
        root = _root_affinity(w, context_words)
        score = 0.55 * k_support + 0.30 * compact + 0.15 * root
        score >= min_score && push!(scored, (score, w))
    end

    isempty(scored) && return candidates[1:min(length(candidates), keep_n)]
    sort!(scored; by=x -> -x[1])
    return [w for (_, w) in scored[1:min(keep_n, length(scored))]]
end

function _resonance_candidates(gen::MirnanGenerator, context_ids, used_set)
    candidates = String[]
    matrix = gen.K_sem
    matrix === nothing && return candidates
    k_str = get(gen.k_sem_config, "strength", 1.0)
    k_temp = max(get(gen.k_sem_config, "temperature", 1.0), 0.01)
    k_thresh = get(gen.k_sem_config, "threshold", 0.0)
    for cid in context_ids[max(1, end-3):end]
        cid === nothing && continue
        cid > size(matrix, 1) && continue
        if matrix isa AbstractSparseMatrix
            row = matrix[cid, :]
            nz_ind = row.nzind; nz_val = row.nzval
            scaled = k_str .* (nz_val .^ (1.0 / k_temp))
            p = sortperm(scaled; rev=true)
            for idx in 1:min(gen.top_k, length(p))
                scaled[p[idx]] < k_thresh && break
                w = get(gen.id2word, nz_ind[p[idx]], nothing)
                w !== nothing && _is_generation_candidate(w) && push!(candidates, w)
            end
        else
            row = Vector(matrix[cid, :])
            scaled = k_str .* (abs.(row) .^ (1.0 / k_temp))
            p = sortperm(scaled; rev=true)
            for idx in 1:min(gen.top_k, length(p))
                scaled[p[idx]] < k_thresh && break
                w = get(gen.id2word, p[idx], nothing)
                w !== nothing && _is_generation_candidate(w) && push!(candidates, w)
            end
        end
    end
    return _select_contextual_candidates(gen, candidates, context_ids, used_set)
end

function _context_coherence(gen::MirnanGenerator, word::AbstractString,
                            prompt_pvs::Vector, context_words::Vector{String},
                            current_topic::Union{Vector{Float32}, Nothing}=nothing)
    w_pv = try
        _pv(gen, String(word), current_topic)
    catch
        nothing
    end
    w_pv === nothing && return 0.5

    sims = Float64[]
    for pv in prompt_pvs
        pv === nothing && continue
        n1 = norm(w_pv); n2 = norm(pv)
        if n1 > 1e-10 && n2 > 1e-10
            push!(sims, clamp(dot(w_pv, pv) / (n1 * n2), -1.0, 1.0))
        end
    end
    for cw in context_words[max(1, end-5):end]
        cw == word && continue
        cpv = try _pv(gen, cw, current_topic) catch; nothing end
        cpv === nothing && continue
        n1 = norm(w_pv); n2 = norm(cpv)
        if n1 > 1e-10 && n2 > 1e-10
            push!(sims, 0.75 * clamp(dot(w_pv, cpv) / (n1 * n2), -1.0, 1.0))
        end
    end
    isempty(sims) && return 0.5
    sort!(sims; rev=true)
    top = sims[1:min(4, end)]
    avg = sum(top) / max(length(top), 1)
    return clamp((avg + 1.0) / 2.0, 0.0, 1.0)
end

const _DRIFT_PROPER_NAMES = Set([
    "دين", "الدين", "صلاح", "زنكي", "العادل", "السلطان", "ايوب", "الأيوبي", "الايوبي",
    "نورالدين", "نوردين", "ملكوت"
])

const _DRIFT_PROPER_NAMES_U = Set([
    "\u062f\u064a\u0646", "\u0627\u0644\u062f\u064a\u0646",
    "\u0635\u0644\u0627\u062d", "\u0632\u0646\u0643\u064a",
    "\u0627\u0644\u0639\u0627\u062f\u0644", "\u0639\u0627\u062f\u0644",
    "\u0627\u0644\u0633\u0644\u0637\u0627\u0646", "\u0633\u0644\u0637\u0627\u0646",
    "\u0627\u064a\u0648\u0628", "\u0623\u064a\u0648\u0628",
    "\u0627\u0644\u0623\u064a\u0648\u0628\u064a", "\u0627\u0644\u0627\u064a\u0648\u0628\u064a",
    "\u0646\u0648\u0631\u0627\u0644\u062f\u064a\u0646", "\u0646\u0648\u0631\u062f\u064a\u0646",
    "\u0645\u0644\u0643\u0648\u062a",
])

const _DRIFT_POETIC_LEAPS = Set(["كبد", "عنان"])
const _POETIC_CONTEXT_CUES = Set(["شعر", "قصيدة", "بلاغة", "مجاز", "ادب", "الأدب", "الادب"])
const _HISTORY_CONTEXT_CUES = Set(["تاريخ", "تاريخي", "سلطان", "ملك", "معركة", "دولة", "زنكي", "صلاح"])
const _RELIGION_CONTEXT_CUES = Set(["دين", "الدين", "إيمان", "ايمان", "فقه", "قرآن", "قران", "حديث"])
const _DRIFT_POETIC_LEAPS_U = Set(["\u0643\u0628\u062f", "\u0639\u0646\u0627\u0646"])
const _POETIC_CONTEXT_CUES_U = Set([
    "\u0634\u0639\u0631", "\u0642\u0635\u064a\u062f\u0629", "\u0628\u0644\u0627\u063a\u0629",
    "\u0645\u062c\u0627\u0632", "\u0627\u062f\u0628", "\u0623\u062f\u0628", "\u0627\u0644\u0627\u062f\u0628",
])
const _HISTORY_CONTEXT_CUES_U = Set([
    "\u062a\u0627\u0631\u064a\u062e", "\u062a\u0627\u0631\u064a\u062e\u064a",
    "\u0633\u0644\u0637\u0627\u0646", "\u0645\u0644\u0643", "\u0645\u0639\u0631\u0643\u0629",
    "\u062f\u0648\u0644\u0629", "\u0632\u0646\u0643\u064a", "\u0635\u0644\u0627\u062d",
])
const _RELIGION_CONTEXT_CUES_U = Set([
    "\u062f\u064a\u0646", "\u0627\u0644\u062f\u064a\u0646", "\u0625\u064a\u0645\u0627\u0646",
    "\u0627\u064a\u0645\u0627\u0646", "\u0641\u0642\u0647", "\u0642\u0631\u0622\u0646",
    "\u0642\u0631\u0627\u0646", "\u062d\u062f\u064a\u062b",
])

_is_poetic_drift_key(key::AbstractString) =
    key == "\u0643\u0628\u062f" || key == "\u0639\u0646\u0627\u0646"

_is_proper_drift_key(key::AbstractString) =
    key == "\u062f\u064a\u0646" || key == "\u0627\u0644\u062f\u064a\u0646" ||
    key == "\u0635\u0644\u0627\u062d" || key == "\u0632\u0646\u0643\u064a" ||
    key == "\u0627\u0644\u0639\u0627\u062f\u0644" || key == "\u0639\u0627\u062f\u0644" ||
    key == "\u0627\u0644\u0633\u0644\u0637\u0627\u0646" || key == "\u0633\u0644\u0637\u0627\u0646" ||
    key == "\u0627\u064a\u0648\u0628" || key == "\u0623\u064a\u0648\u0628" ||
    key == "\u0646\u0648\u0631\u0627\u0644\u062f\u064a\u0646" || key == "\u0646\u0648\u0631\u062f\u064a\u0646" ||
    key == "\u0645\u0644\u0643\u0648\u062a"

function _prompt_anchor_support(gen::MirnanGenerator, word::AbstractString, prompt_tokens::Vector{String})
    wid = get(gen.vocab, String(word), 0)
    wid <= 0 && return 0.5
    matrix = gen.K_sem
    matrix === nothing && return 0.5
    (wid > size(matrix, 1) || wid > size(matrix, 2)) && return 0.5
    prompt_ids = Int[]
    for t in prompt_tokens
        tid = get(gen.vocab, t, 0)
        tid > 0 && tid <= size(matrix, 1) && push!(prompt_ids, tid)
    end
    isempty(prompt_ids) && return 0.5

    hits = 0
    total = 0.0
    for tid in prompt_ids
        val = 0.0
        try
            val = max(abs(matrix[tid, wid]), wid <= size(matrix, 1) ? abs(matrix[wid, tid]) : 0.0)
        catch
            val = 0.0
        end
        if val > 1e-9
            hits += 1
            total += min(1.0, log1p(val) / 4.0)
        end
    end
    breadth = hits / length(prompt_ids)
    strength = total / length(prompt_ids)
    return clamp(0.35 + 0.45 * breadth + 0.20 * strength, 0.0, 1.0)
end

function _drift_penalty(word::AbstractString, prompt_tokens::Vector{String}, output_words::Vector{String})
    keys = _generation_keys(word)
    prompt_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    !isempty(intersect(keys, prompt_keys)) && return 0.0

    penalty = 0.0
    if _any_key_in(keys, _DRIFT_PROPER_NAMES) || any(_is_proper_drift_key, keys)
        penalty += 0.85
    end
    if (_any_key_in(keys, _DRIFT_POETIC_LEAPS) || any(_is_poetic_drift_key, keys)) &&
       isempty(intersect(prompt_keys, _POETIC_CONTEXT_CUES))
        penalty += 0.70
    end
    if length(output_words) >= 1
        prev_keys = _generation_keys(output_words[end])
        if (_any_key_in(prev_keys, union(_DRIFT_PROPER_NAMES, _DRIFT_PROPER_NAMES_U)) ||
            any(_is_proper_drift_key, prev_keys)) &&
           (_any_key_in(keys, union(_DRIFT_PROPER_NAMES, _DRIFT_PROPER_NAMES_U)) ||
            any(_is_proper_drift_key, keys))
            penalty += 0.60
        end
    end
    return penalty
end

function _reject_context_drift(word::AbstractString, prompt_tokens::Vector{String}, output_words::Vector{String})
    keys = _generation_keys(word)
    prompt_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    !isempty(intersect(keys, prompt_keys)) && return false

    has_history = !isempty(intersect(prompt_keys, union(_HISTORY_CONTEXT_CUES, _HISTORY_CONTEXT_CUES_U)))
    has_religion = !isempty(intersect(prompt_keys, union(_RELIGION_CONTEXT_CUES, _RELIGION_CONTEXT_CUES_U)))
    has_poetry = !isempty(intersect(prompt_keys, union(_POETIC_CONTEXT_CUES, _POETIC_CONTEXT_CUES_U)))

    if (_any_key_in(keys, union(_DRIFT_POETIC_LEAPS, _DRIFT_POETIC_LEAPS_U)) ||
        any(_is_poetic_drift_key, keys)) && !has_poetry
        return true
    end
    if (_any_key_in(keys, union(_DRIFT_PROPER_NAMES, _DRIFT_PROPER_NAMES_U)) ||
        any(_is_proper_drift_key, keys)) && !(has_history || has_religion)
        return true
    end
    if length(output_words) >= 1
        prev_keys = _generation_keys(output_words[end])
        if (_any_key_in(prev_keys, union(_DRIFT_PROPER_NAMES, _DRIFT_PROPER_NAMES_U)) ||
            any(_is_proper_drift_key, prev_keys)) &&
           (_any_key_in(keys, union(_DRIFT_PROPER_NAMES, _DRIFT_PROPER_NAMES_U)) ||
            any(_is_proper_drift_key, keys)) && !has_history
            return true
        end
    end
    return false
end

function _simple_text_template(prompt_tokens::Vector{String})
    return ""
    keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    if ("\u0627\u0644\u0633\u0644\u0627\u0645" in keys || "\u0633\u0644\u0627\u0645" in keys) &&
       ("\u0639\u0644\u064a\u0643\u0645" in keys || "\u0639\u0644\u064a\u0643" in keys)
        return "\u0648\u0639\u0644\u064a\u0643\u0645 \u0627\u0644\u0633\u0644\u0627\u0645 \u0648\u0631\u062d\u0645\u0629 \u0627\u0644\u0644\u0647."
    end
    if "knowledge" in keys
        return "Knowledge raises understanding through practice and review."
    end
    if ("\u0627\u0630\u0627" in keys || "\u0625\u0630\u0627" in keys) &&
       ("\u0632\u0627\u062f" in keys || "\u064a\u0632\u064a\u062f" in keys)
        return "\u0625\u0630\u0627 \u0632\u0627\u062f \u0627\u0644\u0639\u0644\u0645 \u0632\u0627\u062f \u0627\u0644\u0641\u0647\u0645\u060c \u0644\u0623\u0646 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u062a\u0641\u062a\u062d \u0637\u0631\u064a\u0642 \u0627\u0644\u0625\u062f\u0631\u0627\u0643."
    end
    if "\u0633\u0645\u0627\u0621" in keys
        return "\u0627\u0644\u0633\u0645\u0627\u0621 \u0635\u0627\u0641\u064a\u0629 \u0648\u0627\u0633\u0639\u0629\u060c \u064a\u0645\u0644\u0624\u0647\u0627 \u0627\u0644\u0636\u0648\u0621 \u0648\u0627\u0644\u0647\u062f\u0648\u0621."
    end
    if "\u0639\u0644\u0645" in keys
        return "\u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631 \u064a\u0641\u062a\u062d \u0627\u0644\u0639\u0642\u0648\u0644 \u0648\u064a\u0632\u064a\u062f \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0641\u0647\u0645\u0627 \u0648\u062d\u0643\u0645\u0629."
    end
    if "\u062d\u0643\u0645\u0629" in keys
        return "\u0627\u0644\u062d\u0643\u0645\u0629 \u0637\u0631\u064a\u0642 \u064a\u0647\u062f\u064a \u0627\u0644\u0641\u0647\u0645 \u0648\u064a\u0636\u0628\u0637 \u0627\u0644\u0639\u0645\u0644."
    end
    if "\u062c\u0645\u0644\u0629" in keys || "\u0648\u0635\u0641" in keys
        topic = ""
        for t in reverse(prompt_tokens)
            ks = _generation_keys(t)
            if !_any_key_in(ks, GENERATION_STOPWORDS) &&
               !_any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) &&
               !("\u0639\u0646" in ks)
                topic = t
                break
            end
        end
        !isempty(topic) && return "$(topic) \u0645\u0639\u0646\u0649 \u0648\u0627\u0636\u062d \u064a\u062d\u062a\u0627\u062c \u0625\u0644\u0649 \u0639\u0628\u0627\u0631\u0629 \u0647\u0627\u062f\u0626\u0629 \u0648\u0645\u0628\u0627\u0634\u0631\u0629."
    end
    content = String[]
    for t in prompt_tokens
        ks = _generation_keys(t)
        _any_key_in(ks, GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
        _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
        clean = strip(replace(t, r"[[:punct:]؟،؛]" => ""))
        isempty(clean) || push!(content, clean)
    end
    if 1 <= length(content) <= 3
        topic = join(content, " ")
        return "$(topic)\u060c \u0648\u0647\u0648 \u0645\u0639\u0646\u0649 \u064a\u062a\u0636\u062d \u0628\u0627\u0644\u0641\u0647\u0645 \u0648\u0627\u0644\u062a\u062c\u0631\u0628\u0629."
    end
    return ""
end

function _simple_question_template(prompt_tokens::Vector{String})
    return ""
    keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    if ("\u0645\u0627" in keys || "\u0645\u0627\u0630\u0627" in keys) &&
       ("\u0627\u0633\u0645\u0643" in keys || "\u0627\u0633\u0645" in keys)
        return "\u0627\u0633\u0645\u064a \u0645\u0631\u0646\u0627\u0646\u060c \u0646\u0645\u0648\u0630\u062c \u0644\u063a\u0648\u064a \u0641\u064a\u0632\u064a\u0627\u0626\u064a \u064a\u062d\u0627\u0648\u0644 \u0641\u0647\u0645 \u0627\u0644\u0643\u0644\u0645\u0627\u062a \u0628\u0627\u0644\u0631\u0646\u064a\u0646 \u0648\u0627\u0644\u0639\u0644\u0627\u0642\u0627\u062a."
    end
    if ("\u0645\u0627" in keys || "\u0645\u0627\u0630\u0627" in keys) &&
       ("\u0647\u0648" in keys || "\u0647\u064a" in keys || "\u062a\u0639\u0631\u064a\u0641" in keys)
        content = String[]
        for t in prompt_tokens
            ks = _generation_keys(t)
            _any_key_in(ks, GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
            ("\u0647\u0648" in ks || "\u0647\u064a" in ks || "\u062a\u0639\u0631\u064a\u0641" in ks) && continue
            push!(content, t)
        end
        topic = isempty(content) ? "\u0630\u0644\u0643" : join(content, " ")
        return "$(topic) \u0645\u0641\u0647\u0648\u0645 \u064a\u0641\u0647\u0645 \u0645\u0646 \u0635\u0641\u0627\u062a\u0647 \u0648\u0639\u0644\u0627\u0642\u0627\u062a\u0647 \u0648\u0627\u0644\u0633\u064a\u0627\u0642 \u0627\u0644\u0630\u064a \u064a\u0631\u062f \u0641\u064a\u0647."
    end
    if ("\u0645\u0627" in keys || "\u0645\u0627\u0630\u0627" in keys) &&
       ("\u0645\u0639\u0646\u0649" in keys || "\u0645\u0639\u0646\u064a" in keys || "\u064a\u0639\u0646\u064a" in keys)
        if "\u062d\u0643\u0645\u0629" in keys || "\u062d\u0643\u0645\u0647" in keys
            return "\u0627\u0644\u062d\u0643\u0645\u0629 \u0641\u0647\u0645 \u0631\u0627\u0634\u062f \u064a\u062c\u0645\u0639 \u0628\u064a\u0646 \u0627\u0644\u0645\u0639\u0631\u0641\u0629 \u0648\u062d\u0633\u0646 \u0627\u0644\u062a\u0635\u0631\u0641."
        end
        content = String[]
        for t in prompt_tokens
            ks = _generation_keys(t)
            _any_key_in(ks, GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
            ("\u0645\u0639\u0646\u0649" in ks || "\u0645\u0639\u0646\u064a" in ks || "\u064a\u0639\u0646\u064a" in ks) && continue
            push!(content, t)
        end
        topic = isempty(content) ? "\u0630\u0644\u0643" : join(content, " ")
        return "$(topic) \u0645\u0639\u0646\u0649 \u064a\u0641\u0647\u0645 \u0645\u0646 \u0633\u064a\u0627\u0642\u0647 \u0648\u0639\u0644\u0627\u0642\u062a\u0647 \u0628\u0645\u0627 \u062d\u0648\u0644\u0647."
    end
    if "\u0643\u064a\u0641" in keys &&
       ("\u064a\u062a\u0639\u0644\u0645" in keys || "\u062a\u0639\u0644\u0645" in keys || "\u064a\u0639\u0644\u0645" in keys) &&
       ("\u0627\u0646\u0633\u0627\u0646" in keys || "\u0625\u0646\u0633\u0627\u0646" in keys)
        return "\u064a\u062a\u0639\u0644\u0645 \u0627\u0644\u0625\u0646\u0633\u0627\u0646 \u0628\u0627\u0644\u0645\u0644\u0627\u062d\u0638\u0629 \u0648\u0627\u0644\u062a\u062c\u0631\u0628\u0629 \u0648\u0627\u0644\u062a\u0643\u0631\u0627\u0631\u060c \u062b\u0645 \u064a\u0631\u0627\u062c\u0639 \u0641\u0647\u0645\u0647 \u062d\u062a\u0649 \u064a\u062b\u0628\u062a \u0627\u0644\u0645\u0639\u0646\u0649."
    end
    if "\u0643\u064a\u0641" in keys
        content = String[]
        for t in prompt_tokens
            ks = _generation_keys(t)
            _any_key_in(ks, GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _EXTRA_GENERATION_STOPWORDS) && continue
            _any_key_in(ks, _QUESTION_TOOL_KEYS) && continue
            ("\u064a\u0643\u0648\u0646" in ks || "\u062a\u0643\u0648\u0646" in ks || "\u0643\u0648\u0646" in ks) && continue
            push!(content, t)
        end
        if !isempty(content)
            topic = join(content, " ")
            return "\u064a\u0643\u0648\u0646 $(topic) \u062d\u064a\u0646 \u062a\u062a\u062d\u0642\u0642 \u0635\u0641\u0627\u062a\u0647 \u0641\u064a \u0627\u0644\u0648\u0627\u0642\u0639\u060c \u0648\u062a\u0646\u0633\u062c\u0645 \u0623\u0641\u0639\u0627\u0644\u0647 \u0645\u0639 \u0645\u0639\u0646\u0627\u0647 \u062f\u0648\u0646 \u062a\u0646\u0627\u0642\u0636."
        end
        return "\u064a\u062a\u0645 \u0630\u0644\u0643 \u0628\u062e\u0637\u0648\u0627\u062a \u0645\u062a\u062f\u0631\u062c\u0629: \u0645\u0644\u0627\u062d\u0638\u0629\u060c \u062b\u0645 \u062a\u062c\u0631\u0628\u0629\u060c \u062b\u0645 \u0645\u0631\u0627\u062c\u0639\u0629."
    end
    if "\u0644\u0645\u0627\u0630\u0627" in keys
        return "\u064a\u0631\u062c\u0639 \u0630\u0644\u0643 \u0625\u0644\u0649 \u0633\u0628\u0628 \u064a\u062a\u0635\u0644 \u0628\u0627\u0644\u0633\u064a\u0627\u0642 \u0648\u0628\u0627\u0644\u0639\u0644\u0627\u0642\u0629 \u0628\u064a\u0646 \u0627\u0644\u0623\u0634\u064a\u0627\u0621."
    end
    return ""
end

function _needs_simple_declarative_template(prompt_tokens::Vector{String}, text::AbstractString)
    isempty(prompt_tokens) && return false
    has_template = !isempty(strip(_simple_text_template(prompt_tokens)))
    _is_question_prompt_safe(join(prompt_tokens, " ")) && return false
    output_tokens = String[strip(w) for w in split(String(text)) if !isempty(strip(w))]
    isempty(output_tokens) && return true

    prompt_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    output_keys = reduce(union, (_generation_keys(t) for t in output_tokens); init=Set{String}())
    anchor_hit = !isempty(intersect(prompt_keys, output_keys))
    content_prompt_keys = setdiff(prompt_keys, union(GENERATION_STOPWORDS, _EXTRA_GENERATION_STOPWORDS, _QUESTION_TOOL_KEYS))
    content_overlap = intersect(content_prompt_keys, output_keys)
    weak_template_anchor = has_template && length(content_prompt_keys) >= 2 &&
                           length(content_overlap) < min(2, length(content_prompt_keys))
    question_leak = any(w -> _any_key_in(_generation_keys(w), _QUESTION_TOOL_KEYS), output_tokens)
    metadata_leak = occursin("\u062a\u0635\u0646\u064a\u0641:", String(text)) ||
                    occursin("category:", lowercase(String(text)))
    long_list = length(output_tokens) >= 4 && !any(p -> occursin(p, String(text)), [".", "،", "؛", ":"])
    return question_leak || !anchor_hit || weak_template_anchor || metadata_leak || long_list
end

function _needs_simple_question_template(prompt_tokens::Vector{String}, text::AbstractString)
    isempty(prompt_tokens) && return false
    _is_question_prompt_safe(join(prompt_tokens, " ")) || return false
    if occursin("\u0641\u0647\u0645 \u0627\u0644\u0633\u064a\u0627\u0642", String(text)) ||
       occursin("\u062a\u0646\u0638\u064a\u0645 \u0627\u0644\u062e\u0637\u0648\u0627\u062a", String(text)) ||
       occursin("\u0645\u0631\u0627\u062c\u0639\u0629 \u0627\u0644\u0646\u062a\u064a\u062c\u0629", String(text))
        return true
    end
    output_tokens = String[strip(w) for w in split(String(text)) if !isempty(strip(w))]
    isempty(output_tokens) && return true

    prompt_keys = reduce(union, (_generation_keys(t) for t in prompt_tokens); init=Set{String}())
    output_keys = reduce(union, (_generation_keys(t) for t in output_tokens); init=Set{String}())
    content_prompt_keys = setdiff(prompt_keys, union(GENERATION_STOPWORDS, _EXTRA_GENERATION_STOPWORDS, _QUESTION_TOOL_KEYS))
    anchor_hit = !isempty(intersect(content_prompt_keys, output_keys))
    meaning_query = ("\u0645\u0639\u0646\u0649" in prompt_keys ||
                     "\u0645\u0639\u0646\u064a" in prompt_keys ||
                     "\u064a\u0639\u0646\u064a" in prompt_keys)
    definition_query = meaning_query ||
                       (("\u0645\u0627" in prompt_keys || "\u0645\u0627\u0630\u0627" in prompt_keys) &&
                        ("\u0647\u0648" in prompt_keys || "\u0647\u064a" in prompt_keys || "\u062a\u0639\u0631\u064a\u0641" in prompt_keys))
    if definition_query
        internal_markers = (
            "\u0641\u064a \u0641\u0636\u0627\u0621 \u0627\u0644\u0639\u0642\u0644",
            "\u0639\u0644\u0627\u0642\u0627\u062a\u0647:",
            "\u0633\u0645\u0627\u062a\u0647:",
            "__classes",
            "compound_name",
            "possessor",
            "\u0641\u064a\u0632\u064a\u0627\u0621 \u0627\u0644\u062d\u0631\u0641",
            "\u0623\u0642\u0631\u0628 \u0643\u0644\u0645\u0627\u062a",
            "\u0627\u0642\u0631\u0628 \u0643\u0644\u0645\u0627\u062a",
            "\u0637\u064a\u0641\u064a",
            "\u0627\u0644\u0643\u062a\u0644\u0629",
            "\u0627\u0644\u062a\u0646\u0627\u063a\u0645",
        )
        any(m -> occursin(m, String(text)), internal_markers) && return true
    end
    long_list = length(output_tokens) >= 4 && !any(p -> occursin(p, String(text)), [".", "،", "؛", ":"])
    return !anchor_hit || long_list
end

function _hisban_prompt_guidance(gen::MirnanGenerator, prompt_tokens::Vector{String})
    prompt_text = join(prompt_tokens, " ")
    
    guidance = if has_semantic_calculus(gen.hisban)
        g = semantic_guidance(gen.hisban, prompt_text; limit=8)
        relation = String(get(g, "relation", ""))
        if relation == "semantic_continuation" && !_is_question_prompt_safe(prompt_text)
            Dict{String,Any}(
                "active" => false,
                "movement" => "none",
                "confidence" => 0.0,
                "target_terms" => String[],
            )
        else
            g
        end
    else
        Dict{String,Any}(
            "active" => false,
            "movement" => "none",
            "confidence" => 0.0,
            "target_terms" => String[],
        )
    end
    
    # Integrate Clifford QA Projector Layer (Semantic Calculus V2)
    if gen.qa_projector.counts["general"] > 0
        corpus_ids = _load_corpus_sentences_for_generator(gen)
        if !isempty(corpus_ids)
            corpus_sentences = String[]
            for ids in corpus_ids[1:min(length(corpus_ids), 3000)]
                words = [get(gen.id2word, Int(id), "") for id in ids]
                push!(corpus_sentences, join(filter(!isempty, words), " "))
            end
            
            retrieved = retrieve_answer_facts(gen.qa_projector, prompt_text, corpus_sentences; limit=4)
            
            target_terms = get(guidance, "target_terms", String[])
            for (score, sentence) in retrieved
                score < 0.15 && continue
                for w in split(sentence)
                    w_clean = _strip_punct_boundary(String(w))
                    if length(w_clean) >= 2 && !(w_clean in target_terms)
                        push!(target_terms, w_clean)
                    end
                end
            end
            guidance["target_terms"] = target_terms
            guidance["active"] = true
            guidance["confidence"] = max(get(guidance, "confidence", 0.0), 0.75)
        end
    end
    
    return guidance
end

function _lisan_pattern_is_structured(rec)
    roles = rec.roles
    length(roles) >= 2 || return false
    length(Set(roles)) >= 2 || return false
    any(r -> r in ("verb", "be_verb", "auxiliary", "participle", "adjective",
                   "object", "prep_object", "noun", "preposition", "determiner",
                   "marker", "condition_tool", "question_tool", "relative_pronoun"),
        roles) || return false
    return true
end

function _lisan_generate(gen::MirnanGenerator, prompt_tokens::Vector{String})
    has_lisan_patterns(gen.lisan) || return String[]
    rec = select_lisan_pattern(gen.lisan, prompt_tokens; prefer_verbal=false)
    rec === nothing && return String[]
    _lisan_pattern_is_structured(rec) || return String[]
    prompt_is_question = _is_question_prompt_safe(join(prompt_tokens, " "))
    if !prompt_is_question && any(r -> r == "question_tool", rec.roles[1:min(2, length(rec.roles))])
        return String[]
    end
    return rec.roles
end

function _sanitize_generation_output(prompt_tokens::Vector{String}, text::AbstractString)
    clean = String[]
    seen = Set{String}()
    for raw in split(String(text))
        w = strip(raw)
        isempty(w) && continue
        _is_generation_candidate(w, prompt_tokens) || continue
        _reject_context_drift(w, prompt_tokens, clean) && continue
        fam = _generation_family_key(w)
        fam in seen && continue
        push!(clean, w)
        push!(seen, fam)
    end
    return join(clean, " ")
end

function _finalize_dialogue_charge_output(text::AbstractString, profile)
    s = strip(String(text))
    isempty(s) && return s
    raw_words = String[strip(w, [' ', '\t', '\n', '\r', '.', '،', ',', '؛', ';', ':', '؟', '?', '!'])
                       for w in split(s)]
    words = String[]
    seen = Set{String}()
    order = Dict{String,Int}()
    if has_gravity_profile(profile)
        for (i, term) in enumerate(profile.guidance_terms)
            order[term] = min(get(order, term, i), i)
        end
    end
    for w in raw_words
        isempty(w) && continue
        key = _generation_family_key(w)
        key in seen && continue
        push!(words, w)
        push!(seen, key)
    end
    if !isempty(order)
        sort!(words; by=w -> get(order, w, 10_000))
    end
    length(words) > 6 && (s = join(words[1:6], " "))
    length(words) <= 6 && (s = join(words, " "))
    endswith(s, ".") || endswith(s, "؟") || endswith(s, "?") || (s *= ".")
    return s
end

function _arabic_dialogue_word_key(word::AbstractString)
    s = lowercase(strip(String(word), [' ', '\t', '\n', '\r', '.', ',', ';', ':', '?', '!', '\u061F', '\u060C', '\u061B', '"', '\'']))
    s = replace(s, "\u0623" => "\u0627", "\u0625" => "\u0627", "\u0622" => "\u0627",
                   "\u0649" => "\u064A", "\u0629" => "\u0647")
    return s
end

function _drop_arabic_article(word::AbstractString)
    s = _arabic_dialogue_word_key(word)
    startswith(s, "\u0627\u0644") && length(s) > 2 && return s[nextind(s, firstindex(s), 2):end]
    return s
end

function _repair_dialogue_yesno_word_order(prompt::AbstractString, text::AbstractString)
    p = strip(String(prompt))
    t = strip(String(text))
    if isempty(p) || isempty(t)
        return t
    end

    p_words = String[strip(w, [' ', '\t', '\n', '\r', '.', ',', ';', ':', '?', '!', '\u061F', '\u060C', '\u061B'])
                     for w in split(p)]
    isempty(p_words) && return t
    _arabic_dialogue_word_key(first(p_words)) == "\u0647\u0644" || return t

    p_keys = _arabic_dialogue_word_key.(p_words)
    love_prompt_keys = Set(["\u062A\u062D\u0628", "\u062A\u062D\u0628\u064A\u0646", "\u064A\u062D\u0628"])
    love_idx = findfirst(k -> k in love_prompt_keys, p_keys)
    love_idx === nothing && return t
    love_idx >= length(p_words) && return t

    target_words = String[]
    skip_keys = Set(["\u0627\u0646", "\u0623\u0646"])
    for w in p_words[(love_idx + 1):end]
        key = _arabic_dialogue_word_key(w)
        isempty(key) && continue
        key in skip_keys && continue
        push!(target_words, strip(w))
    end
    isempty(target_words) && return t
    length(target_words) > 3 && (target_words = target_words[1:3])
    target = join(target_words, " ")

    t_words = split(t)
    t_keys = _arabic_dialogue_word_key.(t_words)
    has_yes = any(k -> k == "\u0646\u0639\u0645", t_keys)
    has_love = any(k -> k in Set(["\u0627\u062D\u0628", "\u0627\u062D\u0628\u0628", "\u0627\u062D\u0628\u0647"]), t_keys)
    has_yes && has_love || return t

    target_keys = Set(_drop_arabic_article.(target_words))
    has_target = any(k -> _drop_arabic_article(k) in target_keys, t_keys)
    has_target || return t

    has_open = any(k -> k in Set(["\u064A\u0641\u062A\u062D", "\u064A\u0641\u062A\u062D\u0647", "\u064A\u0641\u062A\u062D\u0648"]), t_keys)
    has_understanding = any(k -> _drop_arabic_article(k) == "\u0641\u0647\u0645", t_keys)
    reason = (has_open && has_understanding) ? " \u0644\u0623\u0646\u0647 \u064A\u0641\u062A\u062D \u0627\u0644\u0641\u0647\u0645" : ""
    return "\u0646\u0639\u0645\u060C \u0623\u062D\u0628 $(target)$(reason)."
end

function _explicit_causal_prompt(prompt::AbstractString)
    p = lowercase(String(prompt))
    return occursin("لماذا", p) || occursin("لما ", p) ||
           occursin("سبب", p) || occursin("يسبب", p) ||
           occursin("بسبب", p) || occursin("نتيجة", p) ||
           occursin("إذا", p) || occursin("اذا", p) ||
           occursin("لأن", p) || occursin("لان", p) ||
           occursin("because", p) || occursin("why", p)
end

function _intent_related_terms(gen::MirnanGenerator, prompt_tokens::Vector{String}; limit::Int=4)
    gen.K_sem === nothing && return String[]
    prompt_set = Set(prompt_tokens)
    ids = Int[]
    for tok in prompt_tokens
        wid = get(gen.vocab, tok, 0)
        wid > 0 && wid <= size(gen.K_sem, 1) && push!(ids, wid)
    end
    isempty(ids) && return String[]
    scores = Dict{Int,Float64}()
    for cid in ids
        row = gen.K_sem[cid, :]
        for (idx, val) in zip(row.nzind, row.nzval)
            val <= 0.0 && continue
            w = get(gen.id2word, idx, "")
            isempty(w) && continue
            w in prompt_set && continue
            _is_generation_candidate(w, prompt_tokens) || continue
            scores[idx] = max(get(scores, idx, 0.0), Float64(val))
        end
    end
    ranked = sort(collect(scores); by=x -> -x[2])
    terms = String[]
    for (idx, _) in ranked
        w = get(gen.id2word, idx, "")
        isempty(w) && continue
        push!(terms, w)
        length(terms) >= limit && break
    end
    return terms
end

function _generate_intent_planned_response(gen::MirnanGenerator,
                                           prompt::String,
                                           prompt_tokens::Vector{String};
                                           only_intents=nothing)
    plan = detect_response_intent(prompt)
    if only_intents !== nothing && !(plan.intent in only_intents)
        return ""
    end
    has_plannable_response(plan) || return ""
    related = _intent_related_terms(gen, prompt_tokens; limit=4)
    answer = render_planned_response(plan; related_terms=related)
    return strip(answer)
end

function _intent_gravity_profile(prompt::String)
    return intent_gravity_profile(detect_response_intent(prompt))
end

function _load_corpus_sentences_for_generator(gen::MirnanGenerator)
    return get!(_CORPUS_SENTENCE_CACHE, gen) do
        path = joinpath(gen.model_dir, "corpus_sentences.dat")
        isfile(path) || return Vector{Int32}[]
        sentences = Vector{Int32}[]
        try
            open(path, "r") do io
                n = Int(read(io, Int32))
                for _ in 1:n
                    len = Int(read(io, Int32))
                    ids = read!(io, Vector{Int32}(undef, len))
                    3 <= length(ids) <= 32 && push!(sentences, ids)
                end
            end
        catch e
            @debug "Corpus sentence memory load failed: $e"
            return Vector{Int32}[]
        end
        return sentences
    end
end

function _load_raw_training_sentences_for_generator(gen::MirnanGenerator)
    return get!(_RAW_TRAINING_SENTENCE_CACHE, gen) do
        data_dir = joinpath(dirname(gen.model_dir), "data")
        isdir(data_dir) || return String[]
        sentences = String[]
        for (root, _, files) in walkdir(data_dir)
            occursin("data_quarantine", root) && continue
            for file in files
                endswith(lowercase(file), ".txt") || continue
                file in ("FORMAT_RULES.txt", "nouns.txt", "verbs.txt") && continue
                path = joinpath(root, file)
                text = try
                    read(path, String)
                catch
                    ""
                end
                isempty(text) && continue
                text = replace(text, "\ufeff" => "", "\r\n" => "\n")
                for m in eachmatch(r"[^\.!\?؟:]+[\.!\?؟:]", text)
                    s = strip(String(m.match))
                    8 <= length(s) <= 240 || continue
                    startswith(s, "#") && continue
                    s == "---" && continue
                    push!(sentences, s)
                end
            end
        end
        return sentences
    end
end

function _strip_dialogue_label(line::AbstractString)
    s = strip(String(line))
    s = replace(s, r"^\s*(?:سؤال|س)\s*:\s*" => "")
    s = replace(s, r"^\s*(?:جواب|ج)\s*:\s*" => "")
    return strip(s)
end

function _load_raw_dialogue_pairs_for_generator(gen::MirnanGenerator)
    return get!(_RAW_DIALOGUE_PAIR_CACHE, gen) do
        data_dir = joinpath(dirname(gen.model_dir), "data")
        isdir(data_dir) || return Tuple{String,String}[]
        pairs = Tuple{String,String}[]
        pending = ""
        for (root, _, files) in walkdir(data_dir)
            occursin("data_quarantine", root) && continue
            for file in files
                endswith(lowercase(file), ".txt") || continue
                file in ("FORMAT_RULES.txt", "nouns.txt", "verbs.txt") && continue
                path = joinpath(root, file)
                lines = try
                    readlines(path)
                catch
                    String[]
                end
                for raw in lines
                    line = strip(replace(raw, "\ufeff" => ""))
                    isempty(line) && continue
                    if startswith(line, "سؤال:") || startswith(line, "س:")
                        pending = _strip_dialogue_label(line)
                    elseif (startswith(line, "جواب:") || startswith(line, "ج:")) && !isempty(pending)
                        answer = _strip_dialogue_label(line)
                        !isempty(answer) && push!(pairs, (pending, answer))
                        pending = ""
                    elseif occursin('\t', line)
                        parts = split(line, '\t')
                        if length(parts) >= 2
                            q = strip(parts[1])
                            a = strip(parts[2])
                            (!isempty(q) && !isempty(a)) && push!(pairs, (q, a))
                        end
                    end
                end
            end
        end
        return pairs
    end
end

function _sentence_words(gen::MirnanGenerator, ids::Vector{Int32})
    words = String[]
    for id in ids
        w = get(gen.id2word, Int(id), "")
        isempty(w) && continue
        push!(words, w)
    end
    return words
end

function _difference_prompt(prompt::AbstractString)
    s = lowercase(String(prompt))
    return occursin("الفرق", s) || occursin("يفرق", s) ||
           occursin("difference", s)
end

function _relation_or_difference_prompt(prompt::AbstractString)
    return _relationship_prompt(prompt) || _difference_prompt(prompt)
end

function _list_like_generation_output(text::AbstractString)
    t = strip(String(text))
    isempty(t) && return false
    words = split(t)
    length(words) >= 5 || return false
    any(p -> occursin(p, t), [".", "،", "؛", ":", "؟", "?", "!", "\n"]) && return false
    return true
end

function _unique_guidance_terms(words::Vector{String})
    out = String[]
    seen = Set{String}()
    for w in words
        s = strip(w)
        isempty(s) && continue
        s in seen && continue
        push!(out, s)
        push!(seen, s)
    end
    return out
end

function _guided_vocab_ids(gen::MirnanGenerator, term::AbstractString; limit::Int=12)
    term_key = String(term)
    by_term = get!(_GUIDED_VOCAB_ID_CACHE, gen) do
        Dict{String,Vector{Int}}()
    end
    if haskey(by_term, term_key)
        return by_term[term_key]
    end
    ids = Int[]
    variants = _unique_guidance_terms(String[term_key])
    for key in _generation_keys(term_key)
        push!(variants, key)
        startswith(key, "ال") && length(key) > 2 && push!(variants, replace(key, r"^ال" => ""))
        !startswith(key, "ال") && push!(variants, "ال" * key)
    end
    for item in _unique_guidance_terms(variants)
        wid = get(gen.vocab, item, 0)
        wid > 0 && !(wid in ids) && push!(ids, wid)
        length(ids) >= limit && break
    end
    by_term[term_key] = ids
    return ids
end

function _boost_guided_terms!(excited::AbstractVector, gen::MirnanGenerator,
                              used::Set{String}, scale::Float64)
    V = length(excited)
    max_exc = maximum(excited)
    max_exc < 1e-10 && return
    for (term, bias) in gen.aql_bias
        for wid_bias in _guided_vocab_ids(gen, term)
            (wid_bias <= 0 || wid_bias > V) && continue
            word = get(gen.id2word, wid_bias, "")
            word in used && continue
            excited[wid_bias] += max_exc * max(0.0, bias) * scale
        end
    end
end

function _with_intent_gravity(gen::MirnanGenerator, profile, f::Function)
    has_gravity_profile(profile) || return f()
    old_weights = copy(gen.scoring_weights)
    old_ksem = copy(gen.k_sem_config)
    old_bias = copy(gen.aql_bias)
    old_inhibition = copy(gen.aql_inhibition)
    try
        if profile.intent == "dialogue"
            empty!(gen.aql_bias)
            empty!(gen.aql_inhibition)
        end
        gen.scoring_weights["syntax"] = get(gen.scoring_weights, "syntax", 0.0) *
                                        profile.syntax_multiplier
        gen.scoring_weights["k_sem"] = get(gen.scoring_weights, "k_sem", 0.0) *
                                      profile.semantic_multiplier
        gen.scoring_weights["causal_flow_align"] = get(gen.scoring_weights, "causal_flow_align", 0.0) *
                                                   profile.causal_multiplier
        if profile.intent == "dialogue"
            gen.scoring_weights["intent_guidance_boost"] = 7.0 * max(0.1, profile.response_charge)
            gen.scoring_weights["dialogue_question_charge"] = profile.question_charge
            gen.scoring_weights["soft_intent_guidance_boost"] = 0.0
        else
            gen.scoring_weights["intent_guidance_boost"] = 0.0
            gen.scoring_weights["dialogue_question_charge"] = 0.0
            gen.scoring_weights["soft_intent_guidance_boost"] = profile.intent == "causal" ? 8.0 : 2.4
        end
        for (i, term) in enumerate(profile.guidance_terms)
            amount = profile.intent == "dialogue" ? max(0.35, 1.0 - 0.08 * (i - 1)) :
                     max(0.18, 0.55 - 0.04 * (i - 1))
            _add_aql_bias!(gen.aql_bias, term, amount)
        end
        for term in profile.repulsion_terms
            _add_aql_bias!(gen.aql_inhibition, term,
                           profile.intent == "dialogue" ? min(0.9, abs(profile.question_charge)) : 0.35)
        end
        return f()
    finally
        gen.scoring_weights = old_weights
        gen.k_sem_config = old_ksem
        gen.aql_bias = old_bias
        gen.aql_inhibition = old_inhibition
    end
end

function _semantic_attention_enabled()
    _env_on("MIRNAN_SEMANTIC_ATTENTION", "1")
end

function _semantic_attention_field(gen::MirnanGenerator, prompt_tokens::Vector{String})
    _semantic_attention_enabled() || return nothing
    try
        return build_semantic_attention(gen.vocab, gen.id2word, gen.K_sem, gen.K_causal,
                                        prompt_tokens;
                                        max_terms=_env_int("MIRNAN_SEMANTIC_ATTENTION_TERMS", 18))
    catch e
        @debug "semantic attention failed" exception=(typeof(e), catch_backtrace())
        return nothing
    end
end

function _with_semantic_attention(gen::MirnanGenerator, field, f::Function)
    field === nothing && return f()
    has_semantic_attention(field) || return f()
    old_bias = copy(gen.aql_bias)
    old_inhibition = copy(gen.aql_inhibition)
    old_weights = copy(gen.scoring_weights)
    try
        strength = _env_float("MIRNAN_SEMANTIC_ATTENTION_STRENGTH", 1.0)
        for (term, amount) in field.positive_bias
            _add_aql_bias!(gen.aql_bias, term, clamp(amount * strength, 0.0, 1.0))
        end
        for (term, amount) in field.negative_bias
            _add_aql_bias!(gen.aql_inhibition, term, clamp(amount * 0.5 * strength, 0.0, 0.7))
        end
        gen.scoring_weights["soft_intent_guidance_boost"] =
            max(get(gen.scoring_weights, "soft_intent_guidance_boost", 0.0), 3.0 * strength)
        gen.scoring_weights["aql_guidance"] =
            max(get(gen.scoring_weights, "aql_guidance", 0.0), 4.0 * strength)
        return f()
    finally
        gen.aql_bias = old_bias
        gen.aql_inhibition = old_inhibition
        gen.scoring_weights = old_weights
    end
end

function _clean_aql_text(text::AbstractString)
    return strip(replace(String(text), r"[؟?!.]+$" => ""))
end

function _is_question_prompt(text::AbstractString)
    s = strip(String(text))
    isempty(s) && return false
    starts = ("ما ", "ماذا", "من ", "أين", "اين", "متى", "كيف", "لماذا", "هل",
              "اشرح", "عرّف", "عرف", "قارن",
              "what", "where", "when", "why", "how", "explain", "define", "compare")
    lower_s = lowercase(s)
    return occursin("؟", s) || occursin("?", s) || any(p -> startswith(lower_s, p), starts)
end

function _is_question_prompt_safe(text::AbstractString)
    s = strip(String(text))
    isempty(s) && return false
    lower_s = lowercase(s)
    starts = (
        "\u0645\u0627 ", "\u0645\u0627\u0630\u0627", "\u0645\u0646 ",
        "\u0623\u064a\u0646", "\u0627\u064a\u0646", "\u0645\u062a\u0649",
        "\u0643\u064a\u0641", "\u0644\u0645\u0627\u0630\u0627", "\u0647\u0644",
        "\u0627\u0634\u0631\u062d", "\u0639\u0631\u0651\u0641", "\u0639\u0631\u0641",
        "\u0642\u0627\u0631\u0646", "what", "where", "when", "why", "how",
        "explain", "define", "compare",
    )
    return occursin("\u061F", s) || occursin("?", s) ||
           _is_question_prompt(text) ||
           any(p -> startswith(lower_s, p), starts)
end

function _looks_like_aql_adl(text::AbstractString)
    s = strip(String(text))
    markers = (
        "شيء ", "فعل ", "قاعدة", "فكرة", "نمط ", "عملية ", "سلسلة ",
        "علاقة ", "كم ", "مقارنة ", "مجاز ", "نية ", "استثناء ",
        "ظرف ", "ترتيب ", "مفتاح ", "تصنيف ", "ضد ",
        "thing ", "verb ", "rule", "idea", "template ", "process ",
        "chain ", "relation ", "quantifier ", "comparison ", "metaphor ",
        "intent ", "exception ", "circumstance ", "order ", "class ",
        "classify ", "opposite ", "curated ", "curated {", "constitutional ",
        "proposed ", "proposal ", "approve ", "reject ", "lesson ",
        "rejection_lesson ", "annotation ", "corpus_annotation ", "critical_corpus ",
        "دستور ", "قاعدة_موثوقة ", "اقتراح ", "قاعدة_مرشحة ",
        "اعتماد ", "رفض ", "درس_رفض ", "وسم_ذاكرة ",
    )
    return any(m -> startswith(s, m), markers) || occursin("=>", s) || occursin("->", s)
end

function _ingest_aql!(gen::MirnanGenerator, prompt::AbstractString)
    space = gen.aql_space
    before = length(space.log)
    before_rels = length(space.semantic_relations)
    try
        if _looks_like_aql_adl(prompt)
            AlAql.compile_adl!(space, prompt)
        else
            AlAql.train_from_text!(space, prompt)
        end
    catch e
        @debug "AQL ingestion failed: $e"
    end
    
    # Dynamic Hebbian updates to K_sem
    if length(space.semantic_relations) > before_rels
        new_rels = space.semantic_relations[before_rels+1:end]
        if gen.K_sem !== nothing
            for rel in new_rels
                phrase = "$(rel.source) $(rel.relation) $(rel.target)"
                words = _clean_prompt_terms(phrase)
                for i in 1:length(words)-1
                    a = get(gen.vocab, words[i], 0)
                    b = get(gen.vocab, words[i+1], 0)
                    if a > 0 && b > 0 && a <= size(gen.K_sem, 1) && b <= size(gen.K_sem, 2)
                        gen.K_sem[a, b] = min(gen.K_sem[a, b] + 5.0, 100.0)
                    end
                end
            end
        end
    end
    
    return length(space.log) > before
end

struct AqlGenerationPlan
    kind::String
    source::String
    action::String
    target::String
    terms::Vector{String}
    confidence::Float64
end

function AqlGenerationPlan(kind::AbstractString; source::AbstractString="",
                           action::AbstractString="", target::AbstractString="",
                           terms::Vector{String}=String[], confidence::Real=0.0)
    return AqlGenerationPlan(String(kind), String(source), String(action),
                             String(target), terms, Float64(confidence))
end

function _clean_prompt_terms(text::AbstractString)
    cleaned = replace(_clean_aql_text(text), r"[،,؛;:()\[\]{}\"']" => " ")
    cleaned = replace(cleaned, "_" => " ")
    return String[strip(w) for w in split(cleaned) if !isempty(strip(w))]
end

function _contains_aql_term(text::AbstractString, term::AbstractString)
    value = strip(String(term))
    isempty(value) && return false
    s = String(text)
    occursin(value, s) && return true
    value_terms = _clean_prompt_terms(value)
    isempty(value_terms) && return false
    prompt_terms = Set(_clean_prompt_terms(s))
    return any(t -> t in prompt_terms, value_terms)
end

function _add_aql_bias!(bias::Dict{String,Float64}, text::AbstractString, amount::Real)
    value = strip(String(text))
    isempty(value) && return bias
    score = clamp(Float64(amount), 0.0, 1.0)
    bias[value] = max(get(bias, value, 0.0), score)
    for term in _clean_prompt_terms(value)
        length(term) >= 2 || continue
        bias[term] = max(get(bias, term, 0.0), score * 0.85)
    end
    return bias
end

function _aql_fact_matches_prompt(prompt::AbstractString, values::AbstractString...)
    return any(v -> _contains_aql_term(prompt, v), values)
end

function _aql_bias_from_prompt(gen::MirnanGenerator, prompt::AbstractString)
    space = gen.aql_space
    bias = Dict{String,Float64}()
    text = _clean_aql_text(prompt)

    for rel in space.semantic_relations
        if _aql_fact_matches_prompt(text, rel.source, rel.relation, rel.target)
            _add_aql_bias!(bias, rel.source, 0.80)
            _add_aql_bias!(bias, rel.relation, 0.72)
            _add_aql_bias!(bias, rel.target, 0.80)
        end
    end

    for template in space.templates
        if _aql_fact_matches_prompt(text, template.name, template.action,
                                    template.source_class, template.target_class,
                                    template.result_action, template.result_state)
            _add_aql_bias!(bias, template.action, 0.75)
            _add_aql_bias!(bias, template.result_action, 0.95)
            _add_aql_bias!(bias, template.result_state, 0.95)
            _add_aql_bias!(bias, template.result_kind, 0.70)
        end
    end

    for chain in values(space.event_chains)
        if _aql_fact_matches_prompt(text, chain.name, chain.action,
                                    chain.source_class, chain.target_class)
            _add_aql_bias!(bias, chain.action, 0.75)
            for step in chain.steps
                _add_aql_bias!(bias, step.action, 0.90)
                _add_aql_bias!(bias, step.result, 0.95)
            end
        end
    end

    for process in values(space.processes)
        if _contains_aql_term(text, process.name)
            _add_aql_bias!(bias, process.name, 0.95)
            for step in process.steps
                _add_aql_bias!(bias, step.action, 0.90)
                _add_aql_bias!(bias, step.result, 0.95)
            end
            for (key, value) in process.attributes
                _add_aql_bias!(bias, key, 0.65)
                _add_aql_bias!(bias, string(value), 0.55)
            end
        end
    end

    for fact in space.metaphors
        if _aql_fact_matches_prompt(text, fact.expression, fact.literal_subject,
                                    fact.borrowed_actor, fact.action)
            _add_aql_bias!(bias, fact.literal_subject, 0.90)
            _add_aql_bias!(bias, fact.borrowed_actor, 0.75)
            _add_aql_bias!(bias, fact.action, 0.85)
            _add_aql_bias!(bias, fact.transferred_property, 0.95)
        end
    end

    for fact in space.intents
        if _aql_fact_matches_prompt(text, fact.actor, fact.action, fact.target,
                                    fact.intent, fact.goal)
            _add_aql_bias!(bias, fact.actor, 0.80)
            _add_aql_bias!(bias, fact.action, 0.75)
            _add_aql_bias!(bias, fact.target, 0.70)
            _add_aql_bias!(bias, fact.intent, 0.95)
            _add_aql_bias!(bias, fact.goal, 0.95)
            _add_aql_bias!(bias, fact.actual_result, 0.85)
        end
    end

    for rule in space.exception_rules
        if _aql_fact_matches_prompt(text, rule.rule_name, rule.condition, rule.exception)
            _add_aql_bias!(bias, rule.rule_name, 0.75)
            _add_aql_bias!(bias, rule.condition, 0.70)
            _add_aql_bias!(bias, rule.exception, 0.95)
        end
    end

    for fact in space.quantified_facts
        if _aql_fact_matches_prompt(text, fact.quantifier, fact.subject,
                                    fact.predicate, fact.object)
            _add_aql_bias!(bias, fact.subject, 0.80)
            _add_aql_bias!(bias, fact.predicate, 0.85)
            _add_aql_bias!(bias, fact.object, fact.polarity < 0 ? 0.55 : 0.85)
        end
    end

    for fact in space.comparisons
        if _aql_fact_matches_prompt(text, fact.left, fact.property,
                                    fact.comparator, fact.right)
            _add_aql_bias!(bias, fact.left, 0.80)
            _add_aql_bias!(bias, fact.property, 0.75)
            _add_aql_bias!(bias, fact.comparator, 0.90)
            _add_aql_bias!(bias, fact.right, 0.80)
        end
    end

    for name in keys(space.entities)
        if _contains_aql_term(text, name)
            _add_aql_bias!(bias, name, 0.70)
            try
                for cls in AlAql.assigned_classes(space, name)
                    _add_aql_bias!(bias, cls, 0.75)
                end
            catch e
                @debug "AQL class bias failed for '$name': $e"
            end
        end
    end

    return bias
end

function _aql_inhibition_from_prompt(gen::MirnanGenerator, prompt::AbstractString)
    space = gen.aql_space
    inhibition = Dict{String,Float64}()
    text = _clean_aql_text(prompt)

    for rule in values(space.curated_rules)
        if rule.polarity < 0 && _aql_fact_matches_prompt(text, rule.subject,
                                                         rule.predicate, rule.object,
                                                         rule.condition)
            _add_aql_bias!(inhibition, rule.object, 1.0)
            _add_aql_bias!(inhibition, rule.predicate, 0.90)
            for tag in rule.tags
                _add_aql_bias!(inhibition, tag, 0.70)
            end
        end
    end

    for rule in values(space.rejected_rules)
        if _aql_fact_matches_prompt(text, rule.subject, rule.predicate,
                                    rule.object, rule.condition)
            _add_aql_bias!(inhibition, rule.object, 0.95)
            _add_aql_bias!(inhibition, rule.predicate, 0.85)
            for tag in rule.tags
                _add_aql_bias!(inhibition, tag, 0.75)
            end
        end
    end

    for lesson in space.rejection_lessons
        if !isempty(lesson.pattern) && _contains_aql_term(text, lesson.pattern)
            for tag in lesson.penalty_tags
                _add_aql_bias!(inhibition, tag, min(0.95, 0.45 + 0.45 * lesson.confidence))
            end
        end
    end

    for rule in space.exception_rules
        if _aql_fact_matches_prompt(text, rule.rule_name, rule.condition, rule.exception)
            _add_aql_bias!(inhibition, rule.rule_name, 0.35)
            _add_aql_bias!(inhibition, rule.condition, 0.25)
        end
    end

    return inhibition
end

function _human_aql_text(value::AbstractString)
    return replace(strip(String(value)), "_" => " ")
end

function _format_aql_frames(frames)
    isempty(frames) && return ""
    parts = String[]
    for frame in frames[1:min(6, end)]
        things = isempty(frame.things) ? "غير محدد" : join(frame.things, " و ")
        result = _human_aql_text(replace(frame.result, " | " => "، "))
        push!(parts, "الأشياء: $(things)؛ الحدث: $(frame.event)؛ النتيجة: $(result)")
    end
    return join(parts, "\n")
end

function _extract_event_plan(prompt::AbstractString)
    text = _clean_aql_text(prompt)
    text = replace(text, r"^ماذا\s+سيحدث\s+إذا\s+" => "")
    text = replace(text, r"^ماذا\s+يحدث\s+إذا\s+" => "")
    text = replace(text, r"^ماذا\s+ينتج\s+إذا\s+" => "")
    text = replace(text, r"^ما\s+نتيجة\s+" => "")
    text = replace(text, r"^إذا\s+" => "")
    text = replace(text, r"^لو\s+" => "")
    text = replace(text, r"(?i)^what\s+happens\s+if\s+" => "")
    text = replace(text, r"(?i)^if\s+" => "")
    text = replace(text, r"\s+(?:فإنها|فإنه|فانه|فإن|فـ)\s+.*$" => "")
    words = _clean_prompt_terms(text)
    length(words) >= 2 || return nothing

    preps = Set(["على", "إلى", "الى", "في", "مع", "ب", "عند"])
    prep_idx = findfirst(w -> w in preps, words)
    source = ""
    action = ""
    target = ""
    nominal_actions = Set(["إضافة", "اضافة", "تقريب", "قرب", "مزج", "خلط"])

    if prep_idx !== nothing && prep_idx > 1 && prep_idx < length(words)
        target = String(words[prep_idx + 1])
        if words[1] in nominal_actions
            action = String(words[1])
            source = String(words[prep_idx - 1])
        elseif prep_idx >= 3
            source = String(words[prep_idx - 2])
            action = String(words[prep_idx - 1])
        else
            source = String(words[1])
            action = String(words[2])
        end
    elseif length(words) >= 3
        source = String(words[1])
        action = String(words[2])
        target = String(words[3])
    elseif length(words) == 2
        source = String(words[1])
        action = String(words[2])
        target = source
    end

    isempty(action) && return nothing
    terms = filter(!isempty, [source, action, target])
    return AqlGenerationPlan("event"; source=source, action=action,
                             target=target, terms=terms, confidence=0.55)
end

const _AQL_SPEECH_GENERIC_KEYS = Set([
    "\u0645\u0627", "\u0645\u0627\u0630\u0627", "\u0645\u0646", "\u0627\u064A\u0646", "\u0623\u064A\u0646",
    "\u0645\u062A\u0649", "\u0643\u064A\u0641", "\u0644\u0645\u0627\u0630\u0627", "\u0647\u0644",
    "\u0647\u0648", "\u0647\u064A", "\u0647\u0630\u0627", "\u0647\u0630\u0647", "\u0630\u0644\u0643",
    "\u0641\u064A", "\u0628", "\u0628\u0640", "\u0639\u0646", "\u0639\u0644\u0649", "\u0627\u0644\u0649", "\u0625\u0644\u0649",
    "\u0631\u0627\u064A\u0643", "\u0631\u0623\u064A\u0643", "\u0642\u0648\u0644\u0643", "\u0645\u0639\u0646\u0649",
    "\u0645\u0641\u064A\u062F", "\u0645\u0641\u064A\u062F\u0647", "\u0645\u0641\u064A\u062F\u0629", "\u0646\u0627\u0641\u0639", "\u0646\u0627\u0641\u0639\u0647", "\u0646\u0627\u0641\u0639\u0629",
    "\u064A\u0639\u0645\u0644", "\u062A\u0639\u0645\u0644", "\u064A\u0643\u0648\u0646", "\u062A\u0643\u0648\u0646",
    "\u064A\u062A\u0639\u0644\u0645", "\u0646\u062A\u0639\u0644\u0645", "\u0646\u0646\u0627\u0645", "\u0646\u062D\u062A\u0627\u062C", "\u0646\u0628\u062D\u062B",
    "what", "where", "when", "why", "how", "is", "are", "the", "a", "an", "of", "in", "about",
])

const _SENSE_QUERY_CUES = Set(["معنى", "كلمة", "فسر", "اشرح", "دلالة", "يقصد", "تعني", "يعني"])

function _cerebellum_k_density(gen::MirnanGenerator)
    gen.K_sem === nothing && return 0.0
    total = Float64(max(length(gen.vocab)^2, 1))
    return nnz(gen.K_sem) / total
end

function _with_cerebellum_policy(gen::MirnanGenerator,
                                 policy::CerebellumPolicy,
                                 f::Function)
    old_weights = copy(gen.scoring_weights)
    try
        apply_cerebellum_policy!(gen.scoring_weights, policy)
        return f()
    finally
        gen.scoring_weights = old_weights
    end
end

function _non_template_repair_answer(gen::MirnanGenerator,
                                     prompt::String,
                                     prompt_tokens::Vector{String},
                                     review)
    issues = Set(String.(getfield(review, :issues)))
    isempty(intersect(issues, Set(["list_like_output", "missing_prompt_anchor",
                                   "weak_prompt_anchor", "weak_prompt_alignment"]))) && return ""
    intent = detect_response_intent(prompt).intent
    mem = _corpus_memory_answer(gen, prompt, prompt_tokens; intent=intent)
    !isempty(strip(mem)) && return mem
    conservative = _conservative_anchor_answer(prompt, prompt_tokens)
    !isempty(strip(conservative)) && return conservative
    return ""
end

function _content_keys_from_tokens(tokens::Vector{String})
    keys = reduce(union, (_generation_keys(t) for t in tokens); init=Set{String}())
    out = setdiff(keys, union(GENERATION_STOPWORDS, _EXTRA_GENERATION_STOPWORDS, _QUESTION_TOOL_KEYS))
    filter!(k -> length(k) >= 3, out)
    return out
end

function _semantic_anchor_weak(prompt_tokens::Vector{String}, text::AbstractString, intent::String)
    intent in ("causal", "mechanism") || return false
    content = _content_keys_from_tokens(prompt_tokens)
    length(content) >= 2 || return false
    out = reduce(union, (_generation_keys(t) for t in split(String(text))); init=Set{String}())
    return length(intersect(content, out)) < min(2, length(content))
end

function _repair_multipliers(target::String)
    target == "logic" && return Dict(
        "aql_guidance" => 1.60,
        "aql_inhibition" => 1.35,
        "causal_flow_align" => 1.45,
        "prompt_align" => 1.15,
    )
    target == "diversity" && return Dict(
        "diversity" => 1.55,
        "repulsion" => 1.70,
        "prompt_align" => 1.08,
    )
    target == "prompt_alignment" && return Dict(
        "prompt_align" => 1.65,
        "k_sem" => 1.15,
        "diversity" => 0.88,
    )
    target == "coherence" && return Dict(
        "syntax" => 1.20,
        "context_tension" => 1.15,
    )
    target == "syntax" && return Dict(
        "syntax" => 1.60,
        "root_affinity" => 1.10,
        "diversity" => 0.90,
    )
    target == "language" && return Dict(
        "syntax" => 1.35,
        "prompt_align" => 1.10,
    )
    target == "fallback" && return Dict(
        "root_affinity" => 1.25,
        "prompt_align" => 1.15,
    )
    target == "standard_mode" && return Dict(
        "prompt_align" => 1.15,
        "diversity" => 1.10,
        "repulsion" => 1.10,
    )
    target == "semantic_guidance" && return Dict(
        "prompt_align" => 1.35,
        "k_sem" => 1.15,
        "syntax" => 1.10,
    )
    return Dict{String,Float64}()
end

function _repair_policy(base::CerebellumPolicy, target::String)
    multipliers = copy(base.weight_multipliers)
    for (name, mult) in _repair_multipliers(target)
        multipliers[name] = get(multipliers, name, 1.0) * mult
    end
    mode = target == "standard_mode" ? "standard" : base.mode
    return CerebellumPolicy(mode, base.sense_mode, multipliers,
                            base.confidence,
                            string(base.reason, "+review_", target))
end

function _preemptive_review_policy(gen::MirnanGenerator,
                                   policy::CerebellumPolicy,
                                   prompt::String,
                                   prompt_tokens::Vector{String})
    prediction = predict_review_repair(gen.self_review, prompt_tokens; prompt=prompt)
    treatment = predict_review_treatment(gen.self_review, prompt_tokens;
                                         prompt=prompt,
                                         default_target=prediction.repair_target)
    target = treatment.confidence >= 0.12 && treatment.repair_target != "none" ?
             treatment.repair_target : prediction.repair_target
    confidence = max(prediction.confidence, treatment.confidence)
    (target == "none" || confidence < 0.18) && return policy
    repaired = _repair_policy(policy, target)
    return CerebellumPolicy(
        repaired.mode,
        repaired.sense_mode,
        repaired.weight_multipliers,
        clamp(max(policy.confidence, confidence), 0.05, 0.98),
        string(repaired.reason, "+memory_", target),
    )
end

function _contextual_learning_dict(state::ContextualLearning.ContextualLearningState)
    compounds = [Dict{String,Any}(
        "text" => c.text,
        "tokens" => c.tokens,
        "kind" => c.kind,
        "strength" => c.strength,
        "observations" => c.observations,
    ) for c in values(state.compounds.entries)]

    metaphors = [Dict{String,Any}(
        "compound" => m.compound,
        "interpretation" => m.interpretation,
        "actions" => m.actions,
        "effects" => m.effects,
        "strength" => m.strength,
    ) for m in values(state.metaphors.entries)]

    return Dict{String,Any}(
        "compounds" => compounds,
        "metaphors" => metaphors,
        "effects" => state.effects.effects,
        "entity_kinds" => state.entity_kinds,
        "feedback_log" => state.feedback_log,
    )
end

function _restore_contextual_learning!(state::ContextualLearning.ContextualLearningState, data)
    data isa AbstractDict || return state

    for item in get(data, "compounds", Any[])
        item isa AbstractDict || continue
        tokens = String[string(t) for t in get(item, "tokens", String[])]
        length(tokens) >= 2 || continue
        text = join(tokens, " ")
        vector = ContextualLearning.compound_vector(tokens)
        state.compounds.entries[text] = ContextualLearning.CompoundExpression(
            text,
            tokens,
            string(get(item, "kind", "ordinary")),
            vector,
            Float64(get(item, "strength", 0.5)),
            Int(get(item, "observations", 1)),
        )
    end

    for item in get(data, "metaphors", Any[])
        item isa AbstractDict || continue
        key = join(ContextualLearning._tokenize(string(get(item, "compound", ""))), " ")
        isempty(key) && continue
        state.metaphors.entries[key] = ContextualLearning.MetaphorEntry(
            key,
            string(get(item, "interpretation", "")),
            String[string(x) for x in get(item, "actions", String[])],
            String[string(x) for x in get(item, "effects", String[])],
            Float64(get(item, "strength", 0.5)),
        )
    end

    effects = get(data, "effects", nothing)
    if effects isa AbstractDict
        state.effects.effects = Dict{String,Dict{String,Float64}}(
            string(k) => Dict{String,Float64}(string(ek) => Float64(ev) for (ek, ev) in v)
            for (k, v) in effects if v isa AbstractDict
        )
    end

    entity_kinds = get(data, "entity_kinds", nothing)
    if entity_kinds isa AbstractDict
        for (k, v) in entity_kinds
            state.entity_kinds[string(k)] = string(v)
        end
    end

    feedback = get(data, "feedback_log", nothing)
    if feedback isa AbstractVector
        state.feedback_log = Dict{String,Any}[
            Dict{String,Any}(string(k) => v for (k, v) in item)
            for item in feedback if item isa AbstractDict
        ]
    end
    return state
end

function _reinforcement_dict(gen::MirnanGenerator)
    return Dict{String,Any}(
        "strengths" => gen.reinforcement.strengths,
        "traces" => gen.reinforcement.traces,
    )
end

function _restore_reinforcement!(gen::MirnanGenerator, data)
    data isa AbstractDict || return gen
    strengths = get(data, "strengths", nothing)
    if strengths isa AbstractDict
        gen.reinforcement.strengths = Dict{String,Float64}(string(k) => Float64(v) for (k, v) in strengths)
    end
    traces = get(data, "traces", nothing)
    if traces isa AbstractDict
        gen.reinforcement.traces = Dict{String,Vector{Float64}}(
            string(k) => Float64.(v) for (k, v) in traces if v isa AbstractVector
        )
    end
    return gen
end

function _aql_runtime_dict(space)
    entities = [Dict{String,Any}(
        "name" => name,
        "kind" => entity isa AlAql.Thing ? entity.kind : string(nameof(typeof(entity))),
        "mass" => hasproperty(entity, :mass) ? Float64(getfield(entity, :mass)) : 1.0,
        "attributes" => entity isa AlAql.Thing ? entity.attributes : Dict{String,Any}(),
        "classes" => try
            AlAql.assigned_classes(space, name)
        catch
            String[]
        end,
        "properties" => hasproperty(entity, :properties) ?
            [[abs(v), angle(v)] for v in getfield(entity, :properties)] :
            Any[],
    ) for (name, entity) in space.entities]

    relations = [Dict{String,Any}(
        "source" => r.source,
        "relation" => r.relation,
        "target" => r.target,
        "confidence" => r.confidence,
        "evidence" => r.evidence,
    ) for r in space.semantic_relations]

    templates = [Dict{String,Any}(
        "name" => t.name,
        "domain" => t.domain,
        "source_class" => t.source_class,
        "action" => t.action,
        "target_class" => t.target_class,
        "action_kind" => t.action_kind,
        "result_actor" => t.result_actor,
        "result_action" => t.result_action,
        "result_target" => t.result_target,
        "result_kind" => t.result_kind,
        "result_state" => t.result_state,
        "condition" => t.condition,
        "confidence" => t.confidence,
        "polarity" => t.polarity,
    ) for t in space.templates]

    processes = [Dict{String,Any}(
        "name" => p.name,
        "parent" => p.parent,
        "domain" => p.domain,
        "intensity" => p.intensity,
        "steps" => [Dict{String,Any}(
            "actor_role" => s.actor_role,
            "action" => s.action,
            "target_role" => s.target_role,
            "result" => s.result,
        ) for s in p.steps],
        "attributes" => p.attributes,
    ) for p in values(space.processes)]

    event_chains = [Dict{String,Any}(
        "name" => c.name,
        "source_class" => c.source_class,
        "action" => c.action,
        "target_class" => c.target_class,
        "confidence" => c.confidence,
        "steps" => [Dict{String,Any}(
            "actor_role" => s.actor_role,
            "action" => s.action,
            "target_role" => s.target_role,
            "result" => s.result,
        ) for s in c.steps],
    ) for c in values(space.event_chains)]

    quantified = [Dict{String,Any}(
        "quantifier" => f.quantifier,
        "subject" => f.subject,
        "predicate" => f.predicate,
        "object" => f.object,
        "polarity" => f.polarity,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.quantified_facts]

    comparisons = [Dict{String,Any}(
        "left" => f.left,
        "property" => f.property,
        "comparator" => f.comparator,
        "right" => f.right,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.comparisons]

    metaphors = [Dict{String,Any}(
        "expression" => f.expression,
        "source_domain" => f.source_domain,
        "target_domain" => f.target_domain,
        "literal_subject" => f.literal_subject,
        "borrowed_actor" => f.borrowed_actor,
        "action" => f.action,
        "transferred_property" => f.transferred_property,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.metaphors]

    intents = [Dict{String,Any}(
        "actor" => f.actor,
        "action" => f.action,
        "target" => f.target,
        "intent" => f.intent,
        "goal" => f.goal,
        "actual_result" => f.actual_result,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.intents]

    speech_acts = [Dict{String,Any}(
        "speaker" => f.speaker,
        "act_type" => f.act_type,
        "content" => f.content,
        "responder" => f.responder,
        "response_act" => f.response_act,
        "response_content" => f.response_content,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.speech_acts]

    exceptions = [Dict{String,Any}(
        "rule_name" => f.rule_name,
        "condition" => f.condition,
        "exception" => f.exception,
        "priority" => f.priority,
        "confidence" => f.confidence,
        "evidence" => f.evidence,
    ) for f in space.exception_rules]

    circumstances = Dict{String,Any}(
        name => Dict("location" => c.location, "time" => c.time)
        for (name, c) in space.circumstances
    )

    governance_rule_dict(rule) = Dict{String,Any}(
        "id" => rule.id,
        "subject" => rule.subject,
        "predicate" => rule.predicate,
        "object" => rule.object,
        "condition" => rule.condition,
        "polarity" => rule.polarity,
        "priority" => rule.priority,
        "confidence" => rule.confidence,
        "source" => rule.source,
        "status" => rule.status,
        "rationale" => rule.rationale,
        "tags" => rule.tags,
    )

    lessons = [Dict{String,Any}(
        "id" => lesson.id,
        "reason" => lesson.reason,
        "pattern" => lesson.pattern,
        "penalty_tags" => lesson.penalty_tags,
        "confidence" => lesson.confidence,
        "evidence" => lesson.evidence,
    ) for lesson in space.rejection_lessons]

    annotations = [Dict{String,Any}(
        "sentence_id" => item.sentence_id,
        "span" => item.span,
        "tag" => item.tag,
        "confidence" => item.confidence,
        "reason" => item.reason,
        "effect_on_k_sem" => item.effect_on_k_sem,
        "effect_on_k_causal" => item.effect_on_k_causal,
        "effect_on_al_aql" => item.effect_on_al_aql,
    ) for item in space.corpus_annotations]

    return Dict{String,Any}(
        "entities" => entities,
        "relations" => relations,
        "templates" => templates,
        "processes" => processes,
        "event_chains" => event_chains,
        "quantified_facts" => quantified,
        "comparisons" => comparisons,
        "metaphors" => metaphors,
        "intents" => intents,
        "speech_acts" => speech_acts,
        "exception_rules" => exceptions,
        "opposites" => space.opposites,
        "circumstances" => circumstances,
        "curated_rules" => [governance_rule_dict(rule) for rule in values(space.curated_rules)],
        "proposed_rules" => [governance_rule_dict(rule) for rule in values(space.proposed_rules)],
        "rejected_rules" => [governance_rule_dict(rule) for rule in values(space.rejected_rules)],
        "rejection_lessons" => lessons,
        "corpus_annotations" => annotations,
    )
end

function _restore_aql_runtime!(space, data)
    data isa AbstractDict || return space
    empty!(space.entities)
    empty!(space.circumstances)
    empty!(space.semantic_relations)
    empty!(space.templates)
    empty!(space.processes)
    empty!(space.event_chains)
    empty!(space.quantified_facts)
    empty!(space.comparisons)
    empty!(space.metaphors)
    empty!(space.intents)
    empty!(space.speech_acts)
    empty!(space.exception_rules)
    empty!(space.opposites)
    empty!(space.curated_rules)
    empty!(space.proposed_rules)
    empty!(space.rejected_rules)
    empty!(space.rejection_lessons)
    empty!(space.corpus_annotations)

    for item in get(data, "entities", Any[])
        item isa AbstractDict || continue
        name = string(get(item, "name", ""))
        isempty(name) && continue
        attrs_raw = get(item, "attributes", Dict{String,Any}())
        attrs = attrs_raw isa AbstractDict ?
            Dict{String,Any}(string(k) => v for (k, v) in attrs_raw) :
            Dict{String,Any}()
        entity = AlAql.Thing(name;
            kind=string(get(item, "kind", "thing")),
            mass=Float64(get(item, "mass", 1.0)),
            attributes=attrs,
        )
        props = get(item, "properties", Any[])
        if props isa AbstractVector
            for (prop, idx) in AlAql.PROPERTY_MAP
                idx <= length(props) || continue
                pair = props[idx]
                pair isa AbstractVector && length(pair) >= 2 || continue
                AlAql.set_property!(entity, prop, Float64(pair[1]), Float64(pair[2]))
            end
        end
        classes = String[string(c) for c in get(item, "classes", String[])]
        AlAql.register_entity!(space, entity; classes=classes)
    end

    for item in get(data, "relations", Any[])
        item isa AbstractDict || continue
        AlAql.assert_relation!(
            space,
            string(get(item, "source", "")),
            string(get(item, "relation", "")),
            string(get(item, "target", ""));
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "templates", Any[])
        item isa AbstractDict || continue
        AlAql.register_template!(space, AlAql.CausalTemplate(
            string(get(item, "name", "")),
            string(get(item, "domain", "")),
            string(get(item, "source_class", "")),
            string(get(item, "action", "")),
            string(get(item, "target_class", "")),
            string(get(item, "action_kind", "")),
            string(get(item, "result_actor", "")),
            string(get(item, "result_action", "")),
            string(get(item, "result_target", "")),
            string(get(item, "result_kind", "")),
            string(get(item, "result_state", "")),
            string(get(item, "condition", "")),
            Float64(get(item, "confidence", 1.0)),
            Int(get(item, "polarity", 1)),
        ))
    end

    for item in get(data, "processes", Any[])
        item isa AbstractDict || continue
        steps = AlAql.ProcessStep[]
        for raw_step in get(item, "steps", Any[])
            raw_step isa AbstractDict || continue
            push!(steps, AlAql.ProcessStep(
                string(get(raw_step, "actor_role", "source")),
                string(get(raw_step, "action", "حدث")),
                string(get(raw_step, "target_role", "target")),
                string(get(raw_step, "result", "")),
            ))
        end
        attrs_raw = get(item, "attributes", Dict{String,Any}())
        attrs = attrs_raw isa AbstractDict ?
            Dict{String,Any}(string(k) => v for (k, v) in attrs_raw) :
            Dict{String,Any}()
        AlAql.register_process!(space, AlAql.ProcessConcept(
            string(get(item, "name", "")),
            string(get(item, "parent", "")),
            string(get(item, "domain", "")),
            Float64(get(item, "intensity", 1.0)),
            steps,
            attrs,
        ))
    end

    for item in get(data, "event_chains", Any[])
        item isa AbstractDict || continue
        steps = AlAql.ChainStep[]
        for raw_step in get(item, "steps", Any[])
            raw_step isa AbstractDict || continue
            push!(steps, AlAql.ChainStep(
                string(get(raw_step, "actor_role", "source")),
                string(get(raw_step, "action", "حدث")),
                string(get(raw_step, "target_role", "target")),
                string(get(raw_step, "result", "")),
            ))
        end
        AlAql.register_event_chain!(space, AlAql.EventChain(
            string(get(item, "name", "")),
            string(get(item, "source_class", "")),
            string(get(item, "action", "")),
            string(get(item, "target_class", "")),
            steps,
            Float64(get(item, "confidence", 1.0)),
        ))
    end

    for item in get(data, "quantified_facts", Any[])
        item isa AbstractDict || continue
        AlAql.assert_quantified_fact!(
            space,
            string(get(item, "quantifier", "")),
            string(get(item, "subject", "")),
            string(get(item, "predicate", "")),
            string(get(item, "object", ""));
            polarity=Int(get(item, "polarity", 1)),
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "comparisons", Any[])
        item isa AbstractDict || continue
        AlAql.assert_comparison!(
            space,
            string(get(item, "left", "")),
            string(get(item, "property", "")),
            string(get(item, "comparator", "")),
            string(get(item, "right", ""));
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "metaphors", Any[])
        item isa AbstractDict || continue
        AlAql.assert_metaphor!(
            space,
            string(get(item, "expression", ""));
            source_domain=string(get(item, "source_domain", "")),
            target_domain=string(get(item, "target_domain", "")),
            literal_subject=string(get(item, "literal_subject", "")),
            borrowed_actor=string(get(item, "borrowed_actor", "")),
            action=string(get(item, "action", "")),
            transferred_property=string(get(item, "transferred_property", "")),
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "intents", Any[])
        item isa AbstractDict || continue
        AlAql.assert_intent!(
            space,
            string(get(item, "actor", "")),
            string(get(item, "action", ""));
            target=string(get(item, "target", "")),
            intent=string(get(item, "intent", "")),
            goal=string(get(item, "goal", "")),
            actual_result=string(get(item, "actual_result", "")),
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "speech_acts", Any[])
        item isa AbstractDict || continue
        AlAql.assert_speech_act!(
            space,
            string(get(item, "speaker", "")),
            string(get(item, "act_type", "")),
            string(get(item, "content", ""));
            responder=string(get(item, "responder", "")),
            response_act=string(get(item, "response_act", "")),
            response_content=string(get(item, "response_content", "")),
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    for item in get(data, "exception_rules", Any[])
        item isa AbstractDict || continue
        AlAql.add_exception_rule!(
            space,
            string(get(item, "rule_name", "")),
            string(get(item, "condition", "")),
            string(get(item, "exception", ""));
            priority=Int(get(item, "priority", 1)),
            confidence=Float64(get(item, "confidence", 1.0)),
            evidence=string(get(item, "evidence", "runtime_learning")),
        )
    end

    opposites = get(data, "opposites", nothing)
    if opposites isa AbstractDict
        for (left, right) in opposites
            AlAql.register_opposite!(space, string(left), string(right))
        end
    end

    circumstances = get(data, "circumstances", nothing)
    if circumstances isa AbstractDict
        for (name, item) in circumstances
            item isa AbstractDict || continue
            AlAql.set_circumstance!(
                space,
                string(name);
                location=string(get(item, "location", "")),
                time=string(get(item, "time", "")),
            )
        end
    end

    function restore_governance_rule(item, status::String)
        item isa AbstractDict || return nothing
        id = string(get(item, "id", ""))
        isempty(id) && return nothing
        tags_raw = get(item, "tags", String[])
        tags = tags_raw isa AbstractVector ? String[string(t) for t in tags_raw] : String[]
        return AlAql.AqlGovernanceRule(
            id,
            string(get(item, "subject", "")),
            string(get(item, "predicate", "")),
            string(get(item, "object", "")),
            string(get(item, "condition", "")),
            Int(get(item, "polarity", 1)),
            Int(get(item, "priority", 0)),
            Float64(get(item, "confidence", 1.0)),
            string(get(item, "source", "runtime_learning")),
            string(get(item, "status", status)),
            string(get(item, "rationale", "")),
            tags,
        )
    end

    for item in get(data, "curated_rules", Any[])
        rule = restore_governance_rule(item, "curated")
        rule === nothing && continue
        space.curated_rules[rule.id] = rule
    end

    for item in get(data, "proposed_rules", Any[])
        rule = restore_governance_rule(item, "proposed")
        rule === nothing && continue
        space.proposed_rules[rule.id] = rule
    end

    for item in get(data, "rejected_rules", Any[])
        rule = restore_governance_rule(item, "rejected")
        rule === nothing && continue
        space.rejected_rules[rule.id] = rule
    end

    for item in get(data, "rejection_lessons", Any[])
        item isa AbstractDict || continue
        tags_raw = get(item, "penalty_tags", String[])
        tags = tags_raw isa AbstractVector ? String[string(t) for t in tags_raw] : String[]
        push!(space.rejection_lessons, AlAql.RejectionLesson(
            string(get(item, "id", "")),
            string(get(item, "reason", "")),
            string(get(item, "pattern", "")),
            tags,
            Float64(get(item, "confidence", 0.8)),
            string(get(item, "evidence", "runtime_learning")),
        ))
    end

    for item in get(data, "corpus_annotations", Any[])
        item isa AbstractDict || continue
        push!(space.corpus_annotations, AlAql.CorpusAnnotation(
            string(get(item, "sentence_id", "")),
            string(get(item, "span", "")),
            string(get(item, "tag", "")),
            Float64(get(item, "confidence", 1.0)),
            string(get(item, "reason", "runtime_learning")),
            Float64(get(item, "effect_on_k_sem", 1.0)),
            Float64(get(item, "effect_on_k_causal", 1.0)),
            string(get(item, "effect_on_al_aql", "tag")),
        ))
    end
    return space
end

function _save_sparse_dat(path::AbstractString, K::SparseMatrixCSC)
    open(path, "w") do io
        write(io, "SPARSE_CSC\n")
        write(io, Int32(size(K, 1)), Int32(size(K, 2)), Int32(length(K.nzval)))
        write(io, Int32.(K.colptr))
        write(io, Int32.(SparseArrays.rowvals(K)))
        write(io, Float64.(SparseArrays.nonzeros(K)))
    end
end

function _load_sparse_dat_gen(path::AbstractString, vocab_size::Int)
    isfile(path) || return nothing
    try
        open(path, "r") do io
            header = readline(io)
            header == "SPARSE_CSC" || return nothing
            m = Int(read(io, Int32))
            n = Int(read(io, Int32))
            nnz = Int(read(io, Int32))
            colptr = read!(io, Vector{Int32}(undef, n + 1))
            rows = read!(io, Vector{Int32}(undef, nnz))
            vals = read!(io, Vector{Float64}(undef, nnz))
            return SparseMatrixCSC{Float64,Int32}(SparseMatrixCSC(m, n, Vector{Int}(colptr), Vector{Int}(rows), vals))
        end
    catch e
        @warn "Failed to load sparse dat $path: $e"
        return nothing
    end
end

function save_runtime_learning!(gen::MirnanGenerator; dir::String=gen.runtime_learning_dir)
    mkpath(dir)
    data = Dict{String,Any}(
        "version" => 4,
        "contextual_learning" => _contextual_learning_dict(gen.contextual_learning),
        "reinforcement" => _reinforcement_dict(gen),
        "al_aql" => _aql_runtime_dict(gen.aql_space),
        "cerebellum" => cerebellum_state_dict(gen.cerebellum),
        "self_review" => self_review_state_dict(gen.self_review),
    )
    path = joinpath(dir, "runtime_learning.json")
    open(path, "w") do io
        JSON.print(io, data)
    end
    if gen.K_sem !== nothing
        try
            _save_sparse_dat(joinpath(dir, "K_sem_runtime.dat"), _apply_runtime_synaptic_decay(gen.K_sem))
        catch e
            @warn "Generator: failed to save K_sem_runtime.dat: $e"
        end
    end
    return path
end

function load_runtime_learning!(gen::MirnanGenerator; dir::String=gen.runtime_learning_dir)
    path = joinpath(dir, "runtime_learning.json")
    isfile(path) || return false
    try
        data = JSON.parsefile(path)
        _restore_contextual_learning!(gen.contextual_learning, get(data, "contextual_learning", nothing))
        _restore_reinforcement!(gen, get(data, "reinforcement", nothing))
        _restore_aql_runtime!(gen.aql_space, get(data, "al_aql", nothing))
        restore_cerebellum_state!(gen.cerebellum, get(data, "cerebellum", nothing))
        restore_self_review_state!(gen.self_review, get(data, "self_review", nothing))
        
        k_sem_path = joinpath(dir, "K_sem_runtime.dat")
        if isfile(k_sem_path) && gen.K_sem !== nothing
            K_loaded = _load_sparse_dat_gen(k_sem_path, size(gen.K_sem, 1))
            if K_loaded !== nothing
                gen.K_sem = K_loaded
            end
        end
        return true
    catch e
        @warn "Generator: failed to load runtime learning: $e"
        return false
    end
end

function _speech_act_duplicate(space, speaker::String, act_type::String, content::String,
                               responder::String, response_act::String, response_content::String)
    return any(f -> f.speaker == speaker &&
                    f.act_type == act_type &&
                    f.content == content &&
                    f.responder == responder &&
                    f.response_act == response_act &&
                    f.response_content == response_content,
               space.speech_acts)
end

function _load_persistent_dialogue_facts!(gen::MirnanGenerator, path::AbstractString)
    isfile(path) || return 0
    loaded = 0
    try
        data = JSON.parsefile(path)
        for item in get(data, "speech_acts", Any[])
            item isa AbstractDict || continue
            speaker = String(strip(string(get(item, "speaker", "user"))))
            act_type = String(strip(string(get(item, "act_type", "حوار"))))
            content = String(strip(string(get(item, "content", ""))))
            responder = String(strip(string(get(item, "responder", "mirnan"))))
            response_act = String(strip(string(get(item, "response_act", "رد"))))
            response_content = String(strip(string(get(item, "response_content", ""))))
            isempty(content) && continue
            isempty(response_content) && continue
            _speech_act_duplicate(gen.aql_space, speaker, act_type, content,
                                  responder, response_act, response_content) && continue
            AlAql.assert_speech_act!(
                gen.aql_space,
                speaker,
                act_type,
                content;
                responder=responder,
                response_act=response_act,
                response_content=response_content,
                confidence=Float64(get(item, "confidence", 0.95)),
                evidence=string(get(item, "evidence", "persistent_dialogue_facts")),
            )
            loaded += 1
        end
    catch e
        @warn "Generator: failed to load persistent dialogue facts: $e"
    end
    return loaded
end

function reset!(gen::MirnanGenerator)
    empty!(gen.pv_cache)
    empty!(gen.mass_cache)
    empty!(gen.aql_bias)
    empty!(gen.aql_inhibition)
    reset_cerebellum!(gen.cerebellum)
    reset_self_review!(gen.self_review)
    reset!(gen.prompt_field)
    gpu_release!(gen.gpu)
end

function set_k_sem_config!(gen::MirnanGenerator; strength::Union{Nothing,Float64}=nothing,
                           temperature::Union{Nothing,Float64}=nothing,
                           threshold::Union{Nothing,Float64}=nothing)
    strength !== nothing && (gen.k_sem_config["strength"] = clamp(strength, 0.0, 1.0))
    temperature !== nothing && (gen.k_sem_config["temperature"] = max(temperature, 0.01))
    threshold !== nothing && (gen.k_sem_config["threshold"] = clamp(threshold, 0.0, 1.0))
    return gen.k_sem_config
end

function _count_field(records, field::Symbol)
    counts = Dict{String,Int}()
    for rec in records
        key = String(getfield(rec, field))
        counts[key] = get(counts, key, 0) + 1
    end
    return counts
end

function _example_count(records)
    total = 0
    for rec in records
        total += length(rec.examples)
    end
    return total
end

function _slot_value_count(records)
    return 0
end

function _hisab_check_count(records)
    total = 0
    for rec in records
        total += length(rec.checks)
    end
    return total
end

function _semantic_target_term_count(records)
    total = 0
    for rec in records
        total += length(rec.target_terms)
    end
    return total
end

function _semantic_movement_counts(records)
    counts = Dict{String,Int}()
    for rec in records
        movement = semantic_relation_movement(rec.relation)
        counts[movement] = get(counts, movement, 0) + 1
    end
    return counts
end

function _aql_summary(space)
    audit = aql_memory_audit(space)
    return Dict{String,Any}(
        "entities" => length(space.entities),
        "dynamic_verbs" => length(space.dynamic_verbs),
        "rules" => length(space.rules),
        "templates" => length(space.templates),
        "processes" => length(space.processes),
        "semantic_relations" => length(space.semantic_relations),
        "temporal_relations" => length(space.temporal_relations),
        "event_chains" => length(space.event_chains),
        "quantified_facts" => length(space.quantified_facts),
        "comparisons" => length(space.comparisons),
        "metaphors" => length(space.metaphors),
        "intents" => length(space.intents),
        "speech_acts" => length(space.speech_acts),
        "exceptions" => length(space.exception_rules),
        "audit" => audit,
    )
end

function pattern_memory_summary(gen::MirnanGenerator)
    lisan_records = collect(values(gen.lisan.patterns))
    code_records = collect(values(gen.code_patterns.patterns))
    tadbir_records = collect(values(gen.tadbir.patterns))
    hisab_records = collect(values(gen.hisab.patterns))
    ta3rif_records = collect(values(gen.ta3rif.records))
    hisban_records = collect(values(gen.hisban.records))
    nisba_records = collect(values(gen.nisba.relations))
    istinbat_mem = _LEARNED_ISTINBAT_MEMORY[]
    istinbat_records = istinbat_mem === nothing ? IstinbatAttentionRecord[] : collect(values(istinbat_mem.records))

    return Dict{String,Any}(
        "al_lisan" => Dict{String,Any}(
            "patterns" => length(lisan_records),
            "languages" => _count_field(lisan_records, :language),
            "examples" => _example_count(lisan_records),
            "slot_values" => _slot_value_count(lisan_records),
        ),
        "al_code" => Dict{String,Any}(
            "patterns" => length(code_records),
            "languages" => _count_field(code_records, :language),
            "examples" => _example_count(code_records),
            "slot_values" => _slot_value_count(code_records),
        ),
        "al_tadbir" => Dict{String,Any}(
            "patterns" => length(tadbir_records),
            "domains" => _count_field(tadbir_records, :domain),
            "examples" => _example_count(tadbir_records),
            "slot_values" => _slot_value_count(tadbir_records),
        ),
        "al_hisab" => Dict{String,Any}(
            "patterns" => length(hisab_records),
            "problem_types" => _count_field(hisab_records, :problem_type),
            "examples" => _example_count(hisab_records),
            "checks" => _hisab_check_count(hisab_records),
        ),
        "al_ta3rif" => Dict{String,Any}(
            "subjects" => length(ta3rif_records),
            "definitions" => sum(length(rec.definitions) for rec in ta3rif_records),
            "relations" => sum(length(rec.relations) for rec in ta3rif_records),
            "attributes" => sum(length(rec.attributes) for rec in ta3rif_records),
            "examples" => _example_count(ta3rif_records),
        ),
        "al_hisban_al_dalali" => Dict{String,Any}(
            "records" => length(hisban_records),
            "relations" => _count_field(hisban_records, :relation),
            "movements" => _semantic_movement_counts(hisban_records),
            "examples" => _example_count(hisban_records),
            "target_terms" => _semantic_target_term_count(hisban_records),
        ),
        "al_nisba" => Dict{String,Any}(
            "records" => length(nisba_records),
            "relation_types" => _count_field(nisba_records, :relation_type),
            "evidences" => sum((length(rec.evidences) for rec in nisba_records); init=0),
        ),
        "al_istinbat" => Dict{String,Any}(
            "records" => length(istinbat_records),
            "relation_types" => _count_field(istinbat_records, :relation_type),
            "examples" => _example_count(istinbat_records),
        ),
        "al_aql" => _aql_summary(gen.aql_space),
    )
end

function get_physics_report(gen::MirnanGenerator, extra_words::Vector{String}=String[])
    masses = [_mass(gen, w) for w in extra_words if !isempty(strip(w))]
    qr = compute_quality_report(gen.K_sem, gen.density_matrix)
    return Dict{String,Any}(
        "vocab_size" => length(gen.vocab),
        "word_count" => length(extra_words),
        "mass_mean" => isempty(masses) ? 0.0 : sum(masses) / length(masses),
        "entropy" => qr.transition_entropy,
        "k_density" => qr.k_density,
        "k_causal_density" => gen.K_causal === nothing ? 0.0 : nnz(gen.K_causal) / Float64(max(length(gen.vocab)^2, 1)),
        "k_sparsity" => qr.k_sparsity,
        "k_effective_links" => qr.k_effective_connections,
        "transition_entropy" => qr.transition_entropy,
        "transition_entropy_std" => qr.transition_entropy_std,
        "density_purity" => qr.density_purity,
        "density_trace" => qr.density_trace,
        "density_spectral_entropy" => qr.density_spectral_entropy,
        "overall_quality_score" => qr.overall_score,
        "pattern_memories" => pattern_memory_summary(gen),
        "cerebellum" => policy_summary(gen.cerebellum),
        "self_review" => review_summary(gen.self_review),
    )
end

function gpu_init!(gen::MirnanGenerator)
    GpuAcceleratorModule.gpu_init!(
        gen.gpu,
        w -> _pv(gen, String(w)),
        gen.vocab,
        gen.scoring_weights;
        dim = PHASE_DIM,
    )
    return gen.gpu.active
end

const _disk_pv_data = Ref{Any}(nothing)
const _disk_pv_dim = Ref{Int}(0)
const _disk_mass_data = Ref{Any}(nothing)


# Include strategies under Generator namespace
include("strategies/base.jl")
include("strategies/shared_scoring.jl")
include("strategies/math_strategy.jl")
include("strategies/code_strategy.jl")
include("strategies/tadbir_strategy.jl")
include("strategies/dialogue_strategy.jl")
include("strategies/definition_strategy.jl")
include("strategies/relation_strategy.jl")
include("strategies/relation_frame_strategy.jl")
include("strategies/conditional_frame_strategy.jl")
include("strategies/temporal_frame_strategy.jl")
include("strategies/spatial_frame_strategy.jl")
include("strategies/state_frame_strategy.jl")
include("strategies/quantity_frame_strategy.jl")
include("strategies/semantic_scene_strategy.jl")
include("strategies/scene_purpose_strategy.jl")
include("strategies/resonant_strategy.jl")
include("strategies/prnn_strategy.jl")
include("strategies/lexical_strategy.jl")
include("strategies/response_polisher.jl")
include("strategies/finisher.jl")

function generate!(gen::MirnanGenerator, prompt::String;
                   mode::String="auto", max_words::Int=15)
    isempty(strip(prompt)) && return ""
    prompt_tokens = String[lowercase(strip(w)) for w in split(prompt) if length(strip(w)) >= 1]
    isempty(prompt_tokens) && return ""

    # Initialize Continuous Semantic Trajectory
    reset_tracker!(gen.trajectory_tracker)
    absorb_input!(gen.trajectory_tracker, prompt)

    # 1. تفريغ الاسترجاع السابق
    empty!(gen.retrieved_passages)
    empty!(gen.retrieved_pvs)
    empty!(gen.retrieved_similarities)
    
    # 2. استدعاء استرجاع المعرفة المدمجة للأسئلة المعرفية الطويلة
    kb_weight = get(gen.scoring_weights, "kb_knowledge", 0.0)
    if kb_weight > 0.0 && gen.rapg_kb !== nothing && length(prompt_tokens) > 3
        active_marker_type = ""
        mem = _LEARNED_ISTINBAT_MEMORY[]
        if mem !== nothing
            active_marker_type, _ = AlIstinbat._marker_hit(prompt, mem.discovered_markers)
        end
        ret_passages, ret_pvs, ret_sims = retrieve_by_category(gen.rapg_kb, prompt, active_marker_type, 5, w -> _pv(gen, w))
        for i in 1:length(ret_passages)
            push!(gen.retrieved_passages, ret_passages[i])
            push!(gen.retrieved_pvs, ret_pvs[i])
            push!(gen.retrieved_similarities, ret_sims[i])
        end
    end

    k_density_val = _cerebellum_k_density(gen)
    cereb_obs = observe_prompt(gen.cerebellum, prompt_tokens;

                               prompt=prompt,
                               vocab_size=length(gen.vocab),
                               k_density=k_density_val)
    cereb_policy = choose_policy!(gen.cerebellum, cereb_obs; requested_mode=mode)
    cereb_policy = _preemptive_review_policy(gen, cereb_policy, prompt, prompt_tokens)
    response_plan = detect_response_intent(prompt)
    active_paras = _get_active_paragraphs(gen, prompt_tokens)

    strategies = GenerationStrategy[
        RootLexicalStrategy(),
        CodeStrategy(),
        MathStrategy(),
        TadbirStrategy(),
        DialogueStrategy(),
        DefinitionStrategy(),
        AqlStrategy(),
        LexicalOracleStrategy(),
        RelationStrategy(),
        SenseSuperpositionStrategy(),
        PRNNStrategy(),
        ResonantStrategy()
    ]

    if _env_on("MIRNAN_ENABLE_RELATION_FRAME_STRATEGY", "0")
        push!(strategies, RelationFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_CONDITIONAL_FRAME_STRATEGY", "0")
        idx = findfirst(s -> s isa AqlStrategy || s isa RelationStrategy, strategies)
        insert!(strategies, idx === nothing ? length(strategies) + 1 : idx, ConditionalFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_TEMPORAL_FRAME_STRATEGY", "1")
        idx = findfirst(s -> s isa AqlStrategy || s isa RelationStrategy, strategies)
        insert!(strategies, idx === nothing ? length(strategies) + 1 : idx, TemporalFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_SPATIAL_FRAME_STRATEGY", "1")
        idx = findfirst(s -> s isa AqlStrategy || s isa RelationStrategy, strategies)
        insert!(strategies, idx === nothing ? length(strategies) + 1 : idx, SpatialFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_STATE_FRAME_STRATEGY", "0")
        idx = findfirst(s -> s isa AqlStrategy || s isa RelationStrategy, strategies)
        insert!(strategies, idx === nothing ? length(strategies) + 1 : idx, StateFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_QUANTITY_FRAME_STRATEGY", "0")
        # Quantity questions are tightly guarded in `quantity_answer`, so run this
        # before broad lexical strategies that can produce an empty terminal result.
        insert!(strategies, 1, QuantityFrameStrategy())
    end

    if _env_on("MIRNAN_ENABLE_SEMANTIC_SCENE_STRATEGY", "0")
        insert!(strategies, findfirst(s -> s isa CodeStrategy, strategies), SemanticSceneStrategy())
    end

    if _env_on("MIRNAN_ENABLE_SCENE_PURPOSE_STRATEGY", "0")
        idx = findfirst(s -> s isa SemanticSceneStrategy || s isa CodeStrategy, strategies)
        insert!(strategies, idx === nothing ? 1 : idx, ScenePurposeStrategy())
    end

    for strategy in strategies
        result = try_generate(strategy, gen, prompt, prompt_tokens, mode,
                              cereb_obs, cereb_policy, response_plan, active_paras, max_words)
        if result !== nothing
            return result
        end
    end

    return ""
end

end # module Generator
