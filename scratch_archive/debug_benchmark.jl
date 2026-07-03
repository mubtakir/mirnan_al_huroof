push!(LOAD_PATH, joinpath(@__DIR__, "..", "src"))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew.Physics.PRNNGenerator
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew.Physics.PRNNLearner
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew.Physics.PRNNCore
using JSON, SparseArrays, LinearAlgebra

const PROJECT_DIR = joinpath(@__DIR__, "..")
const MODEL_DIR = joinpath(PROJECT_DIR, "model")

vocab_file = joinpath(MODEL_DIR, "vocab.json")
vocab = Dict{String,Int}(k => Int(v) for (k,v) in JSON.parsefile(vocab_file))
id2word = Dict{Int,String}(v => k for (k,v) in vocab)

cs_path = joinpath(MODEL_DIR, "corpus_sentences.dat")
corpus_sentences = Vector{Int32}[]
open(cs_path, "r") do io
    n_sentences = read(io, Int32)
    for _ in 1:n_sentences
        len = read(io, Int32)
        push!(corpus_sentences, read!(io, Vector{Int32}(undef, len)))
    end
end

const N_SENTENCES = min(1000, length(corpus_sentences))
bench_sentences = corpus_sentences[1:N_SENTENCES]

# اختيار كلمة بذرة للتجربة
seed = "جذرها"
prompt_tokens = [seed]

println("--- Detailed Debugging PRNN Standalone ---")

session = PRNNSession(vocab, id2word, bench_sentences, prompt_tokens; N=10000, beta=3.0, max_sentences=100)
println("Active vocab size: ", length(session.active_vocab))
println("Transitions count: ", length(session.transitions))

bv = session.base_vectors
N  = session.N

function print_top_overlaps(z_state, v_ctx, label)
    z_u = unbind_phase(z_state, v_ctx)
    overlaps = []
    for w in session.active_vocab
        ov = real(dot(z_u, bv[w])) / N
        push!(overlaps, (w, ov))
    end
    sort!(overlaps, by=x->x[2], rev=true)
    println("      [$label] Top overlaps:")
    for idx in 1:min(3, length(overlaps))
        println("         - $(overlaps[idx][1]): $(overlaps[idx][2])")
    end
end

state = PRNNState(N)
model = LowRankPRNN(session.transitions, session.beta_coupling)
current_word = prompt_tokens[end]

if haskey(bv, current_word)
    state.z .= bv[current_word]
end

println("Start loop:")
used = Set{String}(prompt_tokens)

for step_idx in 1:3
    global current_word
    println("--- Step $step_idx ---")
    println("   Current word context: \"$current_word\"")
    
    print_top_overlaps(state.z, bv[current_word], "Before Simulation")
    
    # تفصيل محركات الاقتران النشطة
    global_activity = sum(abs2(zi) for zi in state.z) / N
    norm_z = sqrt(global_activity)
    coupling_matches = []
    for (v_curr, v_next, weight) in model.transitions
        overlap = dot(v_curr, state.z)
        similarity = abs(overlap) / (N * max(0.01, norm_z))
        if similarity > 0.15
            # إيجاد الكلمة المطابقة لـ v_curr
            matched_w = "unknown"
            for w in session.active_vocab
                if real(dot(v_curr, bv[w])) / N > 0.99
                    matched_w = w
                    break
                end
            end
            push!(coupling_matches, (matched_w, similarity))
        end
    end
    println("   Coupling matches during this step: ", coupling_matches)
    
    simulate_stuart_landau!(state, model;
        mu=1.0, g_inh=0.5, gamma=2.0, tau_a=1.5, steps=40, dt=0.02)
    
    print_top_overlaps(state.z, bv[current_word], "After Simulation")
    
    next_w, ov = decode(state.z, bv[current_word])
    println("   Decoded next_w = \"$next_w\" with overlap = $ov")

    if isempty(next_w)
        println("   Breaking: next_w is empty")
        break
    end
    if next_w in used
        println("   Breaking: next_w \"$next_w\" already used")
        break
    end
    if ov < 0.10
        println("   Breaking: overlap $ov < 0.10")
        break
    end

    push!(used, next_w)
    state.z .= bv[next_w]
    current_word = next_w
end
