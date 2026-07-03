# scratch/test_reforms.jl - Verification script for PRNN changes
using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "src", "MirnanNew.jl")); using .MirnanNew
using Test
using SparseArrays

println("=== Testing PRNN Reforms ===")

# Test 1: Dimension Unification
println("1. Checking Constants and Dimensions...")
N = MirnanNew.Physics.Constants.TOTAL_DIM
println("   TOTAL_DIM = $N")
@test N == 10000

# Test 2: Global dense phase vector cache in PRNNLearner
println("2. Checking PRNNLearner dense phase vector caching...")
word = "سلام"
t1 = time_ns()
v1 = MirnanNew.Physics.PRNNLearner.build_dense_phase_vector(word, N)
dt1 = time_ns() - t1

t2 = time_ns()
v2 = MirnanNew.Physics.PRNNLearner.build_dense_phase_vector(word, N)
dt2 = time_ns() - t2

println("   First build time:  $(dt1 / 1e6) ms")
println("   Second build time (cached): $(dt2 / 1e6) ms")

@test v1 === v2 # check identity or equality
@test dt2 < dt1 / 2 # must be significantly faster due to cache

# Test 3: Standalone PRNNSession cache and TOTAL_DIM default
println("3. Checking standalone PRNNSession...")
vocab = Dict("العلم" => 1, "نور" => 2)
id2word = Dict(1 => "العلم", 2 => "نور")
corpus = [Int32[1, 2]]
session = MirnanNew.Physics.PRNNGenerator.PRNNSession(vocab, id2word, corpus, ["العلم"])
@test session.N == N
@test length(session.base_vectors["العلم"]) == N

# Test 4: Default mode routing in Generator
println("4. Checking default mode auto routing in Generator...")
gen = MirnanGenerator(vocab, spzeros(2, 2))
# Initially corpus_sentences is empty, so mode="auto" (default) routes to "standard"
@test gen.dialogue_mode == false
res = generate!(gen, "العلم")
println("   Generated output: '$res'")

println("=== Verification Completed Successfully! ===")
