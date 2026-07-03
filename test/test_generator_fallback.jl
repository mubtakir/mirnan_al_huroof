include("../src/MirnanNew.jl")
using .MirnanNew

const Physics = MirnanNew.Physics

println("=" ^ 60)
println("GENERATOR FALLBACK TEST")
println("=" ^ 60)

vocab = Dict(w => i for (i, w) in enumerate([
    "\u0627\u0644\u0639\u0644\u0645",
    "\u0646\u0648\u0631",
    "\u062a\u0639\u0644\u0645",
    "\u0645\u0639\u0644\u0645",
    "\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0636\u064a\u0627\u0621",
    "\u0643\u062a\u0627\u0628",
    "\u0642\u0644\u0645",
]))

gen = Physics.MirnanGenerator(vocab)
result = Physics.generate!(
    gen,
    "\u0627\u0644\u0639\u0644\u0645 \u0646\u0648\u0631";
    mode="standard",
    max_words=3,
)

println("Generated: $result")

if isempty(strip(result))
    error("Fallback generation returned an empty result")
end

expected_context = [
    "\u062a\u0639\u0644\u0645",
    "\u0645\u0639\u0644\u0645",
    "\u0627\u0644\u0639\u0627\u0644\u0645",
    "\u0636\u064a\u0627\u0621",
]
prompt_preserved =
    occursin("\u0627\u0644\u0639\u0644\u0645", result) &&
    occursin("\u0646\u0648\u0631", result)
context_related = any(w -> occursin(w, result), expected_context)

if !(prompt_preserved || context_related)
    error("Fallback generation neither chose a context-related word nor preserved the prompt shape")
end

println("PASSED generator fallback")
