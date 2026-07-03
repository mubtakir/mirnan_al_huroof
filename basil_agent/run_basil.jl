using Pkg
Pkg.activate(joinpath(@__DIR__))

# Load main package
include("src/BasilAgent.jl")
using .BasilAgent

# Start Web server on http://127.0.0.1:5000
# (This matches the Flask port 5000 and is fully compatible with the UI)
BasilAgent.App.start_server("127.0.0.1", 5000)
