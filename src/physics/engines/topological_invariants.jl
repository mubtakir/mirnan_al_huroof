"""TopologicalInvariants — ثوابت طوبولوجية (Berry Phase + Winding Number)."""
module TopologicalInvariants
using LinearAlgebra, Statistics
export TopologicalInvariants, berry_phase, winding_number, topological_similarity

function berry_phase(word_pvs::Vector{<:AbstractVector})
    length(word_pvs) < 2 && return 0.0
    phases = Float64[]
    for i in 1:(length(word_pvs)-1)
        a, b = word_pvs[i], word_pvs[i+1]
        na, nb = norm(a), norm(b)
        (na < 1e-10 || nb < 1e-10) && continue
        overlap = clamp(dot(a,b)/(na*nb), -1.0, 1.0)
        push!(phases, acos(overlap))
    end
    return sum(phases)
end

function winding_number(word_pvs::Vector{<:AbstractVector}; subspace_pairs=nothing)
    length(word_pvs) < 2 && return 0.0
    if subspace_pairs === nothing
        subspace_pairs = [(1,2),(3,4),(5,6),(7,8),(9,10)]
    end
    windings = Float64[]
    for (s0, s1) in subspace_pairs
        angles = [atan(pv[s1], pv[s0]) for pv in word_pvs]
        if length(angles) >= 2
            unwrapped = cumsum([angles[1]; diff(angles) |> (d->[x > π ? x-2π : x < -π ? x+2π : x for x in d])])
            push!(windings, (unwrapped[end] - unwrapped[1])/(2π))
        end
    end
    return isempty(windings) ? 0.0 : mean(windings)
end

function topological_similarity(chain_a, chain_b)
    bp_a, bp_b = berry_phase(chain_a), berry_phase(chain_b)
    wn_a, wn_b = winding_number(chain_a), winding_number(chain_b)
    return 1.0 / (1.0 + abs(bp_a-bp_b) + abs(wn_a-wn_b))
end
end
