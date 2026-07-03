"""
جبر كليفورد الهندسي — Clifford Geometric Algebra.

Multivector22: متجه كليفورد 22D مع الدرجات 0 (سلمي)، 1 (متجه)، و 2 (ثنائي المتجه).
إجمالي المركبات: 1 + 22 + 231 = 254.
يدعم: الجداء الهندسي، المعكوس، القسمة الدلالية، الثنوية.
"""
module CliffordMath

using LinearAlgebra

export Multivector22, from_vector, get_biv_indices, get_scalar_essence,
       normalize!, CLIFFORD_DIM, BIVECTOR_DIM

const CLIFFORD_DIM = 22
const BIVECTOR_DIM = CLIFFORD_DIM * (CLIFFORD_DIM - 1) ÷ 2

# ثوابت مُسبقة الحساب لمؤشرات bivector
const _biv_indices = Tuple{Int,Int}[]
for i in 1:CLIFFORD_DIM
    for j in (i+1):CLIFFORD_DIM
        push!(_biv_indices, (i, j))
    end
end

"""
    Multivector22

متجه كليفورد 22D.
- `s::Float64` — الدرجة 0 (سلمي)
- `v::Vector{Float64}` — الدرجة 1 (متجه 22D)
- `b::Vector{Float64}` — الدرجة 2 (bivector 231D)
- `p::Float64` — السلمي الزائف (pseudoscalar)
"""
mutable struct Multivector22
    s::Float64
    v::Vector{Float64}
    b::Vector{Float64}
    p::Float64

    function Multivector22(scalar::Real=0.0, vector=nothing, bivector=nothing, pseudoscalar::Real=0.0)
        v_arr = zeros(Float64, CLIFFORD_DIM)
        if vector !== nothing
            vec_v = Float64.(collect(vector))
            n = min(length(vec_v), CLIFFORD_DIM)
            v_arr[1:n] .= vec_v[1:n]
        end

        b_arr = zeros(Float64, BIVECTOR_DIM)
        if bivector !== nothing
            vec_b = Float64.(collect(bivector))
            n = min(length(vec_b), BIVECTOR_DIM)
            b_arr[1:n] .= vec_b[1:n]
        end

        return new(Float64(scalar), v_arr, b_arr, Float64(pseudoscalar))
    end
end

"""
    from_vector(v) -> Multivector22

إنشاء متجه كليفورد من متجه 22D عادي.
"""
function from_vector(v)
    return Multivector22(0.0, v)
end

"""
    norm(mv::Multivector22) -> Float64

معيار المتجه: √(s² + Σv² + Σb²).
"""
function LinearAlgebra.norm(mv::Multivector22)
    return sqrt(mv.s^2 + sum(mv.v .^ 2) + sum(mv.b .^ 2) + mv.p^2)
end

"""
    normalize!(mv::Multivector22) -> Multivector22

تطبيع المتجه.
"""
function normalize!(mv::Multivector22)
    n = norm(mv)
    if n > 0
        mv.s /= n
        mv.v ./= n
        mv.b ./= n
        mv.p /= n
    end
    return mv
end

"""
    reverse(mv::Multivector22) -> Multivector22

معكوس كليفورد (conjugate): A† = s + v - b
"""
function Base.reverse(mv::Multivector22)
    return Multivector22(mv.s, copy(mv.v), -copy(mv.b), 0.0)
end

"""
    inv(mv::Multivector22) -> Multivector22

معكوس كليفورد: A⁻¹ = A† / |A|²
"""
function Base.inv(mv::Multivector22)
    rev = reverse(mv)
    n2 = mv.s^2 + sum(mv.v .^ 2) + sum(mv.b .^ 2) + mv.p^2
    if n2 < 1e-12
        return Multivector22(1.0)
    end
    rev.s /= n2
    rev.v ./= n2
    rev.b ./= n2
    return rev
end

"""
    Base.:(/)(a::Multivector22, b::Multivector22) -> Multivector22

قسمة كليفورد: self / other = other⁻¹ * self
"""
function Base.:(/)(a::Multivector22, b::Multivector22)
    return inv(b) * a
end

function Base.:(/)(a::Number, b::Multivector22)
    return Multivector22(a) * inv(b)
end

"""
    dual(mv::Multivector22) -> Multivector22

ثنوية: dual(A) = -p + s·I
"""
function dual(mv::Multivector22)
    return Multivector22(-mv.p, nothing, nothing, mv.s)
end

function Base.:(+)(a::Multivector22, b::Multivector22)
    return Multivector22(a.s + b.s, a.v + b.v, a.b + b.b, a.p + b.p)
end

function Base.:(-)(a::Multivector22, b::Multivector22)
    return Multivector22(a.s - b.s, a.v - b.v, a.b - b.b, a.p - b.p)
end

"""
    _wedge(v1, v2) -> Vector{Float64}

الجداء الخارجي (wedge) بين متجهين لإنتاج bivector 231D.
"""
function _wedge(v1::AbstractVector, v2::AbstractVector)
    b = zeros(Float64, BIVECTOR_DIM)
    k = 0
    for i in 1:CLIFFORD_DIM
        for j in (i+1):CLIFFORD_DIM
            k += 1
            b[k] = v1[i] * v2[j] - v1[j] * v2[i]
        end
    end
    return b
end

"""
    _contract_bv(b::Vector{Float64}, v::Vector{Float64}) -> Vector{Float64}

انكماش bivector مع متجه لإنتاج متجه.
"""
function _contract_bv(b::Vector{Float64}, v::Vector{Float64})
    res = zeros(Float64, CLIFFORD_DIM)
    k = 0
    for i in 1:CLIFFORD_DIM
        for j in (i+1):CLIFFORD_DIM
            k += 1
            val = b[k]
            res[i] -= val * v[j]
            res[j] += val * v[i]
        end
    end
    return res
end

"""
    *(a::Multivector22, b::Multivector22) -> Multivector22

الجداء الهندسي لفضاء كليفورد.
"""
function Base.:(*)(a::Multivector22, b::Multivector22)
    # Scalar
    res_s = a.s * b.s + dot(a.v, b.v)
    # Vector
    res_v = a.s .* b.v .+ b.s .* a.v
    res_v .+= _contract_bv(a.b, b.v)
    res_v .-= _contract_bv(b.b, a.v)
    # Bivector
    res_b = a.s .* b.b .+ b.s .* a.b
    res_b .+= _wedge(a.v, b.v)

    b_norm = norm(a.b)
    persistence = b_norm > 0.5 ? 0.7 : 0.3
    res_b .+= a.b .* persistence
    res_b .+= b.b .* persistence

    # Pseudoscalar
    res_p = a.s * b.p + b.s * a.p

    return Multivector22(res_s, res_v, res_b, res_p)
end

"""
    get_scalar_essence(mv::Multivector22) -> Float64

الجوهر السلمي (المركبة السلمية).
"""
get_scalar_essence(mv::Multivector22) = mv.s

"""
    get_bivector_orientation(mv::Multivector22) -> Float64

شدة اتجاه bivector (معياره).
"""
get_bivector_orientation(mv::Multivector22) = norm(mv.b)

function Base.show(io::IO, mv::Multivector22)
    print(io, "MV22(S:", round(mv.s; digits=2),
          ", |V|:", round(norm(mv.v); digits=2),
          ", |B|:", round(norm(mv.b); digits=2), ")")
end

"""
    get_biv_indices(; dim=22) -> Vector{Tuple{Int,Int}}

مؤشرات (i, j) لمركبات bivector.
"""
function get_biv_indices(; dim::Int=CLIFFORD_DIM)
    indices = Tuple{Int,Int}[]
    for i in 1:dim
        for j in (i+1):dim
            push!(indices, (i, j))
        end
    end
    return indices
end

end # module CliffordMath
