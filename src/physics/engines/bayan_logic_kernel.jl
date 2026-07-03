"""
BayanLogicKernelModule -- a compact Mirnan-local copy of Bayan logic auditing.

The full BayanLanguage tree remains owned by Majnon. This module copies the
small idea Mirnan needs: extract simple logical claims from prompt/output and
detect direct contradiction before the answer is trusted as a learning signal.
"""
module BayanLogicKernelModule

export BayanLogicClaim, BayanLogicAudit, BayanLogicKernel,
       extract_bayan_claims, audit_bayan_logic, bayan_logic_summary

struct BayanLogicClaim
    subject::String
    relation::String
    object::String
    polarity::Int
    source::String
end

struct BayanLogicAudit
    score::Float64
    consistent::Bool
    issues::Vector{String}
    contradictions::Vector{Tuple{BayanLogicClaim,BayanLogicClaim}}
    claims::Vector{BayanLogicClaim}
end

mutable struct BayanLogicKernel
    enabled::Bool
    max_claims::Int
    history::Vector{Dict{String,Any}}
end

BayanLogicKernel(; enabled::Bool=true, max_claims::Int=120) =
    BayanLogicKernel(enabled, max_claims, Dict{String,Any}[])

const AR_NEG = Set(["لا", "ليس", "ليست", "غير", "لن", "لم", "ما"])
const EN_NEG = Set(["not", "never", "no", "isnt", "isn't", "cannot", "can't"])
const AR_COPULA = Set(["هو", "هي", "يكون", "تكون", "كان", "كانت"])
const EN_COPULA = Set(["is", "are", "was", "were", "be", "being"])

function _clean_token(token::AbstractString)
    s = lowercase(strip(String(token)))
    s = replace(s, r"[،؟?!;:,.\(\)\[\]{}\"']" => "")
    return strip(s)
end

function _norm(token::AbstractString)
    s = _clean_token(token)
    s = replace(s, 'أ' => 'ا', 'إ' => 'ا', 'آ' => 'ا', 'ى' => 'ي', 'ة' => 'ه')
    startswith(s, "ال") && length(s) > 2 && (s = s[nextind(s, firstindex(s), 2):end])
    return s
end

function _tokens(text::AbstractString)
    return String[_norm(t) for t in split(String(text)) if !isempty(_norm(t))]
end

function _claim(subject::String, relation::String, object::String,
                polarity::Int, source::String)
    return BayanLogicClaim(subject, relation, object, polarity, source)
end

function _add_claim!(claims::Vector{BayanLogicClaim}, claim::BayanLogicClaim)
    isempty(claim.subject) && return
    isempty(claim.object) && return
    push!(claims, claim)
end

function extract_bayan_claims(text::AbstractString; source::String="text")
    toks = _tokens(text)
    claims = BayanLogicClaim[]
    n = length(toks)
    n < 3 && return claims

    for i in 1:(n - 2)
        a, b, c = toks[i], toks[i + 1], toks[i + 2]
        if b in AR_COPULA || b in EN_COPULA
            _add_claim!(claims, _claim(a, "is", c, 1, source))
        elseif b in AR_NEG || b in EN_NEG
            if i + 2 <= n
                rel = (c in AR_COPULA || c in EN_COPULA) ? "is" : c
                obj = (c in AR_COPULA || c in EN_COPULA) && i + 3 <= n ? toks[i + 3] : c
                _add_claim!(claims, _claim(a, rel, obj, -1, source))
            end
        end
    end

    for i in 1:(n - 3)
        a, b, c, d = toks[i], toks[i + 1], toks[i + 2], toks[i + 3]
        if (b in AR_COPULA || b in EN_COPULA) && (c in AR_NEG || c in EN_NEG)
            _add_claim!(claims, _claim(a, "is", d, -1, source))
        end
    end

    return claims
end

function _same_proposition(a::BayanLogicClaim, b::BayanLogicClaim)
    return a.subject == b.subject && a.relation == b.relation && a.object == b.object
end

function _contradictions(claims::Vector{BayanLogicClaim})
    out = Tuple{BayanLogicClaim,BayanLogicClaim}[]
    for i in 1:length(claims)
        i == length(claims) && break
        for j in (i + 1):length(claims)
            a, b = claims[i], claims[j]
            _same_proposition(a, b) && a.polarity != b.polarity && push!(out, (a, b))
        end
    end
    return out
end

function _record!(kernel::BayanLogicKernel, audit::BayanLogicAudit)
    push!(kernel.history, Dict{String,Any}(
        "score" => audit.score,
        "consistent" => audit.consistent,
        "issues" => copy(audit.issues),
        "claim_count" => length(audit.claims),
        "contradiction_count" => length(audit.contradictions),
    ))
    while length(kernel.history) > kernel.max_claims
        popfirst!(kernel.history)
    end
    return audit
end

function audit_bayan_logic(kernel::BayanLogicKernel,
                           prompt::AbstractString,
                           output::AbstractString)
    if !kernel.enabled
        return BayanLogicAudit(1.0, true, String[], Tuple{BayanLogicClaim,BayanLogicClaim}[], BayanLogicClaim[])
    end
    claims = vcat(extract_bayan_claims(prompt; source="prompt"),
                  extract_bayan_claims(output; source="output"))
    contradictions = _contradictions(claims)
    issues = String[]
    isempty(contradictions) || push!(issues, "bayan_logic_contradiction")

    score = isempty(contradictions) ? 1.0 : max(0.0, 1.0 - 0.45 * length(contradictions))
    audit = BayanLogicAudit(score, isempty(contradictions), issues, contradictions, claims)
    return _record!(kernel, audit)
end

function bayan_logic_summary(kernel::BayanLogicKernel)
    n = length(kernel.history)
    bad = count(item -> !Bool(get(item, "consistent", true)), kernel.history)
    return Dict{String,Any}(
        "enabled" => kernel.enabled,
        "history_count" => n,
        "inconsistent_count" => bad,
    )
end

end # module BayanLogicKernelModule
