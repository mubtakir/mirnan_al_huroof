"""Advanced Physics Engines — المحركات الفيزيائية المتقدمة الكاملة.
PhaseAccumulator (SO(3) rotation) + RefractoryGate + MacroWaveEngine + ResonantBeamformer."""
module AdvancedEngines
using LinearAlgebra, SparseArrays, Statistics
using ..Constants: TOTAL_DIM, PHASE_DIM
using ..WordPhysics: phase_similarity
export Beamformer, VectorizedGravity, PhaseAttention, PhaseAccumulator, RefractoryGate, MacroWaveEngine, ResonantBeamformer

# ═══ Beamformer ═══
mutable struct Beamformer; focus_matrix::Matrix{Float64}; end
Beamformer() = Beamformer(zeros(0,0))
function focus!(bf::Beamformer, pvs::Vector{<:AbstractVector})
    n=length(pvs); n<2 && return; D=length(pvs[1]); bf.focus_matrix = zeros(n,D)
    w=exp.(-0.3.*Float64(0:n-1))
    for i in 1:n; bf.focus_matrix[i,:].=w[i].*Float64.(pvs[i]); end
end
function beamform_score(bf::Beamformer, cpv::AbstractVector)
    size(bf.focus_matrix,1)==0 && return 0.0
    ss=[dot(view(bf.focus_matrix,i,:),cpv)/(norm(view(bf.focus_matrix,i,:))*norm(cpv)+1e-10) for i=1:size(bf.focus_matrix,1)]
    return mean(abs.(ss))
end
mutable struct VectorizedGravity end
function compute_gravity(::VectorizedGravity, masses::Vector{Float64}, pv_mat::Matrix{Float64})
    n=size(pv_mat,1); n<2 && return 0.0; total=0.0
    for i=1:n, j=(i+1):n; r=norm(pv_mat[i,:].-pv_mat[j,:])+1e-6; total+=masses[i]*masses[j]/r^2; end
    return total
end
mutable struct PhaseAttention; alpha::Float64; end
PhaseAttention(;alpha=0.3)=PhaseAttention(alpha)
function attend(pa::PhaseAttention, pvs::Vector{<:AbstractVector})
    n=length(pvs); n<2 && return pvs; mat=reduce(hcat,pvs)'; attn=mat*mat'; attn./=sum(attn;dims=2).+1e-10
    result=copy(mat); for i=1:n; result[i,:]=(1-pa.alpha).*mat[i,:].+pa.alpha.*(attn[i,:]'*mat); end
    return [result[i,:] for i=1:n]
end

# ═══ RESONANT BEAMFORMER ═══
struct ResonantBeamformer; steering_gain::Float64; end
ResonantBeamformer(;gain=2.0)=ResonantBeamformer(gain)
function beamform(rb::ResonantBeamformer, cand_pv, ctx_pvs)
    isempty(ctx_pvs) && return zeros(9958), Float64[], 0.0
    n=length(ctx_pvs); weights=zeros(n)
    for i=1:n; cpv=ctx_pvs[i]; norm(cpv)<1e-10&&continue; weights[i]=exp(rb.steering_gain*phase_similarity(cpv,cand_pv)); end
    w_sum=sum(weights); w_sum<1e-10 && return zeros(9958), Float64[], 0.0
    weights./=w_sum; beam=sum(w.*Float64.(pv[1:min(9958,end)]) for (w,pv)=zip(weights,ctx_pvs))
    nrm=norm(beam); nrm>1e-10 && (beam./=nrm); return beam, weights, maximum(weights)
end

# ═══ PHASE ACCUMULATOR (SO(3) non-commutative) ═══
mutable struct PhaseAccumulator
    base_angle::Float64; rotation_cache::Dict{Int,Matrix{Float64}}
end
PhaseAccumulator(;base_angle=0.15)=PhaseAccumulator(base_angle, Dict())
function _rot3(pa::PhaseAccumulator, pos::Int)
    haskey(pa.rotation_cache,pos) && return pa.rotation_cache[pos]
    a=pos*pa.base_angle
    Rx=[1 0 0;0 cos(a) -sin(a);0 sin(a) cos(a)]
    Ry=[cos(a*0.7) 0 sin(a*0.7);0 1 0;-sin(a*0.7) 0 cos(a*0.7)]
    Rz=[cos(a*0.5) -sin(a*0.5) 0;sin(a*0.5) cos(a*0.5) 0;0 0 1]
    R=Rx*Ry*Rz; pa.rotation_cache[pos]=R; return R
end
function accumulate(pa::PhaseAccumulator, pvs::Vector{<:AbstractVector}; normalize=true)
    n=length(pvs); n==0&&return zeros(9958); n==1&&return Float64.(pvs[1][1:min(9958, length(pvs[1]))])
    acc=zeros(9958)
    for (i,pv)=enumerate(pvs); v=Float64.(pv[1:9958]); norm(v)<1e-10&&continue
        R=_rot3(pa,i); v3=v[1:3]; v3r=R*v3; v[1:3].=v3r; acc.+=v; end
    if normalize; nrm=norm(acc); nrm>1e-10&&(acc./=nrm); end; return acc
end

# ═══ REFRACTORY GATE ═══
mutable struct RefractoryGate
    steps::Int; decay::Float64; strength::Float64; depleted::Vector{Tuple{Vector{Float64},Int,Float64}}
end
RefractoryGate(;steps=3,decay=0.4,strength=1.5)=RefractoryGate(steps,decay,strength,Tuple{Vector{Float64},Int,Float64}[])
function deplete!(rg::RefractoryGate, pv)
    pv===nothing && return; cp=Float64.(pv[1:9958]); nrm=norm(cp); nrm>1e-10&&(cp./=nrm); push!(rg.depleted,(cp,rg.steps,rg.strength))
end
function repulsion(rg::RefractoryGate, cpv)
    isempty(rg.depleted) && return 0.0; total=0.0
    for (pv,st,str) in rg.depleted; f=st/rg.steps; s=phase_similarity(pv,cpv); s>0&&(total-=str*f*s); end; return total
end
function step!(rg::RefractoryGate)
    new=Tuple{Vector{Float64},Int,Float64}[]
    for (pv,st,str) in rg.depleted; st>1&&push!(new,(pv,st-1,str*(1-rg.decay))); end; rg.depleted=new
end

# ═══ MACRO-WAVE ENGINE ═══
mutable struct MacroWaveEngine
    max_concepts::Int; concepts::Dict{String,Dict}; next_id::Int
end
MacroWaveEngine(;max=50)=MacroWaveEngine(max, Dict(), 0)
function entangle(me::MacroWaveEngine, words, get_pv, get_mass)
    length(words)<2 && return nothing
    n=min(4,length(words))
    for nw in[2,3]; n<nw&&continue
        phrase=words[end-nw+1:end]; pkey=join(phrase," "); haskey(me.concepts,pkey)&&return me.concepts[pkey]
        pvs=[get_pv(w) for w in phrase]; any(x->x===nothing,pvs)&&continue
        masses=[get_mass(w) for w in phrase]
        cpv=mean(Float64[p[1:9958] for p in pvs]); nrm=norm(cpv); nrm>1e-10&&(cpv./=nrm)
        cmass=sum(masses); cfreq=exp(mean(log.(max.(masses,1e-10))))
        concept=Dict("id"=>me.next_id,"pv"=>cpv,"mass"=>cmass,"freq"=>cfreq,"words"=>phrase,"key"=>pkey)
        me.concepts[pkey]=concept; me.next_id+=1; length(me.concepts)>me.max_concepts&&pop!(me.concepts,first(keys(me.concepts)))
        return concept
    end; return nothing
end
function score_candidate_via_concepts(me::MacroWaveEngine, cpv, ctx_words)
    isempty(me.concepts)&&return 0.0; best=0.0
    for (_,c) in me.concepts
        sim=phase_similarity(c["pv"],cpv); grav=c["mass"]/(1+length(ctx_words)); best=max(best,sim*grav*0.1)
    end; return best
end
end
