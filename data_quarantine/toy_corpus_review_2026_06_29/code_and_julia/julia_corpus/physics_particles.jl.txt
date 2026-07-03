"""Particles — حروف المعاني والروابط النحوية العربية (Classifier كامل)."""
module Particles
export is_particle, get_particle_role

@enum ParticleRole::UInt8 begin
    ROLE_JARR
    ROLE_NASB
    ROLE_JAZM
    ROLE_ATF
    ROLE_SHART
    ROLE_NAFY
    ROLE_ISTIFHAM
    ROLE_COMPOUND
end

const JARR = Set(["من","إلى","عن","على","في","ب","ل","ك","حتى","منذ","مذ","رب","واو","تالله","عدا","خلا","حاشا"])
const NASB = Set(["لن","كي","لكي","إذن","أن","أن","كأن","لأن"])
const JAZM = Set(["لم","لما","لام","لا"])
const ATF = Set(["و","ف","ثم","أو","أم","بل","لكن","حتى"])
const SHART = Set(["إن","إذما","من","ما","مهما","متى","أيان","أين","أينما","حيثما","أنى","كيفما","أي"])
const NAFY = Set(["لا","ما","لم","لن","ليس","إن","غير"])
const ISTIFHAM = Set(["هل","أ","من","ما","ماذا","متى","أين","كيف","كم","أي","أنى"])
const COMPOUND = Set(["بينما","لطالما","طالما","لأن","كأن","كأنما","إنما","لكن","كيما","لكيما","مما","عما","فيم","لم","بم","ألا","لئلا","كيلا"])

const ALL = union(JARR, NASB, JAZM, ATF, SHART, NAFY, ISTIFHAM, COMPOUND)
const _p_cache = Dict{String,Bool}()

is_particle(word::String) = begin
    length(word) == 0 && return false
    haskey(_p_cache, word) && return _p_cache[word]
    result = word in ALL
    _p_cache[word] = result
    return result
end

function get_particle_role(word::String)
    word in JARR && return ROLE_JARR
    word in NASB && return ROLE_NASB
    word in JAZM && return ROLE_JAZM
    word in ATF && return ROLE_ATF
    word in SHART && return ROLE_SHART
    word in NAFY && return ROLE_NAFY
    word in ISTIFHAM && return ROLE_ISTIFHAM
    word in COMPOUND && return ROLE_COMPOUND
    return nothing
end
end
