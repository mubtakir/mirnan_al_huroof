"""
EnglishMorpheme كامل — صرف إنجليزي (POS tagging, Grammar, Weight Resonance, Stemmer).
"""
module EnglishMorpheme
export EnglishMorphology, EnglishGrammarEngine, EnglishWeightResonance,
       get_en_syntax_anchor, detect_sentence_type, english_stem

# POS tagger
mutable struct EnglishMorphology; pv_cache::Dict{String,Vector{Float64}}; end
EnglishMorphology()=EnglishMorphology(Dict())

const POS_PATTERNS = Dict(
    "VERB_GERUND" => r"ing$", "VERB_PAST" => r"ed$", "VERB_3RD" => r"s$",
    "ADV" => r"ly$", "NOUN_TION" => r"tion$|sion$", "NOUN_MENT" => r"ment$", "NOUN_NESS" => r"ness$",
    "ADJ_FUL" => r"ful$", "ADJ_LESS" => r"less$", "ADJ_ABLE" => r"able$|ible$",
    "ADJ_ER" => r"er$", "ADJ_EST" => r"est$", "NOUN_PL" => r"s$")
const EN_DET = Set(["the","a","an","this","that","these","those"])
const EN_PREP = Set(["in","on","at","to","from","with","by","for","of","about","into","onto","upon","within","without","during","through"])
const EN_CONJ = Set(["and","or","but","so","because","if","when","while","although","since"])
const EN_PRON = Set(["i","you","he","she","it","we","they","me","him","her","us","them","my","your","his","her","its","our","their","mine","yours","hers","ours","theirs"])
const EN_BE = Set(["is","are","was","were","am","be","been","being"])
const EN_MODAL = Set(["can","could","will","would","shall","should","may","might","must"])

function get_pos(em::EnglishMorphology, word::String)
    w=lowercase(word); w in EN_DET && return "DET"; w in EN_PREP && return "PREP"
    w in EN_CONJ && return "CONJ"; w in EN_PRON && return "PRON"
    w in EN_BE && return "VERB_BE"; w in EN_MODAL && return "VERB_MODAL"
    for (tag,pat) in POS_PATTERNS; occursin(pat,w) && return tag; end
    return "NOUN"
end

function transition_score(em::EnglishMorphology, word::String, prev::Union{String,Nothing})
    prev===nothing && return 0.5; pw=lowercase(prev); ww=lowercase(word)
    pp=get_pos(em,prev); wp=get_pos(em,word)
    if pp=="DET"&&wp in ["NOUN","ADJ_FUL","ADJ_LESS","ADJ_ABLE"]; return 0.9
    elseif pp in ["VERB_BE","VERB_MODAL"]&&wp in ["VERB_GERUND","VERB_PAST","ADJ_FUL","ADV"]; return 0.8
    elseif pp=="ADJ_FUL"&&wp in ["NOUN","NOUN_TION","NOUN_MENT"]; return 0.7
    elseif pp=="PREP"&&wp in ["DET","NOUN","PRON"]; return 0.9; end; return 0.4
end

function agreement_score(em::EnglishMorphology, word::String, prev::Union{String,Nothing})
    prev===nothing && return 0.5; ww=word; pw=prev
    if pw=="is"&&!(endswith(ww,"ing")||endswith(ww,"ed")||ww in EN_DET||ww in EN_PRON); return 0.7; end
    if pw=="are"&&!(endswith(ww,"ing")||ww=="not"); return 0.7; end; return 0.5
end

function get_stem(em::EnglishMorphology, word::String)
    w=lowercase(word); w=="went" && return "go"; w=="children" && return "child"
    w=="better" && return "good"; w=="best" && return "good"
    w=="worse" && return "bad"; w=="worst" && return "bad"
    w=="mice" && return "mouse"; w=="feet" && return "foot"
    for (suf, pat, nrem) in [("ing", r"ing$", 3), ("ed", r"ed$", 2), ("ly", r"ly$", 2)]
        occursin(pat,w) && length(w)>4 && return w[1:end-nrem]
    end
    endswith(w,"s")&&!endswith(w,"ss")&&length(w)>3&&return w[1:end-1]
    return w
end

struct EnglishGrammarEngine end

function detect_sentence_type(text::String)
    t=strip(text); isempty(t)&&return "declarative"
    occursin(r"^(what|where|when|why|how|who|which|whose|whom)\b"i,t) && return "interrogative"
    endswith(t,"?") && return "interrogative"
    occursin(r"^(do|don't|please|go|come|take|give|make|let|stop|start|try|tell|show)\b"i,t) && return "imperative"
    return "declarative"
end

struct EnglishWeightResonance
    patterns::Dict{String,Vector{Float64}}; transitions::Dict{String,Dict{String,Float64}}
end
EnglishWeightResonance()=EnglishWeightResonance(
    Dict("noun_sing"=>ones(22)*0.1,"verb_present"=>ones(22)*0.8,"adj_base"=>ones(22)*0.3,"adv"=>ones(22)*0.25,"det"=>ones(22)*0.05,"prep"=>ones(22)*0.08,"pron"=>ones(22)*0.12,"conj"=>ones(22)*0.06),
    Dict("det"=>Dict("noun_sing"=>0.9,"adj_base"=>0.7),"prep"=>Dict("det"=>0.9,"noun_sing"=>0.8),"noun_sing"=>Dict("verb_present"=>0.8,"prep"=>0.6)))

function get_en_syntax_anchor(word::String)
    w=lowercase(word); w in EN_DET && return [0.0,0.8,0.0,0.0,0.0,0.0]
    w in EN_PREP && return [0.0,0.0,1.0,0.0,0.0,0.0]
    w in EN_CONJ && return [0.0,0.0,0.0,0.0,1.0,0.0]
    w in EN_PRON && return [0.0,0.5,0.0,0.0,0.0,0.5]
    return [0.0,1.0,0.0,0.0,0.0,0.0]
end
function english_stem(word::String)
    w=lowercase(word); w=="went"&&return"go"; w=="children"&&return"child"
    endswith(w,"ing")&&length(w)>4&&return w[1:end-3]
    endswith(w,"ed")&&length(w)>4&&return w[1:end-2]
    endswith(w,"s")&&!endswith(w,"ss")&&length(w)>3&&return w[1:end-1]
    return w
end
end
