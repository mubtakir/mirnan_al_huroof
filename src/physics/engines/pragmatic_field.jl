"""PragmaticField — الفضاء القصدي (6D) + PragmaticBindingEngine."""
module PragmaticField
using LinearAlgebra, Statistics

export PragmaticBindingEngine

struct PragmaticBindingEngine
    anchors::Dict{String,Vector{Float64}}
    triggers::Dict{String,Set{String}}
    interrogative::Set{String}
    answer_attractors::Set{String}
end

function PragmaticBindingEngine()
    anchors = Dict(
        "locative" => [0.0,0.8,0.2,0.0,0.1,0.3], "causal" => [0.3,0.1,0.7,0.0,0.2,0.0],
        "temporal" => [0.1,0.2,0.1,0.9,0.0,0.4], "modal" => [0.0,0.0,0.0,0.1,0.9,0.6],
        "relational" => [0.5,0.5,0.1,0.1,0.8,0.8], "neutral" => [0.2,0.2,0.2,0.2,0.2,0.2],
    )
    for (k, v) in anchors; nrm = norm(v); if nrm>1e-10; anchors[k] = v./nrm; end; end
    triggers = Dict(
        "locative" => Set(["في","على","تحت","فوق","أين","حيث","هنا","هناك","نحو","إلى"]),
        "causal" => Set(["لأن","بسبب","لذلك","إذن","لماذا","نتيجة","من أجل","حتى"]),
        "temporal" => Set(["متى","حين","وقت","أمس","غدا","يوم","شهر","سنة","عندما","بعد","قبل"]),
        "modal" => Set(["كيف","هل","أ","ماذا","ربما","قد","سوف","لن","لا","إن","إذا"]),
    )
    interr = Set(["لماذا","كيف","أين","متى","هل","ماذا","من","أي","كم","why","how","what","where","when"])
    answer = Set(["بسبب","لأن","نتيجة","عبر","يعمل","يقع","السبب","تحدث","يتم","because","by","is","due"])
    PragmaticBindingEngine(anchors, triggers, interr, answer)
end

function detect_intent_frame(eng::PragmaticBindingEngine, prompt_words::Vector{String})
    isempty(prompt_words) && return eng.anchors["neutral"], 0.0, false
    is_interr = any(w->lowercase(w) in eng.interrogative, prompt_words)
    scores = Dict{String,Float64}(k=>0.0 for k in keys(eng.anchors))
    for (i, w) in enumerate(reverse(prompt_words[max(1, end-3):end]))
        weight = 1.0/(i+1)
        for (frame, tset) in eng.triggers
            if w in tset; scores[frame] += weight; end
        end
    end
    best_frame = "neutral"; best_score = 0.0
    for (f,s) in scores; if s>best_score; best_score=s; best_frame=f; end; end
    return eng.anchors[best_frame], min(best_score, 1.0), is_interr
end

function compute_pragmatic_score(eng::PragmaticBindingEngine, word_pv, intent_frame, alpha; word=nothing, is_interrogative=false)
    base = 0.0
    if is_interrogative && word !== nothing && word in eng.answer_attractors; base = 2.0; end
    d_start = max(1, length(word_pv)-5) # pragmatic slice: last 6 elements
    phi_rel = length(word_pv) >= 6 ? word_pv[d_start:min(d_start+5, length(word_pv))] : zeros(6)
    if norm(phi_rel) < 1e-10 || norm(intent_frame) < 1e-10; return base; end
    return alpha * mean(cos.(phi_rel .- intent_frame)) + base
end
end
