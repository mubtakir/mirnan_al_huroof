"""SentimentPolarity — تحليل المشاعر في فضاء الطور."""
module SentimentPolarity
export compute_word_polarity, compute_sentence_polarity, sentiment_fidelity

const POSITIVE = Set(["جميل","رائع","ممتاز","طيب","حسن","جيد","سعيد","فرح","نشاط","حب","سلام","خير","نور","أمل","صحة","نجاح","جمال","كرم","شجاع","كريم","صبور","عادل","صادق","أمين","وفى","بهجة","سرور","رضا","يقين","ثقة","good","great","excellent","wonderful","beautiful","nice","happy","love","peace","success","hope","perfect","amazing","fantastic","brilliant","awesome","thank","thanks","grateful"])
const NEGATIVE = Set(["سيء","قبيح","خطأ","مشكلة","صعب","متعب","مزعج","مؤلم","حزين","غضب","كره","حرب","شر","ظلم","فشل","مرض","جهل","خوف","كذب","خيانة","غش","حسد","بغض","نفاق","رياء","غرور","كبر","قسوة","جشع","ظلام","ضيق","هم","غم","كرب","بلاء","bad","terrible","awful","horrible","ugly","wrong","problem","difficult","tired","annoying","painful","sad","anger","hate","war","evil","failure","sick","fear","lie","betray","cheat","fake","broken","error","bug","crash","worse","worst"])
const NEGATORS = Set(["ليس","ليست","لا","لم","لن","ما","غير","دون","not","never","no","without"])

function compute_word_polarity(word::String)
    w = lowercase(strip(word, ['.',',',':',';','!','?','(',')','[',']','"','`']))
    w in POSITIVE && return 1.0
    w in NEGATIVE && return -1.0
    return 0.0
end

function compute_sentence_polarity(text::String)
    words = split(text); isempty(words) && return 0.0
    total = 0.0; n = 0; negate = false
    for w in words
        wc = lowercase(strip(w, ['.',',',':',';','!','?','(',')']))
        if wc in NEGATORS; negate = !negate; continue; end
        pol = compute_word_polarity(wc)
        if pol != 0.0; total += negate ? -pol : pol; n += 1; end
        negate = false
    end
    n == 0 && return 0.0
    return clamp(total/n, -1.0, 1.0)
end

function sentiment_fidelity(word::String, context_words::Vector{String}; cache=nothing)
    isempty(context_words) && return 0.5
    ctx_pol_vec = if cache !== nothing
        get!(cache, "SENT_CTX\0" * join(context_words, "\0")) do
            [compute_sentence_polarity(join(context_words, " "))]
        end
    else
        [compute_sentence_polarity(join(context_words, " "))]
    end
    ctx_pol = ctx_pol_vec[1]
    word_pol = compute_word_polarity(word)
    ctx_pol == 0.0 && return 0.5
    word_pol == 0.0 && return 0.3
    return (ctx_pol * word_pol) > 0.0 ? min(1.0, abs(ctx_pol)+0.3) : max(0.0, 1.0-abs(ctx_pol))
end
end
