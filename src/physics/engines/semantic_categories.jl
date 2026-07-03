"""SemanticCategories — تصنيف دلالي للكلمات."""
module SemanticCategories
export classify_word

const CATEGORY_WORDS = Dict{String,Set{String}}(
    "place" => Set(["بيت","مدرسة","مسجد","سوق","مدينة","قرية","شارع","حديقة","مكتبة","مستشفى"]),
    "person" => Set(["رجل","امرأة","طفل","ولد","بنت","أب","أم","أخ","أخت","صديق"]),
    "object" => Set(["كتاب","قلم","باب","نافذة","طاولة","كرسي","سيارة","هاتف","كوب","ملعقة"]),
    "abstract" => Set(["حب","كره","علم","جهل","حرية","عدل","سلام","حرب","حياة","موت","حق","باطل"]),
    "action" => Set(["كتب","قرأ","مشى","جلس","أكل","شرب","نام","ذهب","جاء","رأى"]),
    "time" => Set(["يوم","أسبوع","شهر","سنة","صباح","مساء","ليل","نهار","ساعة","دقيقة"]),
)

function classify_word(word::String)
    for (cat, words) in CATEGORY_WORDS
        if word in words; return cat; end
    end
    return "other"
end

function category_similarity(cat1::String, cat2::String)
    cat1 == cat2 && return 1.0
    return 0.0
end
end
