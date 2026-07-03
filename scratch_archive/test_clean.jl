const _AR_NORM_MAP_DEST = Dict(
    'آ'=>'ا','أ'=>'ا','إ'=>'ا','ؤ'=>'و','ئ'=>'ي','ة'=>'ه','ى'=>'ي',
)
const _AR_DIACRITICS_SET_DEST = Set([
    0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650, 0x0651, 0x0652, 0x0670,
])

function _clean_arabic_word_dest(w::String)
    chars = Char[]
    for c in w
        code = UInt32(c)
        if code in _AR_DIACRITICS_SET_DEST
            println("Stripped diacritic: ", escape_string(string(c)))
            continue
        end
        if code == 0x0640
            println("Stripped tatweel")
            continue
        end
        push!(chars, get(_AR_NORM_MAP_DEST, c, c))
    end
    return lowercase(String(chars))
end

word1 = "مِرنانَ"
word2 = "مرنان"
println("Word 1 cleaned: '", _clean_arabic_word_dest(word1), "'")
println("Word 2 cleaned: '", _clean_arabic_word_dest(word2), "'")
println("Match? ", _clean_arabic_word_dest(word1) == _clean_arabic_word_dest(word2))
