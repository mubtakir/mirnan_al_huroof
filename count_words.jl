content = read(joinpath(@__DIR__, "data", "corpus", "ma_test_corpus.txt"), String)
words = split(content)
filtered = filter(w -> length(w) >= 2 && any(c -> '\u0600' <= c <= '\u06FF', collect(string(w))), words)
unique_words = unique(filtered)
println("إجمالي الكلمات: ", length(filtered))
println("الكلمات الفريدة: ", length(unique_words))
println()
println("أول 30 كلمة فريدة:")
for w in unique_words[1:min(30, length(unique_words))]
    println("  ", w)
end
