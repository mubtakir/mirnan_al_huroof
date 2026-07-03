const AR_NORM = Dict(
    'آ'=>'ا','أ'=>'ا','إ'=>'ا','ؤ'=>'و','ئ'=>'ي','ة'=>'ه','ى'=>'ي',
)
const AR_DIACRITICS = Set([
    0x064B, 0x064C, 0x064D, 0x064E, 0x064F, 0x0650, 0x0651, 0x0652, 0x0670,
])

function clean_word(w::AbstractString)
    chars = Char[]
    for c in w
        code = UInt32(c)
        code in AR_DIACRITICS && continue
        code == 0x0640 && continue
        push!(chars, get(AR_NORM, c, c))
    end
    return lowercase(String(chars))
end

const QUERY = "وعليكم السَّلَامِ ورحمه وبركاته عَلَيْكُمْ اللَّهُ حفظ ورعايته امان السلامه صديقي يا ختام معَ يتفاعل لينتج حالَهٍ تثبت التجربه المستقره العلميه السببيه زمانياً ومنطقياً يسبق يجعله عله كافيه الكاين فَاِن"

# 1. Clean query words
query_tokens = [clean_word(w) for w in split(QUERY) if length(strip(w)) >= 2]
unique_targets = unique(query_tokens)
println("Searching for $(length(unique_targets)) target words...")

results = []

# 2. Walk directories
scan_dirs = ["data", "basil_agent", "."]
for s_dir in scan_dirs
    isdir(s_dir) || continue
    for (root, dirs, files) in walkdir(s_dir)
        if occursin(".git", root) || occursin(".julia", root) || occursin("model", root) || occursin("julia_corpus_backup", root)
            continue
        end
        for f in files
            ext = lowercase(splitext(f)[2])
            ext in [".txt", ".md", ".jl", ".json"] || continue
            path = joinpath(root, f)
            # Skip this script itself
            basename(path) == "find_corpus_sources.jl" && continue
            try
                content = read(path, String)
                words_in_file = Set(clean_word(w) for w in split(content) if length(strip(w)) >= 2)
                
                matched = intersect(unique_targets, words_in_file)
                if !isempty(matched)
                    pct = round(length(matched) / length(unique_targets) * 100, digits=1)
                    push!(results, (path=path, count=length(matched), percent=pct, matches=matched))
                end
            catch e
            end
        end
    end
end

# 3. Sort by match count descending
sort!(results; by=x->-x.count)

println("\n==================================================")
println("  🔍 Corpus Search Results (Matching Target Words)")
println("==================================================")
for (idx, r) in enumerate(results[1:min(15, end)])
    println("$idx. [Matches: $(r.count) / $(length(unique_targets)) ($(r.percent)%)]")
    println("   Path: $(r.path)")
    match_list = collect(r.matches)
    sample_str = join(match_list[1:min(8, end)], " | ")
    println("   Sample Words Found: $sample_str...")
    println("--------------------------------------------------")
end
