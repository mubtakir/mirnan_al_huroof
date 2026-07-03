"""
test_full.jl - اختبار شامل للمشروع
"""

println("═══════════════════════════════════════════")
println("     اختبار شامل لمشروع مرنان الجديد")
println("═══════════════════════════════════════════")
println("")

# تحميل المشروع
println("1. تحميل المشروع...")
using MirnanNew
println("   ✓ تم تحميل المشروع بنجاح")
println("")

# ═══════════════════════════════════════════════════════
# اختبار المعالجة المسبقة
# ═══════════════════════════════════════════════════════
println("2. اختبار المعالجة المسبقة...")
pp = TextPreprocessor()
normalized = preprocess_text("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"; preprocessor = pp)
println("   المُعالَج: ", normalized)
words = extract_words("مرحبا بالعالم العربي")
println("   الكلمات: ", words)
sentences = extract_sentences("الجملة الأولى. الجملة الثانية! الجملة الثالثة؟")
println("   الجمل: ", sentences)
println("   ✓ المعالجة المسبقة تعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار الدلالة
# ═══════════════════════════════════════════════════════
println("3. اختبار الدلالة...")
analyzer = SemanticAnalyzer()
embedding = embed_word(analyzer, "كتب")
println("   الجذر: ", embedding.root)
println("   طول المتجه: ", length(embedding.semantic_vector))
similarity = compute_semantic_similarity(
    embed_word(analyzer, "كتب"),
    embed_word(analyzer, "قرأ")
)
println("   التشابه (كتب، قرأ): ", round(similarity * 100, digits=1), "%")
println("   ✓ الدلالة تعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار النحو
# ═══════════════════════════════════════════════════════
println("4. اختبار النحو...")
grammar = GrammarAnalyzer()
sentence = analyze_sentence(grammar, "الطالب يقرأ الكتاب")
println("   نوع الجملة: ", sentence.sentence_type)
println("   عدد الكلمات: ", length(sentence.words))
println("   ✓ النحو يعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار قاعدة بيانات الجذور
# ═══════════════════════════════════════════════════════
println("5. اختبار قاعدة بيانات الجذور...")
roots = load_root_db()
println("   عدد الجذور: ", length(roots))
root_info = get_root_info(roots, "كتب")
if root_info !== nothing
    println("   معلومات الجذر 'كتب': ", root_info["meaning"])
    println("   التصنيف: ", root_info["category"])
    println("   المشتقات: ", root_info["derivatives"])
end
stats = get_statistics(roots)
println("   الإحصائيات: ", stats)
println("   ✓ قاعدة بيانات الجذور تعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار الربط المتكامل
# ═══════════════════════════════════════════════════════
println("6. اختبار الربط المتكامل...")
pipeline = MirnanPipeline()
result = analyze_text(pipeline, "الطالب يقرأ الكتاب في المكتبة")
println("   النص الأصلي: ", result.original_text)
println("   النص المُعالَج: ", result.normalized_text)
println("   عدد الكلمات: ", length(result.words))
println("   عدد الجمل: ", length(result.sentences))
println("   الملخص الدلالي:")
println("     - الكلمات الفريدة: ", result.semantic_summary.unique_words)
println("     - متوسط الكثافة: ", round(result.semantic_summary.avg_semantic_density, digits=3))
println("   الملخص النحوي:")
println("     - أنواع الجمل: ", result.grammar_summary.sentence_types)
println("     - الدقة النحوية: ", round(result.grammar_summary.grammatical_accuracy * 100, digits=1), "%")
println("   الملخص الفيزيائي:")
println("     - الطاقة الكلية: ", round(result.physics_summary.total_energy, digits=3))
println("     - الكتلة المتوسطة: ", round(result.physics_summary.avg_mass, digits=3))
println("   الاستجابة:")
println("     ", result.response)
println("   ✓ الربط المتكامل يعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار التقرير التحليلي
# ═══════════════════════════════════════════════════════
println("7. اختبار التقرير التحليلي...")
report = get_analysis_report(result)
println(report)
println("   ✓ التقرير التحليلي يعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار تحليل كلمة
# ═══════════════════════════════════════════════════════
println("8. اختبار تحليل كلمة...")
word_analysis = analyze_word(pipeline, "كتاب", ["كتب", "مكتبة", "قراءة"])
println("   الكلمة: ", word_analysis.word)
println("   الجذر: ", word_analysis.root)
println("   الدور النحوي: ", word_analysis.syntactic_role)
println("   المورفيات: ", word_analysis.morphemes)
println("   التكرار: ", round(word_analysis.frequency, digits=3))
println("   ✓ تحليل الكلمة يعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار تحليل الجملة
# ═══════════════════════════════════════════════════════
println("9. اختبار تحليل الجملة...")
sentence_analysis = analyze_sentence_full(pipeline, "الطالب يقرأ الكتاب في المكتبة", ["كتب", "مكتبة"])
println("   الجملة: ", sentence_analysis.sentence)
println("   النوع: ", sentence_analysis.sentence_type)
println("   التماسك الدلالي: ", round(sentence_analysis.semantic_coherence * 100, digits=1), "%")
println("   الصحة النحوية: ", round(sentence_analysis.grammatical_correctness * 100, digits=1), "%")
println("   ✓ تحليل الجملة يعمل")
println("")

# ═══════════════════════════════════════════════════════
# اختبار تحليل نص طويل
# ═══════════════════════════════════════════════════════
println("10. اختبار تحليل نص طويل...")
long_text = """
مرحبا بكم في مشروع مرنان الجديد. هذا مشروع لغوي فيزيائي ديناميكي.
يهدف إلى بناء نموذج لغوي عربي متكامل. المشروع يحتوي على عدة مكونات:
المعالجة المسبقة، الدلالة العربية، النحو العربي، والمحركات الفيزيائية.
الهدف هو فهم اللغة العربية بشكل أعمق وإنتاج نصوص مفيدة.
"""
result_long = analyze_text(pipeline, long_text)
println("   طول النص: ", length(long_text), " حرف")
println("   عدد الكلمات: ", length(result_long.words))
println("   عدد الجمل: ", length(result_long.sentences))
println("   الحقول الدلالية: ", result_long.semantic_summary.semantic_fields)
println("   ✓ تحليل النص الطويل يعمل")
println("")

println("═══════════════════════════════════════════")
println("     اجتاز جميع الاختبارات بنجاح!")
println("═══════════════════════════════════════════")
