"""
الشد الفيزيائي للفيلامنت — Filament Tension.

قياس طاقة الربط والشد الفيزيائي بين الحروف المكونة للكلمة.
تتفاعل الحروف كأوتار (Filaments) مهتزة؛ التوافق الطوري يقلل الشد (استقرار)،
والتنافر الطوري أو التباين الترددي يزيد الشد (جهد موجي).
"""
module FilamentTension

using LinearAlgebra, Statistics
using ..LetterDB: LetterDatabase, get_vector, get_omega_0
using ..WordPhysics: _parse_word_harakat

export FilamentTensionEngine, compute_tension

mutable struct FilamentTensionEngine
    db::LetterDatabase
    FilamentTensionEngine() = new(LetterDatabase())
end

"""
    compute_tension(eng::FilamentTensionEngine, word::String) -> Float64

حساب الشد الفيزيائي للكلمة. يمثل قيمة بين 0.0 و 1.0.
القيم المنخفضة تعني شدًا متزناً (طاقة ربط قوية وسهولة نطق ورنين أعلى).
القيم المرتفعة تعني تشنجاً موجياً (تنافر الحروف).
"""
function compute_tension(eng::FilamentTensionEngine, word::String)
    parsed = _parse_word_harakat(word)
    isempty(parsed) && return 0.5
    
    letters = [string(l) for (l, _) in parsed]
    n = length(letters)
    n < 2 && return 0.1 # كلمة من حرف واحد لها شد منخفض جداً
    
    tensions = Float64[]
    
    # حساب الشد بين كل حرفين متتاليين
    for i in 1:(n-1)
        l1, l2 = letters[i], letters[i+1]
        
        # الحصول على متجهات الطور للحروف (أول 22 بعداً للتوافق مع أبعاد الحروف الصرفية)
        v1 = Float64.(get_vector(eng.db, l1)[1:22])
        v2 = Float64.(get_vector(eng.db, l2)[1:22])
        
        n1, n2 = norm(v1), norm(v2)
        sim = (n1 > 1e-10 && n2 > 1e-10) ? dot(v1, v2) / (n1 * n2) : 0.0
        
        # الترددات الذاتية للحروف
        w1 = get_omega_0(eng.db, l1)
        w2 = get_omega_0(eng.db, l2)
        
        # نسبة الرنين التوافقي (Harmonic Ratio Resonance)
        # التردد الأكبر على التردد الأصغر
        ratio = max(w1, w2) / (min(w1, w2) + 1e-10)
        
        # إذا كانت النسبة قريبة من الأعداد الصحيحة (مثلاً 1:1 أو 2:1)، يكون الشد منخفضاً
        harmonic_residual = abs(ratio - round(ratio))
        
        # معادلة الشد بين الحرفين:
        # يرتفع الشد عند ضعف التوافق الطوري وزيادة النشوز التوافقي
        pair_tension = 0.5 * (1.0 - sim) + 0.5 * harmonic_residual
        push!(tensions, pair_tension)
    end
    
    # الشد الإجمالي هو متوسط شد الأزواج
    return clamp(mean(tensions), 0.0, 1.0)
end

end # module FilamentTension
