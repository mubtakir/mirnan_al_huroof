import re

def clean_algorithms():
    path = r"C:\Users\allmy\Desktop\aaa\mirnan_julia\docs\ALGORITHMS.md"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    # 1. Clean Section 7
    old_sec7 = """## 7. مصفوفة الكثافة الكمومية (Quantum Density Matrix)

### البناء

```
ρ = Σᵢ wᵢ · |ψᵢ⟩⟨ψᵢ|
```

حيث:
- `|ψᵢ⟩`: متجه طور الكلمة i (10000D، مُطبَّع)
- `wᵢ`: وزن اضمحلال أسي `wᵢ = exp(-decay·(n-i)) / Σ exp(-decay·(n-k))`

### الرنين

```
R(v) = ⟨v| ρ |v⟩
     = vᵀ · ρ · v
     = Σᵢ wᵢ · |⟨v|ψᵢ⟩|²
```

**لماذا هذا أفضل من np.average(pvs)؟**
- يحافظ على العلاقات البينية بين متجهات السياق
- لا يحدث تداخل إتلافي (destructive interference)
- يسمح بقياس درجة "النقاء" و "التشابك"

### النقاء والتشابك

```
Purity = Tr(ρ²)          ∈ (0, 1]
  1.0 = حالة نقية (كلمة واحدة)
  < 1.0 = حالة مختلطة (سياق متعدد الكلمات)

S_vn = -Tr(ρ ln ρ)        إنتروبيا فون نيومان
  تقيس درجة "التشابك" — كلما زادت، كان السياق أكثر ثراءً
```"""

    new_sec7 = """## 7. مصفوفة الكثافة الطورية (Phase Density Matrix)

### البناء

```
ρ = Σᵢ wᵢ · |ψᵢ⟩⟨ψᵢ|
```

Subject/Context represented as a mixed state:
- `|ψᵢ⟩`: متجه طور الكلمة i (10000D، مُطبَّع)
- `wᵢ`: وزن اضمحلال أسي `wᵢ = exp(-decay·(n-i)) / Σ exp(-decay·(n-k))`

### الرنين

```
R(v) = ⟨v| ρ |v⟩
     = vᵀ · ρ · v
     = Σᵢ wᵢ · |⟨v|ψᵢ⟩|²
```

**لماذا هذا أفضل من np.average(pvs)؟**
- يحافظ على العلاقات البينية بين متجهات السياق
- لا يحدث تداخل إتلافي (destructive interference)
- يسمح بقياس درجة "النقاء" و "الاقتران الطوري"

### النقاء والاقتران

```
Purity = Tr(ρ²)          ∈ (0, 1]
  1.0 = حالة نقية (كلمة واحدة)
  < 1.0 = حالة مختلطة (سياق متعدد الكلمات)

S_vn = -Tr(ρ ln ρ)        إنتروبيا فون نيومان الطيفية
  تقيس درجة "الاقتران الطوري" — كلما زادت، كان السياق أكثر ثراءً
```"""

    content = content.replace(old_sec7.replace("\n", "\r\n"), new_sec7.replace("\n", "\r\n"))
    content = content.replace(old_sec7, new_sec7)

    # 2. Remove misplaced Section 19 from Section 18
    # We will locate the start of Section 19 inside 18 and remove until the 3. **ديناميكيات
    misplaced_pattern = r"## 19\. محرك المتجه الموجي والاقتران الطوري التناظري.*?(?=3\. \*\*ديناميكيات)"
    content = re.sub(misplaced_pattern, "", content, flags=re.DOTALL)

    # 3. Clean section 19 at the end of the file
    old_sec19 = """## 19. محرك الكيوبايت والتشابك الطوري الموجي التناظري (Analog Wave Qubit Engine)

يعتمد هذا المحرك على تصميم حوسبة موجية تناظرية مستمرة كبديل متماسك للحاسوب الكمومي التقليدي. يمثل الكيوبايت كحزمة موجية متراكبة ومتداخلة في فضاء طوري كثيف ذي أبعاد $N = 10,000$.

### الخوارزمية الرياضية

1. **انشطار الحزمة (Wave Splitting):**
   تنقسم الموجة الطورية المركبة $\\mathbf{v}$ إلى مركبتين متعامدتين ونظير معاكس (موجة مضادة/هدامة) وفق معادلة مجزئ الحزمة:
   $$ \\mathbf{v}_1 = \\cos(\\theta) \\mathbf{v} $$
   $$ \\mathbf{v}_2 = \\sin(\\theta) \\mathbf{v} \\odot e^{i \\phi_{\\text{shift}}} $$
   $$ \\mathbf{v}_{\\text{anti}} = -\\mathbf{v}_1 = \\mathbf{v}_1 \\odot e^{i \\pi} $$
   حيث يحقق النظام انحفاظ الطاقة الموجية: $\|\\mathbf{v}_1\|^2 + \|\\mathbf{v}_2\|^2 = \|\\mathbf{v}\|^2$.

2. **دمج وتداخل الموجات (Wave Merging):**
   تندمج موجتان $\\mathbf{v}_1$ و $\\mathbf{v}_2$ مع سياق التوليد بفرق طور $\\Delta \\phi$:
   $$ \\mathbf{w}_{\\text{merged}} = \\frac{\\mathbf{v}_1 + \\mathbf{v}_2 \\odot e^{i \\Delta \\phi}}{\|\\mathbf{v}_1 + \\mathbf{v}_2 \\odot e^{i \\Delta \\phi}\|} $$
   حيث تعطي القيمة $\\Delta \\phi = 0$ تداخلاً بناءً تاماً، وتعطي $\\Delta \\phi = \pi$ تداخلاً هداماً متضاداً تاماً.

3. **التشابك والقياس الطوري الهولوغرافي (Entanglement & Measurement):**
   تُشابك الموجة الناتجة من التداخل البناء $\\mathbf{w}_{\\text{constructive}}$ مع حقل موجة المثير الإجمالية $\\mathbf{w}_{\\text{prompt}}$ عبر جداء خارجي هولوغرافي:
   $$ \\mathbf{E}_{AB} = \\mathbf{w}_{\\text{constructive}} \\otimes \\mathbf{w}_{\\text{prompt}}^\\dagger $$
   وعند تمرير موجة السياق الحالية للقياس، يتم استرجاع الموجة التنبؤية بالتشابك:
   $$ \\mathbf{w}_{\\text{projected}} = \\mathbf{E}_{AB} \\mathbf{v}_{\\text{curr}} $$

4. **معادلة تقييم الكيوبايت للمرشح:**
   تُقاس محاذاة الكلمة المرشحة مع الموجات الثلاث:
   $$ S_{\\text{projected}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{projected}})}{N} $$
   $$ S_{\\text{constructive}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{constructive}})}{N} $$
   $$ S_{\\text{destructive}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{destructive}})}{N} $$
   وتُعطى درجة التسجيل النهائية للكيوبايت بالعلاقة:
   $$ Score_{\\text{qubit}} = 4.0 S_{\\text{projected}} + 2.0 S_{\\text{constructive}} - 3.0 S_{\\text{destructive}} $$
   حيث يُخصم التداخل الهدام لمنع الكلمات المتناقضة سياقياً، ويُعزز التداخل البناء والتشابك لتوجيه المعنى بدقة فيزيائية متناهية."""

    new_sec19 = """## 19. محرك المتجه الموجي والاقتران الطوري التناظري (Analog Wave Oscillator & Coupling Engine)

يعتمد هذا المحرك على تصميم حوسبة موجية تناظرية مستمرة كبديل متماسك للحوسبة الرقمية التقليدية. يمثل المتجه الموجي المركب كحزمة موجية متراكبة ومتداخلة في فضاء طوري كثيف ذي أبعاد $N = 10,000$.

### الخوارزمية الرياضية

1. **انشطار الحزمة (Wave Splitting):**
   تنقسم الموجة الطورية المركبة $\\mathbf{v}$ إلى مركبتين متعامدتين ونظير معاكس (موجة مضادة/هدامة) وفق معادلة مجزئ الحزمة:
   $$ \\mathbf{v}_1 = \\cos(\\theta) \\mathbf{v} $$
   $$ \\mathbf{v}_2 = \\sin(\\theta) \\mathbf{v} \\odot e^{i \\phi_{\\text{shift}}} $$
   $$ \\mathbf{v}_{\\text{anti}} = -\\mathbf{v}_1 = \\mathbf{v}_1 \\odot e^{i \\pi} $$
   حيث يحقق النظام انحفاظ الطاقة الموجية: $\|\\mathbf{v}_1\|^2 + \|\\mathbf{v}_2\|^2 = \|\\mathbf{v}\|^2$.

2. **دمج وتداخل الموجات (Wave Merging):**
   تندمج موجتان $\\mathbf{v}_1$ و $\\mathbf{v}_2$ مع سياق التوليد بفرق طور $\\Delta \\phi$:
   $$ \\mathbf{w}_{\\text{merged}} = \\frac{\\mathbf{v}_1 + \\mathbf{v}_2 \\odot e^{i \\Delta \\phi}}{\|\\mathbf{v}_1 + \\mathbf{v}_2 \\odot e^{i \\Delta \\phi}\|} $$
   حيث تعطي القيمة $\\Delta \\phi = 0$ تداخلاً بناءً تاماً، وتعطي $\\Delta \\phi = \\pi$ تداخلاً هداماً متضاداً تاماً.

3. **الاقتران والقياس الطوري الهولوغرافي (Phase Coupling & Measurement):**
   تُقرن الموجة الناتجة من التداخل البناء $\\mathbf{w}_{\\text{constructive}}$ مع حقل موجة المثير الإجمالية $\\mathbf{w}_{\\text{prompt}}$ عبر جداء خارجي هولوغرافي:
   $$ \\mathbf{E}_{AB} = \\mathbf{w}_{\\text{constructive}} \\otimes \\mathbf{w}_{\\text{prompt}}^\\dagger $$
   وعند تمرير موجة السياق الحالية للقياس، يتم استرجاع الموجة التنبؤية بالاقتران الطوري:
   $$ \\mathbf{w}_{\\text{projected}} = \\mathbf{E}_{AB} \\mathbf{v}_{\\text{curr}} $$

4. **معادلة تقييم المتجه الموجي للمرشح:**
   تُقاس محاذاة الكلمة المرشحة مع الموجات الثلاث:
   $$ S_{\\text{projected}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{projected}})}{N} $$
   $$ S_{\\text{constructive}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{constructive}})}{N} $$
   $$ S_{\\text{destructive}} = \\frac{\\text{Re}(\\mathbf{v}_{\\text{cand}}^\\dagger \\cdot \\mathbf{w}_{\\text{destructive}})}{N} $$
   وتُعطى درجة التسجيل النهائية للمتجه الموجي بالعلاقة:
   $$ Score_{\\text{oscillator}} = 4.0 S_{\\text{projected}} + 2.0 S_{\\text{constructive}} - 3.0 S_{\\text{destructive}} $$
   حيث يُخصم التداخل الهدام لمنع الكلمات المتناقضة سياقياً، ويُعزز التداخل البناء والاقتران لتوجيه المعنى بدقة فيزيائية متناهية."""

    content = content.replace(old_sec19.replace("\n", "\r\n"), new_sec19.replace("\n", "\r\n"))
    content = content.replace(old_sec19, new_sec19)

    # 4. Replace other occurrences of quantum terminology in ALGORITHMS.md
    content = content.replace("الكيوبايت", "المذبذب الموجي")
    content = content.replace("التشابك", "الاقتران الطوري")
    content = content.replace("الكمي", "المركب")
    content = content.replace("كمي", "مركب")
    content = content.replace("quantum", "phase")
    content = content.replace("qubit", "oscillator")
    content = content.replace("entangle", "couple")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print("ALGORITHMS.md cleaned.")

def clean_file(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for old, new in replacements:
        content = content.replace(old, new)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"{path} cleaned.")

if __name__ == "__main__":
    clean_algorithms()

    # clean ARCHITECTURE.md
    clean_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\docs\ARCHITECTURE.md",
        [
            ("QuantumEntanglementEngine", "PhaseCouplingEngine"),
            ("WaveQubit", "WaveOscillator"),
            ("فوضى حرارية وتداخل موجي واقتران طوري كمتجه موجي مركب", "فوضى حرارية وتداخل موجي واقتران طوري كمتجه طوري مركب"),
            ("المتجه الموجي المركب (WaveQubit / Continuous State Vector)", "المتجه الموجي المركب (WaveOscillator / Continuous State Vector)"),
            ("الاقتران والقياس الطوري الهولوغرافي (entangle_waves & measure_entanglement)", "الاقتران والقياس الطوري الهولوغرافي (couple_waves & project_coupled_wave)"),
            ("entangle_waves & measure_entanglement", "couple_waves & project_coupled_wave"),
            ("الاقتران الطوري الهولوغرافي", "الاقتران الطوري الهولوغرافي"),
            ("chaos_and_entanglement.jl", "chaos_and_entanglement.jl"),
            ("density_matrix", "density_matrix"),
            ("QuantumDensityMatrix", "PhaseDensityMatrix"),
            ("مصفوفة كثافة", "مصفوفة الكثافة الطورية"),
            ("حوسبة موجية تناظرية واقتران طوري هولوغرافي", "حوسبة موجية تناظرية واقتران طوري هولوغرافي"),
        ]
    )

    # clean API_REFERENCE.md
    clean_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\docs\API_REFERENCE.md",
        [
            ("QuantumDensityMatrix", "PhaseDensityMatrix"),
            ("get_entanglement_entropy", "get_spectral_entropy"),
            ("entangle", "wave_coupling"),
        ]
    )

    # clean EXAMPLES.md
    clean_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\docs\EXAMPLES.md",
        [
            ("QuantumDensityMatrix", "PhaseDensityMatrix"),
            ("get_entanglement_entropy", "get_spectral_entropy"),
        ]
    )

    # clean INDEX.md
    clean_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\docs\INDEX.md",
        [
            ("chaos_and_entanglement.jl", "chaos_and_entanglement.jl"),
        ]
    )
