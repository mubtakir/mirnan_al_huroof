def replace_in_file(path, replacements):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Apply replacements (both raw and CRLF variations)
    for old, new in replacements:
        content = content.replace(old.replace("\n", "\r\n"), new.replace("\n", "\r\n"))
        content = content.replace(old, new)
        
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Refactored: {path}")

if __name__ == "__main__":
    # 1. Refactor src/Mirnan.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\src\Mirnan.jl",
        [
            ("quantum mechanics,", "complex phase dynamics,"),
        ]
    )

    # 2. Refactor src/physics/density_matrix.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\src\physics\density_matrix.jl",
        [
            ("مصفوفة الكثافة الكمومية — Quantum Density Matrix.", "مصفوفة الكثافة الطورية — Phase Density Matrix."),
            ("export QuantumDensityMatrix, build!, resonance, get_trace, get_purity, get_entanglement_entropy",
             "export PhaseDensityMatrix, build!, resonance, get_trace, get_purity, get_spectral_entropy"),
            ("QuantumDensityMatrix", "PhaseDensityMatrix"),
            ("get_entanglement_entropy", "get_spectral_entropy"),
            ("ممثل السياق كمصفوفة كثافة كمومية.", "ممثل السياق كمصفوفة كثافة طورية."),
            ("إنتروبيا فون نيومان: S = -Tr(ρ ln ρ)", "إنتروبيا فون نيومان الطيفية: S = -Tr(ρ ln ρ)"),
        ]
    )

    # 3. Refactor src/physics/chaos_and_entanglement.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\src\physics\chaos_and_entanglement.jl",
        [
            ("Chaos, Entanglement, and Molecular Physics — ThermalChaos, QuantumEntanglement, MolecularBinder.",
             "Chaos, Phase Coupling, and Molecular Physics — ThermalChaos, PhaseCoupling, MolecularBinder."),
            ("export ThermalChaosEngine, QuantumEntanglementEngine, MolecularBinder,\n       WaveQubit, split_wave, merge_waves, entangle_waves, measure_entanglement",
             "export ThermalChaosEngine, PhaseCouplingEngine, MolecularBinder,\n       WaveOscillator, split_wave, merge_waves, couple_waves, project_coupled_wave"),
            ("mutable struct QuantumEntanglementEngine; threshold::Float64; super_gravity::Float64; end\nQuantumEntanglementEngine(; threshold=0.85, super_gravity=3.0) = QuantumEntanglementEngine(threshold, super_gravity)",
             "mutable struct PhaseCouplingEngine; threshold::Float64; super_gravity::Float64; end\nPhaseCouplingEngine(; threshold=0.85, super_gravity=3.0) = PhaseCouplingEngine(threshold, super_gravity)"),
            ("QuantumEntanglementEngine", "PhaseCouplingEngine"),
            ("# ═══ الكيوبايت الموجي التناظري والتشابك المستمر ═══", "# ═══ المذبذب الموجي التناظري والاقتران الطوري المستمر ═══"),
            ("mutable struct WaveQubit\n    psi::Vector{ComplexF64} # المتجه الموجي الكلي (N)\nend",
             "mutable struct WaveOscillator\n    psi::Vector{ComplexF64} # المتجه الطوري الكلي (N)\nend"),
            ("WaveQubit", "WaveOscillator"),
            ("entangle_waves", "couple_waves"),
            ("measure_entanglement", "project_coupled_wave"),
            ("توليد مصفوفة تشابك طوري هولوغرافي", "توليد مصفوفة اقتران طوري هولوغرافي"),
            ("قياس أو إسقاط الموجة المشتبكة عبر موجة اختبار معينة (تحليل التشابك):", "قياس أو إسقاط الموجة المقترنة عبر موجة اختبار معينة (تحليل الاقتران الطوري):"),
            ("W_out = E_ab * test_v", "W_out = C_ab * test_v"),
        ]
    )

    # 4. Refactor src/physics/generator.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\src\physics\generator.jl",
        [
            ("using ..ChaosEntanglement", "using ..ChaosEntanglement"), # keep module name
            ("density_matrix::QuantumDensityMatrix;", "density_matrix::PhaseDensityMatrix;"),
            ("quantum_ent::QuantumEntanglementEngine", "quantum_ent::PhaseCouplingEngine"),
            ("QuantumDensityMatrix(), CausalFlowField(),", "PhaseDensityMatrix(), CausalFlowField(),"),
            ("QuantumEntanglementEngine(),", "PhaseCouplingEngine(),"),
            ("density_matrix::QuantumDensityMatrix", "density_matrix::PhaseDensityMatrix"),
            ("quantum_ent::QuantumEntanglementEngine", "quantum_ent::PhaseCouplingEngine"),
            
            # Modes and helpers
            ("# ═══════════════════ QUANTUM MODE ═══════════════════", "# ═══════════════════ PHASE RESONANCE MODE ═══════════════════"),
            ("function _quantum_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=12)",
             "function _phase_resonance_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=12)"),
            ("_quantum_generate(gen, prompt_tokens; max_words=max_words)", "_phase_resonance_generate(gen, prompt_tokens; max_words=max_words)"),
            ("return _quantum_generate(gen, prompt_tokens; max_words=max_words)", "return _phase_resonance_generate(gen, prompt_tokens; max_words=max_words)"),
            
            # Entangle mode
            ("# ═══════════════════ ANALOG WAVE QUANTUM ENTANGLE MODE ═══════════════════",
             "# ═══════════════════ ANALOG WAVE COUPLING MODE ═══════════════════"),
            ("توليد النص باستخدام الحوسبة الموجية التناظرية والتشابك الكمي الطوري.\nيقوم بتمثيل سياق التوليد ككيوبايتات موجية مستمرة، حيث يتم شطر موجة الكلمة الحالية\nإلى أطوار بناءة وهدامة (متضادة) وتشبيكها هولوغرافياً مع موجة المثير (Prompt Wave)،\nثم فك التشابك لقياس وتحديد الكلمة التالية التي تحقق التداخل البناء وتجنب التداخل الهدام.",
             "توليد النص باستخدام الحوسبة الموجية التناظرية والاقتران الطوري.\nيقوم بتمثيل سياق التوليد كمذبذبات موجية مستمرة، حيث يتم شطر موجة الكلمة الحالية\nإلى أطوار بناءة وهدامة (متضادة) وإقرانها هولوغرافياً مع موجة المثير (Prompt Wave)،\nثم فك الاقتران لقياس وتحديد الكلمة التالية التي تحقق التداخل البناء وتجنب التداخل الهدام."),
            ("function _entangle_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=15)",
             "function _coupling_generate(gen::MirnanGenerator, prompt_tokens::Vector{String}; max_words=15)"),
            ("E = entangle_waves(w_constructive, prompt_wave)", "E = couple_waves(w_constructive, prompt_wave)"),
            ("w_projected = measure_entanglement(E, v_curr)", "w_projected = project_coupled_wave(E, v_curr)"),
            ("qubit_score = 4.0 * sim_projected + 2.0 * sim_constructive - 3.0 * sim_destructive",
             "qubit_score = 4.0 * sim_projected + 2.0 * sim_constructive - 3.0 * sim_destructive"),
            ("total_score = s_std + qubit_score * 3.0", "total_score = s_std + qubit_score * 3.0"),
            ("return _entangle_generate(gen, prompt_tokens; max_words=max_words)", "return _coupling_generate(gen, prompt_tokens; max_words=max_words)"),
            
            # generate! mode routing
            ("elseif mode == \"quantum\"\n        return _quantum_generate(gen, prompt_tokens; max_words=max_words)",
             "elseif mode == \"quantum\" || mode == \"phase_resonance\"\n        return _phase_resonance_generate(gen, prompt_tokens; max_words=max_words)"),
            ("elseif mode == \"entangle\"\n        return _entangle_generate(gen, prompt_tokens; max_words=max_words)",
             "elseif mode == \"entangle\" || mode == \"wave_coupling\" || mode == \"coupling\"\n        return _coupling_generate(gen, prompt_tokens; max_words=max_words)"),
        ]
    )

    # 5. Refactor src/api/server.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\src\api\server.jl",
        [
            ("[\"auto\", \"dialogue\", \"creative\", \"wave\", \"standard\", \"quantum\", \"multiverse\", \"poetic\", \"code\", \"math\", \"physics\", \"attract\", \"synthesize\"]",
             "[\"auto\", \"dialogue\", \"creative\", \"wave\", \"standard\", \"quantum\", \"phase_resonance\", \"multiverse\", \"poetic\", \"code\", \"math\", \"physics\", \"attract\", \"synthesize\"]"),
            ("مصفوفة كثافة كمومية (DM)", "مصفوفة الكثافة الطورية (DM)"),
        ]
    )

    # 6. Refactor ui/index.html
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\ui\index.html",
        [
            ("<option value=\"quantum\">Quantum Superposition</option>",
             "<option value=\"quantum\">Phase Superposition (تراكب طوري مركب)</option>"),
            ("<option value=\"entangle\">Entangle (تشابك كيوبايت موجي)</option>",
             "<option value=\"entangle\">Wave Coupling (ترابط طوري موجي)</option>"),
            ("مصفوفة كثافة كمومية (DM)", "مصفوفة الكثافة الطورية (DM)"),
        ]
    )

    # 7. Refactor test/runtests.jl
    replace_in_file(
        r"C:\Users\allmy\Desktop\aaa\mirnan_julia\test\runtests.jl",
        [
            ("dm = Mirnan.Physics.DensityMatrix.QuantumDensityMatrix()",
             "dm = Mirnan.Physics.DensityMatrix.PhaseDensityMatrix()"),
            ("get_entanglement_entropy(dm)", "get_spectral_entropy(dm)"),
            ("@testset \"Continuous Wave Qubit & Entanglement Physics\" begin",
             "@testset \"Continuous Wave Oscillator & Phase Coupling Physics\" begin"),
            ("split_wave(qubit, theta, phi_shift)", "split_wave(WaveOscillator(v_test), theta, phi_shift)"), # fix structural type
            ("split_wave(v_test, theta, phi_shift)", "split_wave(v_test, theta, phi_shift)"),
            ("split_wave(qubit::WaveQubit,", "split_wave(qubit::WaveOscillator,"),
            ("WaveQubit", "WaveOscillator"),
            ("entangle_waves", "couple_waves"),
            ("measure_entanglement", "project_coupled_wave"),
            ("E = Mirnan.Physics.ChaosEntanglement.entangle_waves(v_a, v_b)",
             "E = Mirnan.Physics.ChaosEntanglement.couple_waves(v_a, v_b)"),
            ("v_measured = Mirnan.Physics.ChaosEntanglement.measure_entanglement(E, v_b)",
             "v_measured = Mirnan.Physics.ChaosEntanglement.project_coupled_wave(E, v_b)"),
        ]
    )
