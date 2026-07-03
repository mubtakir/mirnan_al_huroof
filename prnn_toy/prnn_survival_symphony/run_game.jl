"""
🎮 Survival Symphony — Main Game Loop & Terminal UI
===================================================
الملف الرئيسي لتشغيل اللعبة. يربط واجهة المستخدم للطرفية (ASCII GUI)
بمحرك اللعبة والنموذج الرياضي للـ PRNN لتمكين الطيار الآلي والتعلم التلقائي.

طريقة اللعب:
- A / D : تعديل التردد يدوياً
- Space : تفعيل الدرع الرنيني
- M : تبديل وضع التحكم (يدوي / طيار آلي)
- Q : خروج
"""

using Pkg
# تفعيل بيئة مشروع مرنان الرئيسي
Pkg.activate(joinpath(dirname(dirname(@__DIR__))))

include("prnn_model.jl")
include("game_engine.jl")

using .PRNNModel
using .GameEngine
using REPL
using Random
using LinearAlgebra
using Printf, Statistics, Dates, Serialization

# دالة رسم سهم الطور بناء على زاوية الطور
function get_phase_arrow(phase::Float64)
    # حصر الزاوية بين -pi و pi
    p = atan(sin(phase), cos(phase))
    if -pi/8 <= p < pi/8
        return "→"
    elseif pi/8 <= p < 3pi/8
        return "↗"
    elseif 3pi/8 <= p < 5pi/8
        return "↑"
    elseif 5pi/8 <= p < 7pi/8
        return "↖"
    elseif p >= 7pi/8 || p < -7pi/8
        return "←"
    elseif -7pi/8 <= p < -5pi/8
        return "↙"
    elseif -5pi/8 <= p < -3pi/8
        return "↓"
    else
        return "↘"
    end
end

# أكواد الألوان للطرفية (ANSI Escape Codes)
const RESET   = "\e[0m"
const RED     = "\e[38;5;196m"
const GREEN   = "\e[38;5;46m"
const YELLOW  = "\e[38;5;226m"
const BLUE    = "\e[38;5;45m"
const MAGENTA = "\e[38;5;201m"
const CYAN    = "\e[38;5;51m"
const WHITE   = "\e[38;5;231m"
const GRAY    = "\e[38;5;244m"
const BG_DARK = "\e[48;5;232m"

# دالة تلوين النصوص
color_str(str::String, color::Symbol) = begin
    if color == :red return RED * str * RESET
    elseif color == :green return GREEN * str * RESET
    elseif color == :yellow return YELLOW * str * RESET
    elseif color == :blue return BLUE * str * RESET
    elseif color == :magenta return MAGENTA * str * RESET
    elseif color == :cyan return CYAN * str * RESET
    elseif color == :gray return GRAY * str * RESET
    else return WHITE * str * RESET
    end
end

"""
بناء ورسم إطار واجهة الطرفية بالكامل كـ String واحد لتجنب اهتزاز الشاشة (Flicker-free Rendering).
"""
function draw_screen(state::GameState, nn::PRNN)
    buf = IOBuffer()
    
    # 1. تنظيف الشاشة ونقل المؤشر للأعلى
    print(buf, "\e[H")
    
    # 2. رسم العنوان
    title = " 🧠 SURVIVAL SYMPHONY — سيمفونية النجاة (PRNN)"
    println(buf, color_str("+" * "-"^78 * "+", :cyan))
    println(buf, color_str("|", :cyan) * rpad(title, 78) * color_str("|", :cyan))
    println(buf, color_str("+" * "-"^78 * "+", :cyan))
    
    # 3. إعداد رادار اللعبة (21 عرض × 11 ارتفاع)
    radar_w, radar_h = 23, 11
    grid = fill(" ", radar_h, radar_w)
    
    # رسم درع اللاعب حول المركز إذا كان نشطاً
    cx, cy = 12, 6
    if state.player.shield_active
        # رسم دائرة درع صغيرة
        for (dx, dy) in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
            grid[cy+dy, cx+dx] = color_str("o", :blue)
        end
    end
    
    # رسم اللاعب في المركز
    grid[cy, cx] = state.player.shield_active ? color_str("@", :cyan) : color_str("P", :cyan)
    
    # رسم الكائنات المعادية في الرادار
    for enemy in state.enemies
        ex = cx + round(Int, enemy.distance * cos(enemy.angle) * 8.0)
        ey = cy + round(Int, enemy.distance * sin(enemy.angle) * 4.0)
        
        # حصر الإحداثيات داخل الرادار
        ex = clamp(ex, 2, radar_w - 1)
        ey = clamp(ey, 2, radar_h - 1)
        
        symbol = enemy.is_chameleon ? "C" : (enemy.frequency == 0.2 ? "2" : (enemy.frequency == 0.5 ? "5" : "8"))
        grid[ey, ex] = color_str(symbol, enemy.color)
    end
    
    # رسم الجسيمات البصرية
    for p in state.particles
        px = cx + round(Int, p.x)
        py = cy + round(Int, p.y)
        if 1 <= px <= radar_w && 1 <= py <= radar_h
            grid[py, px] = color_str(p.char, p.color)
        end
    end
    
    # 4. إعداد جدول حالات عصبونات PRNN
    # العقد: 1-3 مستشعرات | 4 تحيز | 5 تردد اللاعب | 6-8 خفية | 9 مخرج تردد | 10 مخرج درع
    node_names = [
        "S1: Threat Dist  ",
        "S2: Freq Diff 1  ",
        "S3: Freq Diff 2  ",
        "B4: Reference Bias",
        "I5: Player Freq  ",
        "H6: Hidden State ",
        "H7: Hidden State ",
        "H8: Hidden State ",
        "O9: Freq Control ",
        "O10: Shield Out  "
    ]
    
    # 5. تجميع الرادار وجدول العصبونات جنباً إلى جنب
    for y in 1:radar_h
        # رسم سطر الرادار
        row_str = ""
        for x in 1:radar_w
            row_str *= grid[y, x]
        end
        
        # رسم سطر العصبون المقابل
        node_str = ""
        if y <= 10
            name = node_names[y]
            amp = abs(nn.z[y])
            amp = isnan(amp) ? 0.0 : amp
            phase = angle(nn.z[y])
            phase = isnan(phase) ? 0.0 : phase
            arrow = get_phase_arrow(phase)
            
            # تمثيل السعة كشريط صغير
            bar_len = clamp(round(Int, amp * 5), 0, 10)
            bar = "█"^bar_len * "░"^(10 - bar_len)
            
            # تلوين مختلف للطبقات
            col = y <= 3 ? :red : (y == 4 ? :blue : (y == 5 ? :green : (y <= 8 ? :yellow : :cyan)))
            
            # صياغة النص
            node_str = color_str("[$y] $name", col) * 
                       " | Amp: " * bar * " ($(round(amp, digits=2)))" *
                       " | Phase: $arrow ($(round(phase/pi, digits=2))π)"
        else
            # السطر الأخير يعرض أعلى الأوزان اقتراناً
            top_weights = Tuple{Int,Int,Float64}[]
            for i in 1:nn.N
                for j in (i+1):nn.N
                    if PRNNModel.is_connected(i, j) && abs(nn.K[i, j]) > 0.15
                        push!(top_weights, (i, j, nn.K[i, j]))
                    end
                end
            end
            sort!(top_weights, by = x -> abs(x[3]), rev=true)
            
            weight_displays = String[]
            for k in 1:min(2, length(top_weights))
                w = top_weights[k]
                push!(weight_displays, "[$(w[1])⚡$(w[2]): $(round(w[3], digits=2))]")
            end
            node_str = color_str("🔗 Top Links: ", :cyan) * join(weight_displays, " ")
        end
        
        # طباعة السطر المجمع
        println(buf, color_str("| ", :cyan) * row_str * color_str(" |  ", :cyan) * node_str)
    end
    
    # رسم خط الفاصل للرادار
    println(buf, color_str("+" * "-"^25 * "+", :cyan))
    
    # 6. لوحة التحكم والإحصائيات
    player = state.player
    health_bar = "█"^round(Int, player.health/10) * "░"^(10 - round(Int, player.health/10))
    energy_bar = "█"^round(Int, player.shield_energy/10) * "░"^(10 - round(Int, player.shield_energy/10))
    
    mode_str = player.autopilot ? color_str("AUTOPILOT (الطيار الآلي)", :cyan) : color_str("MANUAL (تحكم يدوي)", :yellow)
    shield_status = player.shield_active ? color_str("ACTIVE", :blue) : color_str("INACTIVE", :gray)
    
    @printf(buf, "❤️ Health: [%s] %3.0f%%  |  🛡️ Shield: [%s] %3.0f%% (%s)\n", 
            color_str(health_bar, player.health > 40 ? :green : :red), player.health,
            color_str(energy_bar, :blue), player.shield_energy, shield_status)
            
    learn_mode_str = nn.learning_mode == :contrastive ? color_str("CHL (تبايني)", :green) : color_str("Hebbian (معزز)", :magenta)
    @printf(buf, "📊 Score: %-5d | Diff: %-2d | Player Freq: %4.2f Hz | Control: %s | Learn: %s\n", 
            player.score, state.wave_difficulty, player.frequency, mode_str, learn_mode_str)
            
    println(buf, color_str("-"^80, :gray))
    println(buf, color_str("⌨️ [A/D]: Freq | [Space]: Shield | [M]: Autoplay | [T]: Learn | [R]: Report | [C]: Credits | [S]: Save | [L]: Load | [Q]: Exit", :gray))
    
    # ── تقرير أداء الشبكة (عند الطلب) ──
    # ── تقرير أداء الشبكة (عند الطلب) ──
    if state.report_active
        println(buf, color_str("═"^80, :cyan))
        println(buf, color_str("📋 NETWORK PERFORMANCE REPORT — تقرير أداء الشبكة", :cyan))
        println(buf, color_str("─"^80, :gray))
        
        sensor_amp = mean(abs.(nn.z[1:3]))
        hidden_amp = mean(abs.(nn.z[6:8]))
        output_amp = mean(abs.(nn.z[9:10]))
        println(buf, @sprintf("  📈 Avg |z| — Sensors: %.2f | Hidden: %.2f | Outputs: %.2f", sensor_amp, hidden_amp, output_amp))
        
        if !isempty(state.enemies)
            closest = sort(state.enemies, by = e -> e.distance)[1]
            phase_out = angle(nn.z[9])
            pred_freq = clamp(abs(phase_out) / pi, 0.1, 1.0)
            freq_err = abs(pred_freq - closest.frequency)
            match_pct = max(0.0, (1.0 - freq_err / 0.8) * 100)
            println(buf, @sprintf("  🎯 Freq target: %.2f Hz | Pred: %.2f Hz | Error: %.3f | Match: %.0f%%",
                    closest.frequency, pred_freq, freq_err, match_pct))
        end
        
        shield_amp = abs(nn.z[10])
        println(buf, @sprintf("  🛡️ Shield amp: %.2f | Decision: %s", shield_amp,
                shield_amp > 0.8 ? "ON ✓" : (shield_amp < 0.4 ? "OFF ✓" : "BORDERLINE ⚠")))
        
        top_w = Tuple{Int,Int,Float64}[]
        for i in 1:nn.N, j in (i+1):nn.N
            if PRNNModel.is_connected(i, j) && abs(nn.K[i,j]) > 0.1
                push!(top_w, (i, j, nn.K[i,j]))
            end
        end
        sort!(top_w, by = x -> abs(x[3]), rev=true)
        println(buf, color_str("  🔗 Top weights:", :cyan))
        for k in 1:min(3, length(top_w))
            w = top_w[k]
            arrow = w[3] > 0 ? "→" : "←"
            println(buf, @sprintf("    [%d⚡%d]  K=%.3f %s", w[1], w[2], w[3], arrow))
        end
        
        println(buf, color_str("═"^80, :cyan))
    end
    
    # ── شاشة Credits (عند الطلب) ──
    if state.credits_active
        println(buf, color_str("═"^80, :cyan))
        println(buf, color_str("📜 PRNN SURVIVAL SYMPHONY — Technical Credits", :cyan))
        println(buf, color_str("─"^80, :gray))
        println(buf, "  🧠 Dynamics:   Stuart-Landau Complex Oscillators")
        println(buf, "  📐 Equation:   dz/dt = (μ - a - g_inh·⟨|z|²⟩ - |z|² + iω)·z + ΣK·(zⱼ - zᵢ)")
        println(buf, "  ⚙️ Parameters: μ=$(nn.mu)  g_inh=$(nn.g_inh)  γ=$(nn.gamma)  τ_a=$(nn.tau_a)")
        println(buf, "  📚 Learning:   Contrastive Hebbian (CHL) + Reward-modulated Hebbian")
        println(buf, "  🌐 Topology:   $(nn.N) oscillators (3 sensors → 3 hidden → 2 outputs + bias + freq)")
        println(buf)
        println(buf, color_str("  🔬 Resonance Combat: cos(Δφ) modulates damage rate", :green))
        println(buf, "     Constructive interference (aligned phases) → damage × $(1.0 + 1.0)")
        println(buf, "     Destructive interference (opposite phases) → damage × $(1.0 - 0.3)")
        println(buf)
        println(buf, color_str("  🏆 Built with ❤️ using Julia + PRNNModel.jl", :yellow))
        println(buf, color_str("═"^80, :cyan))
    end
    
    # طباعة المحتوى بالكامل للطرفية دفعة واحدة
    print(stdout, String(take!(buf)))
end

"""
دالة التشغيل الرئيسية للعبة.
"""
function play_symphony()
    # 1. إعدادات الطرفية ونظام التشغيل (كشف تلقائي)
    term_type = Sys.iswindows() ? "vt100" : "xterm"
    term = REPL.Terminals.TTYTerminal(term_type, stdin, stdout, stderr)
    
    # إخفاء المؤشر وتنظيف الشاشة
    print(stdout, "\e[?25l\e[2J")
    
    # تفعيل الوضع الخام (Raw mode) للحصول على مدخلات لوحة المفاتيح مباشرة
    REPL.Terminals.raw!(term, true)
    
    # إنشاء قناة لاستقبال الأحرف المدخلة بدون حجب
    input_channel = Channel{Char}(100)
    
    @async begin
        try
            while true
                c = read(stdin, Char)
                put!(input_channel, c)
                if c == 'q' || c == 'Q'
                    break
                end
            end
        catch e
            # تجاهل أخطاء إغلاق القناة عند الإنهاء
        finally
            close(input_channel)
        end
    end
    
    # 2. تهيئة اللعبة والشبكة
    state = init_game()
    nn = PRNN(lr=0.07) # تهيئة شبكة PRNN
    
    # توليد الكائن الأول فوراً
    spawn_enemy!(state)
    
    # متغيرات Benchmark
    benchmark_file = joinpath(@__DIR__, "scores.csv")
    if !isfile(benchmark_file)
        open(benchmark_file, "w") do f
            write(f, "timestamp,score,wave,mode\n")
        end
    end
    
    dt = 0.05
    running = true
    
    try
        while running && state.player.health > 0
            t_frame_start = time_ns()
            
            # 1. فحص وقراءة المدخلات المتاحة في القناة بدون حجب
            while isready(input_channel)
                key = take!(input_channel)
                if key == 'q' || key == 'Q'
                    running = false
                elseif key == 'm' || key == 'M'
                    state.player.autopilot = !state.player.autopilot
                elseif key == 't' || key == 'T'
                    nn.learning_mode = nn.learning_mode == :contrastive ? :hebbian : :contrastive
                elseif key == 'r' || key == 'R'
                    state.report_active = !state.report_active
                    state.credits_active = false
                elseif key == 'c' || key == 'C'
                    state.credits_active = !state.credits_active
                    state.report_active = false
                elseif key == 's' || key == 'S'
                    w_file = joinpath(@__DIR__, "trained_weights_$(Dates.format(now(), "yyyymmdd_HHMMSS")).bin")
                    serialize(w_file, (nn.K, nn.omega))
                elseif key == 'l' || key == 'L'
                    w_files = filter(f -> startswith(f, "trained_weights_"), readdir(@__DIR__))
                    if !isempty(w_files)
                        latest = joinpath(@__DIR__, sort(w_files)[end])
                        K_loaded, omega_loaded = deserialize(latest)
                        nn.K .= K_loaded
                        nn.omega .= omega_loaded
                    end
                elseif key == ' '
                    if !state.player.shield_active && state.player.shield_energy > 15.0
                        state.player.shield_active = true
                    elseif state.player.shield_active
                        state.player.shield_active = false
                    end
                elseif !state.player.autopilot
                    # تحكم يدوي بالتردد
                    if key == 'a' || key == 'A'
                        state.player.frequency = max(0.1, state.player.frequency - 0.05)
                    elseif key == 'd' || key == 'D'
                        state.player.frequency = min(1.0, state.player.frequency + 0.05)
                    end
                end
            end
            
            if !running; break; end
            
            # 2. إعداد المدخلات الطورية والمخرجات المستهدفة للـ PRNN
            # ترتيب المداخل: 1 خطر، 2 فرق1، 3 فرق2، 4 تحيز، 5 تردد اللاعب
            clamped_nodes = [1, 2, 3, 4, 5]
            clamped_phases = Dict{Int, Float64}()
            
            # مستشعر الخطر (مسافة أقرب عدو)
            closest_enemy = nothing
            second_closest = nothing
            
            if !isempty(state.enemies)
                # ترتيب الأعداء حسب المسافة
                sorted_enemies = sort(state.enemies, by = e -> e.distance)
                closest_enemy = sorted_enemies[1]
                if length(sorted_enemies) > 1
                    second_closest = sorted_enemies[2]
                end
            end
            
            # تثبيت أطوار المدخلات
            clamped_phases[4] = 0.0 # التحيز دائماً طور 0
            clamped_phases[5] = state.player.frequency * pi # التردد الحالي للاعب
            
            if closest_enemy !== nothing
                # خطر المسافة
                clamped_phases[1] = (1.2 - clamp(closest_enemy.distance, 0.0, 1.2)) / 1.2 * pi
                # فرق التردد مع الأقرب
                clamped_phases[2] = abs(state.player.frequency - closest_enemy.frequency) * pi
            else
                clamped_phases[1] = 0.0
                clamped_phases[2] = 0.0
            end
            
            if second_closest !== nothing
                clamped_phases[3] = abs(state.player.frequency - second_closest.frequency) * pi
            else
                clamped_phases[3] = 0.0
            end
            
            # 3. حلقة تطور وتعلم الـ PRNN (اختيار نمط التعلم)
            # هدف التدريب: تطابق التردد مع أقرب عدو وتفعيل الدرع إذا كان الخطر كبيراً (< 0.65)
            target_freq = state.player.frequency
            target_shield = state.player.shield_active ? pi/2 : 0.0
            
            if closest_enemy !== nothing
                target_freq = closest_enemy.frequency
                target_shield = closest_enemy.distance < 0.65 ? pi/2 : 0.0
            end
            
            if nn.learning_mode == :contrastive
                # ── التعلم التبايني (CHL) ──
                # أ. تسجيل المرحلة السالبة (الحالة الحرة للمخرج)
                neg_state = save_state(nn)
                simulate_stuart_landau!(nn, clamped_nodes, clamped_phases, 40, 0.02; noise_level=0.005)
                z_neg = copy(nn.z)
                
                # ب. تسجيل المرحلة الموجبة (قفل المخارج 9 و 10 عند الأهداف الصحيحة)
                restore_state!(nn, neg_state)
                pos_clamped_nodes = [clamped_nodes..., 9, 10]
                pos_clamped_phases = copy(clamped_phases)
                pos_clamped_phases[9] = target_freq * pi
                pos_clamped_phases[10] = target_shield
                
                simulate_stuart_landau!(nn, pos_clamped_nodes, pos_clamped_phases, 40, 0.02; noise_level=0.005)
                z_pos = copy(nn.z)
                
                # ج. تحديث الأوزان
                update_contrastive!(nn, z_pos, z_neg)
                
                # د. استرجاع الحالة الحرة
                restore_state!(nn, neg_state)
                simulate_stuart_landau!(nn, clamped_nodes, clamped_phases, 40, 0.02; noise_level=0.005)
                
            else # :hebbian
                # ── التعلم الهيبي المعزز بالمكافأة ──
                neg_state = save_state(nn)
                simulate_stuart_landau!(nn, clamped_nodes, clamped_phases, 40, 0.02; noise_level=0.005)
                
                # حساب المكافأة اللحظية
                reward = 0.6  # أساسي للبقاء
                if closest_enemy !== nothing
                    danger = 1.0 - clamp(closest_enemy.distance, 0.0, 1.2) / 1.2
                    reward -= danger  # اقتراب الخطر يقلل المكافأة
                    is_correct_freq = abs(state.player.frequency - closest_enemy.frequency) <= 0.08
                    is_correct_shield = (closest_enemy.distance < 0.65) == state.player.shield_active
                    if is_correct_freq
                        reward += 0.5  # مكافأة مطابقة التردد
                    end
                    if is_correct_shield
                        reward += 0.3  # مكافأة الدرع الصحيح
                    end
                end
                update_reward_hebbian!(nn, reward)
                
                restore_state!(nn, neg_state)
                simulate_stuart_landau!(nn, clamped_nodes, clamped_phases, 40, 0.02; noise_level=0.005)
            end
            
            # 4. تطبيق قرارات الـ PRNN إذا كان وضع Autopilot نِشطاً
            if state.player.autopilot
                # استخلاص التردد المتوقع من مخرج الطور (العقدة 9)
                phase_out = angle(nn.z[9])
                pred_freq = abs(phase_out) / pi
                # حصر التردد المقاس
                state.player.frequency = clamp(pred_freq, 0.1, 1.0)
                
                # استخلاص قرار الدرع من سعة العقدة 10
                shield_amp = abs(nn.z[10])
                if shield_amp > 0.8 && state.player.shield_energy > 15.0
                    state.player.shield_active = true
                elseif shield_amp < 0.4
                    state.player.shield_active = false
                end
            end
            
            # 5. تحديث محرك اللعبة
            update_game!(state, dt)
            
            # 6. رسم الشاشة
            draw_screen(state, nn)
            
            # 7. الحفاظ على معدل إطارات ثابت (حوالي 20 إطار في الثانية)
            t_frame_end = time_ns()
            elapsed_ms = (t_frame_end - t_frame_start) / 1e6
            sleep_time = max(0.001, (50.0 - elapsed_ms) / 1000.0)
            sleep(sleep_time)
        end
        
        # 3. تسجيل Benchmark
        try
            open(benchmark_file, "a") do f
                ts = now()
                write(f, "$ts,$(state.player.score),$(state.wave_difficulty),$(nn.learning_mode)\n")
            end
            # حساب وعرض الإحصائيات
            lines = readlines(benchmark_file)
            if length(lines) > 1
                chl_scores = Int[]
                hebb_scores = Int[]
                for line in lines[2:end]
                    parts = split(line, ',')
                    if length(parts) >= 4
                        score = parse(Int, parts[2])
                        mode = strip(parts[4])
                        if mode == "contrastive"; push!(chl_scores, score)
                        else; push!(hebb_scores, score)
                        end
                    end
                end
                chl_avg = isempty(chl_scores) ? 0.0 : mean(chl_scores)
                hebb_avg = isempty(hebb_scores) ? 0.0 : mean(hebb_scores)
                
                println(color_str("📊 BENCHMARK (saved to scores.csv)", :cyan))
                println()
                if !isempty(chl_scores)
                    println("  Contrastive CHL ($(length(chl_scores)) games): avg = $(round(Int, chl_avg)) pts")
                end
                if !isempty(hebb_scores)
                    println("  Hebbian RL ($(length(hebb_scores)) games): avg = $(round(Int, hebb_avg)) pts")
                end
                println()
                if chl_avg > hebb_avg && hebb_avg > 0
                    println(color_str("  🏆 CHL يتفوق على Hebbian بنسبة $(round(chl_avg/hebb_avg, digits=2))x!", :green))
                elseif hebb_avg > chl_avg && chl_avg > 0
                    println(color_str("  🏆 Hebbian يتفوق على CHL بنسبة $(round(hebb_avg/chl_avg, digits=2))x!", :yellow))
                end
            end
        catch e
            # تجاهل أخطاء الملف
        end
        
        println()
        println("🎉 شكراً للعب سيمفونية النجاة الرنينية الطورية!")
        println(color_str("+" * "="^40 * "+", :red))
        println()
        
    finally
        # إغلاق الوضع الخام وإظهار مؤشر الطرفية مجدداً بشكل آمن
        REPL.Terminals.raw!(term, false)
        print(stdout, "\e[?25h\e[0m")
    end
end

# تشغيل فوري في حال استدعاء الملف مباشرة
if abspath(PROGRAM_FILE) == @__FILE__
    play_symphony()
end
