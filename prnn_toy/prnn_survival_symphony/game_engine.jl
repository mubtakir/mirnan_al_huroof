"""
🎮 Survival Symphony Game Engine Module
======================================
إدارة حالة وحركة عناصر اللعبة (اللاعب، الكائنات المعادية، الجسيمات البصرية).
يتعامل هذا الملف مع تفاعلات الإفناء الرنيني والتصادمات وتحديث النقاط.
"""
module GameEngine

using Random, LinearAlgebra

export Enemy, Player, Particle, GameState, spawn_enemy!, update_game!, init_game

mutable struct Enemy
    id::Int
    distance::Float64  # المسافة من المركز (0.0 إلى 1.5)
    angle::Float64     # الزاوية بالراديان (0 إلى 2pi)
    frequency::Float64 # التردد المفهومي للعدو (0.1 إلى 1.0)
    phase::Float64     # طور النبض الحالي (0 إلى 2pi)
    health::Float64    # الصحة (0.0 إلى 100.0)
    color::Symbol      # لون العدو للعرض البصري (:red, :yellow, :magenta, :green)
    morph_timer::Int   # عداد تغيير التردد (للمموّه Chameleon فقط)
    is_chameleon::Bool # هل هذا العدو متغير التردد؟
end

mutable struct Player
    frequency::Float64   # التردد الحالي للاعب (0.1 إلى 1.0)
    phase::Float64       # الطور الحالي للاعب (0 إلى 2pi)
    shield_active::Bool  # هل الدرع الرنيني نشط؟
    shield_energy::Float64 # شحن الدرع الحالي (0 إلى 100.0)
    health::Float64      # صحة اللاعب/القلعة (0 إلى 100.0)
    score::Int           # النقاط المسجلة
    autopilot::Bool      # هل الطيار الآلي للـ PRNN مفعل؟
end

mutable struct Particle
    x::Float64
    y::Float64
    vx::Float64
    vy::Float64
    life::Int          # عمر الجسيم بالإطارات
    char::String       # شكل الجسيم البصري
    color::Symbol
end

mutable struct GameState
    enemies::Vector{Enemy}
    player::Player
    particles::Vector{Particle}
    enemy_id_counter::Int
    spawn_timer::Int
    spawn_interval::Int
    wave_difficulty::Int
    report_active::Bool  # عرض تقرير أداء الشبكة
    credits_active::Bool # عرض شاشة Credits
end

"""
تهيئة الحالة البدئية للعبة.
"""
function init_game()
    enemies = Enemy[]
    player = Player(0.3, 0.0, false, 100.0, 100.0, 0, true)
    particles = Particle[]
    return GameState(enemies, player, particles, 1, 0, 45, 1, false, false)
end

"""
توليد كائن معادٍ جديد على حافة الرادار بتردد عشوائي.
"""
function spawn_enemy!(state::GameState)
    state.enemy_id_counter += 1
    
    # 10% احتمال ظهور عدو مموّه متغير التردد (Chameleon) إذا زادت الصعوبة عن 3
    if state.wave_difficulty >= 3 && rand() < 0.1
        angle = rand() * 2 * pi
        push!(state.enemies, Enemy(state.enemy_id_counter, 1.3, angle, 0.35, rand() * 2 * pi, 100.0, :green, 0, true))
        return
    end
    
    freq_options = [0.2, 0.5, 0.8]
    freq = rand(freq_options)
    color = freq == 0.2 ? :red : (freq == 0.5 ? :yellow : :magenta)
    
    angle = rand() * 2 * pi
    push!(state.enemies, Enemy(state.enemy_id_counter, 1.3, angle, freq, rand() * 2 * pi, 100.0, color, 0, false))
end

"""
توليد تأثير جسيمات انفجارية في الطرفية عند إفناء عدو.
"""
function spawn_explosion!(state::GameState, r::Float64, theta::Float64, color::Symbol)
    # تحويل من قطبي إلى كارتيزي لرسم الجسيمات
    cx = r * cos(theta) * 20.0 # معامل تصحيح لتناسب الطول والعرض بالطرفية
    cy = r * sin(theta) * 10.0
    
    chars = ["*", ".", "+", "o", "x"]
    for _ in 1:12
        vx = (rand() - 0.5) * 4.0
        vy = (rand() - 0.5) * 2.0
        life = rand(8:15)
        char = rand(chars)
        push!(state.particles, Particle(cx, cy, vx, vy, life, char, color))
    end
end

"""
تحديث اللعبة لخطوة زمنية واحدة dt.
"""
function update_game!(state::GameState, dt::Float64)
    player = state.player
    
    # 1. تحديث طور اللاعب
    player.phase = mod2pi(player.phase + 2 * pi * player.frequency * dt)
    
    # 2. إدارة شحن واستهلاك الدرع
    if player.shield_active
        player.shield_energy = max(0.0, player.shield_energy - 25.0 * dt)
        if player.shield_energy <= 0.0
            player.shield_active = false
        end
    else
        player.shield_energy = min(100.0, player.shield_energy + 10.0 * dt)
    end
    
    # 3. تحديث الكائنات المعادية
    active_enemies = Enemy[]
    for enemy in state.enemies
        # حركة بطيئة نحو المركز (بسقف لمنع التسارع اللانهائي)
        speed = min(0.35, 0.08 + (state.wave_difficulty * 0.01))
        
        # إذا كان تردد اللاعب بعيداً جداً عن التردد المستهدف، يسرع العدو نحوه عقاباً!
        freq_diff = abs(player.frequency - enemy.frequency)
        if freq_diff > 0.4 * 0.8 # 40% من مدى الترددات الكلي (0.8)
            speed *= 1.5
        end
        
        enemy.distance -= speed * dt
        enemy.phase = mod2pi(enemy.phase + 2 * pi * enemy.frequency * dt)
        
        # إذا كان العدو من نوع Chameleon، يغير تردده كل 3 ثوان (180 إطار)
        if enemy.is_chameleon
            enemy.morph_timer += 1
            if enemy.morph_timer >= 180
                enemy.morph_timer = 0
                new_freqs = [f for f in [0.2, 0.5, 0.8] if f != enemy.frequency]
                enemy.frequency = rand(new_freqs)
                enemy.color = enemy.frequency == 0.2 ? :red : (enemy.frequency == 0.5 ? :yellow : :magenta)
            end
        end
        
        # كشف التصادم مع اللاعب في المركز
        if enemy.distance <= 0.18
            # اصطدام! إلحاق الضرر باللاعب
            damage = 15.0
            if player.shield_active && freq_diff <= 0.1
                # الدرع يحمي إذا كان هناك تقارب ترددي نسبي
                damage = 2.0
            elseif player.shield_active
                damage = 7.0
            end
            player.health = max(0.0, player.health - damage)
            
            # جسيمات حمراء للتصادم
            spawn_explosion!(state, enemy.distance, enemy.angle, :red)
            continue # إزالة العدو من قائمة النشطين
        end
        
        # تفاعل الرنين والإفناء (مع smooth transition حول الحد 0.08)
        if player.shield_active && freq_diff <= 0.12
            interference = cos(player.phase - enemy.phase)
            # smooth decay: full damage at 0, zero at 0.12
            freq_factor = max(0.0, 1.0 - freq_diff / 0.12) ^ 2
            damage_rate = 80.0 * freq_factor
            
            if interference > 0.0
                damage_rate *= (1.0 + 1.0 * interference)
            else
                damage_rate *= (1.0 + 0.3 * interference)
            end
            
            enemy.health -= damage_rate * dt
            
            # شرارات جسيمات رنينية متوهجة حول العدو
            if rand() < 0.4
                spawn_explosion!(state, enemy.distance, enemy.angle, enemy.color)
            end
        end
        
        # فحص بقاء العدو حياً
        if enemy.health > 0.0
            push!(active_enemies, enemy)
        else
            # العدو مات بفعل الرنين!
            player.score += 10
            spawn_explosion!(state, enemy.distance, enemy.angle, enemy.color)
            
            # زيادة الصعوبة تدريجياً مع زيادة النقاط
            if player.score % 50 == 0
                state.wave_difficulty += 1
                state.spawn_interval = max(20, state.spawn_interval - 5)
            end
        end
    end
    state.enemies = active_enemies
    
    # 4. تحديث الجسيمات
    active_particles = Particle[]
    for p in state.particles
        p.x += p.vx
        p.y += p.vy
        p.life -= 1
        if p.life > 0
            push!(active_particles, p)
        end
    end
    state.particles = active_particles
    
    # 5. توليد الأعداء بشكل دوري
    state.spawn_timer += 1
    if state.spawn_timer >= state.spawn_interval
        state.spawn_timer = 0
        spawn_enemy!(state)
    end
end

end # module GameEngine
