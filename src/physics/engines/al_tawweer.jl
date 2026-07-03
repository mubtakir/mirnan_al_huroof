"""
المحسّن التطوري للأوزان — Al-Tawweer (Evolutionary Weight Optimizer)
يركض على الكوربس لتوليد الأوزان الـ 24 لنموذج مِرنان وتحسينها تلقائياً.
"""
module AlTawweerModule

using Random
using LinearAlgebra
using ..Constants
using ..Generator: MirnanGenerator, generate!

export Chromosome, evolve_weights!, evaluate_fitness!

const WEIGHT_KEYS = [
    "align", "gravity", "prompt_align", "k_sem", "carrier", "resonant_chain", 
    "syntax", "diversity", "dccf", "causal_flow_align", "constraint_align", 
    "kb_knowledge", "ppm", "oscillator", "heterodyne", "repulsion", "causal", 
    "dialogue", "intent_align", "associative_plan", "plan_fidelity", "architect", 
    "trajectory", "context_tension"
]

mutable struct Chromosome
    weights::Dict{String, Float64}
    fitness::Float64
end

Chromosome(weights::Dict{String, Float64}) = Chromosome(weights, 0.0)

"""
    evaluate_fitness!(ind::Chromosome, gen::MirnanGenerator, prompts::Vector{String}) -> Float64

حساب اللياقة لكروموسوم أوزان محدد بتشغيله على مجموعة من الأسئلة والتحقق من نقاط المراجعة الذاتية.
"""
function evaluate_fitness!(ind::Chromosome, gen::MirnanGenerator, prompts::Vector{String})::Float64
    # حفظ الأوزان الحالية للمولد واستبدالها مؤقتاً بأوزان الكروموسوم
    original_weights = copy(gen.scoring_weights)
    for (k, v) in ind.weights
        gen.scoring_weights[k] = v
    end
    
    total_score = 0.0
    for prompt in prompts
        try
            ans = generate!(gen, prompt; mode="auto", max_words=12)
            score = 0.0
            if hasproperty(gen.self_review, :last_review) && gen.self_review.last_review !== nothing
                score = gen.self_review.last_review.score
            else
                score = 0.5 # افتراضي
            end
            
            # عقاب التكرار اللفظي المفرط في الإجابة
            words = split(ans)
            if !isempty(words)
                unique_ratio = length(unique(words)) / length(words)
                score *= (0.7 + 0.3 * unique_ratio)
            else
                score = 0.0 # إجابة فارغة
            end
            
            total_score += score
        catch e
            # فشل توليد
            total_score += 0.0
        end
    end
    
    # استرجاع الأوزان الأصلية للمولد
    for (k, v) in original_weights
        gen.scoring_weights[k] = v
    end
    
    ind.fitness = isempty(prompts) ? 0.0 : total_score / length(prompts)
    return ind.fitness
end

"""
    selection(population::Vector{Chromosome}, k::Int=3) -> Chromosome

انتخاب كروموسوم بطريقة البطولة (Tournament Selection).
"""
function selection(population::Vector{Chromosome}, k::Int=3)::Chromosome
    participants = rand(population, k)
    best = participants[1]
    for ind in participants[2:end]
        if ind.fitness > best.fitness
            best = ind
        end
    end
    return best
end

"""
    crossover(p1::Chromosome, p2::Chromosome) -> Tuple{Chromosome, Chromosome}

تزاوج جيني بالمزج الخطي بين كروموسومين أبويين لإنتاج نسلين جديدين.
"""
function crossover(p1::Chromosome, p2::Chromosome)::Tuple{Chromosome, Chromosome}
    w1 = Dict{String, Float64}()
    w2 = Dict{String, Float64}()
    
    alpha = rand()
    for key in WEIGHT_KEYS
        v1 = get(p1.weights, key, 1.0)
        v2 = get(p2.weights, key, 1.0)
        
        w1[key] = alpha * v1 + (1.0 - alpha) * v2
        w2[key] = (1.0 - alpha) * v1 + alpha * v2
    end
    
    return Chromosome(w1), Chromosome(w2)
end

"""
    mutate!(ind::Chromosome, pm::Float64=0.15, sigma::Float64=0.5)

إحداث طفرة جينية عشوائية بإضافة ضوضاء غاوسية للأوزان لضمان استكشاف الفضاء.
"""
function mutate!(ind::Chromosome, pm::Float64=0.15, sigma::Float64=0.5)
    for key in WEIGHT_KEYS
        if rand() < pm
            current = get(ind.weights, key, 1.0)
            noise = randn() * sigma
            # تقييد الأوزان بين 0.0 و 15.0 لضمان الاستقرار الفيزيائي
            ind.weights[key] = clamp(current + noise, 0.0, 15.0)
        end
    end
end

"""
    evolve_weights!(gen::MirnanGenerator, prompts::Vector{String}; generations::Int=5, pop_size::Int=8) -> Dict{String, Float64}

تشغيل عملية التطور الجيني لتحسين الأوزان الـ 24 للمولد وإرجاع الأوزان المتفوقة.
"""
function evolve_weights!(gen::MirnanGenerator, prompts::Vector{String}; generations::Int=5, pop_size::Int=8)::Dict{String, Float64}
    # 1. تهيئة المجتمع الأولي
    population = Chromosome[]
    
    # الكروموسوم الأول هو الأوزان الحالية للمولد كمرجع أساسي
    push!(population, Chromosome(copy(gen.scoring_weights)))
    
    # بقية المجتمع يتم توليدهم بإدخال اضطرابات عشوائية خفيفة على المرجع
    for _ in 2:pop_size
        mutated_weights = copy(gen.scoring_weights)
        for key in WEIGHT_KEYS
            val = get(mutated_weights, key, 1.0)
            mutated_weights[key] = clamp(val + randn() * 1.5, 0.0, 15.0)
        end
        push!(population, Chromosome(mutated_weights))
    end
    
    # تقييم اللياقة الأولية للمجتمع
    for ind in population
        evaluate_fitness!(ind, gen, prompts)
    end
    
    # 2. تشغيل الأجيال الجينية
    for gen_idx in 1:generations
        sort!(population, by=ind -> -ind.fitness)
        best_fit = population[1].fitness
        @info "Al-Tawweer Generation $gen_idx/$generations - Best Fitness: $best_fit"
        
        next_pop = Chromosome[]
        
        # النخبوية: الاحتفاظ بأفضل كروموسومين بدون تعديل
        push!(next_pop, population[1])
        push!(next_pop, population[2])
        
        # توليد باقي المجتمع الجديد
        while length(next_pop) < pop_size
            p1 = selection(population)
            p2 = selection(population)
            
            c1, c2 = crossover(p1, p2)
            
            mutate!(c1)
            mutate!(c2)
            
            push!(next_pop, c1)
            if length(next_pop) < pop_size
                push!(next_pop, c2)
            end
        end
        
        # تقييم لياقة المجتمع الجديد
        for ind in next_pop
            # تجنب إعادة تقييم النخبة لتوفير الوقت
            if ind.fitness == 0.0
                evaluate_fitness!(ind, gen, prompts)
            end
        end
        
        population = next_pop
    end
    
    # 3. إرجاع الأوزان المتفوقة
    sort!(population, by=ind -> -ind.fitness)
    @info "Al-Tawweer finished. Best weights achieved fitness: $(population[1].fitness)"
    return population[1].weights
end

end # module
