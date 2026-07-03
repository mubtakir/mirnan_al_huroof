"""
المذبذب للمزنان الجديد.

يولد إشارات جيبية للحروف والكلمات.
"""

module Oscillator

using LinearAlgebra

using ..Constants
using ..LetterEquations: get_letter_params

export generate_signal, compute_signal_energy, compute_signal_frequency

# ═══════════════════════════════════════════════════════
# توليد الإشارة
# ═══════════════════════════════════════════════════════

"""
    generate_signal(params::Tuple, x_range::Tuple{Float64, Float64}=(-1.0, 1.0);
                   n_samples::Int=200) -> Vector{Float64}
توليد إشارة من معاملات الحرف
"""
function generate_signal(params::Tuple, x_range::Tuple{Float64, Float64}=(-1.0, 1.0);
                        n_samples::Int=200)
    C, A, β, γ, B, W, k, x0, n = params
    
    xs = collect(range(x_range[1], x_range[2], length=n_samples))
    y = zeros(n_samples)
    
    for (i, x) in enumerate(xs)
        linear = A * (β * x + γ)
        
        dx = x - x0
        if abs(dx) < 1e-10
            powered = 0.0
        else
            powered = sign(dx) * (abs(dx))^n
        end
        
        exponent = k * powered
        exponent = clamp(exponent, -50.0, 50.0)
        
        sigmoid = B * W / (1.0 + exp(-exponent))
        
        y[i] = C * (linear + sigmoid)
    end
    
    return y
end

"""
    generate_word_signal(word::String; n_samples::Int=200) -> Vector{Float64}
توليد إشارة الكلمة
"""
function generate_word_signal(word::String; n_samples::Int=200)
    letters = collect(filter(c -> !isspace(c), word))
    if isempty(letters)
        return zeros(n_samples)
    end
    
    signal = zeros(n_samples)
    for letter in letters
        params = get_letter_params(letter)
        if params !== nothing
            # استخدام الحد الأول فقط للتبسيط
            signal .+= generate_signal(params[1], n_samples=n_samples)
        end
    end
    
    return signal / length(letters)
end

# ═══════════════════════════════════════════════════════
# تحليل الإشارة
# ═══════════════════════════════════════════════════════

"""
    compute_signal_energy(signal::Vector{Float64}) -> Float64
حساب طاقة الإشارة
"""
function compute_signal_energy(signal::Vector{Float64})
    return sum(signal .^ 2)
end

"""
    compute_signal_frequency(signal::Vector{Float64}) -> Float64
حساب تردد الإشارة
"""
function compute_signal_frequency(signal::Vector{Float64})
    n = length(signal)
    if n < 2
        return 0.0
    end
    
    # حساب التغيرات
    changes = 0
    for i in 2:n
        if (signal[i] > 0 && signal[i-1] < 0) || (signal[i] < 0 && signal[i-1] > 0)
            changes += 1
        end
    end
    
    return changes / (n - 1)
end

"""
    compute_signal_correlation(signal1::Vector{Float64}, signal2::Vector{Float64}) -> Float64
حساب الارتباط بين إشارتين
"""
function compute_signal_correlation(signal1::Vector{Float64}, signal2::Vector{Float64})
    n = min(length(signal1), length(signal2))
    if n < 2
        return 0.0
    end
    
    s1 = signal1[1:n]
    s2 = signal2[1:n]
    
    n1 = norm(s1)
    n2 = norm(s2)
    
    if n1 < 1e-10 || n2 < 1e-10
        return 0.0
    end
    
    return clamp(dot(s1, s2) / (n1 * n2), -1.0, 1.0)
end

end # module Oscillator
