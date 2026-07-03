"""Θ_vec — Local Thermodynamic State Vector لكل كلمة.

يحول EntropyGate من ثابت شامل إلى حقل محلي لكل كلمة.
T(w) = -log(p(w|context))
k_B(t) = k_B₀ · exp(-α · ΔS_local)
β(t) = β₀ · (1 + γ · |∇F|)

يمنع التبريد المبكر (Early Freeze) أو التشتت المفرط.
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def local_temperature(word, context_words, vocab_size=10000):
    """حساب درجة الحرارة المحلية T(w) لكل كلمة.
    
    T(w) = -log(p(w|context))
    كلما ارتفعت T ← الكلمة أقل توقعاً (استكشاف أعلى)
    """
    if not context_words:
        return 1.0
    freq = sum(1 for w in context_words if w == word)
    p = (freq + 1) / (len(context_words) + vocab_size)
    T = -np.log(max(p, 1e-10))
    return float(T)


def local_entropy(pvs, target, k_B=0.1):
    """حساب الإنتروبيا المحلية لمجموعة pvs.
    
    S = -k_B · Σ p_i · log(p_i) حيث p_i من محاذاة الطور.
    """
    if not pvs or len(pvs) == 0:
        return 0.0
    pvs_arr = np.array([pv[:len(target)] for pv in pvs])
    alignments = np.cos(pvs_arr - target)
    sims = np.mean(alignments, axis=1)
    probs = np.exp(sims - sims.max()) if len(sims) > 0 else np.array([1.0])
    probs = probs / (probs.sum() + 1e-10)
    S = -k_B * np.sum(probs * np.log(probs + 1e-10))
    return float(S)


def compute_theta(word, pv, context_pvs, target, k_B=0.1, beta=1.0):
    """حساب متجه الحالة Θ = [T, S, F, P] لكلمة في سياقها.
    
    T: درجة حرارة محلية (استكشاف)
    S: إنتروبيا محلية (شك/غموض)
    F: طاقة حرة F = E - T·S (استقرار)
    P: ضغط سياقي (توتر نحوي/دلالي)
    """
    if context_pvs and len(context_pvs) > 0:
        ctx_words = []
        T = local_temperature(word, ctx_words)
    else:
        T = 1.0
    
    proxy_pvs = [pv] if pv is not None else []
    S = local_entropy(proxy_pvs, target)
    
    energy = 1.0 - float(np.mean(np.cos(pv[:len(target)] - target))) if pv is not None else 0.5
    F = energy - T * S
    
    if context_pvs:
        ctx_sims = [float(np.mean(np.cos(pv[:len(target)] - cpv[:len(target)]))) 
                    for cpv in context_pvs if cpv is not None]
        P = 1.0 - float(np.mean(ctx_sims)) if ctx_sims else 0.5
    else:
        P = 0.5
    
    return {
        'T': T,
        'S': S,
        'F': F,
        'P': P,
    }


def adjust_k_B(S_local, k_B_base=0.1, alpha=0.5):
    """ضبط k_B ديناميكياً بناءً على الإنتروبيا المحلية.
    
    k_B(t) = k_B₀ · exp(-α · ΔS)
    ΔS = S_local - S_crit
    """
    return k_B_base * np.exp(-alpha * S_local)


def adjust_beta(S_local, beta_base=1.0, gamma=0.3):
    """ضبط β ديناميكياً (معامل التبريد).
    
    β(t) = β₀ · (1 + γ · |∇F|)
    """
    return beta_base * (1.0 + gamma * abs(S_local))
