# -*- coding: utf-8 -*-
r"""
محرك الاندماج الموجي غير الخطي للكلمات — مرنان الحروف
======================================================

الكلمة ليست مجرد مجموع خطي لموجات حروفها. بل تخضع لاندماج هندسي
يُنتج مستويات دلالية ناشئة (bivectors) لا وجود لها في أي حرف منفرد.

الطبقات:
  1. الاندماج الخطي: word = Σ w_i * v(letter_i)
  2. الاندماج الهندسي (جداء كليفورد): word = MV(L1) * MV(L2) * ... * MV(Ln)
  3. الاندماج الموتر: مصفوفة تفاعل 22×22 + تفكيك طيفي
  4. التداخل الطوري المركب: نمذجة الكلمة كتداخل أمواج

مُكيّف لمرنان الحروف (يستخدم LetterDB بدلاً من SemanticPhysicsEngine).
"""

import numpy as np
from typing import List, Dict
from src.physics.clifford_math import Multivector22

DIM_NAMES = [
    "concentration", "internal_external", "stability_motion", "density",
    "temperature", "time_accumulation", "time_peak", "time_discharge",
    "motion_linear", "motion_rotary", "motion_pulse", "motion_stretch",
    "motion_slip", "motion_air", "axis_v", "mass",
    "hardness_solid", "penetration", "charge", "reference_self",
    "space_extensionality", "time_causality"
]


def _get_letter_vectors(word: str, db=None) -> List[np.ndarray]:
    """استخراج متجهات الحروف من قاعدة البيانات."""
    if db is None:
        from src.physics.letter_db import LetterDB
        db = LetterDB()
    import re
    pure = re.sub(r'[\u064B-\u0652]', '', word)
    return [db.get_vector(ch) for ch in pure]


# ============================================================
#  الطبقة 1: الاندماج الخطي
# ============================================================

def linear_fusion(letter_vectors: List[np.ndarray], position_weights: List[float] = None) -> np.ndarray:
    if not letter_vectors:
        return np.zeros(22)
    if position_weights is None:
        position_weights = [3.5, 2.5, 2.0] + [1.5 / (i ** 0.5) for i in range(1, len(letter_vectors) - 2)]
        position_weights = position_weights[:len(letter_vectors)]
    n = len(letter_vectors)
    weighted = np.zeros(22)
    for i, v in enumerate(letter_vectors):
        w = position_weights[i] if i < len(position_weights) else 1.0
        weighted += v * w
    total_w = sum(position_weights[:n])
    result = weighted / total_w if total_w > 0 else weighted
    norm = np.linalg.norm(result)
    return result / norm if norm > 1e-10 else result


# ============================================================
#  الطبقة 2: الاندماج الهندسي (جداء كليفورد)
# ============================================================

def geometric_word_fusion(letter_vectors: List[np.ndarray]) -> Multivector22:
    """الكلمة = جداء كليفورد غير تبديلي لحروفها.

    word = ((((L1 * L2) * L3) * L4) * ...)

    Returns Multivector22 بمركبات: scalar + vector(22D) + bivector(231D)
    """
    if not letter_vectors:
        return Multivector22()

    mv = Multivector22.from_vector(letter_vectors[0])
    mv.normalize()

    for vec in letter_vectors[1:]:
        operator = Multivector22.from_vector(vec)
        mv = mv * operator
        mv.normalize()

    return mv


def decompose_fusion(mv: Multivector22) -> Dict:
    """تفكيك نتيجة الاندماج الهندسي إلى مكوناتها الدلالية."""
    s = abs(mv.s)
    b_norm = np.linalg.norm(mv.b)
    v_norm = np.linalg.norm(mv.v)
    total = s + b_norm + v_norm + 1e-10

    dominant_pairs = []
    if b_norm > 0.01:
        top_idx = np.argsort(np.abs(mv.b))[::-1][:5]
        for idx in top_idx:
            k = 0
            for i in range(22):
                for j in range(i + 1, 22):
                    if k == idx:
                        dominant_pairs.append((DIM_NAMES[i], DIM_NAMES[j], round(float(mv.b[idx]), 4)))
                        break
                    k += 1
                if k > idx:
                    break

    return {
        "shared_essence": round(s, 4),
        "emergent_intensity": round(b_norm, 4),
        "vector_remnant": round(v_norm, 4),
        "fusion_ratio": round(b_norm / total, 4),
        "shared_ratio": round(s / total, 4),
        "emergent_ratio": round(b_norm / total, 4),
        "bivector_dominant_pairs": dominant_pairs,
        "total_magnitude": round(total, 4),
        "multivector": str(mv),
    }


# ============================================================
#  الطبقة 3: الاندماج الموتر (مصفوفة التفاعل 22×22)
# ============================================================

def interaction_tensor(letter_vectors: List[np.ndarray]) -> np.ndarray:
    """مصفوفة التفاعل بين الأبعاد عبر الحروف المتجاورة."""
    if len(letter_vectors) < 2:
        return np.outer(letter_vectors[0], letter_vectors[0]) if letter_vectors else np.zeros((22, 22))
    tensor = np.zeros((22, 22))
    count = 0
    for i in range(len(letter_vectors) - 1):
        v1 = letter_vectors[i]
        v2 = letter_vectors[i + 1]
        tensor += np.outer(v1, v2)
        count += 1
    return tensor / count if count > 0 else tensor


def eigen_fusion(letter_vectors: List[np.ndarray]) -> Dict:
    """تفكيك طيفي لمصفوفة التفاعل: الأنماط المهيمنة للاندماج."""
    tensor = interaction_tensor(letter_vectors)
    try:
        U, s, Vh = np.linalg.svd(tensor, full_matrices=False)
    except np.linalg.LinAlgError:
        U, s, Vh = np.linalg.svd(np.eye(22), full_matrices=False)

    dominant_mode = U[:, 0] * s[0]

    top_dims = np.argsort(np.abs(U[:, 0]))[::-1][:5]
    top_dim_contributions = [(DIM_NAMES[i], round(float(U[i, 0]), 4)) for i in top_dims]

    emergent = dominant_mode / (np.linalg.norm(dominant_mode) + 1e-10)

    return {
        "dominant_eigenvalue": round(float(s[0]), 4),
        "eigenvalue_spectrum": [round(float(x), 4) for x in s[:5]],
        "top_dimensions_eigen": top_dim_contributions,
        "emergent_vector_eigen": emergent.tolist(),
        "energy_concentration": round(float(s[0] ** 2 / (np.sum(s ** 2) + 1e-10)), 4),
    }


# ============================================================
#  الطبقة 4: التداخل الطوري المركب
# ============================================================

def complex_phase_interference(letter_vectors: List[np.ndarray]) -> Dict:
    """نمذجة الكلمة كتداخل أمواج ذات طور مركب."""
    if not letter_vectors:
        return {"coherence": 0.0, "interference_pattern": np.zeros(22)}

    n = len(letter_vectors)
    complex_vectors = []
    for vec in letter_vectors:
        phase = np.pi * vec / 2.0
        z = np.abs(vec) * np.exp(1j * phase)
        complex_vectors.append(z)

    weights = [3.5, 2.5, 2.0] + [1.5 / (i ** 0.5) for i in range(1, n - 2)]
    weights = weights[:n]
    total = np.zeros(22, dtype=complex)
    for i, z in enumerate(complex_vectors):
        total += z * weights[i]
    total /= sum(weights[:n])

    amplitudes = np.abs(total)
    phases = np.angle(total)

    constructive = []
    destructive = []
    for i in range(22):
        amp = float(amplitudes[i])
        if amp > 0.15:
            constructive.append((DIM_NAMES[i], round(amp, 4), round(float(phases[i]), 4)))
        elif amp < 0.05:
            destructive.append((DIM_NAMES[i], round(amp, 4), round(float(phases[i]), 4)))

    coherence = float(np.sum(amplitudes[amplitudes > 0.1]) / (np.sum(amplitudes) + 1e-10))

    return {
        "interference_real": [round(float(x.real), 4) for x in total],
        "interference_imag": [round(float(x.imag), 4) for x in total],
        "constructive_dims": constructive,
        "destructive_dims": destructive,
        "coherence": round(coherence, 4),
    }


# ============================================================
#  تحليل كامل
# ============================================================

def full_word_analysis(word: str, db=None) -> Dict:
    """تحليل كلمة كاملة بكل طرق الاندماج."""
    from src.physics.word_physics import compute_word_phase_vector
    letter_vectors = _get_letter_vectors(word, db)

    linear_vec = compute_word_phase_vector(word)
    geo_mv = geometric_word_fusion(letter_vectors)
    geo_decomp = decompose_fusion(geo_mv)
    eigen_result = eigen_fusion(letter_vectors)
    phase_result = complex_phase_interference(letter_vectors)

    cos_sim = float(np.dot(linear_vec, geo_mv.v) /
                    (np.linalg.norm(linear_vec) * np.linalg.norm(geo_mv.v) + 1e-10))

    import re
    pure = re.sub(r'[\u064B-\u0652]', '', word)

    return {
        "word": word,
        "num_letters": len(pure),
        "letters": pure,
        "linear_geometric_cosine": round(cos_sim, 4),
        "are_methods_equivalent": abs(cos_sim) > 0.95,
        "linear": {
            "vector": [round(float(x), 4) for x in linear_vec],
            "norm": round(float(np.linalg.norm(linear_vec)), 4),
        },
        "geometric": geo_decomp,
        "eigen": eigen_result,
        "phase": phase_result,
    }


def compare_antonyms(word1: str, word2: str, db=None) -> Dict:
    """مقارنة كلمتين متضادتين دلالياً."""
    a1 = full_word_analysis(word1, db)
    a2 = full_word_analysis(word2, db)

    linear_cosine = float(np.dot(
        np.array(a1["linear"]["vector"]),
        np.array(a2["linear"]["vector"])
    ) / (np.linalg.norm(np.array(a1["linear"]["vector"])) *
         np.linalg.norm(np.array(a2["linear"]["vector"])) + 1e-10))

    eig1 = a1["eigen"]["top_dimensions_eigen"]
    eig2 = a2["eigen"]["top_dimensions_eigen"]
    dims1 = set(d[0] for d in eig1)
    dims2 = set(d[0] for d in eig2)

    return {
        "word1": word1, "word2": word2,
        "linear_cosine_similarity": round(linear_cosine, 4),
        "geometric_emergent_diff": round(
            abs(a1["geometric"]["emergent_intensity"] - a2["geometric"]["emergent_intensity"]), 4),
        "shared_eigen_dimensions": list(dims1 & dims2),
        "different_eigen_dimensions": list((dims1 - dims2) | (dims2 - dims1)),
        "phase_coherence_diff": round(
            abs(a1["phase"]["coherence"] - a2["phase"]["coherence"]), 4),
        "fusion_ratio_word1": a1["geometric"]["fusion_ratio"],
        "fusion_ratio_word2": a2["geometric"]["fusion_ratio"],
    }


def analyze_word_family(words: List[str], db=None) -> Dict:
    """تحليل عائلة كلمات تشترك في جذر."""
    results = {}
    for w in words:
        results[w] = full_word_analysis(w, db)

    common_prefix = words[0]
    for w in words[1:]:
        i = 0
        while i < min(len(common_prefix), len(w)) and common_prefix[i] == w[i]:
            i += 1
        common_prefix = common_prefix[:i]

    base_word = common_prefix if common_prefix else words[0]
    evolution = []
    for w in words:
        if w == base_word:
            continue
        added = w[len(common_prefix):] if len(w) > len(common_prefix) else w
        ev = {
            "from": base_word, "to": w, "added_letters": added,
            "emergent_change": round(
                results[w]["geometric"]["emergent_intensity"] -
                results[base_word]["geometric"]["emergent_intensity"], 4),
            "fusion_ratio_change": round(
                results[w]["geometric"]["fusion_ratio"] -
                results[base_word]["geometric"]["fusion_ratio"], 4),
            "coherence_change": round(
                results[w]["phase"]["coherence"] -
                results[base_word]["phase"]["coherence"], 4),
        }
        evolution.append(ev)

    return {
        "word_family": words,
        "common_prefix": common_prefix,
        "evolution": evolution,
        "individual_analyses": results,
    }
