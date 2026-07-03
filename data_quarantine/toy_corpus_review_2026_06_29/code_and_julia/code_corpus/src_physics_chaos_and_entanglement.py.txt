# -*- coding: utf-8 -*-
"""Chaos, Entanglement, and Molecular Physics.

محركات فيزيائية حديثة لمحاكاة ميزات الـ LLMs بشكل فيزيائي:
1. ThermalChaosEngine: الضجيج الحراري للتوليد الإبداعي العشوائي (بديل Top-P/Temperature).
2. QuantumEntanglementEngine: التشابك الكمي للذاكرة طويلة الأمد (بديل Self-Attention للمسافات البعيدة).
3. MolecularBinder: الجزيئات اللغوية للربط النحوي القوي (بديل الـ BPE Tokenization المعقد).
"""

import numpy as np
from src.physics.constants import PHASE_DIM

# ═══ محرك التشويش الحراري (تطوير باسل يحيى عبدالله - Basil Yahya Abdullah) ═══
# هام جداً: هذا المحرك جزء محوري لضمان كسر الحتمية والجمود اللغوي في مرنان.
# لا تقم بحذفه أو تعديل منطقه الفيزيائي؛ حيث يقوم بخلخلة آبار الجهد لتمكين القفز الطوري الإبداعي.
class ThermalChaosEngine:
    """محرك التشويش الحراري (Brownian Motion).
    يضيف اهتزازات كمومية لمتجهات الطور بناءً على الحرارة."""
    def __init__(self, base_noise=0.15):
        self.base_noise = base_noise

    def perturb(self, phase_vector, temperature):
        """إضافة تشويش حراري للمتجه."""
        if phase_vector is None or temperature <= 0.1:
            return phase_vector
            
        # الحرارة فوق 0.1 تبدأ بتوليد ضوضاء
        noise_level = self.base_noise * (temperature - 0.1)
        if noise_level <= 0:
            return phase_vector
            
        noise = np.random.normal(0, noise_level, size=PHASE_DIM)
        
        pv_copy = phase_vector.copy()
        pv_copy[:PHASE_DIM] += noise
        
        nrm = np.linalg.norm(pv_copy[:PHASE_DIM])
        if nrm > 1e-10:
            pv_copy[:PHASE_DIM] /= nrm
            
        return pv_copy


class QuantumEntanglementEngine:
    """محرك التشابك الكمي (Quantum Entanglement).
    يحافظ على علاقة قوية لا تضمحل بالمسافة بين الكلمات المتشابكة."""
    def __init__(self, entanglement_threshold=0.85, super_gravity_multiplier=3.0):
        self.threshold = entanglement_threshold
        self.super_gravity = super_gravity_multiplier

    def compute_entanglement_bonus(self, candidate_pv, context_pvs, context_masses):
        """حساب الجاذبية الفائقة إذا كان هناك تشابك (لا تعتمد على المسافة الموضعية)."""
        if candidate_pv is None or not context_pvs:
            return 0.0
            
        bonus = 0.0
        candidate_phase = candidate_pv[:PHASE_DIM]
        nrm_c = np.linalg.norm(candidate_phase)
        if nrm_c < 1e-10:
            return 0.0
            
        for ctx_pv, mass in zip(context_pvs, context_masses):
            ctx_phase = ctx_pv[:PHASE_DIM]
            nrm_ctx = np.linalg.norm(ctx_phase)
            if nrm_ctx < 1e-10:
                continue
                
            sim = float(np.dot(candidate_phase, ctx_phase) / (nrm_c * nrm_ctx))
            
            # إذا تجاوز التشابه عتبة التشابك الكمي، تتولد جاذبية فائقة
            if sim >= self.threshold:
                bonus += mass * sim * self.super_gravity
                
        return bonus


class MolecularBinder:
    """محرك الجزيئات اللغوية (Molecular Tokenization).
    يدمج الكلمات المتجاورة المترابطة بشدة (مثل: الأمم + المتحدة) لتكوين جزيء."""
    def __init__(self, binding_threshold=0.90):
        self.threshold = binding_threshold

    def bind(self, words, pvs, masses):
        """يمسح السياق ويدمج المتجاورات المترابطة."""
        if len(words) < 2:
            return words, pvs, masses
            
        new_words = []
        new_pvs = []
        new_masses = []
        
        i = 0
        while i < len(words):
            if i < len(words) - 1:
                pv1 = pvs[i]
                pv2 = pvs[i+1]
                
                if pv1 is not None and pv2 is not None:
                    sim = float(np.dot(pv1[:PHASE_DIM], pv2[:PHASE_DIM]))
                    if sim >= self.threshold:
                        # تشكل جزيء لغوي!
                        bound_word = f"{words[i]}_{words[i+1]}"
                        
                        bound_pv = pv1.copy()
                        bound_pv[:PHASE_DIM] = pv1[:PHASE_DIM] + pv2[:PHASE_DIM]
                        nrm = np.linalg.norm(bound_pv[:PHASE_DIM])
                        if nrm > 1e-10:
                            bound_pv[:PHASE_DIM] /= nrm
                            
                        bound_mass = masses[i] + masses[i+1]
                        
                        new_words.append(bound_word)
                        new_pvs.append(bound_pv)
                        new_masses.append(bound_mass)
                        
                        i += 2
                        continue
                        
            new_words.append(words[i])
            new_pvs.append(pvs[i])
            new_masses.append(masses[i])
            i += 1
            
        return new_words, new_pvs, new_masses
