# -*- coding: utf-8 -*-
"""Advanced Physics Engines — المحركات الفيزيائية المتقدمة لمرنان.

أربعة محركات تعالج القصور الهيكلي مقارنة بـ LLMs ببدائل فيزيائية:

1. ResonantBeamformer — تشكيل شعاع طوري (بديل Self-Attention)
   بدل جمع كل المتجهات في "حساء"، نستخدم هوائي مصفوفة طورية (phased array)
   لتوجيه الشعاع نحو الكلمات ذات الصلة فقط.

2. PhaseAccumulator — تراكم طوري غير تبديلي (بديل Positional Encoding)
   بدل A+B=B+A، نضرب كل متجه بمصفوفة دوران تعتمد على الموضع.
   R(θ₁)·v₁ + R(θ₂)·v₂ ≠ R(θ₂)·v₂ + R(θ₁)·v₁

3. RefractoryGate — بوابة الجموح (بديل Repetition Penalty)
   الكلمة المستخدمة حديثاً يُقلب طورها مؤقتاً (×-1) فتصدّ الكلمات المشابهة
   بدل جذبها. يعود الطور تدريجياً بعد N خطوة.

4. MacroWaveEngine — تشابك الموجات الكبرى (بديل الطبقات العميقة)
   عبارات كاملة (مبتدأ+خبر) تتشابك لتشكل "جسيم مفهوم" بتردده وكتلته الخاصين.
   المفاهيم تتفاعل مع مفاهيم أخرى، لا كلمات مع كلمات.
"""

import numpy as np
from scipy.spatial.transform import Rotation
from src.physics.constants import PHASE_DIM
from src.physics.word_physics import phase_similarity


# ═══════════════════════════════════════════════════════════════
# 1. RESONANT BEAMFORMER — تشكيل شعاع طوري
# ═══════════════════════════════════════════════════════════════

class ResonantBeamformer:
    """هوائي مصفوفة طورية للسياق.

    بدل جمع كل متجهات السياق بالتساوي:
      target = mean(all_pvs)  ← "حساء"

    نستخدم تشكيل الشعاع:
      1. لكل كلمة سياق، نحسب "وزن التوجيه" بناءً على تشابهها مع المرشح.
      2. الكلمات الأكثر صلة بالمرشح تحصل على وزن طوري أعلى.
      3. النتيجة: شعاع مركز على الكلمات المهمة فقط.

    هذا يشبه رادار المصفوفة الطورية:
      - كل عنصر هوائي = كلمة سياق
      - تأخير الطور = الوزن بناءً على الصلة
      - الشعاع الناتج = سياق مركز
    """

    def __init__(self, steering_gain=2.0):
        self.steering_gain = steering_gain

    def beamform(self, candidate_pv, context_pvs, context_words=None):
        """تشكيل شعاع سياقي نحو المرشح.

        Args:
            candidate_pv: متجه الكلمة المرشحة (22D)
            context_pvs: قائمة متجهات السياق
            context_words: قائمة كلمات السياق (اختياري)

        Returns:
            (beam_vector, attention_weights, focus_score)
        """
        if not context_pvs:
            return np.zeros(PHASE_DIM), [], 0.0

        n = len(context_pvs)
        weights = np.zeros(n)
        phases = np.zeros((n, PHASE_DIM))

        for i, cpv in enumerate(context_pvs):
            if cpv is None or np.linalg.norm(cpv[:PHASE_DIM]) < 1e-10:
                continue
            sim = phase_similarity(cpv[:PHASE_DIM], candidate_pv[:PHASE_DIM])
            weight = np.exp(self.steering_gain * sim)
            weights[i] = weight
            phases[i] = cpv[:PHASE_DIM]

        if weights.sum() < 1e-10:
            return np.zeros(PHASE_DIM), [], 0.0

        weights = weights / weights.sum()
        beam = np.sum(phases * weights[:, np.newaxis], axis=0)
        beam_norm = np.linalg.norm(beam)
        if beam_norm > 1e-10:
            beam = beam / beam_norm

        focus_score = float(np.max(weights)) if len(weights) > 0 else 0.0

        return beam, weights.tolist(), focus_score

    def compute_beam_context(self, candidate_pv, all_pv, top_n=8):
        """حساب سياق مركز على أهم N كلمة للمرشح."""
        if not all_pv:
            return np.zeros(PHASE_DIM), 0.0

        sims = np.array([phase_similarity(pv[:PHASE_DIM], candidate_pv[:PHASE_DIM])
                         for pv in all_pv if pv is not None])
        if len(sims) == 0:
            return np.zeros(PHASE_DIM), 0.0

        top_indices = np.argsort(sims)[-min(top_n, len(sims)):]
        top_pvs = [all_pv[i] for i in top_indices]
        beam, _, focus = self.beamform(candidate_pv, top_pvs)
        return beam, focus


# ═══════════════════════════════════════════════════════════════
# 2. PHASE ACCUMULATOR — تراكم طوري غير تبديلي
# ═══════════════════════════════════════════════════════════════

class PhaseAccumulator:
    """تراكم غير تبديلي للمتجهات الطورية.

    المشكلة: A + B = B + A — الترتيب لا يهم.
    الحل: كل كلمة تُضرَب بمصفوفة دوران R(θ_pos) قبل الجمع.
      النتيجة: R(θ₁)·A + R(θ₂)·B ≠ R(θ₂)·B + R(θ₁)·A

    رياضياً: نستخدم مصفوفات دوران في فضاء جزئي 3D (SO(3)) داخل الـ 22D.
    الدورانات في SO(3) غير تبديلية: R_x(α)·R_y(β) ≠ R_y(β)·R_x(α)

    لكل موضع i، نطبق دورانًا بزاوية θ_i = i × base_angle على أول 3 أبعاد.
    """

    def __init__(self, base_angle=0.15):
        self.base_angle = base_angle
        self._rotation_cache = {}

    def _get_rotation(self, position):
        if position not in self._rotation_cache:
            angle = position * self.base_angle
            R = Rotation.from_euler('xyz', [angle, angle * 0.7, angle * 0.5]).as_matrix()
            self._rotation_cache[position] = R
        return self._rotation_cache[position]

    def accumulate(self, phase_vectors, normalize=True):
        """تجميع غير تبديلي للمتجهات حسب ترتيبها.

        Args:
            phase_vectors: قائمة (n, PHASE_DIM) مرتبة حسب الموضع
            normalize: هل نطبّع الناتج

        Returns:
            np.ndarray: المتجه المتراكم (PHASE_DIM,)
        """
        n = len(phase_vectors)
        if n == 0:
            return np.zeros(PHASE_DIM)
        if n == 1:
            pv = phase_vectors[0][:PHASE_DIM].copy()
            if normalize:
                nrm = np.linalg.norm(pv)
                if nrm > 1e-10:
                    pv = pv / nrm
            return pv

        accumulated = np.zeros(PHASE_DIM)
        for i, pv in enumerate(phase_vectors):
            v = pv[:PHASE_DIM].copy()
            nrm = np.linalg.norm(v)
            if nrm < 1e-10:
                continue

            R3 = self._get_rotation(i)
            v3 = v[:3]
            v3_rotated = R3 @ v3
            v_rotated = v.copy()
            v_rotated[:3] = v3_rotated

            accumulated += v_rotated

        if normalize:
            nrm = np.linalg.norm(accumulated)
            if nrm > 1e-10:
                accumulated = accumulated / nrm

        return accumulated

    def positional_encoding(self, position, dim=PHASE_DIM):
        """توليد تشفير موضعي جيبي (للدمج مع المتجه قبل التراكم)."""
        pe = np.zeros(dim)
        for i in range(0, dim, 2):
            denominator = 10000 ** (2 * i / dim)
            pe[i] = np.sin(position / denominator)
            if i + 1 < dim:
                pe[i + 1] = np.cos(position / denominator)
        return pe * 0.1


# ═══════════════════════════════════════════════════════════════
# 3. REFRACTORY GATE — بوابة الجموح
# ═══════════════════════════════════════════════════════════════

class RefractoryGate:
    """بوابة استنفاد الطاقة للكلمات المستخدمة حديثاً.

    المبدأ الفيزيائي: كما أن العصبون يدخل في فترة جموح (refractory period)
    بعد الإطلاق، الكلمة بعد توليدها تدخل في حالة "استنفاد طاقة".

    آلية العمل:
      1. عند توليد كلمة: يُضاف متجهها إلى قائمة "المستنفدة" بطور مقلوب (× -1).
      2. في التوليد التالي: المتجهات المقلوبة تصدّ الكلمات المشابهة (تنافر بدل جذب).
      3. بعد N خطوة: يضمحل التأثير تدريجياً (يعود إلى الصفر).

    النتيجة: النظام يُجبر على استكشاف كلمات جديدة بدل التكرار.
    """

    def __init__(self, refractory_steps=3, decay_rate=0.4, inversion_strength=1.5):
        self.refractory_steps = refractory_steps
        self.decay_rate = decay_rate
        self.inversion_strength = inversion_strength
        self._depleted = []  # (pv, remaining_steps, initial_strength)

    def deplete(self, word_pv):
        """استنفاد كلمة بعد استخدامها."""
        if word_pv is None:
            return
        pv = word_pv[:PHASE_DIM].copy()
        nrm = np.linalg.norm(pv)
        if nrm > 1e-10:
            pv = pv / nrm
        self._depleted.append((pv, self.refractory_steps, self.inversion_strength))

    def get_repulsion_field(self, candidate_pv):
        """حساب مجال التنافر من الكلمات المستنفدة.

        Returns:
            float: قيمة سالبة (تنافر) كلما كان المرشح مشابهاً لكلمة مستنفدة.
        """
        if not self._depleted or candidate_pv is None:
            return 0.0

        total_repulsion = 0.0
        for pv, steps, strength in self._depleted:
            decay_factor = steps / self.refractory_steps
            effective_strength = strength * decay_factor
            sim = phase_similarity(pv[:PHASE_DIM], candidate_pv[:PHASE_DIM])
            if sim > 0:
                total_repulsion -= effective_strength * sim

        return total_repulsion

    def step(self):
        """تقدم خطوة زمنية — اضمحلال التأثيرات."""
        new_depleted = []
        for pv, steps, strength in self._depleted:
            new_steps = steps - 1
            if new_steps > 0:
                new_depleted.append((pv, new_steps, strength * (1.0 - self.decay_rate)))
        self._depleted = new_depleted

    def reset(self):
        self._depleted = []


# ═══════════════════════════════════════════════════════════════
# 4. MACRO-WAVE ENGINE — تشابك الموجات الكبرى
# ═══════════════════════════════════════════════════════════════

class MacroWaveEngine:
    """محرك المفاهيم — تشابك العبارات لتكوين جسيمات مفاهيم.

    المبدأ: كما أن الإلكترونات تتحد مع النواة لتشكل ذرة بخصائص جديدة،
    الكلمات تتشابك لتشكل "مفهوماً" (Concept Particle) له:
      - تردده الخاص (متوسط هندسي لترددات مكوناته)
      - كتلته (مجموع كتل مكوناته)
      - متجهه الطوري (متوسط موزون لمتجهات مكوناته)

    أنماط التشابك (phrase patterns):
      - NOUN + ADJ → مفهوم موصوف
      - VERB + NOUN → مفهوم فعل-مفعول
      - SUBJECT + PREDICATE → مفهوم جملة اسمية
    """

    def __init__(self, max_concepts=50):
        self.max_concepts = max_concepts
        self.concepts = {}  # concept_id -> {"pv": ..., "mass": ..., "freq": ..., "words": [...]}
        self.next_id = 0
        self._phrase_patterns = [
            ("noun_adj", 2),
            ("verb_noun", 2),
            ("prep_noun", 2),
            ("noun_conj_noun", 3),
        ]

    def entangle(self, words, get_pv_fn, get_mass_fn):
        """محاولة تشابك آخر n كلمة في مفهوم.

        يرجع المفهوم إن نجح التشابك، وإلا None.
        """
        if len(words) < 2:
            return None

        last_n = min(4, len(words))
        for n_words in [2, 3]:
            if last_n < n_words:
                continue
            phrase = words[-n_words:]
            pvs = [get_pv_fn(w) for w in phrase]
            if any(p is None for p in pvs):
                continue
            masses = [get_mass_fn(w) for w in phrase]

            concept_pv = np.mean([p[:PHASE_DIM] for p in pvs], axis=0)
            nrm = np.linalg.norm(concept_pv)
            if nrm > 1e-10:
                concept_pv = concept_pv / nrm

            concept_mass = sum(masses)
            concept_freq = np.exp(np.mean([np.log(max(m, 1e-10)) for m in masses]))

            phrase_key = " ".join(phrase)
            if phrase_key in self.concepts:
                return self.concepts[phrase_key]

            cid = self.next_id
            self.next_id += 1
            concept = {
                "id": cid,
                "pv": concept_pv,
                "mass": concept_mass,
                "freq": concept_freq,
                "words": phrase,
                "key": phrase_key,
            }
            self.concepts[phrase_key] = concept

            if len(self.concepts) > self.max_concepts:
                oldest = min(self.concepts.keys(), key=lambda k: self.concepts[k]["id"])
                del self.concepts[oldest]

            return concept

    def get_active_concepts(self, context_words, window=4):
        """استخراج المفاهيم النشطة من آخر n كلمة في السياق."""
        active = []
        ctx = context_words[-window:] if len(context_words) > window else context_words

        for n_words in [2, 3]:
            for i in range(len(ctx) - n_words + 1):
                phrase_key = " ".join(ctx[i:i + n_words])
                if phrase_key in self.concepts:
                    active.append(self.concepts[phrase_key])

        return active

    def score_candidate_via_concepts(self, candidate_pv, context_words):
        """تسجيل مرشح عبر المفاهيم النشطة في السياق."""
        active = self.get_active_concepts(context_words)
        if not active or candidate_pv is None:
            return 0.0

        best_score = 0.0
        for concept in active:
            sim = phase_similarity(concept["pv"][:PHASE_DIM], candidate_pv[:PHASE_DIM])
            gravity = concept["mass"] / (1.0 + abs(len(context_words) - len(active)))
            score = sim * gravity * 0.1
            best_score = max(best_score, score)

        return best_score

    def reset(self):
        self.concepts = {}
        self.next_id = 0
