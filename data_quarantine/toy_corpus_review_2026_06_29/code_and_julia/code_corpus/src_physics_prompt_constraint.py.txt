# -*- coding: utf-8 -*-
"""
Prompt Constraint Field — حقل قيد حدودي (Dirichlet Boundary Condition)

الفكرة الفيزيائية: معادلة لابلاس ∇²φ = 0 مع شروط حدودية (Dirichlet).
إذا ثبتنا قيمة الحل φ على الحدود، فإن الحل يتحدد في كامل المجال.

التطبيق: كلمات الـ prompt تُشكّل "عُقداً ثابتة" في فضاء الطور.
عملية التوليد = حل معادلة الانتشار من عقدة إلى أخرى.
كل كلمة مولّدة تُسحب بنابض نحو العقدة المستهدفة التالية.

الرياضيات:
  F_spring(candidate, constraint_i) = -k · (v_cand - v_constraint_i)
  constraint_target = prompt_pvs[⌊progress · len(prompt_pvs)⌋]

أيضاً: استخراج goal_concept — متجه طوري يمثل "وجهة" التوليد الكلية،
مُستخرج من كامل الـ prompt (وليس من كلمات مفتاحية).
"""

import numpy as np
from src.physics.word_physics import phase_similarity


class PromptConstraintField:
    """حقل نوابض يربط التوليد بعُقد prompt ثابتة — حل Dirichlet للانتشار الطوري."""

    def __init__(self, prompt_pvs=None, k_spring=3.0, damping=0.15):
        """
        Args:
            prompt_pvs: قائمة متجهات طورية لكلمات الـ prompt
            k_spring: ثابت النابض — شدة الربط بالعقدة المستهدفة
            damping: تخميد النابض كلما ابتعدنا عن العقدة
        """
        self.constraints = list(prompt_pvs) if prompt_pvs else []
        self.k = k_spring
        self.damping = damping
        self._constraint_progress = 0.0

    def set_prompt(self, prompt_pvs):
        """تعيين قائمة متجهات prompt كعُقد حدودية."""
        self.constraints = list(prompt_pvs)

    @property
    def has_constraints(self):
        return len(self.constraints) > 0

    def get_constraint_at(self, progress_fraction):
        """استرجاع العقدة المستهدفة حسب نسبة التقدم في التوليد.

        Args:
            progress_fraction: gen_pos / total_pos (بين 0 و 1)

        Returns:
            np.ndarray أو None: متجه العقدة المستهدفة
        """
        if not self.constraints:
            return None
        n = len(self.constraints)
        # توزيع العقد: العقدة الأولى عند 0%، الأخيرة عند 100%
        idx = int(np.clip(progress_fraction * (n - 1), 0, n - 1))
        return self.constraints[idx]

    def spring_force(self, candidate_pv, gen_pos, total_pos, centroid_pv=None):
        """مكافأة تقدمية: انحياز لطيف نحو العقدة المستهدفة حسب مرحلة التوليد.

        الفلسفة: القيد لا ينافس الجاذبية العامة — بل يضيف انحيازاً موجهاً
        حسب مرحلة التوليد. في البداية (0%) ينحاز نحو أول كلمة prompt،
        في النهاية (100%) ينحاز نحو آخر كلمة. هذا يخلق بنية متسلسلة.

        bonus = cos(cand, target_word) * force_weight * 0.3
        (انحياز لطيف 30% لا يطغى على الجاذبية)

        Args:
            candidate_pv: متجه المرشح
            gen_pos: رقم الخطوة الحالية
            total_pos: إجمالي الخطوات
            centroid_pv: متجه مركز الكتلة (غير مستخدم حالياً)

        Returns:
            (float, np.ndarray): (مكافأة إضافية [0,0.3], متجه القوة الخام)
        """
        if not self.constraints:
            return 0.0, np.zeros_like(candidate_pv)

        progress = gen_pos / max(total_pos, 1)
        target = self.get_constraint_at(progress)
        if target is None:
            return 0.0, np.zeros_like(candidate_pv)

        target_sim = phase_similarity(candidate_pv, target)

        # قوة النابض — تضمحل مع البعد عن العقدة
        direction = target - candidate_pv
        distance = np.linalg.norm(direction)
        force_magnitude = self.k * np.exp(-self.damping * distance) if distance > 1e-10 else self.k
        force_weight = min(1.0, force_magnitude / self.k)

        # انحياز لطيف 30% — لا يطغى على الجاذبية
        bonus = target_sim * force_weight * 0.3

        force_vec = np.zeros_like(candidate_pv)
        if distance > 1e-10:
            force_vec = force_magnitude * direction / distance

        return bonus, force_vec

    def compute_goal_concept(self, prompt_pvs, alpha=0.6, holographic_kb=None):
        """استخراج goal_concept من كامل prompt — ليس كشفاً بالكلمات المفتاحية.

        الفلسفة: الـ goal_concept هو "مركز الكتلة الطوري" للـ prompt بأكمله،
        مدمجاً مع المعرفة المسترجعة من الذاكرة الهولوغرافية.

        Args:
            prompt_pvs: قائمة متجهات prompt
            alpha: وزن مركز الكتلة مقابل إسقاط القصد (0.6 = توازن)
            holographic_kb: قاعدة معرفية هولوغرافية (اختياري — يُثري الهدف بالحقائق)

        Returns:
            np.ndarray: متجه الهدف (56D)
        """
        if not prompt_pvs:
            return None

        pvs = np.array(prompt_pvs)
        n = len(pvs)

        # أوزان موضعية: الكلمات الأخيرة أثقل (الأحدث = الأهم)
        pos_weights = np.exp(-0.3 * np.arange(n)[::-1])
        pos_weights = pos_weights / pos_weights.sum()

        # مركز الكتلة الطوري
        center = np.average(pvs, axis=0, weights=pos_weights)
        center_norm = np.linalg.norm(center)
        if center_norm > 1e-10:
            center = center / center_norm

        # إسقاط على الفضاء القصدي (آخر 6 أبعاد)
        pragmatic_projection = np.zeros_like(center)
        pragmatic_projection[-6:] = center[-6:]

        # دمج: مركز الكتلة + الإسقاط القصدي
        goal = alpha * center + (1.0 - alpha) * pragmatic_projection

        # ═══ V8.8: إثراء بالذاكرة الهولوغرافية ═══
        if holographic_kb is not None and holographic_kb._built:
            kb_weight = 0.0
            kb_contribution = np.zeros_like(goal)
            for pv in pvs:
                recon = holographic_kb.reconstruct_vector(pv)
                if recon is not None and np.linalg.norm(recon) > 0.01:
                    kb_contribution += recon
                    kb_weight += 1.0
            if kb_weight > 0:
                kb_contribution /= kb_weight
                kb_norm = np.linalg.norm(kb_contribution)
                if kb_norm > 1e-10:
                    kb_contribution /= kb_norm
                # خلط: 70% الهدف الأصلي + 30% المعرفة الهولوغرافية
                goal = 0.7 * goal + 0.3 * kb_contribution

        goal_norm = np.linalg.norm(goal)
        if goal_norm > 1e-10:
            goal = goal / goal_norm

        return goal

    def goal_alignment_score(self, candidate_pv, goal_pv):
        """مدى توافق المرشح مع goal_concept المستخرج.

        Returns:
            float: درجة التوافق [0, 1]
        """
        if goal_pv is None:
            return 0.0
        sim = phase_similarity(candidate_pv, goal_pv)
        return max(0.0, sim)
