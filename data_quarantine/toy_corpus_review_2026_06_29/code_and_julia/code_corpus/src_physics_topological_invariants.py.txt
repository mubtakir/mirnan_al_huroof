# -*- coding: utf-8 -*-
"""
Topological Invariants — ثوابت طوبولوجية للمعنى الهندسي.

المبدأ الفيزيائي: في فيزياء المادة المكثفة، الخواص الطوبولوجية (مثل عدد اللف،
طور بيري) تُصنف الأنظمة إلى أطوار مستقرة لا تعتمد على التفاصيل الدقيقة.
خاصيتان متطابقتان طوبولوجياً لهما نفس الثوابت مهما اختلفت التفاصيل المحلية.

التطبيق في مِران:
- Berry Phase: الطور الهندسي المتراكم عبر سلسلة كلمات — يقيس "انحناء" المعنى.
- Winding Number: عدد لفات المسار حول نقطة الأصل في فضاء جزئي 2D.
- Topological Similarity: مقياس للتشابه بين سلسلتين لغويتين حتى لو اختلفت كلماتهما.

الفرضية: جملتان تعبران عن نفس المعنى (مثلاً "ثلج→ماء→بخار" و "ice→water→steam")
سيكون لهما ثوابت طوبولوجية متشابهة، بينما جمل غير مرتبطة دلالياً ستختلف.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class TopologicalInvariants:
    """حساب الثوابت الطوبولوجية لسلاسل الكلمات في فضاء الطور."""

    def __init__(self, dim=64, n_subspaces=5):
        """
        Args:
            dim: أبعاد فضاء الطور الكلي
            n_subspaces: عدد الفضاءات الجزئية لحساب winding number (متوسط)
        """
        self.dim = dim
        self.n_subspaces = n_subspaces
        # Pre-define interesting subspace pairs (use different slices)
        self.subspace_pairs = []
        for i in range(n_subspaces):
            start = (i * (dim // (n_subspaces + 1))) % (dim - 2)
            self.subspace_pairs.append((start, start + 1))

    def berry_phase(self, word_pvs):
        """حساب الطور الهندسي (Berry Phase) المتراكم عبر مسار الكلمات.

        φ_B = Σ arccos(⟨v_i | v_{i+1}⟩)  لكل i

        Args:
            word_pvs: قائمة متجهات طورية (كل منها 64D)

        Returns:
            float: الطور الهندسي الكلي
        """
        if len(word_pvs) < 2:
            return 0.0

        phases = []
        for i in range(len(word_pvs) - 1):
            a = word_pvs[i]
            b = word_pvs[i + 1]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na < 1e-10 or nb < 1e-10:
                continue
            overlap = float(np.dot(a, b) / (na * nb))
            overlap = np.clip(overlap, -1.0, 1.0)
            phases.append(np.arccos(overlap))
        return float(np.sum(phases))

    def winding_number(self, word_pvs, subspace=None):
        """عدد لفات المسار حول نقطة الأصل في فضاء جزئي 2D.

        يُحسب متوسط عدد اللفات عبر عدة فضاءات جزئية لتقليل الاعتماد
        على إسقاط واحد.

        Args:
            word_pvs: قائمة متجهات طورية
            subspace: زوج (start, end) للأبعاد (None = متوسط كل الفضاءات)

        Returns:
            float: متوسط عدد اللفات
        """
        if len(word_pvs) < 2:
            return 0.0

        if subspace is not None:
            pairs = [subspace]
        else:
            pairs = self.subspace_pairs

        windings = []
        for s0, s1 in pairs:
            xy = np.array([[pv[s0], pv[s1]] for pv in word_pvs])
            angles = np.arctan2(xy[:, 1], xy[:, 0])
            unwrapped = np.unwrap(angles)
            if len(unwrapped) >= 2:
                w = (unwrapped[-1] - unwrapped[0]) / (2.0 * np.pi)
                windings.append(w)

        if not windings:
            return 0.0
        return float(np.mean(windings))

    def compute_invariants(self, word_pvs):
        """حساب كلا الثابتين الطوبولوجيين دفعة واحدة.

        Args:
            word_pvs: قائمة متجهات طورية

        Returns:
            dict: {'berry_phase': float, 'winding_number': float}
        """
        return {
            'berry_phase': self.berry_phase(word_pvs),
            'winding_number': self.winding_number(word_pvs),
        }

    def topological_similarity(self, chain_a, chain_b):
        """مدى التشابه الطوبولوجي بين سلسلتين لغويتين.

        sim = 1 / (1 + |bp_a - bp_b| + |wn_a - wn_b|)

        النتيجة: 1.0 = متطابقتان طوبولوجياً، 0.0 = مختلفتان تماماً.

        Args:
            chain_a: قائمة متجهات طورية للسلسلة الأولى
            chain_b: قائمة متجهات طورية للسلسلة الثانية

        Returns:
            float: درجة التشابه [0, 1]
        """
        inv_a = self.compute_invariants(chain_a)
        inv_b = self.compute_invariants(chain_b)

        delta_bp = abs(inv_a['berry_phase'] - inv_b['berry_phase'])
        delta_wn = abs(inv_a['winding_number'] - inv_b['winding_number'])

        similarity = 1.0 / (1.0 + delta_bp + delta_wn)
        return float(similarity)

    def chain_similarity_from_words(self, words_a, words_b, pv_fn):
        """حساب التشابه الطوبولوجي بين سلسلتين من الكلمات مباشرة.

        Args:
            words_a: قائمة كلمات (نص)
            words_b: قائمة كلمات (نص)
            pv_fn: دالة word → PV

        Returns:
            dict: {'similarity': float, 'bp_a': float, 'bp_b': float,
                   'wn_a': float, 'wn_b': float}
        """
        pvs_a = [pv_fn(w) for w in words_a]
        pvs_b = [pv_fn(w) for w in words_b]

        inv_a = self.compute_invariants(pvs_a)
        inv_b = self.compute_invariants(pvs_b)

        sim = self.topological_similarity(pvs_a, pvs_b)

        return {
            'similarity': sim,
            'bp_a': inv_a['berry_phase'],
            'bp_b': inv_b['berry_phase'],
            'wn_a': inv_a['winding_number'],
            'wn_b': inv_b['winding_number'],
        }

    def batch_similarity_matrix(self, chains, pv_fn):
        """مصفوفة تشابه n×n بين مجموعة من السلاسل.

        Args:
            chains: قائمة قوائم كلمات
            pv_fn: دالة word → PV

        Returns:
            np.ndarray: مصفوفة التشابه (n, n)
        """
        n = len(chains)
        matrix = np.zeros((n, n))
        pvs_list = [[pv_fn(w) for w in chain] for chain in chains]
        for i in range(n):
            for j in range(i, n):
                sim = self.topological_similarity(pvs_list[i], pvs_list[j])
                matrix[i, j] = sim
                matrix[j, i] = sim
        return matrix

    def compute_chain_topology(self, words, pv_fn, prompt_pvs=None):
        """حساب معلومات طوبولوجية كاملة لسلسلة.

        يُستخدم في التسجيل (scoring) لمكافأة السلاسل المتوافقة طوبولوجياً مع prompt.

        Args:
            words: قائمة كلمات السلسلة
            pv_fn: دالة word → PV
            prompt_pvs: متجهات طورية للـ prompt (اختياري — للمقارنة)

        Returns:
            dict: معلومات طوبولوجية كاملة
        """
        pvs = [pv_fn(w) for w in words if w]
        if len(pvs) < 2:
            return {'bp': 0.0, 'wn': 0.0, 'prompt_sim': 0.0}

        inv = self.compute_invariants(pvs)
        result = {
            'bp': inv['berry_phase'],
            'wn': inv['winding_number'],
            'chain_length': len(pvs),
        }

        if prompt_pvs and len(prompt_pvs) >= 2:
            result['prompt_sim'] = self.topological_similarity(prompt_pvs, pvs)
        else:
            result['prompt_sim'] = 0.0

        return result
