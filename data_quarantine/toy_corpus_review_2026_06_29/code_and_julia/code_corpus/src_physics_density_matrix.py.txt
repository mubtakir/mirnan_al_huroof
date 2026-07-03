# -*- coding: utf-8 -*-
"""
مصفوفة الكثافة الكمومية — بديل فيزيائي لمتوسط المتجهات السياقية.

المشكلة: متوسط np.average(pvs) يدمج المتجهات دماراً — تداخل إتلافي
يفقد التفاصيل الحرة والعلاقات البينية بين الكلمات.

الحل: تمثيل السياق كحالة كمومية مختلطة (Mixed State) عبر مصفوفة كثافة:
ρ = Σᵢ pᵢ |ψᵢ⟩⟨ψᵢ|

حيث:
- |ψᵢ⟩: متجه طور الكلمة i (64-بُعد)
- pᵢ: وزن الكلمة (اضمحلال أسي حسب الموضع)
- ρ: مصفوفة 64×64 تحافظ على كل العلاقات البينية والتشابكات

عند تقييم مرشح:
  R(v_cand) = ⟨v_cand| ρ |v_cand⟩ = Σᵢ pᵢ · |⟨v_cand|ψᵢ⟩|²

هذا يحافظ على مساهمات كل كلمة سياقية منفردة — لا دمج إتلافي.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class QuantumDensityMatrix:
    """ممثل السياق كمصفوفة كثافة كمومية (64×64)."""

    def __init__(self, dim=64, decay_rate=0.8):
        self.dim = dim
        self.decay_rate = decay_rate
        self.rho = None
        self.n_words = 0
        self._word_count = 0

    def build(self, pv_list, dims=None):
        """بناء مصفوفة الكثافة من قائمة متجهات طورية.

        Args:
            pv_list: قائمة متجهات الطور (كل منها 64-بُعداً)
            dims: الأبعاد المستخدمة (None = كل الـ 64 بُعداً)

        Returns:
            np.ndarray: مصفوفة الكثافة (dim, dim)
        """
        if dims is not None:
            pv_list = [pv[dims[0]:dims[1]] if isinstance(dims, tuple) else pv[:dims]
                       for pv in pv_list]
            d = dims if isinstance(dims, int) else (dims[1] - dims[0])
        else:
            d = self.dim

        n = len(pv_list)
        if n == 0:
            self.rho = np.zeros((d, d))
            self.n_words = 0
            return self.rho

        weights = np.exp(-self.decay_rate * np.arange(n))[::-1]
        weights = weights / weights.sum()

        rho = np.zeros((d, d))
        for i, pv in enumerate(pv_list):
            v = pv[:d].copy()
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
            rho += weights[i] * np.outer(v, v)

        self.rho = rho
        self.n_words = n
        self._word_count = n
        return self.rho

    def build_subspace(self, pv_list, start, end):
        """بناء مصفوفة كثافة لفضاء جزئي محدد."""
        return self.build(pv_list, dims=(start, end))

    def resonance(self, candidate_pv, dims=None):
        """قياس رنين مرشح مع مصفوفة الكثافة.

        R = ⟨v| ρ |v⟩ = Σᵢ pᵢ · |⟨v|ψᵢ⟩|²

        Args:
            candidate_pv: متجه المرشح
            dims: الأبعاد المستخدمة (None = كل الأبعاد)

        Returns:
            float: درجة الرنين [0, 1]
        """
        if self.rho is None or self.n_words == 0:
            return 0.0

        if dims is not None:
            if isinstance(dims, tuple):
                rho = self.rho[dims[0]:dims[1], dims[0]:dims[1]]
                v = candidate_pv[dims[0]:dims[1]].copy()
            else:
                rho = self.rho[:dims, :dims]
                v = candidate_pv[:dims].copy()
        else:
            rho = self.rho
            v = candidate_pv.copy()

        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm

        resonance = float(v.T @ rho @ v)
        return max(0.0, min(1.0, resonance))

    def resonance_pairwise(self, candidate_pv, pv_list, dims=None):
        """رنين مباشر (بدون تخزين) — للمقارنة مع build+resonance."""
        if not pv_list:
            return 0.0
        if dims is not None:
            if isinstance(dims, tuple):
                v = candidate_pv[dims[0]:dims[1]].copy()
                pvs = [p[dims[0]:dims[1]].copy() for p in pv_list]
            else:
                v = candidate_pv[:dims].copy()
                pvs = [p[:dims].copy() for p in pv_list]
        else:
            v = candidate_pv.copy()
            pvs = [p.copy() for p in pv_list]

        v_norm = np.linalg.norm(v)
        if v_norm > 1e-10:
            v = v / v_norm

        weights = np.exp(-self.decay_rate * np.arange(len(pvs)))[::-1]
        weights = weights / weights.sum()

        resonance = 0.0
        for i, pv_i in enumerate(pvs):
            pv_norm = np.linalg.norm(pv_i)
            if pv_norm > 1e-10:
                pv_i = pv_i / pv_norm
            dot = float(np.dot(v, pv_i))
            resonance += weights[i] * (dot ** 2)

        return max(0.0, min(1.0, resonance))

    def get_trace(self):
        """أثر المصفوفة — يقيس إجمالي "الطاقة" في الحالة المختلطة."""
        if self.rho is None:
            return 0.0
        return float(np.trace(self.rho))

    def get_purity(self):
        """درجة النقاء Tr(ρ²) — 1.0 = حالة نقية، < 1.0 = حالة مختلطة."""
        if self.rho is None:
            return 0.0
        rho2 = self.rho @ self.rho
        return float(np.trace(rho2))

    def get_entanglement_entropy(self):
        """إنتروبيا فون نيومان: S = -Tr(ρ ln ρ).

        تقيس درجة "التشابك" في السياق — كلما زادت، كان السياق أكثر ثراءً.
        """
        if self.rho is None or self.n_words == 0:
            return 0.0
        eigenvals = np.linalg.eigvalsh(self.rho)
        eigenvals = np.maximum(eigenvals, 1e-12)
        entropy = -np.sum(eigenvals * np.log(eigenvals))
        return float(entropy)

    def dominant_context(self, pv_list, top_k=3):
        """استخراج الكلمات الأكثر هيمنة في الحالة الكمومية.

        يقيس Tr(ρ · |ψᵢ⟩⟨ψᵢ|) لكل كلمة سياقية.
        """
        if self.rho is None or not pv_list:
            return []
        scores = []
        for i, pv in enumerate(pv_list):
            v = pv[:self.dim].copy()
            v_norm = np.linalg.norm(v)
            if v_norm > 1e-10:
                v = v / v_norm
            contrib = float(v.T @ self.rho @ v)
            scores.append((contrib, i))
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]
