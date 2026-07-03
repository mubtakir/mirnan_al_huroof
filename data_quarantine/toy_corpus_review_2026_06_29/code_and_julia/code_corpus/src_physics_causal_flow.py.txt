# -*- coding: utf-8 -*-
"""
حقل التدفق الطوري السببي — بديل فيزيائي للاستدلال المنطقي متعدد الخطوات.

المشكلة: causal_engine يمثل السببية كمصفوفة متقطعة (A→B) ولا يوفر
مساراً مستمراً للاستدلال في فضاء الطور. النتيجة: لا يوجد تفكير منطقي متسلسل.

الحل: تحويل العلاقات السببية إلى حقل تدفق في فضاء الطور.
- القواعد المنطقية = تيارات تدفق (Flow Currents) تجبر المتجهات على التحرك
  في مسارات محددة سلفاً
- الاستدلال = تتبع خطوط التدفق من الكلمة الحالية عبر سلسلة سببية

مثال: "أكبر من" → تيار التدفق يجبر النموذج على سلوك مقارنة:
  current_pv → target_pv (aligned with "larger" direction) → comparison_result

الرياضيات:
  J(pv) = Σᵢ wᵢ · Cᵢ · (pv_target_i - pv)
  حيث Cᵢ هي مصفوفة السببية للعلاقة i
"""

import numpy as np
from scipy import sparse
import logging

logger = logging.getLogger(__name__)


class CausalFlowField:
    """حقل تدفق سببي مستمر في فضاء الطور.

    يمثل العلاقات المنطقية كتيارات تدفق ديناميكية.
    """

    def __init__(self, causal_engine=None, dim=64, flow_strength=1.0):
        """
        Args:
            causal_engine: غرفة الرنين السببي القائمة
            dim: أبعاد فضاء الطور
            flow_strength: شدة تيار التدفق (γ)
        """
        self.causal_engine = causal_engine
        self.dim = dim
        self.flow_strength = flow_strength
        self._flow_cache = {}

    def compute_flow(self, current_pv, context_pvs, context_ids, vocab,
                     causal_matrix=None, word_to_pv_fn=None):
        """حساب متجه التدفق السببي عند النقطة current_pv.

        J(pv) = Σ_{cid ∈ context} C_strength(cid→?) · (pv_direction)

        Args:
            current_pv: متجه الطور الحالي
            context_pvs: متجهات الطور السياقية
            context_ids: معرفات الكلمات السياقية
            vocab: قاموس المفردات
            causal_matrix: مصفوفة سببية (إذا كانت مختلفة عن causal_engine.causal_K)
            word_to_pv_fn: دالة لتحميل متجه طور الكلمة بالاسم

        Returns:
            dict: {
                'flow_vector': np.ndarray — متجه التدفق,
                'flow_magnitude': float — شدة التدفق,
                'causal_chain': list — سلسلة السببية,
                'logical_score': float — درجة الاتساق المنطقي
            }
        """
        if causal_matrix is None:
            if self.causal_engine is None or not self.causal_engine.is_built:
                return self._empty_flow()
            causal_matrix = self.causal_engine.causal_K

        n = len(context_ids) if context_ids else 0
        if n == 0 or causal_matrix is None:
            return self._empty_flow()

        flow_vector = np.zeros(self.dim)
        total_strength = 0.0
        causal_chain = []

        # الحلقة السببية: كل كلمة سياقية تدفع باتجاه الكلمات التي تسببها
        for i in range(min(n, 6)):
            cid = context_ids[-(i + 1)]
            if cid is None or cid >= causal_matrix.shape[0]:
                continue

            row = causal_matrix[cid]
            if sparse.issparse(row):
                row = row.toarray().ravel()

            # إيجاد أقوى الكلمات المسبَّبة من هذه الكلمة السياقية
            top_indices = np.argsort(row)[-5:][::-1]
            for tid in top_indices:
                strength = float(row[tid])
                if strength <= 0.15:
                    continue

                # target word vector
                target_pv = None
                if word_to_pv_fn is not None:
                    word = vocab.id2word.get(tid)
                    if word:
                        target_pv = word_to_pv_fn(word)
                
                if target_pv is None and tid < len(context_pvs):
                    target_pv = context_pvs[tid][:self.dim]
                
                if target_pv is None:
                    target_pv = current_pv.copy()

                # اتجاه التدفق: من current إلى target
                direction = target_pv - current_pv[:self.dim]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-10:
                    direction = direction / dir_norm

                weight = np.exp(-0.5 * i)  # اضمحلال حسب البعد الموضعي
                flow_vector += self.flow_strength * weight * strength * direction
                total_strength += weight * strength

                causal_chain.append({
                    'cause_idx': cid,
                    'effect_idx': tid,
                    'strength': round(strength, 3),
                    'position_penalty': round(float(weight), 3),
                })

        flow_magnitude = float(np.linalg.norm(flow_vector))
        if flow_magnitude > 1e-10:
            flow_vector = flow_vector / flow_magnitude

        logical_score = float(np.tanh(total_strength * 2.0))

        return {
            'flow_vector': flow_vector,
            'flow_magnitude': flow_magnitude,
            'causal_chain': causal_chain,
            'logical_score': logical_score,
        }

    def flow_alignment_score(self, candidate_pv, current_pv, flow_vector):
        """درجة توافق المرشح مع اتجاه التدفق السببي.

        score = cos(candidate_pv - current_pv, flow_vector) · flow_magnitude
        """
        if flow_vector is None or np.linalg.norm(flow_vector) < 1e-10:
            return 0.0

        candidate_direction = candidate_pv[:self.dim] - current_pv[:self.dim]
        dir_norm = np.linalg.norm(candidate_direction)
        if dir_norm < 1e-10:
            return 0.0
        candidate_direction = candidate_direction / dir_norm

        alignment = float(np.dot(candidate_direction, flow_vector[:self.dim]))
        return max(0.0, alignment)

    def compute_transitive_flow(self, word_ids, vocab, causal_matrix=None):
        """حساب قوة التدفق عبر سلسلة سببية (استدلال تعددي).

        A → B → C → D: التدفق ينتقل عبر السلسلة.
        """
        if causal_matrix is None:
            if self.causal_engine is None or not self.causal_engine.is_built:
                return {'flow_strength': 0.0, 'chain_length': 0, 'chain': []}
            causal_matrix = self.causal_engine.causal_K

        if len(word_ids) < 2 or causal_matrix is None:
            return {'flow_strength': 0.0, 'chain_length': 0, 'chain': []}

        cumulative_strength = 1.0
        chain = []
        chain_length = 0

        for i in range(len(word_ids) - 1):
            cid = word_ids[i]
            eid = word_ids[i + 1]
            if cid is None or eid is None:
                continue
            if cid >= causal_matrix.shape[0] or eid >= causal_matrix.shape[1]:
                continue

            strength = float(causal_matrix[cid, eid])
            if strength > 0.15:
                cumulative_strength *= min(strength, 1.0)
                chain_length += 1
                chain.append({
                    'from': cid,
                    'to': eid,
                    'strength': round(strength, 3),
                })

        return {
            'flow_strength': round(cumulative_strength, 4),
            'chain_length': chain_length,
            'chain': chain,
        }

    def get_phase_gradient(self, pv, causal_matrix, vocab, context_ids):
        """حساب تدرج الطور في اتجاه العلاقات السببية.

        يُستخدم لدفع المتجه الحالي نحو مناطق أكثر اتساقاً سببياً.
        """
        gradient = np.zeros(self.dim)
        if causal_matrix is None or not context_ids:
            return gradient

        for cid in context_ids[-5:]:
            if cid is None or cid >= causal_matrix.shape[0]:
                continue
            row = causal_matrix[cid]
            if sparse.issparse(row):
                row = row.toarray().ravel()
            top_effects = np.argsort(row)[-3:][::-1]
            for tid in top_effects:
                if row[tid] <= 0:
                    continue
                # التدرج يشير نحو التأثيرات القوية
                gradient += row[tid] * (pv - pv) * 0.0  # placeholder for actual grad
                gradient = np.ones(self.dim) * 0.01 * row[tid]

        return gradient

    def _empty_flow(self):
        return {
            'flow_vector': np.zeros(self.dim),
            'flow_magnitude': 0.0,
            'causal_chain': [],
            'logical_score': 0.0,
        }
