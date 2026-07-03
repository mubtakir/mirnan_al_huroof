# -*- coding: utf-8 -*-
"""K العلاقية — مصفوفة اقتران بنظائر توافقية لنسب العلاقات.

الفلسفة: مصفوفة K الحالية ثنائية — تسجل فقط أن w₁ و w₂ تظهران معاً.
لكن العلاقة بين "أكل" و "تفاحة" ليست كالعلاقة بين "تفاحة" و "حمراء":
- أكل + تفاحة = فاعل + مفعول (علاقة فعلية)
- تفاحة + حمراء = موصوف + صفة (علاقة وصفية)

بدلاً من مصفوفة 3D (مكلفة)، نُشفّر نوع العلاقة كتردد توافقي
ضمن نفس القيمة في K:
  K[i,j] = amplitude × cos(ω_relation × phase_ij)

ω_relation = 0 → علاقة تركيبية (syn)
ω_relation = 1 → علاقة دلالية (sem)
ω_relation = 2 → علاقة فعلية (verb-object)
ω_relation = 3 → علاقة وصفية (adj-noun)
ω_relation = 4 → علاقة سببية (cause-effect)
ω_relation = 5 → علاقة حوارية (dialogue)

عند حساب الرنين نُفكك الإشارة إلى مكوناتها التوافقية.
"""
import numpy as np
import logging
from scipy import sparse

logger = logging.getLogger(__name__)

RELATION_FREQUENCY = {
    'syn': 0,
    'sem': 1,
    'verb_obj': 2,
    'adj_noun': 3,
    'causal': 4,
    'dialogue': 5,
}

RELATION_NAMES = {v: k for k, v in RELATION_FREQUENCY.items()}


class RelationalK:
    """مصفوفة K علاقية بنظائر توافقية."""

    def __init__(self, config=None):
        cfg = config or {}
        self.n_relations = len(RELATION_FREQUENCY)
        self.base_freq = cfg.get('relational_base_freq', 2.0)
        self.harmonic_amplitude = cfg.get('harmonic_amplitude', 0.15)

    def encode(self, amplitude, relation_type='sem'):
        """ترميز علاقة كتردد توافقي ضمن قيمة موجبة.

        Args:
            amplitude: قوة الارتباط (من K_sem, K_syn, إلخ)
            relation_type: نوع العلاقة

        Returns:
            float: القيمة المشفرة بالنظير التوافقي
        """
        if amplitude <= 0:
            return 0.0
        freq = RELATION_FREQUENCY.get(relation_type, 1)
        harmonic = self.harmonic_amplitude * np.sin(self.base_freq * freq)
        return amplitude * (1.0 + harmonic)

    def decode(self, value):
        """فك تشفير القيمة إلى مكوناتها.

        Args:
            value: القيمة المشفرة

        Returns:
            dict: {base_amplitude, harmonics: {relation: contribution}}
        """
        if value <= 0:
            return {'base': 0.0, 'harmonics': {}}

        base = value
        harmonics = {}
        for name, freq in RELATION_FREQUENCY.items():
            harmonic = self.harmonic_amplitude * np.sin(self.base_freq * freq)
            harmonics[name] = value * harmonic / (1.0 + sum(
                self.harmonic_amplitude * np.sin(self.base_freq * f)
                for f in RELATION_FREQUENCY.values()
            ))

        return {'base': base, 'harmonics': harmonics}

    def extract_relation(self, value, relation_type='sem'):
        """استخراج قوة علاقة محددة من قيمة مشفرة.

        Args:
            value: القيمة المشفرة في K
            relation_type: نوع العلاقة المطلوب

        Returns:
            float: قوة العلاقة المحددة
        """
        if value <= 0:
            return 0.0
        freq = RELATION_FREQUENCY.get(relation_type, 1)
        harmonic = np.sin(self.base_freq * freq)
        return abs(value) * (0.7 + 0.3 * max(0, harmonic))

    def build_from_syntax(self, K_sem, K_syn, K_dial=None, corpus_texts=None, vocab=None):
        """بناء مصفوفة K علاقية من K الموجودة + تحليل تركيبي.

        يعمل بالكامل على مصفوفات sparse بدون تحويلها إلى dense.
        الحل الرياضي:
          causal_contribution = elementwise_min(K_sem, K_syn) * 0.3
          base = K_sem + causal_contribution
          ثم نطبق معامل التشفير التوافقي المناسب لكل منطقة.

        المناطق (بأولوية تنازلية):
          1. dial_mask  : حيث K_dial > 0           → encode(..., 'dialogue')
          2. syn_mask   : حيث K_syn > K_sem         → encode(..., 'syn')
          3. باقي K_sem : حيث K_sem > 0             → encode(..., 'sem')
        """
        if K_sem is None:
            return None

        V = K_sem.shape[0]
        logger.info(f"  بناء K علاقية: {V}×{V}")

        # ── تأكد أن كل المصفوفات sparse CSR ──────────────────────────────
        if not sparse.issparse(K_sem):
            K_sem = sparse.csr_matrix(K_sem)
        else:
            K_sem = K_sem.tocsr()

        if K_syn is not None:
            if not sparse.issparse(K_syn):
                K_syn = sparse.csr_matrix(K_syn)
            else:
                K_syn = K_syn.tocsr()

        if K_dial is not None:
            if not sparse.issparse(K_dial):
                K_dial = sparse.csr_matrix(K_dial)
            else:
                K_dial = K_dial.tocsr()

        # ── causal_contribution = elementwise_min(K_sem, K_syn) * 0.3 ────
        # sparse elementwise min: min(A,B) = (A+B - |A-B|) / 2
        # لكن هذا مكلف؛ نستخدم تقريباً أبسط: min ≈ K_syn حين K_syn < K_sem
        if K_syn is not None:
            # sparse multiply = elementwise product (كلاهما sparse → يبقى sparse)
            # نقدّر min(a,b) ≈ a * (b/(a+b+ε)) عبر sparse ops
            # الأبسط: نأخذ K_syn.multiply(K_sem>0) كتقريب جيد للـ min حين K_syn صغير
            # الطريقة الصحيحة والفعالة للـ sparse:
            diff = K_sem - K_syn          # sparse - sparse
            # الأجزاء حيث K_syn >= K_sem → diff <= 0
            # elementwise max(0, K_syn) عبر K_syn ذاتها (قيم موجبة فقط)
            # min(K_sem, K_syn) = K_sem - max(0, K_sem - K_syn)
            diff_pos = diff.copy()
            diff_pos.data = np.maximum(diff_pos.data, 0)   # max(0, K_sem - K_syn)
            causal = (K_sem - diff_pos).multiply(0.3)      # min(K_sem,K_syn) * 0.3
        else:
            causal = sparse.csr_matrix(K_sem.shape, dtype=np.float64)

        # ── base = K_sem + causal ─────────────────────────────────────────
        base = K_sem + causal   # sparse + sparse

        # ── معاملات التشفير التوافقي ──────────────────────────────────────
        coeff_sem  = 1.0 + self.harmonic_amplitude * np.sin(self.base_freq * RELATION_FREQUENCY['sem'])
        coeff_syn  = 1.0 + self.harmonic_amplitude * np.sin(self.base_freq * RELATION_FREQUENCY['syn'])
        coeff_dial = 1.0 + self.harmonic_amplitude * np.sin(self.base_freq * RELATION_FREQUENCY['dialogue'])

        # ── ابدأ بالطبقة الأساسية (sem) ──────────────────────────────────
        result = base.multiply(coeff_sem)   # الكل بمعامل sem كقاعدة

        # ── طبقة syn: حيث K_syn > K_sem (وكلاهما > 0) ───────────────────
        if K_syn is not None:
            # syn_mask: المواضع حيث K_syn > K_sem
            syn_gt = diff.copy()   # K_sem - K_syn
            syn_gt.data = np.where(syn_gt.data < 0, 1.0, 0.0)  # 1 حيث K_syn > K_sem
            # اطبق تصحيح المعامل فقط على هذه المواضع
            delta_syn = coeff_syn - coeff_sem   # فرق المعاملين
            result = result + base.multiply(syn_gt).multiply(delta_syn)

        # ── طبقة dialogue: حيث K_dial > 0 ───────────────────────────────
        if K_dial is not None:
            # dial_mask: المواضع حيث K_dial > 0
            dial_mask = K_dial.copy()
            dial_mask.data = np.ones_like(dial_mask.data)  # 1 لكل nonzero
            delta_dial = coeff_dial - coeff_sem
            result = result + base.multiply(dial_mask).multiply(delta_dial)

        return result.tocsr()

    def relational_score(self, word_id, context_ids, K, relation_type='sem'):
        """حساب درجة علاقية لنوع محدد من العلاقات.

        Args:
            word_id: معرّف الكلمة المرشحة
            context_ids: معرّفات السياق
            K: مصفوفة K (قد تكون علاقية)
            relation_type: نوع العلاقة

        Returns:
            float: الدرجة العلاقية
        """
        if K is None or word_id is None or word_id >= K.shape[1]:
            return 0.0

        total_score = 0.0
        for cid in context_ids:
            if cid is None or cid >= K.shape[0]:
                continue
            raw_val = float(K[cid, word_id])
            if raw_val > 0:
                total_score += self.extract_relation(raw_val, relation_type)

        return total_score / max(len(context_ids), 1)