# -*- coding: utf-8 -*-
"""تغذية راجعة متماسكة — تقييم ذاتي وتصحيح عبر قياس الترابط.

الفلسفة: النظام الفيزيائي يعرف حالة النظام من طاقته وإنتروبياه.
بعد التوليد، نقيس الترابط الداخلي للجملة المُولّدة:
- رنين تسلسلي: مدى توافق كل كلمة مع سابقتها
- انسجام طوري: مدى توافق المتجهات الطورية في سلسلة واحدة
- طاقة متسقة: هل تتناقص الطاقة بسلاسة أم فيها قفزات؟
- إنتروبيا هيكلية: هل الجملة متماسكة أم فوضوية؟

هذا ليس error signal من backpropagation — إنه قياس فيزيائي
لحالة النظام بعد التوليد، يُستخدم لضبط المعاملات ديناميكياً.
"""
import numpy as np
import logging
from src.physics.word_physics import phase_similarity
from src.physics.constants import TOTAL_DIM

logger = logging.getLogger(__name__)


class CoherenceFeedback:
    """محرك التغذية الراجعة المتماسكة — تقييم ذاتي وتصحيح."""

    def __init__(self, config=None):
        cfg = config or {}
        self.min_words = cfg.get('coherence_min_words', 3)
        self.coherence_threshold = cfg.get('coherence_threshold', 0.3)
        self结构调整_rate = cfg.get('structure_adjust_rate', 0.1)
        self.beta_adjust_rate = cfg.get('beta_adjust_rate', 0.05)
        self.kb_adjust_rate = cfg.get('kb_adjust_rate', 0.02)
        self.history = []

    def measure_sequential_resonance(self, words, all_pv, vocab):
        """قياس الرنين التسلسلي — مدى توافق كل كلمة مع سابقتها.

        Returns:
            float: متوسط التشابه الطوري بين الكلمات المتتالية
        """
        if len(words) < 2:
            return 1.0

        similarities = []
        for i in range(1, len(words)):
            wid_prev = vocab.word2id.get(words[i - 1])
            wid_curr = vocab.word2id.get(words[i])
            if wid_prev is not None and wid_curr is not None:
                if wid_prev < len(all_pv) and wid_curr < len(all_pv):
                    sim = phase_similarity(
                        all_pv[wid_prev][:22],
                        all_pv[wid_curr][:22]
                    )
                    similarities.append(sim)

        return float(np.mean(similarities)) if similarities else 0.5

    def measure_phase_coherence(self, words, all_pv, vocab):
        """قياس الانسجام الطوري — مدى تماسك المتجهات كبوقع واحد.

        Returns:
            float: التماسك الطوري (0=فوضوي, 1=متماسك)
        """
        if len(words) < self.min_words:
            return 0.5

        pvs = []
        for w in words:
            wid = vocab.word2id.get(w)
            if wid is not None and wid < len(all_pv):
                pvs.append(all_pv[wid])

        if len(pvs) < 2:
            return 0.5

        pv_matrix = np.array([p[:22] for p in pvs])
        mean_pv = np.mean(pv_matrix, axis=0)
        mean_norm = np.linalg.norm(mean_pv)
        if mean_norm < 1e-10:
            return 0.5

        alignments = [phase_similarity(p[:22], mean_pv) for p in pvs]
        coherence = float(np.mean(alignments))
        return coherence

    def measure_energy_smoothness(self, words, generator):
        """قياس سلاسة الطاقة — هل تتناقص بسلاسة أم فيها قفزات؟

        Returns:
            float: سلاسة الطاقة (0=متقطع, 1=سلس)
        """
        if len(words) < 2:
            return 1.0

        masses = []
        for w in words:
            try:
                m = generator._dyn_mass(w)
                masses.append(m)
            except Exception:
                masses.append(1.0)

        if len(masses) < 2:
            return 1.0

        diffs = np.abs(np.diff(masses))
        max_mass = max(masses) if masses else 1.0
        normalized_diffs = diffs / max(max_mass, 0.01)
        smoothness = 1.0 - float(np.mean(normalized_diffs))
        return max(0.0, min(1.0, smoothness))

    def measure_structural_entropy(self, words, morpho=None):
        """قياس الإنتروبيا الهيكلية — تنوع أقسام الكلام.

        Returns:
            float: إنتروبيا هيكلية (منخفضة=مملة، عالية=فوضوية، 0.7=مثالية)
        """
        if len(words) < 3:
            return 0.5

        pos_tags = []
        for w in words:
            if morpho:
                try:
                    pos = morpho.get_pos(w)
                    pos_tags.append(pos if pos else 'unknown')
                except Exception:
                    pos_tags.append('unknown')
            else:
                pos_tags.append('unknown')

        unique_pos = set(pos_tags)
        if not unique_pos:
            return 0.5

        n = len(pos_tags)
        probs = [pos_tags.count(p) / n for p in unique_pos]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(len(unique_pos)) if len(unique_pos) > 1 else 1.0
        normalized = entropy / max(max_entropy, 0.01)
        return normalized

    def evaluate(self, generated_words, generator):
        """تقييم شامل للجملة المُولّدة — كل المقاييس الفيزيائية.

        Args:
            generated_words: قائمة الكلمات المُولّدة
            generator: مثيل Generator

        Returns:
            dict: تقرير شامل
        """
        if len(generated_words) < self.min_words:
            return {
                'coherence': 0.5,
                'sequential_resonance': 0.5,
                'phase_coherence': 0.5,
                'energy_smoothness': 1.0,
                'structural_entropy': 0.5,
                'overall': 0.5,
                'adjustments': {},
            }

        seq_res = self.measure_sequential_resonance(
            generated_words, generator._all_pv, generator.vocab)
        phase_coh = self.measure_phase_coherence(
            generated_words, generator._all_pv, generator.vocab)
        energy_sm = self.measure_energy_smoothness(
            generated_words, generator)
        struct_ent = self.measure_structural_entropy(
            generated_words, generator.morpho)

        coherence = (
            0.3 * seq_res +
            0.3 * phase_coh +
            0.2 * energy_sm +
            0.2 * (1.0 - abs(struct_ent - 0.7) / 0.7)
        )

        adjustments = self._compute_adjustments(coherence, seq_res, phase_coh)

        report = {
            'coherence': float(coherence),
            'sequential_resonance': float(seq_res),
            'phase_coherence': float(phase_coh),
            'energy_smoothness': float(energy_sm),
            'structural_entropy': float(struct_ent),
            'overall': float(coherence),
            'adjustments': adjustments,
        }

        self.history.append(report)
        return report

    def _compute_adjustments(self, coherence, seq_res, phase_coh):
        """حساب التعديلات المطلوبة بناءً على التقييم.

        مثل نظام فيزيائي يُعدّل درجة حرارته:
        - إذا كان الترابط منخفضاً: نزيد β (نشاط أقل = أكثر تحكماً)
        - إذا كان الترابط عالياً جداً: نخفض β (مزيد من الحرية)
        - إذا كان التسلسل متقطعاً: نزيد أوزان الجاذبية
        """
        adjustments = {}

        if coherence < 0.3:
            adjustments['beta_delta'] = 0.3
            adjustments['k_B_delta'] = -0.1
            adjustments['description'] = 'ترابط منخفض — نزيد التحكم'
        elif coherence < 0.5:
            adjustments['beta_delta'] = 0.1
            adjustments['k_B_delta'] = -0.05
            adjustments['description'] = 'ترابط متوسط — تعديل طفيف'
        elif coherence > 0.85:
            adjustments['beta_delta'] = -0.1
            adjustments['k_B_delta'] = 0.05
            adjustments['description'] = 'ترابط عالٍ — مزيد من الحرية'
        else:
            adjustments['beta_delta'] = 0.0
            adjustments['k_B_delta'] = 0.0
            adjustments['description'] = 'ترابط جيد — لا تعديل'

        if seq_res < 0.3:
            adjustments['gravity_boost'] = 1.3
            adjustments['syntax_boost'] = 1.2
        elif seq_res > 0.8:
            adjustments['gravity_boost'] = 0.9
            adjustments['syntax_boost'] = 1.0
        else:
            adjustments['gravity_boost'] = 1.0
            adjustments['syntax_boost'] = 1.0

        return adjustments

    def apply_feedback(self, report, generator):
        """تطبيق التغذية الراجعة على المُولّد — ضبط فيزيائي.

        Args:
            report: تقرير التقييم من evaluate()
            generator: مثيل Generator
        """
        adj = report.get('adjustments', {})

        beta_delta = adj.get('beta_delta', 0.0)
        kb_delta = adj.get('k_B_delta', 0.0)

        if abs(beta_delta) > 0.01:
            generator.beta = max(0.1, min(10.0, generator.beta + beta_delta))

        if abs(kb_delta) > 0.01:
            generator.entropy.k_B = max(0.1, min(10.0, generator.entropy.k_B + kb_delta))

        gravity_boost = adj.get('gravity_boost', 1.0)
        if abs(gravity_boost - 1.0) > 0.01 and 'gravity' in generator.W:
            generator.W['gravity'] *= gravity_boost

        syntax_boost = adj.get('syntax_boost', 1.0)
        if abs(syntax_boost - 1.0) > 0.01 and 'syntax' in generator.W:
            generator.W['syntax'] *= syntax_boost

    def identify_weak_positions(self, generated_words, generator):
        """تحديد المواضع الضعيفة في التوليد — كلمات تكسر الرنين.

        Returns:
            list: أزواج (موضع_الكسر, كلمة_ضعيفة, درجة_الضعف)
        """
        if len(generated_words) < 3:
            return []

        weak_positions = []
        for i in range(1, len(generated_words) - 1):
            w_prev = generated_words[i - 1]
            w_curr = generated_words[i]
            w_next = generated_words[i + 1]

            wid_prev = generator.vocab.word2id.get(w_prev)
            wid_curr = generator.vocab.word2id.get(w_curr)
            wid_next = generator.vocab.word2id.get(w_next)

            if any(wid is None or wid >= len(generator._all_pv)
                   for wid in [wid_prev, wid_curr, wid_next]):
                continue

            sim_prev = phase_similarity(
                generator._all_pv[wid_prev][:22],
                generator._all_pv[wid_curr][:22])
            sim_next = phase_similarity(
                generator._all_pv[wid_curr][:22],
                generator._all_pv[wid_next][:22])

            avg_surround = (sim_prev + sim_next) / 2.0
            if avg_surround < 0.15:
                weakness = 1.0 - avg_surround
                weak_positions.append((i, w_curr, float(weakness)))

        return sorted(weak_positions, key=lambda x: -x[2])