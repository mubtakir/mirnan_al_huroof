# -*- coding: utf-8 -*-
"""مولّد الاسترخاء الرنيني — توليد متعدد الممرات عبر الاسترخاء الفيزيائي.

الفلسفة: فيزياء المواد — التبريد يُحسّن البلورة.
الممر الأول: توليد سريع (حرارة عالية = إنتروبيا عالية = حرية أكبر)
الممر الثاني: تنقيح (تبريد = تقليص الخيارات = دقة أعلى)
الممر الثالث: صقل (تجميد = اختيار أفضل مرشح)

هذا ليس chain-of-thought — لا نولّد كلمات وسيطة.
بل نولّد مساراً طورياً كاملاً، نقيّم تماسكه، نعيد توليد
الأجزاء الضعيفة فقط — كتنقيح بلورة بالنقر.

آلية الاسترخاء:
1. Pass 1 (Hot): توليد بـ β منخفض = حرارة عالية = استكشاف
2. Evaluate: قياس الترابط عبر CoherenceFeedback
3. Pass 2 (Warm): إعادة توليد المواضع الضعيفة بـ β أعلى
4. Pass 3 (Cold): تنقيح نهائي بـ β عالي = دقة عالية
"""
import numpy as np
import logging
from src.physics.coherence_feedback import CoherenceFeedback
from src.physics.constants import TOTAL_DIM

logger = logging.getLogger(__name__)


class RelaxationPass:
    """ممر استرخاء واحد — حالة حرارية محددة."""

    def __init__(self, name, beta, max_words, weak_threshold=0.3):
        self.name = name
        self.beta = beta
        self.max_words = max_words
        self.weak_threshold = weak_threshold


class MultiPassGenerator:
    """مولّد الاسترخاء الرنيني — توليد متعدد الممرات."""

    def __init__(self, generator, config=None):
        cfg = config or {}
        self.gen = generator
        self.feedback = CoherenceFeedback(config)
        self.n_passes = cfg.get('relaxation_passes', 3)
        self.beta_schedule = cfg.get('beta_schedule', None)
        self.min_coherence = cfg.get('min_coherence', 0.3)
        self.convergence_threshold = cfg.get('convergence_threshold', 0.05)
        self.history = []

    def _default_beta_schedule(self, base_beta):
        """جدول تبريد — β يزداد تدريجياً (حرارة تنخفض)."""
        if self.beta_schedule:
            return self.beta_schedule
        return [
            base_beta * 0.6,
            base_beta * 1.0,
            base_beta * 1.8,
        ][:self.n_passes]

    def generate(self, prompt, max_words=12, mode='standard', base_beta=None):
        """توليد متعدد الممرات.

        Args:
            prompt: النص المدخل
            max_words: الحد الأقصى للكلمات
            mode: نمط التوليد
            base_beta: β الأساسي (يُأخذ من Generator إن لم يُحدد)

        Returns:
            dict: {result, passes_report}
        """
        if base_beta is None:
            base_beta = self.gen.beta

        beta_schedule = self._default_beta_schedule(base_beta)
        passes_report = []

        original_beta = self.gen.beta
        original_kb = self.gen.entropy.k_B

        result_words = None
        prev_coherence = 0.0

        for pass_idx, beta in enumerate(beta_schedule):
            self.gen.beta = beta
            self.gen.entropy.k_B = original_kb

            if pass_idx == 0:
                result = self.gen.generate(prompt, max_words=max_words, mode=mode)
                if not result or not result.strip():
                    self.gen.beta = original_beta
                    return {'result': '', 'passes_report': passes_report}
                result_words = result.split()

            elif pass_idx > 0 and result_words:
                result_words, coherence = self._refine_pass(
                    prompt, result_words, max_words, beta, mode
                )

            if result_words:
                report = self.feedback.evaluate(result_words, self.gen)
                report['pass'] = pass_idx
                report['beta'] = float(beta)
                passes_report.append(report)

                coherence = report.get('overall', 0.0)
                if abs(coherence - prev_coherence) < self.convergence_threshold and pass_idx > 0:
                    logger.info(f"  استرخاء: تقارب في الممر {pass_idx+1}")
                    break
                prev_coherence = coherence

                if coherence >= 0.7:
                    logger.info(f"  استرخاء: ترابط كافٍ في الممر {pass_idx+1}")
                    break

                self.feedback.apply_feedback(report, self.gen)

        self.gen.beta = original_beta
        self.gen.entropy.k_B = original_kb

        final_result = ' '.join(result_words) if result_words else ''

        self.history.append({
            'prompt': prompt,
            'result': final_result,
            'passes': len(passes_report),
            'coherence': passes_report[-1]['overall'] if passes_report else 0.0,
        })

        return {
            'result': final_result,
            'passes_report': passes_report,
        }

    def _refine_pass(self, prompt, result_words, max_words, beta, mode):
        """ممر تنقيح — إعادة توليد المواضع الضعيفة فقط.

        يحلل الكلماتWeak ويستبدلها بمرشحات بديلة
        من مصفوفة K مع β أعلى (أكثر تحكماً).
        """
        weak_positions = self.feedback.identify_weak_positions(
            result_words, self.gen)

        if not weak_positions:
            coherence_report = self.feedback.evaluate(result_words, self.gen)
            return result_words, coherence_report.get('overall', 0.5)

        refined_words = result_words.copy()

        for pos, weak_word, weakness in weak_positions[:3]:
            if pos <= 0 or pos >= len(refined_words) - 1:
                continue

            prev_word = refined_words[pos - 1]
            wid = self.gen.vocab.word2id.get(prev_word)

            if wid is None or self.gen.K is None or wid >= self.gen.K.shape[0]:
                continue

            row = self.gen.K[wid].toarray().ravel()
            candidates = []
            order = np.argsort(row)[::-1]
            for cid in order[:50]:
                if row[cid] > 0:
                    cw = self.gen.vocab.id2word.get(cid)
                    if cw and cw not in refined_words and len(cw) >= 2:
                        cw_pv = self.gen._get_pv_fast(cw)
                        context_pvs = [
                            self.gen._get_pv_fast(refined_words[j])
                            for j in range(max(0, pos - 2), min(len(refined_words), pos + 2))
                            if j != pos
                        ]
                        if context_pvs:
                            target = np.mean(context_pvs, axis=0)
                            from src.physics.word_physics import phase_similarity
                            sim = phase_similarity(cw_pv[:22], target[:22])
                            candidates.append((cw, sim * row[cid]))

            if candidates:
                candidates.sort(key=lambda x: -x[1])
                best_replacement = candidates[0][0]
                refined_words[pos] = best_replacement

        coherence_report = self.feedback.evaluate(refined_words, self.gen)
        return refined_words, coherence_report.get('overall', 0.5)

    def _dialogue_relaxation(self, prompt, max_words=12):
        """استرخاء رنيني مخصص للحوار."""
        beta_schedule = [
            self.gen.beta * 0.5,
            self.gen.beta * 1.0,
            self.gen.beta * 1.5,
        ]

        original_beta = self.gen.beta
        prev_result = None

        for pass_idx, beta in enumerate(beta_schedule):
            self.gen.beta = beta
            self.gen.dialogue_mode = True

            prompt_tokens = [self.gen.vocab.id2word[self.gen.vocab.get(w)]
                            for w in prompt.split()
                            if self.gen.vocab.get(w) is not None]
            if not prompt_tokens:
                continue

            result = self.gen._dialogue_generate(prompt_tokens, prompt, max_words)
            self.gen.dialogue_mode = False

            if result and (prev_result is None or result != prev_result):
                prev_result = result

        self.gen.beta = original_beta
        return prev_result or ''