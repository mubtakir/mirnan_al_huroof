# -*- coding: utf-8 -*-
"""معايرة الرنين — ضبط ذاتي لأوزان التسجيل عبر تقليل طاقة الرنين.

الفلسفة: كترونة آلة موسيقية — نعزف نغمة، نقارنها بمرجع، نشدّ أو نرخي الوتر.
المرجع هنا هو مصفوفة K نفسها: إذا تنبأنا بالكلمة التالية بنجاح من K،
فالأوزان صحيحة. إذا فشلنا، نُعدّل الأوزان نحو زيادة رنين التنبؤ الصحيح.

الآلية:
1. نأخذ عينات من النص المرجعي (كلمات متسلسلة)
2. نُولّد من كل عينة باستخدام الأوزان الحالية
3. نقيس رنين التوليد مع المتوقع (coherence_energy)
4. نُعدّل كل وزن بنسبة انحراف رنينه عن المتوسط
5. نُكرر حتى الاستقرار

هذا ليس backpropagation — إنه معايرة رنينية:
- لا توجد مشتقات
- لا توجد شبكة عصبية
- الوزن يُعدّل بناءً على مدى رنين إشارته مع التنبؤ الصحيح
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ResonanceCalibrator:
    """معاير رنيني ذاتي لأوزان التسجيل."""

    def __init__(self, config=None):
        cfg = config or {}
        self.learning_rate = cfg.get('calibration_lr', 0.02)
        self.min_samples = cfg.get('calibration_min_samples', 50)
        self.max_iterations = cfg.get('calibration_max_iterations', 10)
        self.convergence_threshold = cfg.get('calibration_convergence', 0.001)
        self.history = []

    def _measure_signal_resonance(self, weight_name, score_components, correct_word, all_pvs, K, vocab):
        """قياس رنين إشارة واحدة مع التنبؤ الصحيح.

        Returns:
            float: مدى توافق هذه الإشارة مع اختيار الكلمة الصحيحة
        """
        if weight_name not in score_components:
            return 0.0
        signal = score_components[weight_name]
        if correct_word is None:
            return abs(signal)
        wid = vocab.word2id.get(correct_word)
        if wid is None:
            return abs(signal)
        correct_pv = all_pvs[wid] if wid < len(all_pvs) else None
        if correct_pv is None:
            return abs(signal)
        sign = 1.0 if signal >= 0 else -1.0
        magnitude = abs(signal)
        correct_resonance = 0.0
        if K is not None and wid < K.shape[0]:
            row = K[wid].toarray().ravel() if hasattr(K[wid], 'toarray') else K[wid]
            if len(row) > 0:
                correct_resonance = float(np.mean(row[row > 0])) if np.any(row > 0) else 0.0
        return sign * min(magnitude, 1.0) * (1.0 + correct_resonance)

    def calibrate(self, generator, reference_texts, config=None):
        """معايرة الأوزان باستخدام نصوص مرجعية.

        Args:
            generator: مثيل Generator بالأوزان الحالية
            reference_texts: قائمة نصوص مرجعية
            config: إعدادات إضافية

        Returns:
            dict: الأوزان المُعايرة
        """
        cfg = config or {}
        lr = cfg.get('lr', self.learning_rate)
        max_iter = cfg.get('max_iterations', self.max_iterations)
        convergence = cfg.get('convergence', self.convergence_threshold)

        vocab = generator.vocab
        K = generator.K_sem

        sequences = self._extract_sequences(reference_texts, vocab, self.min_samples)
        if len(sequences) < 5:
            logger.warning(f"تسلسلات غير كافية للمعايرة: {len(sequences)}")
            return generator.W.copy()

        current_W = generator.W.copy()
        logger.info(f"  معايرة رنينية: {len(sequences)} تسلسل، {len(current_W)} وزن")

        for iteration in range(max_iter):
            weight_scores = {k: [] for k in current_W}
            coherence_scores = []

            for seq in sequences:
                if len(seq) < 4:
                    continue
                context = seq[:-1]
                correct = seq[-1]

                context_pvs = []
                context_ids = []
                context_words = []
                for w in context:
                    wid = vocab.word2id.get(w)
                    if wid is not None and wid < len(generator._all_pv):
                        context_pvs.append(generator._all_pv[wid])
                        context_ids.append(wid)
                        context_words.append(w)

                if not context_ids:
                    continue

                correct_id = vocab.word2id.get(correct)
                if correct_id is None:
                    continue
                correct_pv = generator._all_pv[correct_id] if correct_id < len(generator._all_pv) else None
                if correct_pv is None:
                    continue

                prompt_pv = [generator._get_pv_fast(w) for w in context_words[:3]]
                used = set(context_words)
                prev_word = context_words[-1] if context_words else None
                beta_cur = generator.beta
                k_B_cur = generator.entropy.k_B

                all_pv = context_pvs
                correct_score = generator._score(
                    correct, used, all_pv, prompt_pv,
                    gen_pos=len(context), total_pos=len(context) + 6,
                    prev_word=prev_word, context_ids=context_ids,
                    context_words=context_words, k_B_cur=k_B_cur,
                    beta_cur=beta_cur, S=None
                )
                coherence_scores.append(correct_score)

                k_val = 0.0
                if K is not None and context_ids and correct_id is not None:
                    last_cid = context_ids[-1] if context_ids else None
                    if last_cid is not None and last_cid < K.shape[0] and correct_id < K.shape[1]:
                        k_val = float(K[last_cid, correct_id])

                for wname in weight_scores:
                    contribution = self._estimate_contribution(
                        generator, wname, correct, correct_pv, context_pvs,
                        correct_score, k_val
                    )
                    weight_scores[wname].append(contribution)

            if not coherence_scores:
                break

            avg_coherence = np.mean(coherence_scores)
            prev_W = current_W.copy()

            for wname in current_W:
                if wname not in weight_scores or not weight_scores[wname]:
                    continue
                scores = weight_scores[wname]
                mean_s = np.mean(scores)
                std_s = np.std(scores) if len(scores) > 1 else 1.0
                if std_s > 0.01:
                    alignment = mean_s / std_s
                    adjustment = lr * np.tanh(alignment)
                    current_W[wname] = max(0.001, current_W[wname] + adjustment)

            total = sum(current_W.values())
            if total > 0:
                scale = 0.95 / total
                current_W = {k: v * scale for k, v in current_W.items()}

            delta = sum(abs(current_W.get(k, 0) - prev_W.get(k, 0)) for k in prev_W) / max(len(prev_W), 1)
            self.history.append({
                'iteration': iteration,
                'coherence': float(avg_coherence),
                'delta': float(delta),
            })
            logger.info(f"    جولة {iteration+1}: coherence={avg_coherence:.4f}, delta={delta:.6f}")

            if delta < convergence:
                logger.info(f"  تقارب بعد {iteration+1} جولات")
                break

        return current_W

    def _estimate_contribution(self, generator, weight_name, correct_word, correct_pv,
                               context_pvs, total_score, k_val):
        """تقدير مساهمة وزن واحد في رنين الكلمة الصحيحة."""
        w = generator.W.get(weight_name, 0.0)
        if w < 1e-10:
            return 0.0
        if total_score == 0:
            return 0.0

        if len(context_pvs) > 0:
            target = np.mean(context_pvs, axis=0)
            correct_align = float(np.dot(correct_pv[:22], target[:22])) / (
                max(np.linalg.norm(correct_pv[:22]), 1e-10) * max(np.linalg.norm(target[:22]), 1e-10)
            )
        else:
            correct_align = 0.0

        signal_categories = {
            'align': 'phase', 'prompt_align': 'phase', 'diversity': 'phase',
            'syntax': 'structural', 'syntax_gate': 'structural', 'syntax_phase': 'structural',
            'morpho': 'structural', 'morpho_trans': 'structural', 'irab': 'structural',
            'sem': 'semantic', 'phil_semantic': 'semantic', 'root_align': 'semantic',
            'syn': 'structural', 'contextual_spectra': 'semantic',
            'semantic_density': 'semantic', 'global_resonance': 'semantic',
            'dialogue_gravity': 'dialogue', 'dialogue_spectral': 'dialogue',
            'intent_align': 'dialogue', 'associative_plan': 'dialogue',
            'plan_fidelity': 'dialogue', 'dialogue': 'dialogue',
            'gravity': 'physical', 'spectral': 'physical', 'thermo': 'physical',
            'resonance_gate': 'physical', 'dccf': 'physical', 'ppm': 'physical',
        }

        category = signal_categories.get(weight_name, 'other')
        category_k_map = {
            'phase': 1.0,
            'semantic': 1.2,
            'structural': 0.8,
            'dialogue': 1.0,
            'physical': 1.0,
            'other': 0.5,
        }

        base_contribution = w * abs(correct_align + k_val * 0.01)
        category_boost = category_k_map.get(category, 0.5)

        return base_contribution * category_boost

    def _extract_sequences(self, texts, vocab, min_samples):
        """استخراج تسلسلات كلمات من النصوص المرجعية."""
        sequences = []
        if isinstance(texts, str):
            texts = [texts]
        for text in texts:
            words = text.split()
            for i in range(3, min(len(words), 15)):
                for j in range(0, len(words) - i):
                    seq = words[j:j + i + 1]
                    if all(w in vocab.word2id for w in seq):
                        sequences.append(seq)
                        if len(sequences) >= min_samples * 2:
                            return sequences
        return sequences

    def apply_weights(self, generator, new_weights):
        """تطبيق الأوزان المُعايرة على المُولّد."""
        total = sum(new_weights.values())
        if total > 0:
            scale = 0.95 / total
            new_weights = {k: v * scale for k, v in new_weights.items()}
        generator.W = new_weights
        logger.info(f"  تم تطبيق {len(new_weights)} وزن مُعاير")