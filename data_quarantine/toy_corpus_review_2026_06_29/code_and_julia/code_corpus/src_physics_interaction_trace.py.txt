# -*- coding: utf-8 -*-
"""أثر التفاعل — حقل طوري مستمر يتراكم من تفاعلات المستخدم.

الفلسفة: كالمغناطيس — كل تجربة تُشكّل المجال قليلاً.
الاستخدامات المتكررة تكوّن مناطق مُمغنطة (attractors) في فضاء الطور.
غير أن كل تفاعل يترك أثراً يضمحل تدريجياً — مثل موجة في بركة ماء.

يُحفظ الحقل على القرص ويمكن استعادته بين الجلسات.
يُعرف النظام:
- ما هي مواضيع المستخدم المفضلة
- ما مستوى تعقيده (كلمات بسيطة أم معقدة)
- ما أنماط الحوار المعتادة
- ما نواياه المتكررة

هذا ليس تدريباً على بيانات — إنه تشكيل حقل بالخبرة.
"""
import numpy as np
import os
import json
import logging
from src.physics.word_physics import phase_similarity
from src.physics.constants import TOTAL_DIM, PHASE_DIM

logger = logging.getLogger(__name__)


class InteractionTrace:
    """أثر التفاعل — حقل طوري مستمر يشكّله المستخدم."""

    def __init__(self, config=None):
        cfg = config or {}
        self.decay_rate = cfg.get('trace_decay_rate', 0.95)
        self.max_attractors = cfg.get('trace_max_attractors', 50)
        self.learning_rate = cfg.get('trace_learning_rate', 0.05)
        self.attractors = {}
        self.topic_history = []
        self.intent_frequency = {}
        self.complexity_avg = 0.0
        self.n_interactions = 0
        self.user_style = {
            'avg_words': 0.0,
            'uses_diacritics': False,
            'uses_english': False,
            'politeness_level': 0.5,
        }

    def observe(self, text, intent='STATEMENT', vocab=None, all_pv=None):
        """مراقبة تفاعل المستخدم — تحديث الحقل الطوري.

        Args:
            text: نص المدخل
            intent: القصد المكتشف
            vocab: المعجم
            all_pv: مصفوفة المتجهات
        """
        self.n_interactions += 1

        words = text.split()
        n_words = len(words)

        self.user_style['avg_words'] = (
            self.user_style['avg_words'] * 0.9 + n_words * 0.1
        )

        has_diacritics = any(any('\u064B' <= c <= '\u0656' for c in w) for w in words)
        if has_diacritics:
            self.user_style['uses_diacritics'] = True

        has_english = any(any(c.isascii() and c.isalpha() for c in w) for w in words)
        if has_english:
            self.user_style['uses_english'] = True

        self.intent_frequency[intent] = self.intent_frequency.get(intent, 0) + 1

        complexity = self._estimate_complexity(words)
        self.complexity_avg = self.complexity_avg * 0.9 + complexity * 0.1

        if vocab and all_pv is not None:
            self._update_attractors(words, vocab, all_pv)

        self._apply_decay()

    def _estimate_complexity(self, words):
        """تقدير تعقيد مدخل المستخدم (0=بسيط, 1=معقد)."""
        if not words:
            return 0.5
        avg_len = np.mean([len(w) for w in words])
        rare_fraction = 0.0
        unique_fraction = len(set(words)) / max(len(words), 1)
        score = min(1.0, (avg_len - 3) / 8.0) * 0.5 + rare_fraction * 0.3 + unique_fraction * 0.2
        return max(0.0, min(1.0, score))

    def _update_attractors(self, words, vocab, all_pv):
        """تحديث مناطق الجذب الطورية — هوب فيزيائي."""
        word_ids = [vocab.word2id.get(w) for w in words
                     if vocab.word2id.get(w) is not None and vocab.word2id.get(w) < len(all_pv)]
        if not word_ids:
            return

        pvs = all_pv[word_ids]
        mean_pv = np.mean(pvs, axis=0)
        mean_norm = np.linalg.norm(mean_pv)
        if mean_norm < 1e-10:
            return
        mean_pv /= mean_norm

        best_match = None
        best_sim = 0.0
        for topic_key, attractor in self.attractors.items():
            sim = phase_similarity(mean_pv[:PHASE_DIM], attractor['pv'][:PHASE_DIM])
            if sim > best_sim:
                best_sim = sim
                best_match = topic_key

        if best_match is not None and best_sim > 0.5:
            self.attractors[best_match]['pv'] = (
                (1 - self.learning_rate) * self.attractors[best_match]['pv']
                + self.learning_rate * mean_pv
            )
            pv_norm = np.linalg.norm(self.attractors[best_match]['pv'])
            if pv_norm > 1e-10:
                self.attractors[best_match]['pv'] /= pv_norm
            self.attractors[best_match]['mass'] += 1.0
            self.attractors[best_match]['words'].extend(
                [w for w in words if w not in self.attractors[best_match]['words'][:20]]
            )
            self.attractors[best_match]['words'] = self.attractors[best_match]['words'][-20:]
        else:
            topic_key = f"topic_{len(self.attractors)}"
            self.attractors[topic_key] = {
                'pv': mean_pv,
                'mass': 1.0,
                'words': words[:20],
                'intent': 'UNKNOWN',
            }
            if len(self.attractors) > self.max_attractors:
                min_key = min(self.attractors, key=lambda k: self.attractors[k]['mass'])
                del self.attractors[min_key]

    def _apply_decay(self):
        """اضمحلال الحقل — الأقدام القديمة تخفت."""
        for key in list(self.attractors.keys()):
            self.attractors[key]['mass'] *= self.decay_rate
            if self.attractors[key]['mass'] < 0.01:
                del self.attractors[key]

        self.intent_frequency = {
            k: v * self.decay_rate
            for k, v in self.intent_frequency.items()
            if v * self.decay_rate > 0.01
        }

    def get_context_boost(self, word_pv, intent=None):
        """حساب تعزيز الحقل الطوري لكلمة مرشحة.

        Returns:
            float: تعزيز إيجابي إذا كانت الكلمة في منطقة جذب المستخدم
        """
        boost = 0.0

        for key, attractor in self.attractors.items():
            if attractor.get('pv') is not None and attractor['mass'] > 0.1:
                sim = phase_similarity(word_pv[:PHASE_DIM], attractor['pv'][:PHASE_DIM])
                boost += sim * min(attractor['mass'] / 10.0, 1.0) * 0.3

        if intent and intent in self.intent_frequency:
            freq = self.intent_frequency[intent]
            boost += 0.1 * min(freq / 5.0, 1.0)

        return min(boost, 1.0)

    def get_recommended_beta(self):
        """توصية بـ β بناءً على تعقيد المستخدم."""
        if self.complexity_avg > 0.7:
            return 2.5
        elif self.complexity_avg > 0.4:
            return 2.0
        else:
            return 1.5

    def get_dominant_intent(self):
        """القصد الأكثر شيوعاً."""
        if not self.intent_frequency:
            return 'STATEMENT'
        return max(self.intent_frequency, key=self.intent_frequency.get)

    def save(self, filepath):
        """حفظ أثر التفاعل على القرص."""
        data = {
            'attractors': {},
            'intent_frequency': self.intent_frequency,
            'complexity_avg': self.complexity_avg,
            'n_interactions': self.n_interactions,
            'user_style': self.user_style,
        }
        for key, att in self.attractors.items():
            data['attractors'][key] = {
                'pv': att['pv'].tolist(),
                'mass': float(att['mass']),
                'words': att['words'][:20],
                'intent': att.get('intent', 'UNKNOWN'),
            }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load(self, filepath):
        """تحميل أثر التفاعل من القرص."""
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.intent_frequency = data.get('intent_frequency', {})
            self.complexity_avg = data.get('complexity_avg', 0.0)
            self.n_interactions = data.get('n_interactions', 0)
            self.user_style = data.get('user_style', self.user_style)
            for key, att in data.get('attractors', {}).items():
                self.attractors[key] = {
                    'pv': np.array(att['pv']),
                    'mass': att['mass'],
                    'words': att.get('words', []),
                    'intent': att.get('intent', 'UNKNOWN'),
                }
            logger.info(f"  أثر التفاعل: {len(self.attractors)} مناطق جذب، {self.n_interactions} تفاعل")
        except Exception as e:
            logger.warning(f"  فشل تحميل أثر التفاعل: {e}")