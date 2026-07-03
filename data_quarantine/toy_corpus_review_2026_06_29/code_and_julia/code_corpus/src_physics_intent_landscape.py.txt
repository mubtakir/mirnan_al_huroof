# -*- coding: utf-8 -*-
"""مشهد القصد — القصد كبئر جهد محتمل في فضاء الطور.

الفلسفة: القصد ليس نقطة — إنه بئر في سطح الطاقة المحتملة.
بئر عميق = قصد واضح، بئر واسع = قصد غامض.
النظام "يتدحرج" نحو أعمق بئر — هذه هي آلية اتخاذ القرار.

كل قصد له:
- مركز (center_pv): المتجه الطوري الأساسي
- عمق (depth): ثقة القصد (0-1)
- عرض (width): غموض القصد (الحيز)
- تلال محيطة (ridges): قصود متنافسة

التصنيف يحدث عبر إسقاط المدخل على السطح ثم
التدحرج نحو أعمق بئر — كرة في وعاء.
"""
import numpy as np
import logging
from src.physics.word_physics import phase_similarity
from src.physics.constants import TOTAL_DIM, PHASE_DIM

logger = logging.getLogger(__name__)


class IntentWell:
    """بئر قصد — منطقة جذب في فضاء الطور."""

    def __init__(self, name, center_pv, depth=1.0, width=0.5, keywords=None):
        self.name = name
        self.center_pv = center_pv
        self.depth = depth
        self.width = width
        self.keywords = keywords or []
        self.visits = 0
        self.last_activation = 0

    def potential(self, pv):
        """حساب الجهد المحتمل عند متجه معين — كلما أنخفض = أقرب للبئر."""
        if np.linalg.norm(self.center_pv) < 1e-10 or np.linalg.norm(pv) < 1e-10:
            return 0.0
        sim = phase_similarity(pv[:PHASE_DIM], self.center_pv[:PHASE_DIM])
        distance = 1.0 - max(sim, 0.0)
        potential = -self.depth * np.exp(-distance ** 2 / (2 * self.width ** 2))
        return float(potential)

    def gradient(self, pv):
        """حساب تدرج الجهد — اتجاه القوة المؤثرة على pv."""
        if np.linalg.norm(self.center_pv) < 1e-10 or np.linalg.norm(pv) < 1e-10:
            return np.zeros(TOTAL_DIM)
        sim = phase_similarity(pv[:PHASE_DIM], self.center_pv[:PHASE_DIM])
        distance = 1.0 - max(sim, 0.0)
        direction = self.center_pv[:TOTAL_DIM] - pv[:TOTAL_DIM]
        dir_norm = np.linalg.norm(direction)
        if dir_norm < 1e-10:
            return np.zeros(TOTAL_DIM)
        grad_magnitude = self.depth * distance / (self.width ** 2) * np.exp(
            -distance ** 2 / (2 * self.width ** 2))
        return grad_magnitude * direction / dir_norm


class IntentLandscape:
    """مشهد القصد — سطح الجهد المحتمل مع آبار القصد."""

    DEFAULT_INTENTS = {
        'GREETING': {
            'phrases': ['السلام عليكم', 'مرحبا', 'أهلا', 'صباح الخير', 'hello', 'hi'],
            'depth': 0.9, 'width': 0.4,
        },
        'QUESTION': {
            'phrases': ['ما هو', 'كيف', 'لماذا', 'أين', 'متى', 'هل', 'what', 'how', 'why', 'where', 'when'],
            'depth': 0.85, 'width': 0.5,
        },
        'COMMAND': {
            'phrases': ['افعل', 'اجعل', 'اكتب', 'do', 'make', 'write', 'create', 'أريد', 'أرجو'],
            'depth': 0.8, 'width': 0.45,
        },
        'REQUEST': {
            'phrases': ['أرجو', 'لو سمحت', 'هل يمكنك', 'please', 'can you', 'could you'],
            'depth': 0.75, 'width': 0.5,
        },
        'FAREWELL': {
            'phrases': ['مع السلامة', 'وداعاً', 'إلى اللقاء', 'goodbye', 'bye', 'see you'],
            'depth': 0.9, 'width': 0.35,
        },
        'OPINION': {
            'phrases': ['أعتقد', 'في رأيي', 'من وجهة نظري', 'i think', 'in my opinion', 'believe'],
            'depth': 0.7, 'width': 0.6,
        },
        'THANK': {
            'phrases': ['شكراً', 'جزاك الله', 'thanks', 'thank you', 'ممتاز', 'رائع'],
            'depth': 0.85, 'width': 0.35,
        },
        'COMPLAINT': {
            'phrases': ['لا يعجبني', 'مشكلة', 'خطأ', 'problem', 'wrong', 'bad', 'لا أريد'],
            'depth': 0.7, 'width': 0.55,
        },
        'SUGGESTION': {
            'phrases': ['ما رأيك', 'ربما', 'قد يكون', 'maybe', 'perhaps', 'suggest'],
            'depth': 0.65, 'width': 0.55,
        },
        'STATEMENT': {
            'phrases': ['هذا', 'ذلك', 'في الواقع', 'actually', 'in fact', 'the'],
            'depth': 0.5, 'width': 0.7,
        },
    }

    def __init__(self, config=None):
        cfg = config or {}
        self.wells = {}
        self.step_counter = 0
        self.adaptation_rate = cfg.get('intent_adaptation_rate', 0.01)

    def build_from_vocab(self, vocab, all_pv):
        """بناء آبار القصد من المتجهات الطورية للكلمات المفتاحية.

        Args:
            vocab: معجم الكلمات
            all_pv: مصفوفة المتجهات الطورية (V × TOTAL_DIM)
        """
        for intent_name, intent_def in self.DEFAULT_INTENTS.items():
            phrase_pvs = []
            for phrase in intent_def['phrases']:
                words = phrase.split()
                word_ids = [vocab.word2id.get(w) for w in words]
                valid_ids = [wid for wid in word_ids if wid is not None and wid < len(all_pv)]
                if valid_ids:
                    phrase_pvs.append(np.mean(all_pv[valid_ids], axis=0))

            if phrase_pvs:
                center = np.mean(phrase_pvs, axis=0)
                center_norm = np.linalg.norm(center)
                if center_norm > 1e-10:
                    center /= center_norm
            else:
                rng = np.random.RandomState(hash(intent_name) % 2**31)
                center = rng.randn(TOTAL_DIM) * 0.1
                center /= np.linalg.norm(center)

            self.wells[intent_name] = IntentWell(
                name=intent_name,
                center_pv=center,
                depth=intent_def['depth'],
                width=intent_def['width'],
                keywords=intent_def['phrases'],
            )

        logger.info(f"  مشهد القصد: {len(self.wells)} بئر")

    def detect(self, text, vocab=None, all_pv=None):
        """كشف القصد عبر التدحرج في مشهد الجهد المحتمل.

        Args:
            text: النص المدخل
            vocab: المعجم (اختياري)
            all_pv: مصفوفة المتجهات (اختياري)

        Returns:
            dict: {intent, confidence, landscape, well_scores}
        """
        words = text.split()
        word_ids = []
        if vocab and all_pv is not None:
            for w in words:
                wid = vocab.word2id.get(w)
                if wid is not None and wid < len(all_pv):
                    word_ids.append(wid)

        well_scores = {}
        for name, well in self.wells.items():
            keyword_score = 0.0
            for kw in well.keywords:
                if kw in text or kw in words:
                    keyword_score = max(keyword_score, 1.0)
                    break

            if word_ids and all_pv is not None:
                word_pvs = all_pv[word_ids]
                input_pv = np.mean(word_pvs, axis=0)
                potential = well.potential(input_pv)
                well_scores[name] = -potential + 0.3 * keyword_score
            else:
                well_scores[name] = 0.1 * keyword_score

        if not well_scores:
            return {
                'intent': 'STATEMENT',
                'confidence': 0.3,
                'landscape': {},
                'well_scores': {},
            }

        best_intent = max(well_scores, key=well_scores.get)
        scores = list(well_scores.values())
        max_score = max(scores) if scores else 0.0
        second_max = sorted(scores)[-2] if len(scores) > 1 else 0.0
        confidence = min(1.0, max(0.0, (max_score - second_max) / max(max_score, 0.01)))

        landscape = {name: float(score) for name, score in well_scores.items()}

        self.step_counter += 1
        if best_intent in self.wells:
            self.wells[best_intent].visits += 1
            self.wells[best_intent].last_activation = self.step_counter

        return {
            'intent': best_intent,
            'confidence': float(confidence),
            'landscape': landscape,
            'well_scores': well_scores,
        }

    def adapt(self, text, response, vocab, all_pv):
        """تكييف الآبار بناءً على التفاعل — التعلم من الخبرة.

        مثل تشكيل وعاء البلاستيك — كل تفاعل يشكّل البئر قليلاً.
        """
        words = text.split()
        word_ids = [vocab.word2id.get(w) for w in words
                     if vocab.word2id.get(w) is not None and vocab.word2id.get(w) < len(all_pv)]
        if not word_ids:
            return

        input_pv = np.mean(all_pv[word_ids], axis=0)

        resp_words = response.split()
        resp_ids = [vocab.word2id.get(w) for w in resp_words
                    if vocab.word2id.get(w) is not None and vocab.word2id.get(w) < len(all_pv)]
        if not resp_ids:
            return

        resp_pv = np.mean(all_pv[resp_ids], axis=0)

        detected = self.detect(text, vocab, all_pv)
        intent = detected['intent']
        confidence = detected['confidence']

        if intent in self.wells and confidence > 0.4:
            well = self.wells[intent]
            shift = self.adaptation_rate * confidence * (resp_pv - well.center_pv)
            shift_norm = np.linalg.norm(shift)
            if shift_norm > 0.05:
                shift = shift * (0.05 / shift_norm)

            well.center_pv = well.center_pv + shift
            center_norm = np.linalg.norm(well.center_pv)
            if center_norm > 1e-10:
                well.center_pv /= center_norm

            well.depth = min(1.0, well.depth + 0.01 * confidence)

    def get_target_well(self, intent):
        """الحصول على بئر القصد للاستخدام في التخطيط."""
        if intent in self.wells:
            return self.wells[intent]
        return self.wells.get('STATEMENT', None)