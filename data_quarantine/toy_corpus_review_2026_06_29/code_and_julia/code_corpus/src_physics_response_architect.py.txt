# -*- coding: utf-8 -*-
"""محرك بناء الرد — يُخطط هيكل الرد قبل توليد كلمة واحدة.

الفلسفة: الفيزيائي لا يكتب معادلة عشوائياً — يُخطط البنية أولاً.
كذلك محرك بناء الرد:
1. يحلل قصد المستخدم (سؤال، تحية، طلب...)
2. يُحدد نوع الرد (إجابة، تحية مقابلة، تنفيذ طلب...)
3. يُحدد المفاهيم المفتاحية (ما الذي يجب أن يُذكر في الرد)
4. يبني مساراً طورياً يوجه التوليد عبر معالم

هذا ليس chain-of-thought — إنه هندسة معمارية:
كل رد له أساس (foundation) وأعمدة (pillars) وسقف (roof).
المسار الطوري يُمثّل هذه الهندسة في فضاء 64D.
"""
import numpy as np
import logging
from src.physics.word_physics import phase_similarity, compute_word_mass
from src.physics.constants import TOTAL_DIM, PHASE_DIM

logger = logging.getLogger(__name__)

RESPONSE_PATTERNS = {
    'GREETING': {
        'type': 'reciprocal',
        'structure': ['greeting_return', 'blessing'],
        'examples': [
            ('وعليكم السلام ورحمة الله', 'السلام عليكم'),
            ('مرحبا بك', 'مرحبا'),
            ('أهلا وسهلا', 'أهلا'),
        ],
    },
    'QUESTION': {
        'type': 'informative',
        'structure': ['acknowledgment', 'definition', 'elaboration', 'closing'],
        'examples': [
            ('هو العلم الذي يدرس', 'ما هو'),
            ('يمكن تعريفه بأنه', 'ما هو'),
            ('هو عبارة عن', 'ما هي'),
        ],
    },
    'THANK': {
        'type': 'reciprocal',
        'structure': ['acknowledgment', 'politeness'],
        'examples': [
            ('العفو', 'شكرا'),
            ('لا شكر على واجب', 'شكرا'),
            ('عفوا', 'شكرا'),
        ],
    },
    'FAREWELL': {
        'type': 'reciprocal',
        'structure': ['farewell_return', 'wish'],
        'examples': [
            ('مع السلامة', 'مع السلامة'),
            ('في أمان الله', 'مع السلامة'),
            ('إلى اللقاء', 'إلى اللقاء'),
        ],
    },
    'COMMAND': {
        'type': 'action',
        'structure': ['acknowledgment', 'execution'],
        'examples': [],
    },
    'OPINION': {
        'type': 'discursive',
        'structure': ['engagement', 'perspective', 'evidence', 'synthesis'],
        'examples': [],
    },
    'REQUEST': {
        'type': 'responsive',
        'structure': ['acknowledgment', 'response', 'clarification'],
        'examples': [],
    },
    'SUGGESTION': {
        'type': 'discursive',
        'structure': ['consideration', 'evaluation', 'alternative'],
        'examples': [],
    },
    'COMPLAINT': {
        'type': 'empathetic',
        'structure': ['empathy', 'explanation', 'solution'],
        'examples': [],
    },
    'STATEMENT': {
        'type': 'elaborative',
        'structure': ['acknowledgment', 'addition'],
        'examples': [],
    },
}

TOPIC_KEYWORDS = {
    'physics': ['فيزياء', 'فيزيائي', 'كم', 'ذرة', 'طاقة', 'قوة', 'حركة', 'جاذبية', 'موجة', 'ضوء', 'physics', 'quantum', 'energy', 'force', 'gravity', 'wave'],
    'math': ['رياضيات', 'حساب', 'جبر', 'هندسة', 'معادلة', 'عدد', 'math', 'equation', 'algebra'],
    'language': ['لغة', 'عربية', 'إنجليزية', 'كلمة', 'معنى', 'نحو', 'لغوي', 'language', 'grammar', 'word'],
    'philosophy': ['فلسفة', 'وجود', 'حقيقة', 'عقل', 'وعي', 'فكر', 'philosophy', 'consciousness', 'truth'],
    'technology': ['تقنية', 'حاسوب', 'برمجة', 'ذكاء', 'اصطناعي', 'technology', 'computer', 'programming', 'AI'],
    'health': ['صحة', 'مرض', 'طب', 'علاج', 'جسم', 'health', 'medicine', 'disease'],
    'general': ['شيء', 'أمر', 'حال', 'علم', 'معرفة', 'thing', 'matter'],
}


class ResponseArchitect:
    """مهندس الرد — يبني هيكل الرد قبل التوليد."""

    def __init__(self, vocab=None, K_sem=None, all_pv=None, config=None):
        self.vocab = vocab
        self.K_sem = K_sem
        self.all_pv = all_pv
        self.config = config or {}
        self.pattern_db = {}
        self._build_pattern_db()

    def _build_pattern_db(self):
        """بناء قاعدة بيانات الأنماط من RESPONSE_PATTERNS."""
        for intent, pattern in RESPONSE_PATTERNS.items():
            for example in pattern.get('examples', []):
                if len(example) == 2:
                    response_text, trigger = example
                    self.pattern_db.setdefault(intent, []).append({
                        'trigger': trigger,
                        'response': response_text,
                    })

    def analyze_prompt(self, prompt, intent_result=None):
        """تحليل المدخل وبناء خطة الرد.

        Returns:
            dict: خطة الرد {intent, response_type, structure, 
                  key_concepts, target_concepts, min_length, max_length}
        """
        intent = intent_result.get('intent', 'STATEMENT') if intent_result else 'STATEMENT'
        confidence = intent_result.get('confidence', 0.5) if intent_result else 0.5

        pattern = RESPONSE_PATTERNS.get(intent, RESPONSE_PATTERNS['STATEMENT'])

        key_concepts = self._extract_key_concepts(prompt)

        topic = self._detect_topic(prompt)

        min_length, max_length = self._estimate_length(intent, confidence)

        return {
            'intent': intent,
            'confidence': confidence,
            'response_type': pattern['type'],
            'structure': pattern['structure'],
            'key_concepts': key_concepts,
            'topic': topic,
            'min_length': min_length,
            'max_length': max_length,
        }

    def _extract_key_concepts(self, prompt):
        """استخراج المفاهيم المفتاحية من المدخل."""
        words = prompt.split()
        key_concepts = []
        for w in words:
            wid = self.vocab.word2id.get(w) if self.vocab else None
            if wid is not None and self.K_sem is not None and wid < self.K_sem.shape[0]:
                row = self.K_sem[wid].toarray().ravel()
                top_ids = np.argsort(row)[-10:]
                for tid in top_ids:
                    if row[tid] > 0:
                        tw = self.vocab.id2word.get(tid, '') if self.vocab else ''
                        if len(tw) >= 3 and tw not in words and tw not in key_concepts:
                            key_concepts.append(tw)
        return key_concepts[:5]

    def _detect_topic(self, prompt):
        """كشف الموضوع العام من المدخل."""
        prompt_lower = prompt.lower()
        best_topic = 'general'
        best_score = 0
        for topic, keywords in TOPIC_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw in prompt_lower)
            if score > best_score:
                best_score = score
                best_topic = topic
        return best_topic

    def _estimate_length(self, intent, confidence):
        """تقدير طول الرد المناسب حسب القصد."""
        length_map = {
            'GREETING': (2, 5),
            'THANK': (1, 4),
            'FAREWELL': (2, 5),
            'QUESTION': (5, 15),
            'COMMAND': (3, 10),
            'REQUEST': (3, 12),
            'OPINION': (5, 15),
            'SUGGESTION': (4, 12),
            'COMPLAINT': (4, 10),
            'STATEMENT': (3, 10),
        }
        return length_map.get(intent, (3, 10))

    def build_response_trajectory(self, plan, vocab=None, all_pv=None):
        """بناء مسار طوري للرد بناءً على الخطة.

        لكل مرحلة في بنية الرد، نحسب متجهاً هدفياً
        يوجه التوليد نحو المفاهيم المناسبة.
        """
        if vocab is None:
            vocab = self.vocab
        if all_pv is None:
            all_pv = self.all_pv

        structure = plan.get('structure', ['acknowledgment'])
        key_concepts = plan.get('key_concepts', [])
        n_stages = len(structure)

        trajectory = []

        for stage_idx, stage_name in enumerate(structure):
            progress = (stage_idx + 1) / max(n_stages, 1)

            if stage_name == 'greeting_return':
                target_words = ['وعليكم', 'السلام', 'مرحبا', 'أهلا', 'welcome', 'hello']
            elif stage_name == 'blessing':
                target_words = ['ورحمة', 'الله', 'بركاته', 'بك', 'peace']
            elif stage_name == 'acknowledgment':
                target_words = ['نعم', 'بالطبع', 'حسنا', 'أفهم', 'yes', 'of', 'course', 'indeed', 'الحق']
            elif stage_name == 'definition':
                target_words = key_concepts[:3] if key_concepts else ['هو', 'تعني', 'عبارة', 'معنى', 'means', 'is']
            elif stage_name == 'elaboration':
                target_words = key_concepts[:5] if key_concepts else ['كما', 'و', 'حيث', 'لأن', 'also', 'because']
            elif stage_name == 'closing':
                target_words = ['بشكل', 'عام', 'إذن', 'ولذلك', 'therefore', 'thus', 'overall']
            elif stage_name == 'politeness':
                target_words = ['العفو', 'ولا', 'شكر', 'welcome', 'thank', 'please']
            elif stage_name == 'farewell_return':
                target_words = ['مع', 'السلامة', 'أمان', 'الله', 'bye', 'peace', 'safe']
            elif stage_name == 'wish':
                target_words = ['أتمنى', 'إن', 'شاء', 'hopefully', 'wish']
            elif stage_name == 'empathy':
                target_words = ['أفهم', 'صحيح', 'معك', 'حق', 'understand', 'right']
            elif stage_name == 'explanation':
                target_words = key_concepts[:3] + ['لأن', 'حيث', 'because', 'since']
            elif stage_name == 'solution':
                target_words = ['يمكن', 'حل', 'اقتراح', 'can', 'solution', 'try']
            elif stage_name == 'execution':
                target_words = key_concepts[:3] + ['تم', 'إليك', 'here', 'done']
            elif stage_name == 'addition':
                target_words = key_concepts[:3] + ['أيضا', 'كذلك', 'also', 'moreover']
            elif stage_name == 'perspective':
                target_words = ['من', 'ناحية', 'يمكن', 'رأي', 'perspective', 'opinion']
            elif stage_name == 'evidence':
                target_words = ['على', 'سبيل', 'مثال', 'ليكن', 'example', 'instance', 'fact']
            elif stage_name == 'synthesis':
                target_words = ['خلاصة', 'إذن', 'بشكل', 'overall', 'therefore', 'conclusion']
            elif stage_name == 'clarification':
                target_words = ['أي', 'بمعنى', 'تحديداً', 'specifically', 'mean']
            elif stage_name == 'consideration':
                target_words = ['أرى', 'من', 'الممكن', 'possible', 'consider']
            elif stage_name == 'evaluation':
                target_words = ['جيد', 'ممتاز', 'مفيد', 'good', 'great', 'useful']
            elif stage_name == 'alternative':
                target_words = ['بدلا', 'أو', 'أيضا', 'alternatively', 'instead']
            else:
                target_words = key_concepts[:3] if key_concepts else []

            target_pv = np.zeros(TOTAL_DIM)
            n_valid = 0
            if vocab is not None and all_pv is not None:
                for tw in target_words:
                    twid = vocab.word2id.get(tw)
                    if twid is not None and twid < len(all_pv):
                        target_pv += all_pv[twid]
                        n_valid += 1
                if n_valid > 0:
                    target_pv /= n_valid

            pv_norm = np.linalg.norm(target_pv)
            if pv_norm > 1e-10:
                target_pv /= pv_norm

            tightness = 0.2 + 0.6 * progress

            trajectory.append({
                'name': stage_name,
                'target_pv': target_pv,
                'tightness': tightness,
                'progress': progress,
                'target_words': target_words,
            })

        return trajectory

    def get_guidance_for_step(self, trajectory, step, total_steps):
        """الحصول على التوجيه للخطوة الحالية.

        Returns:
            dict: {target_pv, tightness, target_words, stage_name}
        """
        if not trajectory:
            return None

        progress = step / max(1, total_steps)
        n_stages = len(trajectory)
        stage_idx = min(int(progress * n_stages), n_stages - 1)

        current = trajectory[stage_idx]

        if stage_idx < n_stages - 1:
            next_stage = trajectory[stage_idx + 1]
            local_progress = (progress * n_stages) - stage_idx
            blended_pv = current['target_pv'] * (1 - local_progress) + next_stage['target_pv'] * local_progress
            blended_tightness = current['tightness'] * (1 - local_progress) + next_stage['tightness'] * local_progress
        else:
            blended_pv = current['target_pv']
            blended_tightness = current['tightness']

        pv_norm = np.linalg.norm(blended_pv)
        if pv_norm > 1e-10:
            blended_pv /= pv_norm

        return {
            'target_pv': blended_pv,
            'tightness': blended_tightness,
            'target_words': current.get('target_words', []),
            'stage_name': current['name'],
        }

    def compute_architect_score(self, word, guidance, vocab=None, K_sem=None):
        """تقييم مدى توافق كلمة مع خطة البناء المعماري.

        Returns:
            float: درجة التوافق المعماري (-1 إلى 1)
        """
        if guidance is None:
            return 0.0

        target_words = guidance.get('target_words', [])
        tightness = guidance.get('tightness', 0.5)
        target_pv = guidance.get('target_pv')

        score = 0.0

        if word in target_words:
            score += 1.0 * tightness

        if vocab is not None and K_sem is not None:
            wid = vocab.word2id.get(word)
            if wid is not None and target_pv is not None:
                word_pv = None
                if hasattr(self, 'all_pv') and self.all_pv is not None and wid < len(self.all_pv):
                    word_pv = self.all_pv[wid]
                if word_pv is not None and np.linalg.norm(target_pv) > 1e-10:
                    sim = phase_similarity(word_pv[:PHASE_DIM], target_pv[:PHASE_DIM])
                    score += sim * tightness * 0.5

        return score