"""DialogueEngine — يُفضّل مسارات الحوار عبر تبديل K + anti-completion bias.

بدون قوالب. يعلّم النظام أن الحوار مختلف عن الفيزياء عبر K_dialogue منفصل.
anti-completion scoring يمنع الاستكمال الحرفي لكلمات المستخدم.
"""
import numpy as np
from typing import List, Optional, Tuple

_SENTENCE_STARTERS = {
    'إن', 'قد', 'لقد', 'سوف', 'هل', 'ما', 'من', 'هذا', 'هذه',
    'ذلك', 'تلك', 'هناك', 'هنا', 'عندما', 'حيث', 'بينما', 'ربما',
    'كان', 'كانت', 'ليس', 'يكون', 'تكون', 'أصبح', 'يجب', 'يمكن',
    'لا', 'لن', 'لم', 'إذا', 'لو', 'كل', 'بعض', 'نفس', 'ذات',
    'في', 'على', 'عن', 'منذ', 'حتى', 'ثم', 'أو', 'بل',
    'the', 'a', 'an', 'this', 'that', 'these', 'those',
    'i', 'you', 'he', 'she', 'it', 'we', 'they',
    'there', 'here', 'what', 'where', 'when', 'why', 'how',
    'وعليكم', 'أهلا', 'مرحبا', 'بالطبع', 'حسناً', 'نعم', 'بالتأكيد',
}


class DialogueEngine:
    def __init__(self,
                 intent_detector=None,
                 associative_memory=None,
                 anti_completion_weight: float = 3.0,
                 starter_bonus: float = 3.0):
        self.intent_detector = intent_detector
        self.associative_memory = associative_memory
        self.anti_completion_weight = anti_completion_weight
        self.starter_bonus = starter_bonus

    def compute_dialogue_score(self,
                               candidate_pv: np.ndarray,
                               candidate_word: str,
                               intent_name: str = "STATEMENT",
                               last_user_pv: Optional[np.ndarray] = None,
                               context_words: Optional[List[str]] = None) -> float:
        score = 0.0

        # عقوبة استكمال — تمنع تكرار كلمة المستخدم (باستخدام threshold عالٍ لتفادي العقوبة الزائفة)
        if last_user_pv is not None and candidate_pv is not None:
            continuation_sim = float(np.dot(candidate_pv, last_user_pv) /
                (np.linalg.norm(candidate_pv) * np.linalg.norm(last_user_pv) + 1e-10))
            if continuation_sim > 0.97:
                factor = (continuation_sim - 0.97) / 0.03
                score -= self.anti_completion_weight * factor

        # مكافأة بدء جملة
        if candidate_word in _SENTENCE_STARTERS:
            score += self.starter_bonus

        # عقوبة التكرار
        if context_words and candidate_word in context_words[-4:]:
            score -= 2.0

        return score

    def detect_need_for_dialogue(self, user_text: str) -> Tuple[bool, str, float]:
        if self.intent_detector is None:
            return False, "STATEMENT", 0.0
        result = self.intent_detector.detect(user_text)
        intent = result['intent']
        confidence = result['confidence']
        is_dialogue = confidence > 0.5 and intent in (
            "GREETING", "QUESTION", "COMMAND", "REQUEST",
            "FAREWELL", "OPINION", "SUGGESTION", "COMPLAINT", "THANK")
        return is_dialogue, intent, confidence

    @staticmethod
    def should_stop_after_sentence(words: List[str]) -> bool:
        if not words:
            return False
        if len(words) >= 2 and words[-1] in ('.', '!', '؟', '?', '…'):
            return True
        if len(words) >= 3:
            if words[-2] in ('الله',):
                return True
        return False
