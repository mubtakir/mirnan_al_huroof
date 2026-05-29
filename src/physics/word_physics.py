"""فيزياء الكلمة — المحرك الأساسي لمرنان الحروف.

كل حرف عربي = متجه 22 بُعداً فيزيائياً.
كل كلمة = تراكب موجات حروفها ← بصمة فيزيائية فريدة.

الخوارزمية:
  1. كل حرف → متجه 22D من قاعدة البيانات.
  2. ترجيح موضعي: الحرف الأول × 1.8، الأخير × 1.4، الوسطى × 1.0.
  3. جمع المتجهات الموزونة ÷ عدد الحروف.
  4. تطبيع L2.
"""
import numpy as np
from src.physics.constants import PHASE_DIM
from src.physics.letter_db import LetterDB

_NORMALIZE = str.maketrans({
    '\u0622': '\u0627',  # آ → ا
    '\u0623': '\u0627',  # أ → ا
    '\u0625': '\u0627',  # إ → ا
    '\u0624': '\u0648',  # ؤ → و
    '\u0626': '\u064A',  # ئ → ي
    '\u0629': '\u0647',  # ة → ه
    '\u0649': '\u064A',  # ى → ي
})

_letter_db = None

def get_letter_db():
    global _letter_db
    if _letter_db is None:
        _letter_db = LetterDB()
    return _letter_db


def _normalize_letters(word: str) -> str:
    """تطبيع الحروف: توحيد أشكال الألف والهمزات + إزالة الحركات."""
    return word.lower().translate(_NORMALIZE)


def compute_word_phase_vector(word: str) -> np.ndarray:
    """حساب المتجه الطوري 22D للكلمة مع ترجيح موضعي.

    - الحرف الأول (فاء) : وزن 3.5
    - الحرف الثاني (عين): وزن 2.5
    - الحرف الثالث (لام): وزن 2.0
    - الباقي تنازلي
    - الكلمات القصيرة (1-2 حروف): متوسط بسيط.
    """
    db = get_letter_db()
    letters = _normalize_letters(word)
    chars = list(letters)
    n = len(chars)

    if n <= 2:
        vectors = [db.get_vector(ch) for ch in chars if db.has(ch)]
        if not vectors:
            return np.zeros(PHASE_DIM)
        return np.mean(vectors, axis=0)

    position_weights = [3.5, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]
    vectors = []
    weights = []

    for i, ch in enumerate(chars):
        if db.has(ch):
            w = position_weights[i] if i < len(position_weights) else position_weights[-1]
            vectors.append(db.get_vector(ch))
            weights.append(w)

    if not vectors:
        return np.zeros(PHASE_DIM)

    vectors = np.array(vectors)
    weights = np.array(weights)
    weighted_sum = np.sum(vectors * weights[:, np.newaxis], axis=0)
    total_weight = np.sum(weights)
    mean_vector = weighted_sum / total_weight

    nrm = np.linalg.norm(mean_vector)
    if nrm > 1e-10:
        mean_vector = mean_vector / nrm

    return mean_vector


def phase_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """حساب التشابه الطوري (Cosine Similarity) بين متجهين."""
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))
