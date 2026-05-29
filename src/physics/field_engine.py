"""محرك حقل الجذب والتنافر — مرنان الحروف النظيف.

يقوم بحساب التجاذب والتنافر الفيزيائي للكلمات بناءً على تشابه المتجهات الطورية لحروفها.
يدعم الفضاءات الطورية الثلاثة:
  - المادية (22D): خصائص صوتية-فيزيائية
  - الفلسفية (28D): معاني الحروف الدلالية
  - المدمجة (50D): 22 مادي + 28 فلسفي
"""
import os
import time
import numpy as np
from scipy import sparse
from src.physics.word_physics import compute_word_phase_vector
from src.semantics.arabic_semantics import CharacterSemanticEmbedding, NUM_ARABIC_LETTERS

PHYSICAL_DIM = 22
SEMANTIC_DIM = NUM_ARABIC_LETTERS  # 29
TOTAL_DIM = PHYSICAL_DIM + SEMANTIC_DIM  # 51D


def remove_diacritics(text: str) -> str:
    diacritics = '\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0640'
    return ''.join(c for c in text if c not in diacritics)


def _detect_word_language(word: str) -> str:
    """اكتشاف لغة الكلمة: arabic, english, أو other (رموز وإيموجي)."""
    word = remove_diacritics(word)
    has_arabic = False
    has_latin = False
    for ch in word:
        cp = ord(ch)
        if 0x0600 <= cp <= 0x06FF or 0x0750 <= cp <= 0x077F:
            has_arabic = True
        elif ord('a') <= cp <= ord('z') or ord('A') <= cp <= ord('Z'):
            has_latin = True
        elif ch in '-_\'.':
            pass
        elif cp >= 128:
            return "other"
    if has_arabic:
        return "arabic"
    if has_latin:
        return "english"
    return "other"


def _is_valid_arabic_word(word: str) -> bool:
    """استبعاد الكلمات غير العربية أو المشوهة."""
    if not word:
        return False
    stripped = remove_diacritics(word)
    if len(stripped) < 2:
        return False
    # استبعاد الرموز الرياضية والعلمية والإيموجي والأحرف غير العربية المعزولة
    invalid_chars = set('²³¹⁰₁₂₃₄₅₆₇₈₉⁺⁻ªº*/~εμΔΣ∫∂√∞≈≠≤≥←→↖↑↓↔⇒⇐⇑⇓∀∃∄∅∉∌∎∴∵∶∷≁≂≃≄≅≆≇≈≉≊≋≌≍≎≏≐≑≒≓≔≕≖≗')
    emoji_ranges = [
        (0x1F300, 0x1F9FF), (0x2600, 0x27BF), (0x1FA00, 0x1FA6F),
        (0x1F600, 0x1F64F), (0x2700, 0x27BF), (0x1F680, 0x1F6FF),
        (0x1F900, 0x1F9FF), (0x2B50, 0x2B55), (0x231A, 0x231B),
    ]
    for ch in stripped:
        if ch in invalid_chars:
            return False
        if ch.isdigit():
            return False
        cp = ord(ch)
        for lo, hi in emoji_ranges:
            if lo <= cp <= hi:
                return False
    return True


class FieldEngine:
    def __init__(self, vocab):
        self.vocab = vocab
        import model as mirnan_al_huroof_model
        self.benchmark_vocab = mirnan_al_huroof_model.load_benchmark_vocab()

        self.all_pvs = None
        self.benchmark_pvs = None
        self.words_list = []
        self.benchmark_words_list = []
        self.sem_embed = CharacterSemanticEmbedding()
        self.K_sem = None
        self.arabic_indices = np.array([], dtype=np.int64)
        self.english_indices = np.array([], dtype=np.int64)
        self.benchmark_arabic_indices = np.array([], dtype=np.int64)
        self.benchmark_english_indices = np.array([], dtype=np.int64)

        self._build_pv_matrix()
        self._load_k_matrix()

    def _build_pv_matrix(self):
        t0 = time.time()
        n_words = len(self.vocab)
        self.all_pvs = np.zeros((n_words, TOTAL_DIM))
        self.words_list = [self.vocab.id2word[i] for i in range(n_words)]

        for i in range(n_words):
            word = self.words_list[i]
            phys_vec = compute_word_phase_vector(word)
            sem_vec = self.sem_embed.get_word_semantic_vector(word)
            self.all_pvs[i] = np.concatenate([phys_vec, sem_vec])

        self.arabic_indices = np.array(
            [i for i, w in enumerate(self.words_list) if _detect_word_language(w) == "arabic"],
            dtype=np.int64
        )
        self.english_indices = np.array(
            [i for i, w in enumerate(self.words_list) if _detect_word_language(w) == "english"],
            dtype=np.int64
        )

        n_bench = len(self.benchmark_vocab)
        self.benchmark_pvs = np.zeros((n_bench, TOTAL_DIM))
        self.benchmark_words_list = [self.benchmark_vocab.id2word[i] for i in range(n_bench)]

        for i in range(n_bench):
            word = self.benchmark_words_list[i]
            phys_vec = compute_word_phase_vector(word)
            sem_vec = self.sem_embed.get_word_semantic_vector(word)
            self.benchmark_pvs[i] = np.concatenate([phys_vec, sem_vec])

        self.benchmark_arabic_indices = np.array(
            [i for i, w in enumerate(self.benchmark_words_list) if _detect_word_language(w) == "arabic"],
            dtype=np.int64
        )
        self.benchmark_english_indices = np.array(
            [i for i, w in enumerate(self.benchmark_words_list) if _detect_word_language(w) == "english"],
            dtype=np.int64
        )

        print(f"[FieldEngine] Built {TOTAL_DIM}D phase matrices: "
              f"Full ({n_words:,} words), Benchmark ({n_bench:,} words) "
              f"in {time.time()-t0:.2f}s")

    def rebuild(self):
        self._build_pv_matrix()

    def _load_k_matrix(self):
        try:
            k_path = os.path.join(os.path.dirname(os.path.dirname(
                os.path.dirname(__file__))), 'model', 'K_sem.npz')
            if os.path.exists(k_path):
                self.K_sem = sparse.load_npz(k_path)
                print(f"[FieldEngine] Loaded K_sem matrix from {k_path}")
            else:
                print("[FieldEngine] K_sem matrix not found. Using pure letter physics.")
        except Exception as e:
            print(f"[FieldEngine] Warning: Could not load K_sem: {e}")

    def find_attraction_repulsion(self, target_word: str, space_type: str = "combined",
                                   top_k: int = 15, use_hebbian: bool = False,
                                   vocab_type: str = "full"):
        """البحث عن الكلمات الأكثر تجاذباً وتنافراً للكلمة المستهدفة.

        الفضاءات:
          - physical      : 22D فيزيائي
          - philosophical : 28D دلالي فلسفي
          - combined      : 50D مدمج (افتراضي)

        يكتشف لغة الكلمة المُدخلة تلقائياً ويُظهر نتائج من نفس اللغة فقط.
        """
        target_word_clean = remove_diacritics(target_word)
        input_lang = _detect_word_language(target_word_clean)
        if input_lang == "other":
            return {"attracted": [], "repelled": []}

        target_phys = compute_word_phase_vector(target_word)
        target_sem = self.sem_embed.get_word_semantic_vector(target_word)
        target_pv = np.concatenate([target_phys, target_sem])

        if vocab_type == "benchmark":
            space_matrix = self.benchmark_pvs
            words_list = self.benchmark_words_list
            vocab = self.benchmark_vocab
        else:
            space_matrix = self.all_pvs
            words_list = self.words_list
            vocab = self.vocab

        if space_type == "physical":
            space_matrix_sliced = space_matrix[:, :PHYSICAL_DIM]
            target_pv_sliced = target_pv[:PHYSICAL_DIM]
        elif space_type == "philosophical":
            space_matrix_sliced = space_matrix[:, PHYSICAL_DIM:]
            target_pv_sliced = target_pv[PHYSICAL_DIM:]
        else:
            space_matrix_sliced = space_matrix
            target_pv_sliced = target_pv

        n_t = np.linalg.norm(target_pv_sliced)
        if n_t < 1e-10:
            return {"attracted": [], "repelled": []}

        norms = np.linalg.norm(space_matrix_sliced, axis=1)
        norms = np.maximum(norms, 1e-10)
        similarities = np.dot(space_matrix_sliced, target_pv_sliced) / (norms * n_t)

        scores = similarities.copy()

        if use_hebbian and self.K_sem is not None:
            target_idx = self.vocab.get(target_word)
            if target_idx is not None:
                try:
                    k_row = self.K_sem[target_idx].toarray().ravel()
                    for idx in range(len(words_list)):
                        full_idx = self.vocab.get(words_list[idx])
                        if full_idx is not None and k_row[full_idx] > 0:
                            scores[idx] = similarities[idx] * (1.0 + k_row[full_idx] * 1.5)
                except Exception as e:
                    print(f"[FieldEngine] Error applying K_sem boost: {e}")

        sorted_indices = np.argsort(scores)

        target_norm = remove_diacritics(target_word)
        target_norm_len = len(target_norm)
        attracted = []
        repelled = []

        seen_attracted = set()
        seen_attracted.add(target_norm)
        for idx in reversed(sorted_indices):
            w = words_list[idx]
            w_lang = _detect_word_language(w)
            if w_lang != input_lang or w_lang == "other":
                continue
            w_clean = remove_diacritics(w)
            if (not w or not _is_valid_arabic_word(w) or w_clean in seen_attracted
                    or len(w_clean) < 2):
                continue
            if target_norm_len > 1 and (w_clean in target_norm or target_norm in w_clean):
                continue
            seen_attracted.add(w_clean)
            attracted.append((w, float(scores[idx])))
            if len(attracted) >= top_k:
                break

        seen_repelled = set()
        seen_repelled.add(target_norm)
        for idx in sorted_indices:
            w = words_list[idx]
            w_lang = _detect_word_language(w)
            if w_lang != input_lang or w_lang == "other":
                continue
            w_clean = remove_diacritics(w)
            if (not w or not _is_valid_arabic_word(w) or w_clean in seen_repelled
                    or len(w_clean) < 2):
                continue
            if target_norm_len > 1 and (w_clean in target_norm or target_norm in w_clean):
                continue
            seen_repelled.add(w_clean)
            repelled.append((w, float(scores[idx])))
            if len(repelled) >= top_k:
                break

        return {
            "attracted": attracted,
            "repelled": repelled
        }
