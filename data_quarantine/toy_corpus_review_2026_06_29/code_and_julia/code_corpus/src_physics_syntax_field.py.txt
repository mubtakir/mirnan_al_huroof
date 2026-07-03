"""SyntaxField — فضاء نحوي 6D بمرتكزات غير متعامدة.

يحول الكلمات إلى متجهات نحوية 6D عبر 6 مرتكزات:
verb, noun, prep, adj, adverb, pronoun.
يستخدم Kuramoto mean-cos لقياس التوافق النحوي.
"""
import numpy as np
from src.physics.constants import SYNTAX_DIMS
from src.grammar.irab_engine import IrabEngine
from src.grammar.arabic_grammar import ArabicGrammar
from src.grammar.particles import JAR_PREPS, CONJUNCTIONS, NASB_NOUNS, NEGATIONS
from src.grammar.irab_engine import IrabEngine
from src.grammar.arabic_grammar import ArabicGrammar
from src.semantics.arabic_morphology_analyzer import analyze_word

KANA_VERBS = {"كان", "كانت", "ليس", "ليست", "أصبح", "صار", "ما زال", "يكون", "تكون", "أمسى", "بات", "ليسوا"}
JZM_PARTICLES = set(IrabEngine.JZM_VERB_PARTICLES)
VERB_PREFIXES = set('أنيت')

SYNTAX_ANCHORS = {
    "verb": np.array([1.00, 0.20, 0.00, 0.00, 0.15, 0.10]),
    "noun": np.array([0.20, 1.00, 0.70, 0.00, 0.30, 0.40]),
    "prep": np.array([0.00, 0.70, 1.00, 0.00, 0.10, 0.00]),
    "part": np.array([0.10, 0.10, 0.00, 1.00, 0.10, 0.00]),
    "conj": np.array([0.20, 0.20, 0.10, 0.05, 1.00, 0.05]),
    "kana": np.array([0.10, 0.50, 0.00, 0.00, 0.05, 1.00]),
}


def _l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def _get_syntax_anchor(word, morpho=None, context_words=None):
    if word in KANA_VERBS:
        return SYNTAX_ANCHORS["kana"]
    if word in JAR_PREPS:
        return SYNTAX_ANCHORS["prep"]
    if word in NASB_NOUNS or word in JZM_PARTICLES or word in NEGATIONS:
        return SYNTAX_ANCHORS["part"]
    if word in CONJUNCTIONS:
        return SYNTAX_ANCHORS["conj"]
        
    # استخدام المحلل الصرفي الفطري القوي للكلمات الأخرى
    analysis = analyze_word(word, context_words=context_words)
    word_type = analysis.get('type', 'اسم')
    
    if word_type == 'فعل':
        return SYNTAX_ANCHORS["verb"]
    elif word_type == 'صفة':
        # الصفة تلحق بالاسم لكن بكتلة أضعف، للتبسيط سنعاملها كاسم هنا أو نستخدم Anchor خاص بها لو وجد
        return SYNTAX_ANCHORS["noun"]
    elif word_type == 'حال':
        # الحال منصوب، يمكن أن يأخذ بعض أبعاد الأفعال لأنه مشتق منها
        return SYNTAX_ANCHORS["noun"]
    elif word_type == 'أداة':
        return SYNTAX_ANCHORS["part"]
        
    return SYNTAX_ANCHORS["noun"]


def _is_english_word(word):
    return bool(__import__('re').search(r'[a-zA-Z]', word))

def compute_syntax_vector(word, vocab=None, K=None, morpho=None, context_words=None):
    if _is_english_word(word):
        from src.physics.english_morpheme import get_en_syntax_anchor
        return _l2(get_en_syntax_anchor(word))
    return _l2(_get_syntax_anchor(word, morpho, context_words))


def expected_syntax(prev_word, vocab, K, morpho=None, context_words=None):
    pid = vocab.get(prev_word)
    if pid is None or pid >= K.shape[0]:
        return np.zeros(SYNTAX_DIMS)
    row = K[pid].toarray().ravel()
    total = np.zeros(SYNTAX_DIMS)
    weight_sum = 0.0
    for tid in np.argsort(row)[-30:]:
        if row[tid] <= 0.001:
            continue
        tword = vocab.id2word.get(tid)
        if tword:
            total += row[tid] * _l2(_get_syntax_anchor(tword, morpho, context_words))
            weight_sum += row[tid]
    if weight_sum > 1e-10:
        total = total / weight_sum
    return _l2(total)


class SyntaxFieldCache:
    def __init__(self, vocab, K, morpho=None):
        self._cache = {}
        self._exp_cache = {}
        self.vocab = vocab
        self.K = K
        self.morpho = morpho

    def get(self, word):
        if word not in self._cache:
            if word and len(word) >= 2:
                self._cache[word] = compute_syntax_vector(word, self.vocab, self.K, self.morpho)
            else:
                self._cache[word] = np.zeros(SYNTAX_DIMS)
        return self._cache[word]

    def get_expected(self, prev_word):
        if prev_word not in self._exp_cache:
            self._exp_cache[prev_word] = expected_syntax(prev_word, self.vocab, self.K, self.morpho)
        return self._exp_cache[prev_word]
