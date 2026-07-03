"""EnglishMorphology — محرك الصرف والنحو الإنجليزية.

محاكاة MorphoPhasicEngine + ArabicGrammar للغة الإنجليزية.
تصنيف الكلمات (POS)، تحليل الصرف (زمن، عدد، نفي)،
كشف بنية الجملة، واستخراج الجذع.
"""
import re
import numpy as np
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.letter_db import LetterDB
from src.physics.constants import TOTAL_DIM

_POS_ORDER = ['det', 'noun', 'verb', 'adj', 'adv', 'prep', 'pron', 'conj', 'part', 'aux', 'num', 'intj']

_POS_VECTORS = {
    'det':   np.array([1.0, 0.0, 0.0, 0.2, 0.0]),
    'noun':  np.array([0.0, 1.0, 0.0, 0.1, 0.0]),
    'verb':  np.array([0.0, 0.0, 1.0, 0.1, 0.3]),
    'adj':   np.array([0.3, 0.0, 0.0, 1.0, 0.0]),
    'adv':   np.array([0.2, 0.0, 0.0, 0.0, 1.0]),
    'prep':  np.array([0.5, 0.0, 0.0, 0.0, 0.0]),
    'pron':  np.array([0.8, 0.5, 0.0, 0.0, 0.0]),
    'conj':  np.array([0.6, 0.0, 0.0, 0.0, 0.0]),
    'part':  np.array([0.4, 0.0, 0.3, 0.0, 0.0]),
    'aux':   np.array([0.0, 0.0, 0.8, 0.0, 0.2]),
    'num':   np.array([0.2, 0.7, 0.0, 0.1, 0.0]),
    'intj':  np.array([0.0, 0.0, 0.0, 0.5, 0.5]),
}

_TENSE_VECTORS = {
    'present':  np.array([1.0, 0.0, 0.0]),
    'past':     np.array([0.0, 1.0, 0.0]),
    'future':   np.array([0.0, 0.0, 1.0]),
    'infinitive': np.array([0.5, 0.0, 0.5]),
    'imperative': np.array([0.0, 0.5, 0.5]),
}

_NUMBER_VECTORS = {
    'singular': np.array([1.0, 0.0]),
    'plural':   np.array([0.0, 1.0]),
}

_PERSON_VECTORS = {
    'first':  np.array([1.0, 0.0, 0.0]),
    'second': np.array([0.0, 1.0, 0.0]),
    'third':  np.array([0.0, 0.0, 1.0]),
}

_IRREGULAR_PLURALS = {
    'man': 'men', 'woman': 'women', 'child': 'children', 'person': 'people',
    'tooth': 'teeth', 'foot': 'feet', 'mouse': 'mice', 'goose': 'geese',
    'ox': 'oxen', 'sheep': 'sheep', 'deer': 'deer', 'fish': 'fish',
    'chief': 'chiefs', 'roof': 'roofs', 'belief': 'beliefs', 'proof': 'proofs',
    'leaf': 'leaves', 'wolf': 'wolves', 'knife': 'knives', 'wife': 'wives',
    'shelf': 'shelves', 'half': 'halves', 'life': 'lives', 'self': 'selves',
}

_NEGATIONS = {'not', "n't", 'never', 'no', 'nothing', 'nowhere', 'none', 'nobody', 'neither', 'nor'}
_ARTICLES = {'the', 'a', 'an'}
_DETERMINERS = {'this', 'that', 'these', 'those', 'some', 'any', 'every', 'each', 'all', 'both', 'few', 'many', 'much', 'several', 'no', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
_PREPOSITIONS = {'in', 'on', 'at', 'to', 'for', 'with', 'by', 'from', 'of', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'under', 'over', 'across', 'along', 'against', 'among', 'around', 'behind', 'beneath', 'beside', 'beyond', 'down', 'inside', 'near', 'off', 'out', 'outside', 'past', 'round', 'through', 'throughout', 'toward', 'towards', 'underneath', 'up', 'upon', 'via', 'within', 'without'}
_CONJUNCTIONS = {'and', 'or', 'but', 'yet', 'so', 'for', 'nor', 'because', 'although', 'while', 'since', 'if', 'when', 'as', 'until', 'unless', 'after', 'before', 'though', 'whereas', 'whether', 'both', 'either', 'neither', 'not', 'only', 'rather', 'than'}
_PRONOUNS = {'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their', 'mine', 'yours', 'hers', 'ours', 'theirs', 'myself', 'yourself', 'himself', 'herself', 'itself', 'ourselves', 'yourselves', 'themselves', 'this', 'that', 'these', 'those', 'who', 'whom', 'what', 'which', 'whose', 'someone', 'anyone', 'everyone', 'no one', 'something', 'anything', 'everything', 'nothing', 'somebody', 'anybody', 'everybody', 'nobody'}
_AUXILIARIES = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'done', 'doing', 'will', 'would', 'shall', 'should', 'can', 'could', 'may', 'might', 'must', 'need', 'dare', 'used'}
_INTERJECTIONS = {'oh', 'ah', 'wow', 'hey', 'hi', 'hello', 'goodbye', 'bye', 'yes', 'no', 'well', 'alas', 'bravo', 'hooray', 'ouch', 'oops', 'hmm', 'uh', 'um', 'aha', 'phew', 'ugh', 'yay', 'ooh', 'aah', 'yeah', 'nah', 'nope', 'yep'}

_VERB_NOUN_HOMONYMS = {
    'walk', 'talk', 'play', 'work', 'run', 'jump', 'sleep', 'eat', 'drink',
    'read', 'write', 'speak', 'listen', 'watch', 'look', 'help', 'love', 'hate',
    'dream', 'fight', 'fly', 'ride', 'sing', 'dance', 'smile', 'cry', 'laugh',
    'cook', 'wash', 'clean', 'brush', 'dress', 'kiss', 'hug', 'call', 'answer',
    'question', 'name', 'place', 'time', 'hand', 'head', 'eye', 'ear', 'nose',
    'mouth', 'face', 'arm', 'leg', 'foot', 'back', 'side', 'end', 'start',
    'change', 'move', 'stop', 'break', 'cut', 'open', 'close', 'turn', 'push',
    'pull', 'fill', 'empty', 'cover', 'spread', 'roll', 'fold', 'wrap',
    'flow', 'shine', 'rise', 'rule', 'hope', 'live', 'serve', 'share',
    'miss', 'pass', 'cross', 'touch', 'reach', 'teach', 'watch', 'match',
    'search', 'catch', 'fetch', 'launch', 'approach', 'charge', 'exchange',
    'race', 'face', 'place', 'force', 'balance', 'dance', 'glance',
    'promise', 'practice', 'notice', 'produce', 'increase', 'decrease',
    'release', 'replace', 'reduce', 'introduce', 'believe',
    'broke', 'spoke', 'woke', 'drove', 'rode', 'wrote', 'chose', 'froze',
    'took', 'shook', 'forsook', 'mistook',
    'waken', 'strengthen', 'threaten', 'hasten', 'moisten', 'brighten',
    'darken', 'soften', 'harden', 'sweeten', 'shorten', 'lengthen',
    'widen', 'deepen', 'broaden', 'thicken', 'straighten',
}

# ========== POS Tagging (rule-based) ==========

_SUFFIX_TO_POS = [
    (re.compile(r'(tion|sion)\b'), 'noun'),
    (re.compile(r'(ment|ness|ity|ance|ence|ship|dom|hood|ism|ist|ure|age|al|ry|cy|ics)\b'), 'noun'),
    (re.compile(r'(ant|ent|eer|or|ian|ee|ess|logist)\b'), 'noun'),
    (re.compile(r'ly\b'), 'adv'),
    (re.compile(r'(ous|ive|able|ible|ful|less|ic|ical|ar|ory|ary|escent|ulent|oid|like|worthy|some)\b'), 'adj'),
    (re.compile(r'(ify|ize|ise|ate)\b'), 'verb'),
    (re.compile(r'(er|est)\b'), 'adj'),
    (re.compile(r'ate\b'), 'verb'),
    (re.compile(r'y\b'), 'adj'),
]

def _tag_pos(word):
    w = word.lower().strip()
    if w in _ARTICLES or w in _DETERMINERS:
        return 'det'
    if w in _PRONOUNS:
        return 'pron'
    if w in _PREPOSITIONS:
        return 'prep'
    if w in _CONJUNCTIONS:
        return 'conj'
    if w in _AUXILIARIES:
        return 'aux'
    if w in _INTERJECTIONS:
        return 'intj'
    if w in _NEGATIONS:
        return 'part'
    if re.match(r'^[+-]?\d+(\.\d+)?(st|nd|rd|th)?$', w) or w in {'first','second','third','fourth','fifth','sixth','seventh','eighth','ninth','tenth'}:
        return 'num'

    from src.semantics.english_semantics import EnglishRootExtractor
    _ext = EnglishRootExtractor()
    if w in _ext.irregular_verbs:
        return 'verb'
    if w in _IRREGULAR_PLURALS:
        return 'noun'
    if w in _IRREGULAR_PLURALS.values():
        return 'noun'

    if w.endswith('ing') and len(w) > 4:
        base = w[:-3]
        if len(base) >= 2 and (base in _VERB_NOUN_HOMONYMS or (len(base) > 2 and base[-1] == base[-2] and base[:-1] in _VERB_NOUN_HOMONYMS)):
            return 'verb'
    if w.endswith('ed') and len(w) > 3:
        base = w[:-2]
        if len(base) >= 2 and (base in _VERB_NOUN_HOMONYMS or (len(base) > 2 and base[-1] == base[-2] and base[:-1] in _VERB_NOUN_HOMONYMS)):
            return 'verb'
    if w.endswith('en') and len(w) >= 4:
        base = w[:-2]
        if len(base) >= 2 and (base in _VERB_NOUN_HOMONYMS or base + 'e' in _VERB_NOUN_HOMONYMS):
            return 'verb'
    for pattern, pos in _SUFFIX_TO_POS:
        if pattern.search(w):
            return pos
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss') and not w.endswith('is') and not w.endswith('us'):
        base = w[:-1]
        if base in _VERB_NOUN_HOMONYMS:
            return 'verb'
        if base.endswith('e') and base[:-1] in _VERB_NOUN_HOMONYMS:
            return 'verb'
        if w.endswith('es') and len(w) > 4:
            base2 = w[:-2]
            if base2 in _VERB_NOUN_HOMONYMS:
                return 'verb'
        return 'noun'

    if w in _VERB_NOUN_HOMONYMS:
        return 'verb'

    return 'noun'


# ========== Morphological Analysis ==========

_PREFIXES = ['un', 'im', 'in', 'ir', 'il', 'dis', 'non', 'anti', 'de', 'mis', 'pre', 're', 'over', 'under', 'out', 'en', 'em', 'be', 'co', 'fore', 'inter', 'mid', 'sub', 'super', 'trans', 'semi', 'mini', 'micro', 'macro', 'multi', 'poly', 'bi', 'tri', 'uni', 'mono', 'auto', 'counter', 'extra', 'hyper', 'mega', 'post', 'pre', 'pro', 'pseudo']


def _plural_of(word):
    w = word.lower()
    for sing, plur in _IRREGULAR_PLURALS.items():
        if w == plur:
            return sing, 'plural'
        if w == sing:
            return sing, 'singular'
    if w.endswith('ies') and len(w) > 4:
        return w[:-3] + 'y', 'plural'
    if w.endswith('ves') and len(w) > 4:
        return w[:-3] + 'f', 'plural'
    if w.endswith('es') and len(w) > 4:
        return w[:-2], 'plural'
    if w.endswith('s') and len(w) > 3 and not w.endswith('ss') and not w.endswith('is') and not w.endswith('us'):
        return w[:-1], 'plural'
    return w, 'singular'


def _tense_of(word):
    w = word.lower()
    if w in {'will', 'shall', 'would', 'going'}:
        return 'future'
    if w in _AUXILIARIES:
        if w in {'was', 'were', 'did', 'had'}:
            return 'past'
        if w in {'am', 'is', 'are', 'has', 'does', 'have', 'do'}:
            return 'present'
    from src.semantics.english_semantics import EnglishRootExtractor
    _ext = EnglishRootExtractor()
    if w in _ext.irregular_verbs:
        _stem = _ext.irregular_verbs[w]
        if w != _stem:
            return 'past'
    if w.endswith('ed') and len(w) > 3:
        return 'past'
    if w.endswith('ing'):
        return 'present'
    if w.endswith('s') and len(w) > 3:
        return 'present'
    return 'present'


def _person_of(word, pos):
    w = word.lower()
    if pos == 'pron':
        if w in {'i', 'we', 'me', 'us', 'my', 'our', 'mine', 'ours', 'myself', 'ourselves'}:
            return 'first'
        if w in {'you', 'your', 'yours', 'yourself', 'yourselves'}:
            return 'second'
        if w in {'he', 'she', 'it', 'they', 'him', 'her', 'them', 'his', 'its', 'their',
                  'hers', 'theirs', 'himself', 'herself', 'itself', 'themselves'}:
            return 'third'
    if pos == 'verb':
        if w in {'am'}:
            return 'first'
        if w in {'is', 'was', 'has', 'does'}:
            return 'third'
        if w in {'are', 'were', 'have', 'do'}:
            if w in {'are', 'were', 'have', 'do'}:
                return 'second'
    return 'third'


def _has_negation(word):
    w = word.lower()
    return w in _NEGATIONS or w.endswith("n't")


def _analyze_morphology(word):
    w = word.lower().strip()
    pos = _tag_pos(w)
    if pos in ('verb', 'adv', 'adj'):
        from src.semantics.english_semantics import EnglishRootExtractor
        _ext = EnglishRootExtractor()
        if w in _ext.irregular_verbs:
            stem = _ext.irregular_verbs[w]
            number = 'singular'
        elif w.endswith('ing') and len(w) > 4:
            stem = w[:-3]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in 'lsz':
                stem = stem[:-1]
            number = 'singular'
        elif w.endswith('ed') and len(w) > 3:
            stem = w[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in 'lsz':
                stem = stem[:-1]
            number = 'singular'
        elif w.endswith('ies') and len(w) > 4:
            stem = w[:-3] + 'y'
            number = 'singular'
        elif w.endswith('sses') and len(w) > 5:
            stem = w[:-2]
            number = 'singular'
        elif w.endswith('shes') or w.endswith('ches') or w.endswith('xes') or w.endswith('zes'):
            if len(w) > 5:
                stem = w[:-2]
                number = 'singular'
        elif w.endswith('es') and len(w) > 4:
            stem = w[:-1]
            number = 'singular'
        elif w.endswith('s') and len(w) > 3 and not w.endswith('ss'):
            stem = w[:-1]
            number = 'singular'
            stem = w[:-1]
            number = 'singular'
        elif w.endswith('ly') and len(w) > 4:
            stem = w[:-2]
            if stem.endswith('i'):
                stem = stem[:-1] + 'y'
            number = 'singular'
        elif w.endswith('er') and len(w) > 4:
            stem = w[:-2]
            number = 'singular'
        elif w.endswith('est') and len(w) > 4:
            stem = w[:-3]
            number = 'singular'
        else:
            stem, number = _plural_of(w)
    else:
        stem, number = _plural_of(w)
    tense = _tense_of(w)
    person = _person_of(w, pos)
    negated = _has_negation(w)
    has_prefix = any(w.startswith(p) for p in _PREFIXES)
    return {
        'word': w,
        'pos': pos,
        'stem': stem,
        'number': number,
        'tense': tense,
        'person': person,
        'negated': negated,
        'has_prefix': has_prefix,
    }


# ========== English Grammar ==========

_SENTENCE_STARTERS_EN = {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'my', 'our', 'your', 'his', 'her', 'its', 'their', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'there', 'here'}
_QUESTION_WORDS = {'what', 'where', 'when', 'why', 'how', 'who', 'whom', 'whose', 'which'}
_IMPERATIVE_STARTERS = {'please', 'do', 'don\'t', 'never', 'always', 'let\'s'}
_LINKING_VERBS = {'be', 'am', 'is', 'are', 'was', 'were', 'been', 'being', 'become', 'seem', 'appear', 'look', 'feel', 'sound', 'taste', 'smell', 'remain', 'stay', 'keep', 'grow', 'turn', 'prove'}


def detect_sentence_type(words):
    if not words:
        return 'declarative'
    first = words[0].lower()
    if first in _QUESTION_WORDS:
        return 'interrogative'
    if first in {'do', 'does', 'did', 'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'}:
        if len(words) > 1:
            return 'interrogative'
    if first in {'please', 'let\'s', 'don\'t', 'do', 'never', 'always'}:
        return 'imperative'
    if first in _IMPERATIVE_STARTERS:
        return 'imperative'
    return 'declarative'


def subject_verb_agreement(subject, verb):
    s = subject.lower()
    v = verb.lower()
    third_sing = s in {'he', 'she', 'it', 'this', 'that'} or (s.endswith('s') and not s.endswith('ss') and not s.endswith('is') and not s.endswith('us'))
    if third_sing:
        if v in {'are', 'were', 'have', 'do'}:
            return 0.0
        if v in {'is', 'was', 'has', 'does'} or v.endswith('s'):
            return 1.0
    else:
        if v in {'is', 'was', 'has', 'does'} or v.endswith('s'):
            return 0.0
        if v in {'am', 'are', 'were', 'have', 'do'} or (not v.endswith('s') and v not in _AUXILIARIES):
            return 1.0
    return 0.5


def article_usage(word, prev_word):
    w = word.lower()
    p = prev_word.lower() if prev_word else ''
    if p not in {'a', 'an'}:
        return 1.0
    if w and w[0] in 'aeiou':
        if p == 'an':
            return 1.0
        else:
            return 0.0
    else:
        if p == 'a':
            return 1.0
        else:
            return 0.0


# ========== Grammar check for sentence ==========

def grammar_check(words):
    if len(words) < 2:
        return 0.0
    score = 0.0
    count = 0
    for i, w in enumerate(words):
        if i == 0:
            continue
        prev = words[i - 1]
        morph = _analyze_morphology(w)
        prev_morph = _analyze_morphology(prev)
        pos = morph['pos']
        prev_pos = prev_morph['pos']
        if prev_pos == 'det':
            if pos in {'noun', 'adj', 'adv'}:
                score += 1.0
            elif pos == 'verb':
                score += 0.3
        if prev_pos == 'prep':
            if pos in {'noun', 'pron', 'det'}:
                score += 1.0
        if prev in {'a', 'an'} and pos == 'noun':
            score += article_usage(w, prev)
            count += 1
        if prev_pos == 'aux' and pos != 'verb' and pos != 'adv':
            score -= 0.5
        if pos == 'aux' and prev_pos in {'noun', 'pron'}:
            score += 0.5
        count += 1
    return max(0.0, score / max(count, 1))


# ========== English Stemmer ==========

class EnglishStemmer:
    def __init__(self):
        from src.semantics.english_semantics import EnglishRootExtractor
        self._extractor = EnglishRootExtractor()

    def stem(self, word):
        w = word.lower().strip()
        if not w:
            return w, 0.0
        if w in self._extractor.irregular_verbs:
            return self._extractor.irregular_verbs[w], 1.0
        if w.endswith('ing') and len(w) > 4:
            s = w[:-3]
            if len(s) > 2 and s[-1] == s[-2] and s[-1] not in 'lsz':
                s = s[:-1]
            return s, 0.8
        if w.endswith('ed') and len(w) > 3:
            s = w[:-2]
            if len(s) > 2 and s[-1] == s[-2] and s[-1] not in 'lsz':
                s = s[:-1]
            return s, 0.8
        if w.endswith('ies') and len(w) > 4:
            return w[:-3] + 'y', 0.8
        if w.endswith('es') and len(w) > 4:
            return w[:-2], 0.8
        if w.endswith('s') and len(w) > 3 and not w.endswith('ss') and not w.endswith('is') and not w.endswith('us'):
            return w[:-1], 0.7
        if w.endswith('ly') and len(w) > 4:
            s = w[:-2]
            if s.endswith('i'):
                s = s[:-1] + 'y'
            return s, 0.6
        if w.endswith('er') and len(w) > 4:
            s = w[:-2]
            if len(s) > 2 and s[-1] == s[-2]:
                s = s[:-1]
            return s, 0.6
        if w.endswith('est') and len(w) > 4:
            s = w[:-3]
            if len(s) > 2 and s[-1] == s[-2]:
                s = s[:-1]
            return s, 0.6
        if w.endswith('ness') and len(w) > 5:
            return w[:-4], 0.6
        if w.endswith('tion') and len(w) > 6:
            return w[:-4], 0.6
        if w.endswith('ment') and len(w) > 5:
            return w[:-4], 0.6
        return w, 0.3


# ========== EnglishMorphology Engine ==========

_POS_TRANSITION_SCORES = {
    ('det', 'noun'): 1.0, ('det', 'adj'): 0.8,
    ('adj', 'noun'): 1.0, ('adj', 'verb'): 0.3,
    ('noun', 'verb'): 0.9, ('noun', 'prep'): 0.7, ('noun', 'conj'): 0.4, ('noun', 'pron'): 0.2,
    ('verb', 'noun'): 0.7, ('verb', 'adv'): 0.9, ('verb', 'prep'): 0.8, ('verb', 'adj'): 0.3,
    ('adv', 'verb'): 0.8, ('adv', 'adj'): 0.9, ('adv', 'adv'): 0.4,
    ('prep', 'noun'): 0.9, ('prep', 'pron'): 0.8, ('prep', 'det'): 0.5,
    ('pron', 'verb'): 0.9, ('pron', 'noun'): 0.5,
    ('aux', 'verb'): 1.0, ('aux', 'adv'): 0.6,
    ('conj', 'det'): 0.7, ('conj', 'noun'): 0.6, ('conj', 'pron'): 0.7, ('conj', 'verb'): 0.6,
    ('verb', 'conj'): 0.5, ('noun', 'conj'): 0.5,
    ('det', 'det'): 0.0, ('prep', 'prep'): 0.2, ('conj', 'conj'): 0.3,
}


class EnglishMorphology:
    def __init__(self):
        self._stemmer = EnglishStemmer()
        self._morph_cache = {}
        self._pos_cache = {}

    def analyze(self, word):
        if word in self._morph_cache:
            return self._morph_cache[word]
        result = _analyze_morphology(word)
        self._morph_cache[word] = result
        return result

    def get_pos(self, word):
        if word in self._pos_cache:
            return self._pos_cache[word]
        pos = _tag_pos(word)
        self._pos_cache[word] = pos
        return pos

    def get_stem(self, word):
        s, _ = self._stemmer.stem(word)
        return s

    def compute_morph_phase(self, word):
        morph = self.analyze(word)
        pos_vec = _POS_VECTORS.get(morph['pos'], _POS_VECTORS['noun'])
        tense_vec = _TENSE_VECTORS.get(morph['tense'], _TENSE_VECTORS['present'])
        num_vec = _NUMBER_VECTORS.get(morph['number'], _NUMBER_VECTORS['singular'])

        base = np.zeros(TOTAL_DIM)
        base[:5] = pos_vec
        base[5:8] = tense_vec
        base[8:10] = num_vec

        if morph['negated']:
            base[10] = -0.5
        if morph['has_prefix']:
            base[11] = 0.3

        word_pv = compute_extended_phase_vector(word)
        morph_pv = base * 0.4 + word_pv * 0.6
        nrm = np.linalg.norm(morph_pv)
        if nrm > 1e-10:
            morph_pv = morph_pv / nrm
        return morph_pv

    def score(self, word, prev_word=None):
        morph_pv = self.compute_morph_phase(word)
        word_pv = compute_extended_phase_vector(word)
        align = float(np.mean(np.cos(morph_pv - word_pv)))
        score = max(0.0, (align - 0.7) * 3.0)
        return score

    def transition_score(self, word, prev_word=None):
        if prev_word is None:
            return 0.0
        cur_pos = self.get_pos(word)
        prev_pos = self.get_pos(prev_word)
        key = (prev_pos, cur_pos)
        base = _POS_TRANSITION_SCORES.get(key, 0.3)
        w_pv = compute_extended_phase_vector(word)
        p_morph = self.compute_morph_phase(prev_word)
        align = float(np.mean(np.cos(w_pv - p_morph)))
        return max(0.0, base * 0.5 + (align - 0.6) * 2.0 * 0.5)

    def has_matching_temporal(self, word, prev_word):
        if prev_word is None:
            return False
        wm = self.analyze(word)
        pm = self.analyze(prev_word)
        return wm['tense'] == pm['tense']

    def agreement_score(self, word, prev_word):
        if prev_word is None:
            return 0.0
        wm = self.analyze(word)
        pm = self.analyze(prev_word)
        if pm['pos'] in {'noun', 'pron'} and wm['pos'] == 'verb':
            return subject_verb_agreement(prev_word, word)
        if pm['pos'] == 'det' and wm['pos'] in {'noun', 'adj'}:
            return article_usage(word, prev_word)
        return 0.5


class EnglishGrammarEngine:
    def score_sentence(self, words):
        if not words or len(words) < 2:
            return 0.0
        gs = grammar_check(words)
        stype = detect_sentence_type(words)
        bonus = 0.0
        if stype == 'declarative':
            first_pos = _tag_pos(words[0])
            if first_pos in {'det', 'pron', 'noun'}:
                bonus = 0.3
            if len(words) >= 3:
                mid = len(words) // 2
                mid_pos = _tag_pos(words[mid])
                if mid_pos == 'verb':
                    bonus += 0.3
        elif stype == 'interrogative':
            bonus = 0.2
        elif stype == 'imperative':
            first_pos = _tag_pos(words[0])
            if first_pos in {'verb', 'part'}:
                bonus = 0.3
        return min(1.0, gs + bonus)


# ========== English Weight Resonance (14 weights mirroring Arabic) ==========

_EN_WEIGHT_EMBEDDINGS = {
    "noun_sing":       np.array([0.10, 0.90, 0.00, 0.20, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "noun_pl":         np.array([0.00, 0.85, 0.00, 0.30, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "noun_proper":     np.array([0.00, 0.95, 0.00, 0.10, 0.00, 0.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "verb_present":    np.array([0.00, 0.10, 0.90, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "verb_past":       np.array([0.00, 0.00, 0.85, 0.00, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "verb_gerund":     np.array([0.00, 0.20, 0.80, 0.00, 0.10, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "verb_participle": np.array([0.00, 0.20, 0.70, 0.20, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "adj_base":        np.array([0.00, 0.30, 0.00, 0.80, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "adj_comp":        np.array([0.00, 0.20, 0.00, 0.70, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "adj_sup":         np.array([0.00, 0.20, 0.00, 0.65, 0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "adv":             np.array([0.00, 0.10, 0.10, 0.30, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "det":             np.array([0.90, 0.30, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "prep":            np.array([0.40, 0.50, 0.00, 0.00, 0.00, 0.70, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
    "pron":            np.array([0.70, 0.40, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
}

_EN_WEIGHT_TRANSITIONS = {
    "det":             {"noun_sing": 0.95, "noun_pl": 0.85, "adj_base": 0.80, "adj_comp": 0.50, "noun_proper": 0.60},
    "noun_sing":       {"verb_present": 0.80, "verb_past": 0.60, "prep": 0.70, "adj_base": 0.50, "verb_gerund": 0.30, "conj": 0.40, "pron": 0.20},
    "noun_pl":         {"verb_present": 0.80, "verb_past": 0.60, "prep": 0.70, "adj_base": 0.50},
    "noun_proper":     {"verb_present": 0.80, "verb_past": 0.60, "prep": 0.40},
    "pron":            {"verb_present": 0.90, "verb_past": 0.70, "verb_gerund": 0.50, "adj_base": 0.30},
    "verb_present":    {"noun_sing": 0.50, "noun_pl": 0.50, "adv": 0.80, "prep": 0.70, "adj_base": 0.30, "verb_gerund": 0.20},
    "verb_past":       {"noun_sing": 0.50, "noun_pl": 0.50, "adv": 0.70, "prep": 0.60, "verb_participle": 0.40},
    "verb_gerund":     {"noun_sing": 0.50, "prep": 0.40, "adv": 0.50, "verb_present": 0.20},
    "verb_participle": {"noun_sing": 0.40, "prep": 0.30, "adv": 0.50, "adj_base": 0.40},
    "adj_base":        {"noun_sing": 0.85, "noun_pl": 0.75, "adv": 0.30, "verb_present": 0.20, "prep": 0.20, "conj": 0.30},
    "adj_comp":        {"noun_sing": 0.80, "noun_pl": 0.70, "adv": 0.20},
    "adj_sup":         {"noun_sing": 0.80, "noun_pl": 0.70},
    "adv":             {"verb_present": 0.80, "verb_past": 0.70, "adj_base": 0.80, "adj_comp": 0.70, "adj_sup": 0.60, "adv": 0.30},
    "prep":            {"noun_sing": 0.85, "noun_pl": 0.80, "pron": 0.60, "verb_gerund": 0.50, "det": 0.40, "noun_proper": 0.50},
}

_EN_POS_TO_WEIGHT = {
    'det': 'det', 'noun': 'noun_sing', 'verb': 'verb_present',
    'adj': 'adj_base', 'adv': 'adv', 'prep': 'prep',
    'pron': 'pron', 'conj': 'det', 'aux': 'verb_present',
    'num': 'noun_sing', 'intj': 'adv', 'part': 'adv',
}


class EnglishWeightResonance:
    def __init__(self, english_morph=None):
        self.english_morph = english_morph or EnglishMorphology()
        self._weight_pv_cache = {}
        for name, vec in _EN_WEIGHT_EMBEDDINGS.items():
            nrm = np.linalg.norm(vec)
            self._weight_pv_cache[name] = vec / nrm if nrm > 1e-10 else vec

    def get_weight(self, word):
        morph = self.english_morph.analyze(word)
        pos = morph['pos']
        w = morph['word']
        if pos == 'noun':
            if morph['number'] == 'plural':
                return 'noun_pl'
            if w[0].isupper() and len(w) > 1:
                return 'noun_proper'
            return 'noun_sing'
        if pos == 'verb':
            if morph['tense'] == 'past':
                if w in {'was', 'were', 'had', 'did', 'went', 'said', 'made', 'took', 'got', 'found'}:
                    return 'verb_past'
                if w.endswith('ed') and len(w) > 3:
                    return 'verb_participle'
                return 'verb_past'
            if w.endswith('ing'):
                return 'verb_gerund'
            return 'verb_present'
        if pos == 'adj':
            if w.endswith('er') and len(w) > 3:
                return 'adj_comp'
            if w.endswith('est') and len(w) > 3:
                return 'adj_sup'
            return 'adj_base'
        if pos == 'aux':
            if w in {'was', 'were', 'had', 'did'}:
                return 'verb_past'
            return 'verb_present'
        return _EN_POS_TO_WEIGHT.get(pos, 'noun_sing')

    def get_weight_pv(self, weight_name):
        return self._weight_pv_cache.get(weight_name, np.zeros(22))

    def resonance(self, w1_weight, w2_weight):
        if w1_weight is None or w2_weight is None:
            return 0.0
        pv1 = self._weight_pv_cache.get(w1_weight)
        pv2 = self._weight_pv_cache.get(w2_weight)
        if pv1 is None or pv2 is None:
            return 0.0
        return float(np.mean(np.cos(pv1 - pv2)))

    def transition_score(self, prev_weight, word_weight):
        if prev_weight is None or word_weight is None:
            return 0.0
        trans = _EN_WEIGHT_TRANSITIONS.get(prev_weight, {})
        return trans.get(word_weight, 0.2)

    def weight_density(self, words):
        weight_counts = {}
        for w in words:
            weight = self.get_weight(w)
            weight_counts[weight] = weight_counts.get(weight, 0) + 1
        if not weight_counts:
            return 0.0
        max_count = max(weight_counts.values())
        return max_count / max(len(words), 1)


# ========== English Syntax Anchors ==========

_EN_SYNTAX_ANCHORS = {
    "noun": np.array([0.20, 1.00, 0.10, 0.00, 0.30, 0.10]),
    "verb": np.array([1.00, 0.20, 0.00, 0.00, 0.15, 0.10]),
    "adj":  np.array([0.10, 0.50, 0.00, 1.00, 0.10, 0.00]),
    "adv":  np.array([0.10, 0.10, 0.00, 0.30, 1.00, 0.00]),
    "det":  np.array([0.00, 0.80, 0.70, 0.00, 0.00, 0.00]),
    "prep": np.array([0.00, 0.70, 1.00, 0.00, 0.10, 0.00]),
    "pron": np.array([0.50, 0.50, 0.10, 0.00, 0.10, 0.00]),
    "conj": np.array([0.20, 0.20, 0.10, 0.05, 0.05, 1.00]),
}


def get_en_syntax_anchor(word):
    w = word.lower().strip()
    if w in _ARTICLES or w in _DETERMINERS:
        return _EN_SYNTAX_ANCHORS["det"]
    if w in _PREPOSITIONS:
        return _EN_SYNTAX_ANCHORS["prep"]
    if w in _CONJUNCTIONS:
        return _EN_SYNTAX_ANCHORS["conj"]
    if w in _PRONOUNS:
        return _EN_SYNTAX_ANCHORS["pron"]
    pos = _tag_pos(w)
    if pos in _EN_SYNTAX_ANCHORS:
        return _EN_SYNTAX_ANCHORS[pos]
    return _EN_SYNTAX_ANCHORS["noun"]
