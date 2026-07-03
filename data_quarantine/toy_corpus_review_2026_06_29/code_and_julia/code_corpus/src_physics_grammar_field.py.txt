import os
import json
import numpy as np
from src.physics.constants import PHASE_DIM
from src.physics.word_physics import compute_word_phase_vector, get_letter_db


_OP_MAP = {"+1": 1, "-1": -1, "0": 0, "r": 0, "i": 0}


def _word_category(word):
    db = get_letter_db()
    score = 0
    for ch in word:
        info = db.data.get(ch)
        if info:
            op = info.get("operator", "0")
            score += _OP_MAP.get(op, 0)
    if score > 0:
        return 1
    if score < 0:
        return -1
    return 0


class SyntaxField:
    """SyntaxField يعمل على مستوى bigram / تلازم كلمتين — يعتمد على أطوار الكلمات الكاملة (22D).
    
    هذا يختلف عن syntax_field.py الذي يعمل على مراسي نحوية 6D (فعل، اسم، حرف...).
    يُستعمل SyntaxField هذا في synchronize و model_io و model_bundle.
    """
    def __init__(self, dim=PHASE_DIM):
        self.dim = dim
        self.bigram_cos = {}
        self.bigram_count = {}
        self.cat_transition = {}
        self.cat_count = {}

    def observe(self, words):
        total = len(words)
        for i in range(total - 1):
            pair = (words[i], words[i + 1])
            self.bigram_count[pair] = self.bigram_count.get(pair, 0) + 1
        for i in range(total - 1):
            c1 = _word_category(words[i])
            c2 = _word_category(words[i + 1])
            key = (c1, c2)
            self.cat_transition[key] = self.cat_transition.get(key, 0) + 1
            self.cat_count[c1] = self.cat_count.get(c1, 0) + 1
            self.cat_count[c2] = self.cat_count.get(c2, 0) + 1

    def finalize(self, vocab):
        self.bigram_cos_mean = {}
        for pair, cnt in self.bigram_count.items():
            if cnt < 2:
                continue
            pv1 = compute_word_phase_vector(pair[0])
            pv2 = compute_word_phase_vector(pair[1])
            cos_val = float(np.mean(np.cos(pv1 - pv2)))
            self.bigram_cos_mean[pair] = cos_val
        total_trans = sum(self.cat_transition.values())
        self.cat_trans_prob = {}
        for (c1, c2), cnt in self.cat_transition.items():
            self.cat_trans_prob[(c1, c2)] = cnt / max(total_trans, 1)

    def transition_align(self, prev_word, candidate):
        pair = (prev_word, candidate)
        align = self.bigram_cos_mean.get(pair, 0.0)
        c1 = _word_category(prev_word)
        c2 = _word_category(candidate)
        cat_boost = self.cat_trans_prob.get((c1, c2), 0.0) * 3.0
        return align + cat_boost

    def cohesion_score(self, prev_word, candidate):
        """
        Symbolic cohesive forces based on definiteness and conjunctions.
        """
        if not prev_word or not candidate:
            return 0.0
            
        score = 0.0
        prev_def = prev_word.startswith("ال") and len(prev_word) > 3
        cand_def = candidate.startswith("ال") and len(candidate) > 3
        cand_conj = candidate.startswith("و") and len(candidate) > 2
        
        # If prev word is indefinite noun, and candidate is definite (Mudaf - Mudaf Ilayh)
        # e.g., انخفاض الانتروبيا
        if not prev_def and not prev_word.startswith("و") and cand_def:
            score += 2.0  # Strong pull for construct state
            
        # Noun-Adjective Definiteness Matching
        if prev_def and cand_def:
            score += 1.5
            
        # Repel conjunctions immediately after an indefinite noun
        if not prev_def and cand_conj:
            score -= 1.5
            
        return score

    def cat_label(self, c):
        return {1: "بنائي", -1: "تدميري", 0: "محايد"}.get(c, "?")

    def save(self, path: str):
        data = {
            'dim': self.dim,
            'bigram_cos': {f'{k[0]}||{k[1]}': v for k, v in self.bigram_cos.items()},
            'bigram_count': {f'{k[0]}||{k[1]}': v for k, v in self.bigram_count.items()},
            'cat_transition': {f'{k[0]}||{k[1]}': v for k, v in self.cat_transition.items()},
            'cat_count': {str(k): v for k, v in self.cat_count.items()},
            'bigram_cos_mean': {f'{k[0]}||{k[1]}': v for k, v in getattr(self, 'bigram_cos_mean', {}).items()},
            'cat_trans_prob': {f'{k[0]}||{k[1]}': v for k, v in getattr(self, 'cat_trans_prob', {}).items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        obj.dim = data.get('dim', PHASE_DIM)
        obj.bigram_cos = {tuple(k.split('||')): v for k, v in data.get('bigram_cos', {}).items()}
        obj.bigram_count = {tuple(k.split('||')): v for k, v in data.get('bigram_count', {}).items()}
        obj.cat_transition = {tuple(k.split('||')): v for k, v in data.get('cat_transition', {}).items()}
        obj.cat_count = {int(k): v for k, v in data.get('cat_count', {}).items()}
        obj.bigram_cos_mean = {tuple(k.split('||')): v for k, v in data.get('bigram_cos_mean', {}).items()}
        obj.cat_trans_prob = {tuple(k.split('||')): v for k, v in data.get('cat_trans_prob', {}).items()}
        return obj
