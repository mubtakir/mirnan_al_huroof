"""AttractorMemory — الذاكرة الزمنية المتلاشية (Temporal Decay).

تخزن المخرجات السابقة كجواذب طورية وتسترجعها كسياق.
تستخدم اضمحلالاً أسياً exp(-decay × age) لتقليل تأثير
الماضي السحيق تلقائياً.
"""
import numpy as np
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM


class AttractorMemory:
    def __init__(self, decay=0.01, merge_cos=0.92):
        self.centers = []       # list of phi_center (TOTAL_DIM)
        self.sigmas = []        # list of bandwidth σ
        self.word_seqs = []     # list of word lists
        self.coupling_masks = []  # list of {word: weight} dicts
        self.ages = []          # list of ages for temporal decay
        self.decay = decay
        self.merge_cos = merge_cos
        self._pv_cache = {}

    def tick(self):
        for i in range(len(self.ages)):
            self.ages[i] += 1

    def _get_pv(self, word):
        if word not in self._pv_cache:
            self._pv_cache[word] = compute_extended_phase_vector(word)
        return self._pv_cache[word]

    def observe(self, words, sigma=None):
        if not words:
            return
        pvs = np.array([self._get_pv(w) for w in words])
        phi_center = np.mean(pvs, axis=0)
        nrm = np.linalg.norm(phi_center)
        if nrm > 1e-10:
            phi_center = phi_center / nrm

        if sigma is None:
            norms = np.linalg.norm(pvs, axis=1)
            diffs = pvs - phi_center
            dists = np.linalg.norm(diffs, axis=1)
            sigma = float(np.mean(dists)) + 0.01

        mask = {}
        for w in set(words):
            mask[w] = 1.0

        for i in range(len(self.centers)):
            sim = float(np.mean(np.cos(self.centers[i] - phi_center)))
            if sim > self.merge_cos:
                n1 = len(self.word_seqs[i])
                n2 = len(words)
                self.centers[i] = (n1 * self.centers[i] + n2 * phi_center) / (n1 + n2)
                nrm = np.linalg.norm(self.centers[i])
                if nrm > 1e-10:
                    self.centers[i] = self.centers[i] / nrm
                self.sigmas[i] = min(self.sigmas[i], sigma)
                for w, wt in mask.items():
                    self.coupling_masks[i][w] = self.coupling_masks[i].get(w, 0.0) + wt
                self.word_seqs[i].extend(words)
                return i

        self.centers.append(phi_center)
        self.sigmas.append(sigma)
        self.word_seqs.append(words[:])
        self.coupling_masks.append(mask)
        self.ages.append(0)
        return len(self.centers) - 1

    def resonate(self, phi_current, top_k=3):
        if not self.centers:
            return []
        scores = []
        for i, (c, s, age) in enumerate(zip(self.centers, self.sigmas, self.ages)):
            dim = min(len(phi_current), len(c))
            diff = phi_current[:dim] - c[:dim]
            dist2 = float(np.sum(diff ** 2))
            score = float(np.exp(-dist2 / (2.0 * s ** 2 + 1e-10)))
            decay_factor = float(np.exp(-self.decay * age))
            score *= decay_factor
            scores.append((score, i))
        scores.sort(key=lambda x: -x[0])
        return scores[:top_k]

    def retrieve_context(self, phi_current, max_words=10):
        hits = self.resonate(phi_current, top_k=2)
        if not hits:
            return []
        words = []
        seen = set()
        for score, idx in hits:
            if score < 0.1:
                continue
            for w in self.word_seqs[idx]:
                if w not in seen:
                    words.append(w)
                    seen.add(w)
                    if len(words) >= max_words:
                        return words
        return words

    def get_closest_pv(self, word, phi_current):
        w_pv = self._get_pv(word)
        hits = self.resonate(phi_current, top_k=1)
        if hits and hits[0][0] > 0.3:
            idx = hits[0][1]
            blend = 0.7 * w_pv + 0.3 * self.centers[idx]
            nrm = np.linalg.norm(blend)
            if nrm > 1e-10:
                blend = blend / nrm
            return blend
        return w_pv

    def save(self, path):
        data = {
            'dim': TOTAL_DIM,
            'centers': [c.tolist() for c in self.centers],
            'sigmas': self.sigmas,
            'word_seqs': self.word_seqs,
            'coupling_masks': self.coupling_masks,
            'ages': self.ages,
            'decay': self.decay,
            'merge_cos': self.merge_cos,
        }
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path, decay=0.01, merge_cos=0.92):
        import json
        import numpy as np
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        obj = cls(decay=data.get('decay', decay), merge_cos=data.get('merge_cos', merge_cos))
        obj.centers = [np.array(c) for c in data.get('centers', [])]
        for i in range(len(obj.centers)):
            if len(obj.centers[i]) != TOTAL_DIM:
                old = obj.centers[i]
                padded = np.zeros(TOTAL_DIM)
                n = min(len(old), TOTAL_DIM)
                padded[:n] = old[:n]
                nrm = np.linalg.norm(padded)
                if nrm > 1e-10:
                    padded /= nrm
                obj.centers[i] = padded
        obj.sigmas = data.get('sigmas', [])
        obj.word_seqs = data.get('word_seqs', [])
        obj.coupling_masks = data.get('coupling_masks', [])
        obj.ages = data.get('ages', [0] * len(obj.centers))
        return obj

    def observe_sentence(self, words, sentence_pv=None, sigma=None):
        if not words or len(words) < 2:
            return
        if sentence_pv is None:
            pvs = [self._get_pv(w) for w in words]
            sentence_pv = np.mean(pvs, axis=0)
            nrm = np.linalg.norm(sentence_pv)
            if nrm > 1e-10:
                sentence_pv = sentence_pv / nrm
        if sigma is None:
            sigma = 0.08
        mask = {}
        for w in set(words):
            mask[w] = 1.0
        for i in range(len(self.centers)):
            sim = float(np.mean(np.cos(self.centers[i] - sentence_pv)))
            if sim > self.merge_cos:
                self.centers[i] = (self.centers[i] + sentence_pv) / 2.0
                nrm = np.linalg.norm(self.centers[i])
                if nrm > 1e-10:
                    self.centers[i] = self.centers[i] / nrm
                self.sigmas[i] = min(self.sigmas[i], sigma)
                for w, wt in mask.items():
                    self.coupling_masks[i][w] = self.coupling_masks[i].get(w, 0.0) + wt
                self.word_seqs[i].extend(words)
                return i
        self.centers.append(sentence_pv)
        self.sigmas.append(sigma)
        self.word_seqs.append(words[:])
        self.coupling_masks.append(mask)
        self.ages.append(0)
        return len(self.centers) - 1

    def clear(self):
        self.centers.clear()
        self.sigmas.clear()
        self.word_seqs.clear()
        self.coupling_masks.clear()
        self.ages.clear()

    @property
    def size(self):
        return len(self.centers)
