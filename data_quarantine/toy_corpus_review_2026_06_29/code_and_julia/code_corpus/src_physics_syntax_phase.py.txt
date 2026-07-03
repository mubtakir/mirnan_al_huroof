import numpy as np
import json
from collections import defaultdict
from src.physics.word_physics import compute_extended_phase_vector


class SyntaxTransitionField:
    def __init__(self):
        self.stability = {}
        self._pv_cache = {}
        self.avg_delta_all = 2.0  # default for unseen pairs

    def _get_pv(self, word):
        if word not in self._pv_cache:
            self._pv_cache[word] = compute_extended_phase_vector(word)
        return self._pv_cache[word]

    def extract(self, texts):
        pair_deltas = defaultdict(list)
        deltas_all = []
        for text in texts:
            words = text.split()
            for i in range(len(words) - 1):
                w1, w2 = words[i], words[i + 1]
                pv1 = self._get_pv(w1)
                pv2 = self._get_pv(w2)
                delta = float(np.linalg.norm(pv2 - pv1))
                pair_deltas[(w1, w2)].append(delta)
                deltas_all.append(delta)
        if deltas_all:
            self.avg_delta_all = float(np.mean(deltas_all))

        self.stability = {}
        for (w1, w2), deltas in pair_deltas.items():
            avg_delta = np.mean(deltas)
            self.stability[(w1, w2)] = float(np.exp(-avg_delta * 2.0))
        return self.stability

    def gate(self, prev_word, candidate_word):
        pair = (prev_word, candidate_word)
        if pair in self.stability:
            return self.stability[pair]
        pv1 = self._get_pv(prev_word)
        pv2 = self._get_pv(candidate_word)
        delta = float(np.linalg.norm(pv2 - pv1))
        return float(np.exp(-delta * 2.0))

    def save(self, path):
        data = {
            "avg_delta_all": self.avg_delta_all,
            "pairs": {}
        }
        for (w1, w2), s in self.stability.items():
            data["pairs"][f"{w1}||{w2}"] = {"stability": s}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.stability = {}
        self.avg_delta_all = data.get("avg_delta_all", 2.0)
        for key, d in data["pairs"].items():
            w1, w2 = key.split("||")
            self.stability[(w1, w2)] = d["stability"]
        return self.stability
