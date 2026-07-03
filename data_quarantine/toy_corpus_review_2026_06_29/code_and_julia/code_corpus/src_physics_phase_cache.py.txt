"""PhaseVectorCache — مخبأ موحد للمتجهات الطورية، لتجنب تكرار النمط في 8+ ملفات."""
from src.physics.word_physics import compute_extended_phase_vector


class PhaseVectorCache:
    """مخبأ للمتجهات الطورية (compute_extended_phase_vector)."""

    def __init__(self):
        self._cache = {}

    def get(self, word: str):
        if word not in self._cache:
            self._cache[word] = compute_extended_phase_vector(word)
        return self._cache[word]

    def clear(self):
        self._cache.clear()

    def __contains__(self, word):
        return word in self._cache

    def __len__(self):
        return len(self._cache)
