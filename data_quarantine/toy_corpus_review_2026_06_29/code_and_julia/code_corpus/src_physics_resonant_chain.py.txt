import numpy as np


def _zeta_response(x: float, s: float = 2.0) -> float:
    """استجابة ترددية مستوحاة من دالة زيتا — خط التوازن σ=1/2.

    تُعطي أقصى رنين عند cos≈0.5 (خط الاستواء)، وتُثبّط الأطراف.
    ζ(x) = 1 / (1 + (x-0.5)²)^(s/2)  — نواة لورنتزية متمركزة عند 0.5
    """
    delta = x - 0.5
    return 1.0 / (1.0 + delta * delta) ** (s * 0.5)


class ResonantChain:
    """Each adjacent word pair in a sentence forms an LC tank circuit.

    C (semantic capacitance) — plate area ∝ masses, dielectric ∝ cos(θ)
    L (semantic inductance)  — coupling strength from phase alignment
    f_res = 1 / (2*pi*sqrt(L*C))
    """

    def __init__(self):
        self._freq_history = []

    def capacitance(self, mass_i: float, mass_j: float, cos_theta: float) -> float:
        return max(mass_i, 1e-12) * max(mass_j, 1e-12) * (1.0 + max(cos_theta, -1.0))

    def inductance(self, pv_i: np.ndarray, pv_j: np.ndarray) -> float:
        diff = np.dot(pv_i, pv_j)
        diff = max(-1.0, min(1.0, diff))
        return 1.0 / (1.0 + abs(np.sin(np.arccos(diff) * 0.5)))

    def resonant_freq(self, L: float, C: float) -> float:
        return 1.0 / (2.0 * np.pi * np.sqrt(max(L * C, 1e-24)))

    def pair_freq(self, mass_i: float, mass_j: float, pv_i: np.ndarray, pv_j: np.ndarray) -> float:
        cos_theta = float(np.dot(pv_i, pv_j))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        C = self.capacitance(mass_i, mass_j, cos_theta)
        L = self.inductance(pv_i, pv_j)
        return self.resonant_freq(L, C)

    def sentence_coherence(self, masses: list, pvs: list) -> float:
        if len(masses) < 2:
            return 0.0
        self._freq_history = []
        for i in range(len(masses) - 1):
            f = self.pair_freq(masses[i], masses[i + 1], pvs[i], pvs[i + 1])
            self._freq_history.append(f)
        freqs = np.array(self._freq_history)
        mean_f = np.mean(freqs)
        var_f = np.mean((freqs - mean_f) ** 2)
        max_var = (mean_f ** 2) * 0.25 if mean_f > 0 else 1.0
        coherence = 1.0 - np.sqrt(var_f) / (np.sqrt(max_var) + 1e-12)
        return float(max(0.0, min(1.0, coherence)))

    def score_candidate(self, mass_prev: float, mass_cand: float,
                        pv_prev: np.ndarray, pv_cand: np.ndarray,
                        prev_freqs: list = None) -> float:
        f_cand = self.pair_freq(mass_prev, mass_cand, pv_prev, pv_cand)
        # ═══ زيتا: تعزيز الرنين عند cos≈0.5 (خط التوازن) ═══
        cos_theta = float(np.dot(pv_prev, pv_cand))
        cos_theta = max(-1.0, min(1.0, cos_theta))
        zeta_boost = _zeta_response(cos_theta)
        if prev_freqs and len(prev_freqs) > 0:
            mean_prev = np.mean(prev_freqs)
            std_prev = np.std(prev_freqs) + 1e-12
            delta = abs(f_cand - mean_prev)
            score = np.exp(-0.5 * (delta / std_prev) ** 2)
        else:
            score = 1.0
        return float(score * zeta_boost)  # زيتا: تضخيم الرنين المتوازن
