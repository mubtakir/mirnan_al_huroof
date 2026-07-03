"""EntropyGate — بوابة الإنتروبيا كمحرك انهيار الدالة الموجية.

يراقب إنتروبيا النظام S(t) ويقرر متى تنهار الدالة الموجية
عندما S(t) < S_crit. يقوم بالتبريد والتسخين الذاتي عبر
ضبط k_B و β ديناميكياً.
"""
import logging
import numpy as np
from src.physics.constants import BOLTZMANN_KB
from src.physics.word_physics import phase_similarity

logger = logging.getLogger(__name__)


class EntropyGate:
    def __init__(self, S_crit=2.5, k_B=BOLTZMANN_KB, beta_0=2.0):
        self.S_crit = S_crit
        self.k_B = k_B
        self.beta_0 = beta_0
        self.k_B_orig = k_B
        self.corrections_applied = 0

    def compute_S(self, pv_list, target_pv):
        if not pv_list or target_pv is None:
            return 0.0
        sims = np.array([phase_similarity(pv, target_pv) for pv in pv_list])
        sims = sims - sims.min()
        s_sum = sims.sum()
        if s_sum < 1e-10:
            return 0.0
        p = sims / s_sum
        S = -float(np.sum(p * np.log(p + 1e-10)))
        return S

    def correct(self, S, candidate_words, pv_list, target_pv, k_B_cur, beta_cur):
        if S < self.S_crit:
            return k_B_cur, beta_cur, 1.0
        k_B_new = k_B_cur / 2.0
        beta_new = min(beta_cur * 1.5, 6.0)
        self.corrections_applied += 1
        logger.debug("EntropyGate correction #%d: S=%.3f >= S_crit=%.3f, k_B: %.4f->%.4f, beta: %.2f->%.2f",
                     self.corrections_applied, S, self.S_crit, k_B_cur, k_B_new, beta_cur, beta_new)
        return k_B_new, beta_new, 1.5

    def evaluate(self, pv_list, target_pv, k_B_cur, beta_cur):
        S = self.compute_S(pv_list, target_pv)
        return self.correct(S, None, pv_list, target_pv, k_B_cur, beta_cur)

    def reset(self):
        self.corrections_applied = 0
