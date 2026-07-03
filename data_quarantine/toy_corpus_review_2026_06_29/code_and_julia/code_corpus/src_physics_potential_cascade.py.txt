"""PotentialCascadeLayer — طبقة الهوي التراكمي غير التدميرية.

المبدأ:
  كل كلمة تولَّد تخلق بئر جهد في فضاء الطور 64D.
  الكلمة التالية تهوي حتمياً إلى أعمق نقطة في هذا الحقل التراكمي.
  لا softmax, لا temperature, لا عينة عشوائية.

المعادلة:
  V_total(c) = -λ Σ_i [ m_i · max(0, cosΔφ_{i,c}) / (d_pos(i) + δ)^γ ]

حيث:
  m_i      = الكتلة الدلالية للكلمة i في السياق
  cosΔφ    = التوافق الطوري (الرنين الدلالي)
  d_pos(i) = المسافة الموضعية (1 للسابقة مباشرة، 2 للتي قبلها...)
  γ        = أس اضمحلال المسافة (2.0 ≈ جاذبية كولوم)
  δ        = ثابت تجنب التفرد
  λ        = معامل شدة الطبقة
  𝕀_syntax = حاجز نحوي: 0 إذا مخالف، 1 إذا صحيح
"""
import numpy as np
from typing import List, Tuple, Optional
from src.physics.constants import TOTAL_DIM, PHASE_DIM


class PotentialCascadeLayer:
    def __init__(self,
                 lambda_cascade: float = 3.0,
                 gamma: float = 2.0,
                 delta: float = 0.3,
                 phase_lock_threshold: float = 0.65,
                 repulsion_strength: float = 2.5,
                 friction_decay: float = 0.3):
        self.lambda_cascade = lambda_cascade
        self.gamma = gamma
        self.delta = delta
        self.phase_lock_threshold = phase_lock_threshold
        self.repulsion_strength = repulsion_strength
        self.friction_decay = friction_decay

    def compute_score(self,
                      candidate_pv: np.ndarray,
                      context_pvs: List[np.ndarray],
                      context_masses: List[float],
                      syntax_valid: bool,
                      used_words: Optional[list] = None,
                      word_to_pv_fn=None) -> float:
        if not syntax_valid:
            return -np.inf

        if not context_pvs:
            return 0.0

        last_pv = context_pvs[-1]
        align = float(np.dot(candidate_pv, last_pv) /
                      (np.linalg.norm(candidate_pv) * np.linalg.norm(last_pv) + 1e-10))
        if align < self.phase_lock_threshold:
            return -np.inf

        attraction = 0.0
        n_ctx = len(context_pvs)
        for i, (ctx_pv, mass) in enumerate(zip(context_pvs, context_masses)):
            pos_dist = n_ctx - i
            cos_phase = float(np.dot(candidate_pv, ctx_pv) /
                              (np.linalg.norm(candidate_pv) * np.linalg.norm(ctx_pv) + 1e-10))
            cos_phase = max(0.0, cos_phase)
            attraction += (mass * cos_phase) / ((pos_dist + self.delta) ** self.gamma)

        repulsion = 0.0
        if used_words and word_to_pv_fn:
            for w in used_words:
                w_pv = word_to_pv_fn(w)
                if w_pv is not None:
                    d = np.linalg.norm(candidate_pv - w_pv) + 1e-10
                    repulsion += self.repulsion_strength / (d ** 2 + 1e-10)

        if n_ctx > 0:
            friction = self.friction_decay * n_ctx * float(
                np.linalg.norm(candidate_pv - context_pvs[-1]))
        else:
            friction = 0.0

        potential = -attraction + repulsion + friction
        if potential >= 0.0:
            return 0.0
        return self.lambda_cascade * abs(potential)
