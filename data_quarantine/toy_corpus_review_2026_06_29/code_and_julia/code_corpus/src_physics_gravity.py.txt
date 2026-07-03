import numpy as np
from src.physics.constants import GRAVITY_G, PHASE_DIM


def gravitational_force(m_i: float, v_i: np.ndarray,
                        m_j: float, v_j: np.ndarray,
                        distance: float) -> np.ndarray:
    r_sq = max(distance ** 2, 0.01)
    magnitude = GRAVITY_G * m_i * m_j / r_sq
    delta = v_j - v_i
    norm = np.linalg.norm(delta)
    if norm < 1e-12:
        return np.zeros(PHASE_DIM)
    return magnitude * delta / norm
