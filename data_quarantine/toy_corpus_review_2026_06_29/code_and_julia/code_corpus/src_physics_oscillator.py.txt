import numpy as np
from src.physics.constants import PHASE_DIM
from src.physics.gravity import gravitational_force


class OscillatorEngine:
    def __init__(self, coupling_K=None):
        self.coupling = coupling_K
        self.dim = PHASE_DIM

    def _derivative(self, omega, phases, masses, phase_vectors, coupling_sub, temperature=0.0):
        n = len(omega)
        dphi = np.zeros((n, self.dim))
        for i in range(n):
            kuramoto = np.zeros(self.dim)
            gravity = np.zeros(self.dim)
            for j in range(n):
                if j == i:
                    continue
                d = phases[j] - phases[i]
                kuramoto += coupling_sub[i, j] * np.sin(d)
                f = gravitational_force(
                    masses[i], phase_vectors[i],
                    masses[j], phase_vectors[j],
                    abs(j - i)
                )
                gravity += f
            noise = np.random.normal(0, temperature * 0.1, self.dim)
            dphi[i] = omega[i] + kuramoto + gravity + noise
        return dphi

    def simulate(self, omega, phases, masses, phase_vectors,
                 coupling_sub, dt=0.01, steps=100, temperature=0.0):
        n = len(omega)
        history = np.zeros((steps + 1, n, self.dim))
        history[0] = phases.copy()
        for step in range(steps):
            k1 = self._derivative(omega, phases, masses, phase_vectors, coupling_sub, temperature)
            k2 = self._derivative(omega, phases + 0.5*dt*k1, masses, phase_vectors, coupling_sub, temperature)
            k3 = self._derivative(omega, phases + 0.5*dt*k2, masses, phase_vectors, coupling_sub, temperature)
            k4 = self._derivative(omega, phases + dt*k3, masses, phase_vectors, coupling_sub, temperature)
            phases = phases + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            history[step + 1] = phases.copy()
        return phases, history
