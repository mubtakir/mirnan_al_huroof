/**
 * Phase-Resonant Neural Network (PRNN) — Stuart-Landau Complex Dynamics
 * ====================================================================
 * Each neuron is a complex oscillator z = re + i·im with amplitude |z|
 * and phase φ = atan2(im, re). The network evolves via:
 *
 *   dz_i/dt = (μ - a_i - g_inh·⟨|z|²⟩ - |z_i|² + i·ω_i) · z_i
 *             + Σ_j K_ij · (z_j - z_i) + noise
 *
 *   da_i/dt = (-a_i + γ·|z_i|²) / τ_a    (neural fatigue)
 *
 * Learning: Contrastive Hebbian (CHL) + Reward-modulated Hebbian,
 * both amplitude-weighted: Re(z_i · conj(z_j)) = r_i·r_j·cos(φ_i - φ_j)
 */

class PRNN {
    constructor() {
        this.N = 8;

        // Stuart-Landau complex state
        this.z_re = new Float32Array(this.N);
        this.z_im = new Float32Array(this.N);
        this.phases = new Float32Array(this.N);
        this.r = new Float32Array(this.N);
        this.a = new Float32Array(this.N);
        this.omega = new Float32Array(this.N);

        // Coupling matrix (symmetric)
        this.K = Array.from({ length: this.N }, () => new Float32Array(this.N));

        // Stuart-Landau parameters
        this.mu = 1.0;
        this.g_inh = 0.3;
        this.gamma = 2.0;
        this.tau_a = 1.5;
        this.dt = 0.05;

        // Learning
        this.learningRate = 0.05;
        this.weightDecay = 0.0005;
        this.maxWeight = 2.0;

        this.init();
    }

    init() {
        for (let i = 0; i < this.N; i++) {
            const phase = (Math.random() - 0.5) * 2 * Math.PI;
            this.z_re[i] = Math.cos(phase);
            this.z_im[i] = Math.sin(phase);
            this.phases[i] = phase;
            this.r[i] = 1.0;
            this.a[i] = 0.0;
            this.omega[i] = (Math.random() - 0.5) * 0.02;
        }
        // Bias node (index 3): always at phase 0, amplitude 1, no drift
        this.omega[3] = 0.0;
        this.z_re[3] = 1.0; this.z_im[3] = 0.0;
        this.phases[3] = 0.0; this.r[3] = 1.0; this.a[3] = 0.0;

        // Small random symmetric weights (only along allowed topology)
        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.N; j++) {
                if (i === j) {
                    this.K[i][j] = 0.0;
                } else if (this.isConnected(i, j)) {
                    this.K[i][j] = (Math.random() - 0.5) * 0.5;
                    this.K[j][i] = this.K[i][j];
                } else {
                    this.K[i][j] = 0.0;
                }
            }
        }
    }

    /** Node index mapping:
     *  0: Left Sensor
     *  1: Center Sensor
     *  2: Right Sensor
     *  3: Bias (reference ground, always phase=0)
     *  4,5,6: Hidden layer
     *  7: Output (steering)
     */
    isConnected(i, j) {
        if (i === j) return false;
        const isIn = (k) => k >= 0 && k <= 2;
        const isBias = (k) => k === 3;
        const isHid = (k) => k >= 4 && k <= 6;
        const isOut = (k) => k === 7;

        if (isIn(i) && isHid(j)) return true;
        if (isIn(j) && isHid(i)) return true;
        if (isBias(i) && (isHid(j) || isOut(j))) return true;
        if (isBias(j) && (isHid(i) || isOut(i))) return true;
        if (isHid(i) && isHid(j)) return true;
        if (isHid(i) && isOut(j)) return true;
        if (isHid(j) && isOut(i)) return true;
        return false;
    }

    /** One Euler step of Stuart-Landau dynamics */
    stepSL(clampedNodes, clampedPhases, noiseLevel) {
        const N = this.N, dt = this.dt;
        const dz_re = new Float32Array(N);
        const dz_im = new Float32Array(N);

        // 1. Enforce clamped nodes as z = exp(i·phase)
        for (const idx of clampedNodes) {
            const ph = clampedPhases[idx] !== undefined ? clampedPhases[idx] : this.phases[idx];
            this.z_re[idx] = Math.cos(ph);
            this.z_im[idx] = Math.sin(ph);
            this.a[idx] = 0.0;
        }

        // 2. Global activity ⟨|z|²⟩
        let gAct = 0;
        for (let i = 0; i < N; i++) gAct += this.z_re[i] ** 2 + this.z_im[i] ** 2;
        gAct /= N;

        // 3. Compute derivatives
        for (let i = 0; i < N; i++) {
            if (clampedNodes.includes(i)) { dz_re[i] = 0; dz_im[i] = 0; continue; }
            const re = this.z_re[i], im = this.z_im[i];
            const m2 = re * re + im * im;

            let cr = 0, ci = 0;
            for (let j = 0; j < N; j++) {
                if (this.K[i][j] !== 0) {
                    cr += this.K[i][j] * (this.z_re[j] - re);
                    ci += this.K[i][j] * (this.z_im[j] - im);
                }
            }

            const F = this.mu - this.a[i] - this.g_inh * gAct - m2;
            dz_re[i] = F * re - this.omega[i] * im + cr;
            dz_im[i] = this.omega[i] * re + F * im + ci;
        }

        // 4. Integrate with Langevin noise
        for (let i = 0; i < N; i++) {
            if (!clampedNodes.includes(i)) {
                const n = noiseLevel * Math.sqrt(dt) * (Math.random() - 0.5) * 2;
                this.z_re[i] += dt * dz_re[i] + n;
                this.z_im[i] += dt * dz_im[i] + n;
            }
            this.phases[i] = Math.atan2(this.z_im[i], this.z_re[i]);
            this.r[i] = Math.hypot(this.z_re[i], this.z_im[i]);
            if (!clampedNodes.includes(i)) {
                this.a[i] += dt * (-this.a[i] + this.gamma * (this.z_re[i] ** 2 + this.z_im[i] ** 2)) / this.tau_a;
            }
        }
    }

    /** Run multi-step settle */
    settle(clampedNodes = [], clampedPhases = {}, steps = 80, dt = 0.05, noiseLevel = 0.02) {
        this.dt = dt;
        for (let s = 0; s < steps; s++) this.stepSL(clampedNodes, clampedPhases, noiseLevel);
    }

    /** Save a snapshot of the complex state */
    saveState() { return { re: [...this.z_re], im: [...this.z_im] }; }

    /** Restore a previously-saved complex state */
    restoreState(s) {
        for (let i = 0; i < this.N; i++) {
            this.z_re[i] = s.re[i];
            this.z_im[i] = s.im[i];
            this.phases[i] = Math.atan2(s.im[i], s.re[i]);
            this.r[i] = Math.hypot(s.re[i], s.im[i]);
        }
    }

    /**
     * Reward-modulated Hebbian (amplitude-weighted)
     *   ΔK_ij = η · R · Re(z_i · conj(z_j))
     */
    updateRewardHebbian(reward) {
        for (let i = 0; i < this.N; i++) {
            for (let j = i + 1; j < this.N; j++) {
                if (!this.isConnected(i, j)) continue;
                const coh = this.z_re[i] * this.z_re[j] + this.z_im[i] * this.z_im[j];
                const dk = this.learningRate * reward * coh;
                this.K[i][j] += dk;
                this.K[i][j] *= (1.0 - this.weightDecay);
                this.K[i][j] = Math.max(-this.maxWeight, Math.min(this.maxWeight, this.K[i][j]));
                this.K[j][i] = this.K[i][j];
            }
        }
    }

    /**
     * Contrastive Hebbian Learning (amplitude-weighted)
     *   ΔK_ij = η · [Re(z⁺ᵢ·conj(z⁺ⱼ)) - Re(z⁻ᵢ·conj(z⁻ⱼ))]
     * Pass the complex state of both phases.
     */
    updateContrastive(zPos, zNeg) {
        for (let i = 0; i < this.N; i++) {
            for (let j = i + 1; j < this.N; j++) {
                if (!this.isConnected(i, j)) continue;
                const pos = zPos.re[i] * zPos.re[j] + zPos.im[i] * zPos.im[j];
                const neg = zNeg.re[i] * zNeg.re[j] + zNeg.im[i] * zNeg.im[j];
                const dk = this.learningRate * (pos - neg);
                this.K[i][j] += dk;
                this.K[i][j] *= (1.0 - this.weightDecay);
                this.K[i][j] = Math.max(-this.maxWeight, Math.min(this.maxWeight, this.K[i][j]));
                this.K[j][i] = this.K[i][j];
            }
        }
    }
}
