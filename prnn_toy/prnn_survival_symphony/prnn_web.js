class PRNN {
    constructor() {
        this.N = 10;
        this.z_re = new Float32Array(this.N);
        this.z_im = new Float32Array(this.N);
        this.phases = new Float32Array(this.N);
        this.r = new Float32Array(this.N);
        this.a = new Float32Array(this.N);
        this.omega = new Float32Array(this.N);
        this.K = Array.from({ length: this.N }, () => new Float32Array(this.N));
        this.mu = 1.0;
        this.g_inh = 0.4;
        this.gamma = 2.0;
        this.tau_a = 1.5;
        this.dt = 0.02;
        this.learningRate = 0.07;
        this.weightDecay = 0.0005;
        this.maxWeight = 2.0;
        this.learningMode = 'contrastive';
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
            this.omega[i] = (Math.random() - 0.5) * 0.05;
        }
        this.omega[3] = 0.0; // bias
        this.z_re[3] = 1.0; this.z_im[3] = 0.0;
        this.phases[3] = 0.0; this.r[3] = 1.0; this.a[3] = 0.0;

        for (let i = 0; i < this.N; i++) {
            for (let j = 0; j < this.N; j++) {
                if (i === j) this.K[i][j] = 0.0;
                else if (this.isConnected(i, j)) {
                    this.K[i][j] = (Math.random() - 0.5) * 0.5;
                    this.K[j][i] = this.K[i][j];
                } else this.K[i][j] = 0.0;
            }
        }
    }

    isConnected(i, j) {
        if (i === j) return false;
        const isIn = k => k >= 0 && k <= 2;
        const isBias = k => k === 3;
        const isPlayerIn = k => k === 4;
        const isHid = k => k >= 5 && k <= 7;
        const isOut = k => k === 8 || k === 9;

        if ((isIn(i) || isPlayerIn(i)) && isOut(j)) return false;
        if ((isIn(j) || isPlayerIn(j)) && isOut(i)) return false;
        if ((isIn(i) || isPlayerIn(i)) && (isIn(j) || isPlayerIn(j))) return false;
        if (isBias(i) && !(isHid(j) || isOut(j))) return false;
        if (isBias(j) && !(isHid(i) || isOut(i))) return false;
        return true;
    }

    stepSL(clampedNodes, clampedPhases, noiseLevel) {
        const N = this.N, dt = this.dt;
        const dz_re = new Float32Array(N), dz_im = new Float32Array(N);

        for (const idx of clampedNodes) {
            const ph = (clampedPhases[idx] !== undefined) ? clampedPhases[idx] : this.phases[idx];
            this.z_re[idx] = Math.cos(ph);
            this.z_im[idx] = Math.sin(ph);
            this.a[idx] = 0.0;
        }

        const freeCount = N - clampedNodes.length;
        let gAct = 0;
        for (let i = 0; i < N; i++) {
            if (!clampedNodes.includes(i)) gAct += this.z_re[i] ** 2 + this.z_im[i] ** 2;
        }
        gAct /= freeCount > 0 ? freeCount : N;

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

        const noise = noiseLevel * Math.sqrt(dt);
        for (let i = 0; i < N; i++) {
            if (!clampedNodes.includes(i)) {
                this.z_re[i] += dt * dz_re[i] + noise * (Math.random() - 0.5) * 2;
                this.z_im[i] += dt * dz_im[i] + noise * (Math.random() - 0.5) * 2;
                if (isNaN(this.z_re[i]) || isNaN(this.z_im[i])) {
                    const p = Math.random() * 2 * Math.PI;
                    this.z_re[i] = Math.cos(p); this.z_im[i] = Math.sin(p);
                }
            }
            this.phases[i] = Math.atan2(this.z_im[i], this.z_re[i]);
            this.r[i] = Math.hypot(this.z_re[i], this.z_im[i]);
            if (!clampedNodes.includes(i)) {
                this.a[i] += dt * (-this.a[i] + this.gamma * (this.z_re[i] ** 2 + this.z_im[i] ** 2)) / this.tau_a;
            }
        }
    }

    settle(clampedNodes = [], clampedPhases = {}, steps = 40, dt = 0.02, noiseLevel = 0.01) {
        this.dt = dt;
        for (let s = 0; s < steps; s++) this.stepSL(clampedNodes, clampedPhases, noiseLevel);
    }

    saveState() { return { re: [...this.z_re], im: [...this.z_im] }; }

    restoreState(s) {
        for (let i = 0; i < this.N; i++) {
            this.z_re[i] = s.re[i];
            this.z_im[i] = s.im[i];
            this.phases[i] = Math.atan2(s.im[i], s.re[i]);
            this.r[i] = Math.hypot(s.re[i], s.im[i]);
        }
    }

    updateContrastive(zPos, zNeg) {
        for (let i = 0; i < this.N; i++) {
            for (let j = i + 1; j < this.N; j++) {
                if (!this.isConnected(i, j)) continue;
                const pos = zPos.re[i] * zPos.re[j] + zPos.im[i] * zPos.im[j];
                const neg = zNeg.re[i] * zNeg.re[j] + zNeg.im[i] * zNeg.im[j];
                let dk = this.learningRate * (pos - neg);
                this.K[i][j] += dk;
                this.K[i][j] *= (1.0 - this.weightDecay);
                this.K[i][j] = Math.max(-this.maxWeight, Math.min(this.maxWeight, this.K[i][j]));
                this.K[j][i] = this.K[i][j];
            }
        }
    }

    updateRewardHebbian(reward) {
        for (let i = 0; i < this.N; i++) {
            for (let j = i + 1; j < this.N; j++) {
                if (!this.isConnected(i, j)) continue;
                const coh = this.z_re[i] * this.z_re[j] + this.z_im[i] * this.z_im[j];
                let dk = this.learningRate * reward * coh;
                this.K[i][j] += dk;
                this.K[i][j] *= (1.0 - this.weightDecay);
                this.K[i][j] = Math.max(-this.maxWeight, Math.min(this.maxWeight, this.K[i][j]));
                this.K[j][i] = this.K[i][j];
            }
        }
    }
}
