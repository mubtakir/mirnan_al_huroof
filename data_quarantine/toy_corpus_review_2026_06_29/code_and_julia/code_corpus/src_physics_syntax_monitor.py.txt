import numpy as np
from src.physics.syntax_field import compute_syntax_vector, expected_syntax, SYNTAX_ANCHORS


class SyntaxMonitor:
    def __init__(self, vocab, K, morpho=None, window=5, alarm_R=0.85):
        self.vocab = vocab
        self.K = K
        self.morpho = morpho
        self.window = window
        self.alarm_R = alarm_R
        self.labels = ['فعل', 'اسم', 'ح.جر', 'أداة', 'عطف', 'كان']
        self.reset()

    def reset(self):
        self.R_history = []
        self.align_history = []
        self.dim_drifts = []
        self.transitions = []
        self.alarms = []
        self.errors_by_cat = {}

    def step(self, prev_word, chosen_word, w_pv_full, syn_align_raw=0.0):
        exp = expected_syntax(prev_word, self.vocab, self.K)
        w2v = compute_syntax_vector(chosen_word, self.vocab, self.K, self.morpho)
        nrm_e = np.linalg.norm(exp)
        nrm_w = np.linalg.norm(w2v)
        if nrm_e < 1e-10 or nrm_w < 1e-10:
            self.R_history.append(0.0)
            dim_drift = np.zeros(6)
        else:
            u = w2v / nrm_w
            v = exp / nrm_e
            per_dim = np.cos(u - v)
            R = float(np.mean(per_dim))
            self.R_history.append(R)
            dim_drift = 1.0 - per_dim
            self.dim_drifts.append(dim_drift)
            if R < self.alarm_R:
                rebel_dim = int(np.argmax(dim_drift))
                self.alarms.append({
                    'step': len(self.R_history),
                    'prev': prev_word,
                    'chosen': chosen_word,
                    'R': R,
                    'rebel_dim': rebel_dim,
                    'rebel_label': self.labels[rebel_dim],
                    'per_dim': np.round(per_dim, 4).tolist()
                })
        self.align_history.append(syn_align_raw)

    def report(self):
        R = np.array(self.R_history)
        n = len(R)
        if n == 0:
            return {'verdict': 'فارغ', 'R_mean': 0, 'R_min': 0}
        R_mean = float(np.mean(R))
        R_min = float(np.min(R))
        R_std = float(np.std(R))
        decay = float(np.polyfit(np.arange(n), R, 1)[0]) if n > 5 else 0.0
        drift_events = len(self.alarms)
        dim_drifts_arr = np.array(self.dim_drifts) if self.dim_drifts else np.zeros((1, 6))
        mean_dim_drift = np.mean(dim_drifts_arr, axis=0) if dim_drifts_arr.size > 0 else np.zeros(6)
        rebel_counts = {}
        for a in self.alarms:
            l = a['rebel_label']
            rebel_counts[l] = rebel_counts.get(l, 0) + 1
        if R_mean >= 0.85 and R_min >= 0.70:
            verdict = 'متماسك'
        elif R_mean >= 0.75 and R_min >= 0.50:
            verdict = 'منزلق'
        else:
            verdict = 'حرج'
        return {
            'verdict': verdict,
            'R_mean': R_mean,
            'R_min': R_min,
            'R_std': R_std,
            'R_decay': decay,
            'drift_events': drift_events,
            'rebel_counts': rebel_counts,
            'mean_dim_drift': np.round(mean_dim_drift, 4).tolist(),
            'n_steps': n
        }

    def summary_line(self):
        r = self.report()
        return (f'SyntaxLock={r["R_mean"]:.3f} '
                f'min={r["R_min"]:.3f} '
                f'decay={r["R_decay"]:.4f}/step '
                f'drifts={r["drift_events"]} '
                f'verdict={r["verdict"]}')
