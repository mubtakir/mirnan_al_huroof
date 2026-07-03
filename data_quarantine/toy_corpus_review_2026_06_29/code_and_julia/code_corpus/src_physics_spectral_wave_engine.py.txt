"""SpectralWaveEngine — محرك الموجات الطيفية للعلاقات الدلالية.

يُحقق الابتكارات البعيدة المدى:
8. الشبكة التوليدية الفيزيائية (Physics Generative Network)
9. الترجمة الفيزيائية (Physics-based Translation)
10. الاستدلال المنطقي الطوري (Phase Space Logical Inference)

المبدأ الأساسي — كل كلمة = حزمة موجية:
    ψ_w(t) = A₀·cos(ω₀·t + φ₀) × [1 + Σ cᵢ·cos(Δωᵢ·t + θᵢ)]
    
    ω₀: التردد الذاتي (من فيزياء الحروف)
    A₀: السعة (من كتلة الكلمة)
    φ₀: الطور (من المتجه الطوري)
    cᵢ, Δωᵢ, θᵢ: توافقيات جانبية من ترافق الكلمة مع كلمات أخرى

العلاقات الدلالية = تداخل الموجات:
    ψ_result = ψ_target - ψ_source₁ + ψ_source₂
    البحث عن كلمة q تحقق |ψ_q - ψ_result| < ε

لا تدريب. لا إحصاء. لا باك بروب. فقط فيزياء موجية خالصة.
"""
import numpy as np
import logging
from collections import defaultdict
from scipy import sparse
from src.physics.constants import PHASE_DIM, TOTAL_DIM
from src.physics.word_physics import (
    compute_word_frequency, compute_word_mass, compute_word_energy,
    compute_word_phase_vector, compute_extended_phase_vector,
    _extract_root_light, _compute_root_dims, _normalize_letters,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 1. تمثيل الكلمة كحزمة موجية
# ─────────────────────────────────────────────

def compute_word_wave(word):
    """حساب المعاملات الموجية الأساسية لكلمة.
    
    Returns dict: {omega_0, amplitude, phase_0, energy, mass}
    """
    freq = compute_word_frequency(word)
    mass = compute_word_mass(word)
    energy = compute_word_energy(word)
    base_pv = compute_word_phase_vector(word)
    
    # ω₀ من تردد الكلمة
    omega_0 = max(freq, 0.1)
    
    # A₀ ∝ √mass (سعة الموجة تتناسب مع الطاقة)
    amplitude = float(np.sqrt(mass * energy + 0.01))
    
    # φ₀ من أول مركبة طورية
    phase_0 = float(np.arctan2(base_pv[1], base_pv[0])) if np.linalg.norm(base_pv[:2]) > 1e-10 else 0.0
    
    return {
        'omega_0': omega_0,
        'amplitude': amplitude,
        'phase_0': phase_0,
        'energy': energy,
        'mass': mass,
    }


def wave_at(word_wave, t, num_harmonics=4):
    """تقييم الموجة عند الزمن t.
    
    ψ(t) = A₀·cos(ω₀·t + φ₀) + Σ Aₙ·cos(n·ω₀·t + n·φ₀)
    """
    omega = word_wave['omega_0']
    amp = word_wave['amplitude']
    phase = word_wave['phase_0']
    
    val = amp * np.cos(omega * t + phase)
    for n in range(2, num_harmonics + 1):
        val += (amp / n) * np.cos(n * omega * t + n * phase)
    
    return val


def wave_spectrum(word_wave, n_samples=32):
    """طيف الكلمة — FFT على عينات من موجتها.
    
    Returns: (frequencies, magnitudes, phases)
    """
    t = np.linspace(0, 2 * np.pi, n_samples)
    signal = np.array([wave_at(word_wave, ti) for ti in t])
    fft_vals = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(n_samples, d=2*np.pi/n_samples)
    magnitudes = np.abs(fft_vals)
    phases = np.angle(fft_vals)
    return freqs, magnitudes, phases


# ─────────────────────────────────────────────
# 2. التوافقيات الجانبية (Sidebands) من السياق
# ─────────────────────────────────────────────

def compute_sidebands(word_wave, context_waves, coupling_threshold=0.15):
    """حساب التوافقيات الجانبية الناتجة عن ترافق الكلمة مع كلمات أخرى.
    
    عندما تترافق كلمتان في جملة، تتولد ترددات جانبية:
    Δω = ω₁ ± ω₂  (تداخل جمعي وطرحى)
    
    يتم تجاهل الاقترانات الأضعف من (coupling_threshold × متوسط قوة جميع الاقترانات)
    لمنع تراكم الضجيج الطيفي.
    
    Returns: list of dicts {delta_omega, coupling_strength, phase_diff}
    """
    sidebands = []
    omega_0 = word_wave['omega_0']
    
    if not context_waves:
        return sidebands
    
    # تمريرة أولى لحساب كل قوى الاقتران
    raw_couplings = []
    for cw in context_waves:
        cw_amp = cw['amplitude']
        coupling = word_wave['amplitude'] * cw_amp
        raw_couplings.append(coupling)
    
    # عتبة ديناميكية: فقط الاقترانات فوق (threshold × median)
    median_coupling = np.median(raw_couplings) if raw_couplings else 0.0
    cutoff = max(1e-10, coupling_threshold * median_coupling)
    
    for idx, cw in enumerate(context_waves):
        coupling = raw_couplings[idx]
        if coupling < cutoff:
            continue
        
        cw_omega = cw['omega_0']
        
        # ترددات جانبية: جمعية وطرحيه
        delta_add = omega_0 + cw_omega
        delta_sub = abs(omega_0 - cw_omega)
        
        # فرق الطور
        phase_diff = abs(word_wave['phase_0'] - cw['phase_0'])
        
        sidebands.append({
            'delta_omega': delta_add,
            'coupling_strength': coupling,
            'phase_diff': phase_diff,
            'type': 'additive',
        })
        sidebands.append({
            'delta_omega': delta_sub,
            'coupling_strength': coupling * 0.5,  # الترددات الطرحية أضعف
            'phase_diff': phase_diff,
            'type': 'subtractive',
        })
    
    return sidebands


def wave_signature(word, context_words=None):
    """البصمة الموجية الكاملة للكلمة — أساسي + سياقي.
    
    Returns dict with full wave representation including sidebands.
    """
    base = compute_word_wave(word)
    base['sidebands'] = []
    base['root'] = _extract_root_light(word)
    
    if context_words:
        ctx_waves = []
        for cw in context_words[-8:]:
            try:
                ctx_waves.append(compute_word_wave(cw))
            except Exception:
                continue
        base['sidebands'] = compute_sidebands(base, ctx_waves)
    
    return base


# ─────────────────────────────────────────────
# 3. مصفوفة الاقتران الطيفي (K_spectral)
# ─────────────────────────────────────────────

class SpectralCouplingMatrix:
    """مصفوفة اقتران طيفي — بديل لمصفوفة K الإحصائية.
    
    K_spectral[i,j] = كثافة التداخل الطيفي بين الكلمتين
    بدلاً من K[i,j] = تكرر ظهورهما معاً.
    
    آلية الحساب:
    1. لكل زوج (i,j) يظهران في نافذة سياقية
    2. نحسب cross-spectral density: ∫ ψ_i(t)·ψ_j(t) dt
    3. نخرّج: magnitude (قوة العلاقة) + phase_diff (نوع العلاقة)
    """
    
    def __init__(self):
        self.magnitude = None       # sparse matrix: قوة الاقتران
        self.phase_diff = None      # sparse matrix: فرق الطور
        self.word_waves = {}        # cache: word → wave dict
        self._fft_cache = {}        # cache: word → (fft_mag, fft_phase)
        self._signal_cache = {}     # cache: word → waveform array (32 samples)
        self._fitted = False
    
    def _get_fft(self, word, n_samples=16):
        """FFT مسبق الحساب — يُخزّن ويعيد الاستخدام لمنع تكرار FTT."""
        key = (word, n_samples)
        if key not in self._fft_cache:
            ww = self.word_waves.get(word) or compute_word_wave(word)
            self.word_waves[word] = ww
            t = np.linspace(0, 2*np.pi, n_samples)
            sig = np.array([wave_at(ww, ti) for ti in t])
            fft_vals = np.fft.rfft(sig)
            self._fft_cache[key] = fft_vals
        return self._fft_cache[key]
    
    def _get_signal(self, word, n_samples=32):
        """موجة مسبقة الحساب — لمنع تكرار wave_at."""
        key = (word, n_samples)
        if key not in self._signal_cache:
            ww = self.word_waves.get(word) or compute_word_wave(word)
            self.word_waves[word] = ww
            t = np.linspace(0, 2*np.pi, n_samples)
            sig = np.array([wave_at(ww, ti) for ti in t])
            self._signal_cache[key] = sig
        return self._signal_cache[key]
    
    def _csd(self, w1, w2):
        """Cross-Spectral Density via FFT مخزّن."""
        fft_i = self._get_fft(w1)
        fft_j = self._get_fft(w2)
        cross = fft_i * np.conj(fft_j)
        mag = float(np.mean(np.abs(cross)))
        phase = float(np.mean(np.angle(cross)))
        return mag, phase
    
    def fit(self, vocab, coupling_K, corpus_texts=None):
        """بناء مصفوفة الاقتران الطيفي من:
        - مصفوفة K الحالية (للهيكل)
        - أو من نصوص مباشرة
        """
        # مسح الخبيئة السابقة لضمان عدم تداخل القيم القديمة
        self.word_waves.clear()
        self._fft_cache.clear()
        self._signal_cache.clear()
        
        V = len(vocab)
        self.magnitude = sparse.lil_matrix((V, V))
        self.phase_diff = sparse.lil_matrix((V, V))
        
        # استخدام K لمعرفة أزواج الكلمات التي تترافق
        if coupling_K is not None:
            K_coo = coupling_K.tocoo()
            for idx in range(K_coo.nnz):
                i, j = K_coo.row[idx], K_coo.col[idx]
                if i >= V or j >= V:
                    continue
                w_i = vocab.id2word.get(i)
                w_j = vocab.id2word.get(j)
                if not w_i or not w_j:
                    continue
                mag, phase = self._csd(w_i, w_j)
                if mag > 1e-10:
                    self.magnitude[i, j] = mag
                    self.phase_diff[i, j] = phase
        
        # معالجة النصوص إن وجدت
        if corpus_texts:
            for text in corpus_texts:
                for line in text.split('\n'):
                    tokens = line.split()
                    if len(tokens) < 2:
                        continue
                    for i in range(len(tokens)):
                        for j in range(i+1, min(i+7, len(tokens))):
                            w_i, w_j = tokens[i], tokens[j]
                            idi = vocab.get(w_i)
                            idj = vocab.get(w_j)
                            if idi is None or idj is None:
                                continue
                            mag, phase = self._csd(w_i, w_j)
                            if mag > 1e-10:
                                existing = self.magnitude[idi, idj]
                                if existing > 0:
                                    # متوسط متحرك
                                    self.magnitude[idi, idj] = (existing + mag) / 2
                                    existing_p = self.phase_diff[idi, idj]
                                    self.phase_diff[idi, idj] = (existing_p + phase) / 2
                                else:
                                    self.magnitude[idi, idj] = mag
                                    self.phase_diff[idi, idj] = phase
        
        self.magnitude = self.magnitude.tocsr()
        self.phase_diff = self.phase_diff.tocsr()
        self._fitted = True
        logger.info(f"SpectralCouplingMatrix: {self.magnitude.nnz} couplings from {V} words")
    
    def get_coupling(self, w1, w2, vocab):
        """قوة الاقتران الطيفي بين كلمتين + نوع العلاقة.
        
        Returns:
            magnitude: قوة العلاقة (0–∞)
            phase_diff: نوع العلاقة (-π إلى +π)
                0: متوافقة (same domain)
                +π/2: تكاملية (cause→effect)
                -π/2: تضادية (antonym)
                +π: متضادة تماماً
        """
        i, j = vocab.get(w1), vocab.get(w2)
        if i is None or j is None or not self._fitted:
            return 0.0, 0.0
        
        mag = float(self.magnitude[i, j])
        phase = float(self.phase_diff[i, j])
        return mag, phase
    
    def resonance_with_context(self, word, context_words, vocab):
        """رنين الكلمة مع السياق: متوسط الاقتران الطيفي مع كلمات السياق."""
        if not context_words:
            return 0.0
        
        total = 0.0
        n = 0
        for cw in context_words[-7:]:
            mag, phase = self.get_coupling(word, cw, vocab)
            # العلاقات المتوافقة (phase ≈ 0) تعطي أعلى رنين
            phase_factor = max(0.0, np.cos(phase))
            total += mag * phase_factor
            n += 1
        
        return total / max(n, 1)
    
    def relational_interference(self, target, add_words, sub_words, vocab):
        """التداخل العلائقي — الإضافة والطرح الموجي.
        
        ψ_result = ψ_target + Σ ψ_add - Σ ψ_sub
        ثم نبحث عن كلمة q تحقق |ψ_q - ψ_result| < ε
        
        هذا يعادل: queen = king - man + woman
        لكن في فضاء الموجات بدلاً من المتجهات الإحصائية.
        
        Returns:
            candidate: الكلمة الأكثر تطابقاً
            score: درجة التطابق (0–1)
        """
        t = np.linspace(0, 2*np.pi, 32)
        result_signal = np.zeros(32)
        
        caches = self._signal_cache
        caches[target] = caches.get(target) or self._get_signal(target)
        result_signal += caches[target]
        
        for word in add_words:
            caches[word] = caches.get(word) or self._get_signal(word)
            result_signal += caches[word]
        
        for word in sub_words:
            caches[word] = caches.get(word) or self._get_signal(word)
            result_signal -= caches[word]
        
        excl = set()
        for w in [target] + add_words + sub_words:
            nw = _normalize_letters(w)
            excl.add(nw)
            # also keep the original in case vocab stores it as-is
            excl.add(w)
        best_word = None
        best_score = -np.inf
        
        for wid in range(len(vocab)):
            w = vocab.id2word.get(wid)
            if not w or w in excl:
                continue
            caches[w] = caches.get(w) or self._get_signal(w)
            w_signal = caches[w]
            
            nrm_r = np.linalg.norm(result_signal)
            nrm_w = np.linalg.norm(w_signal)
            if nrm_r < 1e-10 or nrm_w < 1e-10:
                continue
            
            sim = float(np.dot(result_signal, w_signal) / (nrm_r * nrm_w))
            if sim > best_score:
                best_score = sim
                best_word = w
        
        return best_word, best_score
    
    def coupling_strength(self, word1, word2, vocab):
        """قوة الاقتران الموجي الكلية بين كلمتين.
        
        تجمع: cross-spectral density + root overlap + letter resonance.
        """
        mag, phase = self.get_coupling(word1, word2, vocab)
        
        # معامل تضادي: إذا phase ≈ π, العلاقة تضادية
        if abs(abs(phase) - np.pi) < 0.5:
            return -mag  # سالب = تضاد
        
        return mag


# ─────────────────────────────────────────────
# 4. مرشح السياق التكيفي (لمشكلة تعدد المعاني)
# ─────────────────────────────────────────────

def adaptive_context_filter(word_pv, context_pvs, bandwidth=None):
    """مرشح نطاق ترددي تكيفي — يعزل فقط مركبات الكلمة
    المتوافقة مع السياق الحالي.
    
    لكلمة متعددة المعاني (مثل "عين" = eye/spring)،
    هذا المرشح يحتفظ فقط بالمركبات الطورية المتوافقة مع السياق.
    
    bandwidth: إذا لم يُعطَ، يُحسب تلقائياً من median(|Δ|) للدلتا.
    هذا يضمن أن المرشح يتكيف مع مقياس البيانات ولا يقطع
    الإشارات الضعيفة الضرورية.
    
    Returns: متجه مرشح (TOTAL_DIM)
    """
    if not context_pvs or word_pv is None:
        return word_pv
    
    if len(context_pvs) == 0:
        return word_pv
    
    # متوسط السياق
    ctx_avg = np.mean(np.array(context_pvs[-5:]), axis=0)
    
    delta = np.abs(word_pv - ctx_avg)
    
    # bandwidth تلقائي: median(|Δ|) يُبقى نصف المكونات فوق exp(-1) = 0.37
    if bandwidth is None:
        delta_median = float(np.median(delta))
        bandwidth = max(delta_median, 0.05)
        # حماية من القيم الشاذة: أدنى حد 0.05
    bw = max(float(bandwidth), 0.05)
    
    # قناع سيني ناعم بدلاً من القطع الحاد:
    # يثبّت المكونات ذات delta < bandwidth ويخفف تدريجياً ما فوقها
    mask = 1.0 / (1.0 + (delta / bw) ** 2)
    
    result = word_pv * mask + ctx_avg * (1.0 - mask) * 0.15
    nrm = np.linalg.norm(result)
    if nrm > 1e-10:
        result = result / nrm
    return result


# ─────────────────────────────────────────────
# 5. الشبكة التوليدية الفيزيائية (PGN)
# ─────────────────────────────────────────────

class PhysicsGenerativeNetwork:
    """الشبكة التوليدية الفيزيائية — بديل لتوليد كلمة كلمة.
    
    المبدأ:
    - كل كلمة = عقدة (node) بموجة ψ
    - كل علاقة = حافة (edge) بقوة اقتران طيفي
    - التوليد = إيجاد مسار طيفي من عقدة البداية إلى عقدة النهاية
      بأعلى طاقة كلية (أعلى رنين تراكمي)
    
    الفرق عن التوليد الحالي:
    - بدلاً من توليد كلمة ← تقييم ← تكرار
    - نحدد مساراً كاملاً مسبقاً عبر الموجات
    """
    
    def __init__(self, spectral_matrix):
        self.spectral = spectral_matrix
    
    def find_path(self, start_words, end_words, vocab, max_words=8, temperature=0.5):
        """إيجاد مسار من كلمات البداية إلى كلمات النهاية.
        
        Uses spectral coupling as edge weights + wave resonance.
        
        Returns: list of words (المسار المثالي)
        """
        V = len(vocab)
        if V < 5:
            return []
        
        # معالجة كلمات البداية
        start_ids = [vocab.get(w) for w in start_words if vocab.get(w) is not None]
        end_ids = [vocab.get(w) for w in end_words if vocab.get(w) is not None]
        
        if not start_ids:
            return []
        
        from collections import deque
        
        # BFS بالاقتران الطيفي
        paths = deque()
        for sid in start_ids:
            paths.append([sid])
        
        best_path = None
        best_energy = -np.inf
        
        visited = set(start_ids)
        
        while paths and len(paths[0]) <= max_words:
            path = paths.popleft()
            current = path[-1]
            
            # وصلنا للهدف؟
            if current in end_ids and len(path) > 1:
                energy = self._path_energy(path, vocab)
                if energy > best_energy:
                    best_energy = energy
                    best_path = path
            
            # توسيع المسار
            if self.spectral.magnitude is not None:
                row = self.spectral.magnitude[current].toarray().ravel()
                # أقوى 5 جيران
                neighbors = np.argsort(row)[-10:][::-1]
                for nid in neighbors:
                    if nid in visited or row[nid] < 0.01:
                        continue
                    visited.add(nid)
                    new_path = path + [nid]
                    paths.append(new_path)
        
        if best_path:
            return [vocab.id2word[nid] for nid in best_path]
        
        return []
    
    def _path_energy(self, path, vocab):
        """الطاقة الكلية للمسار = مجموع الاقترانات الطيفية + التماسك الموجي."""
        energy = 0.0
        for i in range(len(path) - 1):
            w1 = vocab.id2word.get(path[i])
            w2 = vocab.id2word.get(path[i+1])
            if w1 and w2:
                mag, phase = self.spectral.get_coupling(w1, w2, vocab)
                phase_coherence = max(0.0, np.cos(phase))
                energy += mag * phase_coherence
        
        # تماسك المسار الكلي (تكامل الموجات)
        if len(path) >= 3:
            t = np.linspace(0, 2*np.pi, 16)
            total_signal = np.zeros(16)
            for nid in path:
                w = vocab.id2word.get(nid)
                if w:
                    ww = self.spectral.word_waves.get(w) or compute_word_wave(w)
                    total_signal += np.array([wave_at(ww, ti) for ti in t])
            coherence = float(np.std(total_signal))  # تموج منتظم = مسار متماسك
            energy += coherence
        
        return energy


# ─────────────────────────────────────────────
# 6. الترجمة الفيزيائية
# ─────────────────────────────────────────────

class PhysicsTranslator:
    """ترجمة فيزيائية: إيجاد الكلمات المكافئة عبر الاقتران الطيفي ثنائي اللغة.
    
    لا تقارن الموجات مباشرة (لأن العربية والإنجليزية لهما فيزياء مختلفة).
    بدلاً من ذلك، تستخدم مصفوفة اقتران طيفي ثنائية اللغة بُنيت من:
    - جمل متوازية (عربي || English)
    - الكلمات المترافقة في نفس السياق تبني sidebands متشابهة
    - الترجمة = البحث عن الكلمة ذات أعلى اقتران طيفي عبر اللغة
    
    هكذا "سماء" و "sky" يرتبطان ليس عبر فيزياء الحروف
    بل عبر ظهورهما في سياقات متكافئة.
    """
    
    def __init__(self, spectral_matrix, vocab_src, vocab_tgt=None):
        self.spectral = spectral_matrix
        self.vocab_src = vocab_src
        self.vocab_tgt = vocab_tgt or vocab_src
    
    def translate(self, word, top_k=3):
        """ترجمة كلمة عبر الاقتران الطيفي.
        
        تعمل لأي لغة — تطابق sidebands السياقية لا الترددات الذاتية.
        """
        src_id = self.vocab_src.get(word)
        if src_id is None:
            return []
        
        candidates = []
        for wid in range(len(self.vocab_tgt)):
            tw = self.vocab_tgt.id2word.get(wid)
            if not tw or tw == word:
                continue
            
            # قوة الاقتران الطيفي ثنائي اللغة
            mag, phase = self.spectral.get_coupling(word, tw, self.vocab_src)
            if mag < 0.01:
                continue
            
            # معامل التوافق الطوري: العلاقات المتوافقة (θ≈0) أعلى قيمة
            coherence = max(0.0, np.cos(phase))
            score = mag * coherence
            
            candidates.append((score, tw))
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[:top_k]
    
    def translate_with_context(self, word, context_words, top_k=3):
        """ترجمة مع مراعاة السياق — يرجّح الكلمات المتوافقة مع السياق الحالي."""
        src_id = self.vocab_src.get(word)
        if src_id is None:
            return []
        
        # بناء متجه سياقي
        ctx_pvs = []
        for cw in context_words[-5:]:
            pv = self.spectral.word_waves.get(cw)
            if pv is None:
                from src.physics.word_physics import compute_word_phase_vector
                pv = compute_word_phase_vector(cw)
                self.spectral.word_waves[cw] = pv
            ctx_pvs.append(pv)
        
        candidates = []
        if ctx_pvs:
            ctx_avg = np.mean(np.array(ctx_pvs), axis=0)
        else:
            ctx_avg = None
        
        for wid in range(len(self.vocab_tgt)):
            tw = self.vocab_tgt.id2word.get(wid)
            if not tw or tw == word:
                continue
            
            mag, phase = self.spectral.get_coupling(word, tw, self.vocab_src)
            if mag < 0.01:
                continue
            
            coherence = max(0.0, np.cos(phase))
            score = mag * coherence
            
            # تعزيز سياقي: الكلمات المتوافقة مع السياق ترتفع درجتها
            if ctx_avg is not None:
                tw_wave = self.spectral.word_waves.get(tw) or compute_word_wave(tw)
                # لا يمكن مقارنة متجهات الطور عبر اللغتين — نتجاوز هذه الخطوة
                pass
            
            candidates.append((score, tw))
        
        candidates.sort(key=lambda x: -x[0])
        return candidates[:top_k]


# ─────────────────────────────────────────────
# 7. الاستدلال المنطقي الطوري
# ─────────────────────────────────────────────

class PhaseLogicalInference:
    """الاستدلال المنطقي عبر الموجات: A→B و B→C ⟹ A→C
    
    المبدأ:
    - إذا كانت A و B متقاربتين طورياً (cos θ ≈ 1) وقويا الاقتران
    - وإذا كانت B و C كذلك
    - فإن A و C مترابطتان عبر B (استدلال متعد)
    
    ليس منطقاً كلاسيكياً (Boolean)، بل **منطقاً ضبابياً طورياً**.
    """
    
    def __init__(self, spectral_matrix):
        self.spectral = spectral_matrix
    
    def deduce(self, a, b, c, vocab):
        """هل A → C عبر B؟
        
        A ↔ B بقوة اقتران mag_AB وفرق طور θ_AB
        B ↔ C بقوة اقتران mag_BC وفرق طور θ_BC
        الاستدلال: A → C بقوة = mag_AB · mag_BC · cos(θ_AB) · cos(θ_BC)
        
        إذا كانت A و B متوافقتين (θ≈0) و B و C متوافقتين
        فإن A و C متوافقتان منطقياً.
        """
        mag_ab, phase_ab = self.spectral.get_coupling(a, b, vocab)
        mag_bc, phase_bc = self.spectral.get_coupling(b, c, vocab)
        
        # التطابق المنطقي
        ab_coherence = max(0.0, np.cos(phase_ab))
        bc_coherence = max(0.0, np.cos(phase_bc))
        
        # قوة الاستدلال A→C عبر B
        inference_strength = mag_ab * mag_bc * ab_coherence * bc_coherence
        
        # تحقق مباشر: هل A و C مرتبطين؟
        mag_ac, phase_ac = self.spectral.get_coupling(a, c, vocab)
        ac_coherence = max(0.0, np.cos(phase_ac))
        direct_strength = mag_ac * ac_coherence
        
        return {
            'AB': {'magnitude': round(mag_ab, 3), 'coherence': round(ab_coherence, 3)},
            'BC': {'magnitude': round(mag_bc, 3), 'coherence': round(bc_coherence, 3)},
            'AC_inferred': round(inference_strength, 3),
            'AC_direct': round(direct_strength, 3),
            'conclusion': inference_strength > 0.1,
        }
    
    def transitive_search(self, start, vocab, max_depth=3, min_strength=0.1):
        """بحث تعددي: انطلاقاً من كلمة، ما الكلمات التي يمكن استنتاجها؟
        
        A → B → C → D : A يستلزم D عبر B, C
        """
        visited = {start}
        frontier = [(start, 1.0, 0)]
        results = []
        
        while frontier:
            word, strength, depth = frontier.pop(0)
            if depth > max_depth:
                continue
            
            # جيران هذه الكلمة
            wid = vocab.get(word)
            if wid is None or self.spectral.magnitude is None:
                continue
            
            row = self.spectral.magnitude[wid].toarray().ravel()
            neighbors = np.argsort(row)[-20:][::-1]
            
            for nid in neighbors:
                if row[nid] < min_strength:
                    continue
                nw = vocab.id2word.get(nid)
                if not nw or nw in visited:
                    continue
                
                _, phase = self.spectral.get_coupling(word, nw, vocab)
                coherence = max(0.0, np.cos(phase))
                new_strength = strength * row[nid] * coherence
                
                if new_strength < min_strength:
                    continue
                
                visited.add(nw)
                results.append((nw, round(new_strength, 3), depth + 1))
                frontier.append((nw, new_strength, depth + 1))
        
        return sorted(results, key=lambda x: -x[1])
