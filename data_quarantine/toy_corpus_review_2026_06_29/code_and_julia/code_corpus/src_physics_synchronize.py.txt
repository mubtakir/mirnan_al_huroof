"""synchronize — تدريب المزامنة لمحرك مرنان.

يبني المعجم (vocab) ومصفوفة الاقتران K والمجال النحوي (syntax)
من نصوص التدريب. لا يستخدم backpropagation — فقط مزامنة Hebbian.

تدعم save_k_mmap / load_k_mmap للتخزين المعيّن بالذاكرة (Memory-Mapped).
"""
import re
import numpy as np
from scipy import sparse
from src.physics.particles import is_particle
from src.physics.word_physics import compute_word_phase_vector, _normalize_letters, phase_similarity
from src.physics.grammar_field import SyntaxField
from src.physics.carrier_engine import CarrierWaveEngine


def _tokenize(text):
    # Match any sequence of characters that are not whitespace or common punctuation.
    # This allows Arabic text as well as emojis/symbols (like 🌸) to be treated as tokens.
    return re.findall(r'[^\s\.,:;!\?(){}\[\]"\'،؛؟]+', text)


class Vocabulary:
    def __init__(self):
        self.word2id = {}
        self.id2word = {}
        self.next_id = 0

    def add(self, word):
        w = _normalize_letters(word)
        if w not in self.word2id:
            self.word2id[w] = self.next_id
            self.id2word[self.next_id] = w
            self.next_id += 1
        return self.word2id[w]

    def get(self, word, default=None):
        return self.word2id.get(_normalize_letters(word), default)

    def add_word(self, word):
        """إضافة كلمة جديدة بعد بناء المعجم (ديناميكي).

        الكلمة تضاف بدون تعديل المصفوفة K.
        تُرجع id الكلمة (جديد أو موجود).
        """
        w = _normalize_letters(word)
        if w in self.word2id:
            return self.word2id[w]
        wid = self.next_id
        self.word2id[w] = wid
        self.id2word[wid] = w
        self.next_id += 1
        return wid

    def load_supplemental(self, json_path):
        """تحميل كلمات إضافية من ملف JSON.

        Args:
            json_path: مسار ملف JSON يحتوي على قائمة كلمات بالمفتاح 'words'

        Returns:
            int: عدد الكلمات المضافة فعلياً
        """
        import json, os, logging
        logger = logging.getLogger(__name__)
        if not os.path.exists(json_path):
            logger.warning(f"  Supplemental vocab not found: {json_path}")
            return 0
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = data.get('words', [])
        added = 0
        for w in words:
            prev = self.word2id.get(_normalize_letters(w))
            self.add(w)
            if self.word2id.get(_normalize_letters(w)) != prev or prev is None:
                added += 1
        logger.info(f"  ✓ Loaded {added} supplemental words from {json_path}")
        return added

    def __len__(self):
        return self.next_id


def synchronize(corpus_texts, window=5, alpha=0.25, mode='sem', vocab=None, domain=None,
                particle_weight: float = 0.15):
    if mode == 'syn':
        window = 2
    new_vocab = vocab is None
    if new_vocab:
        vocab = Vocabulary()
    initial_size = max(vocab.next_id or 1, 1)
    cooc = sparse.lil_matrix((initial_size, initial_size))
    phase_sim = sparse.lil_matrix((initial_size, initial_size))
    word_freq = np.zeros(initial_size)
    syntax = SyntaxField()
    _pv_cache = {}

    def _cached_pv(w):
        if w not in _pv_cache:
            _pv_cache[w] = compute_word_phase_vector(w)
        return _pv_cache[w]

    for text in corpus_texts:
        for line in text.split('\n'):
            # تنسيق TAB في كوربس الحوار
            if mode == 'dial' and '\t' in line:
                parts = line.split('\t', 1)
                if len(parts) == 2:
                    q_tokens = _tokenize(parts[0])
                    a_tokens = _tokenize(parts[1])
                    tokens = q_tokens + a_tokens
                else:
                    tokens = _tokenize(line)
            else:
                tokens = _tokenize(line)
            if len(tokens) < 2:
                continue
            # فاصل السياق: نهاية السؤال ↔ بداية الجواب
            context_break = len(q_tokens) if (mode == 'dial' and '\t' in line and len(parts) == 2) else None
            ids = [vocab.add(t) for t in tokens]
            V = len(vocab)
            if V > cooc.shape[0]:
                cooc.resize((V, V))
                phase_sim.resize((V, V))
                word_freq = np.pad(word_freq, (0, V - len(word_freq)), 'constant')
            for i, idi in enumerate(ids):
                word_freq[idi] += 1.0
                for j in range(i+1, min(i+window+1, len(ids))):
                    idj = ids[j]
                    # تعزيز خاص لروابط السؤال↔الجواب في كوربس الحوار
                    dial_boost = 2.0
                    if context_break is not None:
                        if i < context_break and j >= context_break:
                            dial_boost = 3.0
                        elif i == context_break - 1 and j == context_break:
                            dial_boost = 5.0
                    cooc[idi, idj] = cooc[idi, idj] + dial_boost
                    cooc[idj, idi] = cooc[idj, idi] + dial_boost
                    pvi = _cached_pv(vocab.id2word[idi])
                    pvj = _cached_pv(vocab.id2word[idj])
                    if np.linalg.norm(pvi) < 1e-10 or np.linalg.norm(pvj) < 1e-10:
                        continue
                    sim = phase_similarity(pvi, pvj)
                    phase_sim[idi, idj] = phase_sim[idi, idj] + sim * dial_boost
                    phase_sim[idj, idi] = phase_sim[idj, idi] + sim * dial_boost
            syntax.observe([vocab.id2word[id] for id in ids])

    V = len(vocab)
    cooc = cooc.tocsr()
    phase_sim = phase_sim.tocsr()
    cooc_coo = cooc.tocoo()
    data = np.zeros(cooc_coo.nnz)
    total_freq = word_freq.sum()
    wi = vocab.id2word
    for idx in range(cooc_coo.nnz):
        i, j = cooc_coo.row[idx], cooc_coo.col[idx]
        c = cooc_coo.data[idx]
        ps = phase_sim[i, j] / c
        expected = word_freq[i] * word_freq[j] / total_freq
        pmi = np.log((c + 1) / (expected + 1))
        if ps > 0.15:
            val = c * ps * (max(pmi, 0.0) ** alpha)
        elif pmi < -0.5:
            val = c * abs(ps) * pmi * 0.2
        else:
            val = 0.0
        # عقوبة الحروف النحوية — تمنع هيمنتها على K
        if val != 0.0:
            pi = is_particle(wi.get(i, ''))
            pj = is_particle(wi.get(j, ''))
            if pi and pj:
                val *= (particle_weight ** 2)
            elif pi or pj:
                val *= particle_weight
        data[idx] = val
    K = sparse.coo_matrix((data, (cooc_coo.row, cooc_coo.col)), shape=(V, V))
    K = K.tocsr()
    syntax.finalize(vocab)

    return vocab, K, syntax


def synchronize_with_carrier(corpus_texts, window=5, alpha=0.25, mode='sem', vocab=None):
    """تدريب المزامنة + بناء أطياف الموجة الحاملة."""
    vocab, K, syntax = synchronize(corpus_texts, window=window, alpha=alpha, mode=mode, vocab=vocab)
    carrier = CarrierWaveEngine()
    carrier.build_from_corpus(corpus_texts, vocab)
    return vocab, K, syntax, carrier


def assimilate_text(corpus_texts, vocab, K_old, window=5, alpha_blend=0.1):
    """
    التعلم التزايدي لمصفوفة K:
    يقوم بجمع الأوزان الجديدة مع القديمة بشكل متناثر لتفادي إعادة الحساب من الصفر.
    يعيد المصفوفة K_new المحدثة.
    """
    # نبني مصفوفة K للنص الجديد فقط
    _, K_new_raw, _ = synchronize(corpus_texts, window=window, alpha=0.25, mode='sem' if window>2 else 'syn', vocab=vocab)
    
    # يجب مطابقة الأبعاد إذا كان المعجم قد زاد
    old_size = K_old.shape[0]
    new_size = vocab.next_id if vocab.next_id > 0 else 1
    
    if new_size > old_size:
        K_old_resized = sparse.lil_matrix((new_size, new_size))
        # نسخ القديم
        K_old_coo = K_old.tocoo()
        for i, j, v in zip(K_old_coo.row, K_old_coo.col, K_old_coo.data):
            K_old_resized[i, j] = v
        K_old = K_old_resized.tocsr()
    
    if K_new_raw.shape[0] < new_size:
        K_new_raw.resize((new_size, new_size))
        
    # دمج التحديثات بنسبة الزخم (Momentum)
    # القديم يحتفظ بوزنه (1 - alpha_blend) والجديد يضاف
    K_updated = K_old * (1.0 - alpha_blend) + K_new_raw * alpha_blend
    return K_updated.tocsr()


def save_k_mmap(K, base_path):
    """حفظ مصفوفة K المتناثرة كملفات Memory-Mapped لتقليل استهلاك الـ RAM"""
    import os
    os.makedirs(os.path.dirname(os.path.abspath(base_path)), exist_ok=True)
    
    with open(f"{base_path}_shape.txt", "w", encoding="utf-8") as f:
        f.write(f"{K.shape[0]},{K.shape[1]}")
        
    data = np.memmap(f"{base_path}_data.dat", dtype=np.float64, mode='w+', shape=K.data.shape)
    data[:] = K.data.astype(np.float64)[:]
    data.flush()
    
    indices = np.memmap(f"{base_path}_indices.dat", dtype=np.int32, mode='w+', shape=K.indices.shape)
    indices[:] = K.indices.astype(np.int32)[:]
    indices.flush()
    
    indptr = np.memmap(f"{base_path}_indptr.dat", dtype=np.int32, mode='w+', shape=K.indptr.shape)
    indptr[:] = K.indptr.astype(np.int32)[:]
    indptr.flush()


def load_k_mmap(base_path):
    """تحميل مصفوفة K المتناثرة من ملفات Memory-Mapped دون حجز الـ RAM"""
    import os
    if not os.path.exists(f"{base_path}_shape.txt"):
        return None
        
    with open(f"{base_path}_shape.txt", "r", encoding="utf-8") as f:
        shape_str = f.read().strip()
        shape = tuple(map(int, shape_str.split(",")))
        
    data = np.memmap(f"{base_path}_data.dat", dtype=np.float64, mode='r')
    indices = np.memmap(f"{base_path}_indices.dat", dtype=np.int32, mode='r')
    indptr = np.memmap(f"{base_path}_indptr.dat", dtype=np.int32, mode='r')
    
    return sparse.csr_matrix((data, indices, indptr), shape=shape)
