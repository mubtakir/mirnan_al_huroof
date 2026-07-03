import re
import os
import logging
import numpy as np
from collections import defaultdict
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM

logger = logging.getLogger(__name__)


def _tokenize(text):
    return re.findall(r'[^\s\.,:;!\?(){}\[\]"\'،؛؟]+', text)


def _cached_pv(word, cache):
    if word not in cache:
        cache[word] = compute_extended_phase_vector(word)
    return cache[word]


def build_contexts_map(corpus_texts, vocab, half_window=7):
    contexts_map = defaultdict(list)
    pv_cache = {}
    for text in corpus_texts:
        for line in text.split('\n'):
            tokens = _tokenize(line)
            ids = [vocab.get(t) for t in tokens if vocab.get(t) is not None]
            if len(ids) < 2:
                continue
            for i, wid in enumerate(ids):
                left = max(0, i - half_window)
                right = min(len(ids), i + half_window + 1)
                ctx_ids = ids[left:i] + ids[i+1:right]
                if len(ctx_ids) < 2:
                    continue
                ctx_words = [vocab.id2word[cid] for cid in ctx_ids]
                ctx_pvs = [_cached_pv(w, pv_cache) for w in ctx_words]
                ctx_vec = np.mean(ctx_pvs, axis=0)
                nrm = np.linalg.norm(ctx_vec)
                if nrm > 1e-10:
                    ctx_vec = ctx_vec / nrm
                contexts_map[wid].append(ctx_vec)
    return dict(contexts_map)


class GlobalSpectralMemory:
    """الذاكرة الطيفية العالمية — تحل مشكلة الرنين المحلي.

    تبني توقيعاً طيفياً لكل كلمة من كل سياقاتها في الكوربوس،
    باستخدام Coherence-Weighted Summation بدلاً من المتوسط البسيط.
    """

    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.gss_cache = {}
        self.ground_state = None
        self.is_loaded = False

    def build_global_signatures(self, vocab, contexts_map):
        self.gss_cache = {}
        all_vecs = []
        for vec_list in contexts_map.values():
            all_vecs.extend(vec_list)
        if all_vecs:
            self.ground_state = np.mean(all_vecs, axis=0)
            gs_nrm = np.linalg.norm(self.ground_state)
            if gs_nrm > 1e-10:
                self.ground_state = self.ground_state / gs_nrm
        else:
            self.ground_state = np.zeros(TOTAL_DIM)
        for word_id, vectors in contexts_map.items():
            if not vectors:
                continue
            vectors = np.array(vectors)
            ref_vec = np.mean(vectors, axis=0)
            ref_nrm = np.linalg.norm(ref_vec)
            if ref_nrm < 1e-10:
                continue
            ref_vec = ref_vec / ref_nrm
            cosines = np.dot(vectors, ref_vec)
            weights = np.maximum(cosines, 0.0)
            w_sum = np.sum(weights)
            if w_sum < 1e-10:
                continue
            weighted_sum = np.sum(vectors * weights[:, np.newaxis], axis=0)
            sig_nrm = np.linalg.norm(weighted_sum)
            if sig_nrm < 1e-10:
                continue
            signature = weighted_sum / sig_nrm
            pure = signature - self.ground_state
            pure_nrm = np.linalg.norm(pure)
            if pure_nrm > 1e-10:
                pure = pure / pure_nrm
            self.gss_cache[word_id] = pure
        self._save_to_disk()
        self.is_loaded = True

    def get_global_resonance(self, candidate_id, context_vector):
        if not self.is_loaded or candidate_id not in self.gss_cache:
            return 0.0
        signature = self.gss_cache[candidate_id]
        score = np.dot(context_vector, signature)
        return float(np.tanh(score * 2.0))

    def save(self, path):
        """حفظ الذاكرة الطيفية إلى ملف.

        Args:
            path: مسار الملف (بدون امتداد)، e.g. 'data/model_gss'
        """
        if not self.gss_cache:
            return
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        # Fast save (whole dict in one binary file)
        np.save(path + '_dict.npy', self.gss_cache, allow_pickle=True)
        
        # Slow npz fallback saved ONLY for small caches (< 2000 items) to prevent hangs
        if len(self.gss_cache) < 2000:
            np.savez_compressed(
                path + '.npz',
                **{str(k): v for k, v in self.gss_cache.items()}
            )
        else:
            # Save a dummy npz just to satisfy os.path.exists check
            np.savez(path + '.npz', dummy=np.array([1]))
            
        if self.ground_state is not None:
            np.save(path + '_ground.npy', self.ground_state)

    def load(self, path):
        """تحميل الذاكرة الطيفية من ملف.

        Args:
            path: مسار الملف (بدون امتداد)، e.g. 'data/model_gss'
        """
        gss_file_fast = path + '_dict.npy'
        gss_file_slow = path + '.npz'
        ground_file = path + '_ground.npy'
        
        self.gss_cache = {}
        
        # 1. Try fast load
        if os.path.exists(gss_file_fast):
            try:
                self.gss_cache = np.load(gss_file_fast, allow_pickle=True).item()
            except Exception as e:
                logger.warning(f"Failed to load fast GSS dict {gss_file_fast}: {e}. Falling back to slow load.")
        
        # 2. Fallback to slow load if fast load failed/not present
        if not self.gss_cache and os.path.exists(gss_file_slow):
            try:
                import zipfile
                import io
                import time
                
                # Check if it is a dummy npz
                is_dummy = False
                with zipfile.ZipFile(gss_file_slow) as archive:
                    names = archive.namelist()
                    if 'dummy.npy' in names:
                        is_dummy = True
                
                if not is_dummy:
                    logger.info(f"Loading/Migrating slow GSS npz ({len(names)} files) via optimized zip reader... This may take a minute.")
                    t_start = time.time()
                    temp_cache = {}
                    with zipfile.ZipFile(gss_file_slow) as archive:
                        for name in names:
                            if name.endswith('.npy'):
                                key = name[:-4]
                                file_bytes = archive.read(name)
                                arr = np.load(io.BytesIO(file_bytes))
                                temp_cache[int(key)] = arr
                    self.gss_cache = temp_cache
                    
                    # Save fast file immediately so next load is instantaneous
                    np.save(gss_file_fast, self.gss_cache, allow_pickle=True)
                    elapsed = time.time() - t_start
                    logger.info(f"GSS cache successfully migrated to fast format in {elapsed:.2f}s.")
            except Exception as e:
                logger.error(f"Error loading slow GSS npz: {e}")
                
        if os.path.exists(ground_file):
            self.ground_state = np.load(ground_file)
            
        self.is_loaded = bool(self.gss_cache)
        return self.is_loaded

    def _save_to_disk(self):
        self.save(os.path.join(self.data_dir, 'spectral_gss').replace('\\', '/'))

    def load_from_disk(self):
        return self.load(os.path.join(self.data_dir, 'spectral_gss').replace('\\', '/'))

