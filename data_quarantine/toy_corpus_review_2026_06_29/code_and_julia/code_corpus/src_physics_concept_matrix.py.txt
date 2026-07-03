import numpy as np
from scipy import sparse

class ConceptMatrix:
    def __init__(self, gss_cache):
        self.gss_cache = gss_cache
        self.K_conc = None
        
    def build(self, vocab_size, top_k=100):
        cooc = sparse.lil_matrix((vocab_size, vocab_size))
        
        items = list(self.gss_cache.items())
        valid_items = [(i, gss) for i, gss in items if i < vocab_size]
        
        for i, gss_i in valid_items:
            similarities = []
            for j, gss_j in valid_items:
                if i == j: continue
                sim = float(np.dot(gss_i, gss_j))
                if sim > 0.05: # عتبة أساسية خفيفة للتنظيف
                    similarities.append((j, sim))
            
            # الرنين الانتقائي: الاحتفاظ بأقوى top_k روابط مفاهيمية فقط
            similarities.sort(key=lambda x: x[1], reverse=True)
            for j, sim in similarities[:top_k]:
                cooc[i, j] = sim
                cooc[j, i] = sim  # فرض التناظر
                
                    
        self.K_conc = cooc.tocsr()
        return self.K_conc

def build_multi_k(corpus_texts, vocab=None):
    from src.physics.synchronize import synchronize
    from src.physics.spectral_memory import build_contexts_map, GlobalSpectralMemory
    
    # 1. بناء المفردات مرة واحدة فقط عبر أول synchronize
    vocab, K_sem, syntax = synchronize(corpus_texts, mode='sem', vocab=vocab)
    
    # 2. K_syn (النافذة القصيرة) — يعيد استخدام vocab نفسه
    _, K_syn, _ = synchronize(corpus_texts, mode='syn', vocab=vocab)
    
    # 3. K_conc (المفاهيم من التوقيعات الطيفية)
    spectral_memory = GlobalSpectralMemory()
    c_map = build_contexts_map(corpus_texts, vocab, half_window=7)
    spectral_memory.build_global_signatures(vocab, c_map)
    
    cm = ConceptMatrix(spectral_memory.gss_cache)
    K_conc = cm.build(len(vocab))
    
    return K_syn, K_sem, K_conc, vocab, syntax
