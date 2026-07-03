"""RotatingAnchor — التعلم الذاتي للتمثيل الدلالي.

الابتكار الخامس: بدون أي تصنيف بشري، يكتشف النظام العناقيد الدلالية
في فضاء الطور ويبني "مراسي دلالية" (Semantic Anchors).

المبدأ — "المرساة الدوارة":
1. خذ كل كلمات المعجم مع متجهاتها الطورية (64D)
2. اختزل الأبعاد: PCA (64D → 2D)
3. جمّع: DBSCAN يكتشف العناقيد تلقائياً
4. لكل عنقود، احسب متوسط المتجه = "مرساة دلالية"
5. الكلمات من نفس العنقود متجاورة في فضاء الطور → معنى

لا تدريب. لا باك بروب. فقط PCA + عناقيد.
"""
import numpy as np
import logging
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


class RotatingAnchor:
    """محرك المراسي الدلالية.
    
    يبني مراسي من معجم موجود، ثم يقيس انتماء أي كلمة إلى كل مرساة.
    """
    
    def __init__(self, n_components=3, eps=0.5, min_samples=3):
        self.n_components = n_components
        self.eps = eps
        self.min_samples = min_samples
        self.anchors = []         # list of (anchor_pv, cluster_id, label_words)
        self._pca = None
        self._labels = None
        self._word_to_anchor = {}  # word → list of (anchor_idx, distance)
        self._fitted = False
    
    def fit(self, vocab, all_pv_matrix):
        """بناء المراسي من معجم + مصفوفة متجهات.
        
        Args:
            vocab: Vocabulary object with word2id, id2word
            all_pv_matrix: np.ndarray (V, TOTAL_DIM)
        """
        V = len(vocab)
        if V < 10:
            logger.warning("Vocabulary too small for clustering (< 10 words)")
            self._fitted = False
            return
        
        # 1. PCA — اختزال الأبعاد
        pv_sample = all_pv_matrix[:min(V, 5000)]  # عينة 5000 كلمة للسرعة
        self._pca = PCA(n_components=self.n_components)
        reduced = self._pca.fit_transform(pv_sample)
        
        # 2. DBSCAN — تجميع
        clusterer = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self._labels = clusterer.fit_predict(reduced)
        
        # 3. بناء المراسي
        cluster_map = defaultdict(list)
        for i, label in enumerate(self._labels):
            cluster_map[label].append(i)
        
        self.anchors = []
        for label, indices in cluster_map.items():
            if label == -1:
                continue  # ضوضاء — لا نبني مرساة
            words_in_cluster = [vocab.id2word[idx] for idx in indices 
                              if idx in vocab.id2word]
            # متوسط المتجهات الطورية للمراسي
            cluster_pvs = np.array([pv_sample[idx] for idx in indices])
            anchor_pv = np.mean(cluster_pvs, axis=0)
            nrm = np.linalg.norm(anchor_pv)
            if nrm > 1e-10:
                anchor_pv = anchor_pv / nrm
            
            self.anchors.append({
                'id': label,
                'pv': anchor_pv,
                'words': words_in_cluster[:10],  # أول 10 كلمات
                'size': len(words_in_cluster),
            })
            
            for word in words_in_cluster:
                if word not in self._word_to_anchor:
                    self._word_to_anchor[word] = []
                word_pv = all_pv_matrix[vocab.word2id[word]]
                dist = float(np.mean(np.cos(word_pv[:len(anchor_pv)] - anchor_pv)))
                self._word_to_anchor[word].append((label, dist))
        
        logger.info(f"RotatingAnchor: {len(self.anchors)} clusters from {V} words")
        self._fitted = True
    
    def anchor_affinity(self, word, word_pv):
        """انتماء كلمة إلى المراسي الدلالية.
        
        Returns 0.0–1.0: كم تنتمي هذه الكلمة إلى أي عنقود دلالي.
        """
        if not self._fitted or not self.anchors:
            return 0.0
        
        # هل هذه الكلمة معروفة في العناقيد؟
        if word in self._word_to_anchor:
            anchors = self._word_to_anchor[word]
            return max(d for _, d in anchors)
        
        # كلمة جديدة — احسب المسافة إلى أقرب مرساة
        if word_pv is None:
            return 0.0
        
        best = 0.0
        for anchor in self.anchors:
            anchor_pv = anchor['pv']
            common_dims = min(len(word_pv), len(anchor_pv))
            sim = float(np.mean(np.cos(word_pv[:common_dims] - anchor_pv[:common_dims])))
            if sim > best:
                best = sim
        return max(0.0, best)
    
    def semantic_density(self, word, context_words, context_pvs):
        """الكثافة الدلالية: هل الكلمة في نفس العنقود الدلالي للسياق؟
        
        إذا كانت الكلمة والكلمات السابقة تنتمي لنفس العنقود،
        فهي منسجمة دلالياً مع السياق.
        """
        if not self._fitted or not context_pvs:
            return 0.0
        
        # احسب انتماء كل كلمة سياق إلى المراسي
        context_anchors = defaultdict(int)
        n = 0
        for cw, cpv in zip(context_words[-5:], context_pvs[-5:]):
            if cw is None or cpv is None:
                continue
            best_anchor = None
            best_dist = 0.0
            for anchor in self.anchors:
                anchor_pv = anchor['pv']
                common = min(len(cpv), len(anchor_pv))
                sim = float(np.mean(np.cos(cpv[:common] - anchor_pv[:common])))
                if sim > best_dist:
                    best_dist = sim
                    best_anchor = anchor['id']
            if best_anchor is not None and best_dist > 0.6:
                context_anchors[best_anchor] += 1
                n += 1
        
        if n == 0:
            return 0.0
        
        # هل الكلمة تنتمي لأحد المراسي الرائجة في السياق؟
        word_anchors = self._word_to_anchor.get(word, [])
        for aid, _ in word_anchors:
            if aid in context_anchors:
                return min(1.0, context_anchors[aid] / n + 0.3)
        
        return 0.0
    
    def get_anchor_report(self):
        if not self._fitted or not self.anchors:
            return {"clusters": 0, "anchors": []}
        return {
            "clusters": len(self.anchors),
            "anchors": [{
                'id': a['id'],
                'size': a['size'],
                'words': a['words'],
            } for a in sorted(self.anchors, key=lambda x: -x['size'])[:10]],
        }
