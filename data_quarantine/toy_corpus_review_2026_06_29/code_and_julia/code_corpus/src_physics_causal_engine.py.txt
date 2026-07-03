import re
import numpy as np
from scipy import sparse


def _tokenize(text):
    return re.findall(r'[^\s\.,:;!\?(){}\[\]"\'،؛؟]+', text)


class CausalPhaseEngine:
    """غرفة الرنين المنطقي — تستنتج العلاقات السببية من ترتيب الكلمات في الجمل.

    المبدأ: إذا كانت A تظهر قبل B باستمرار في الجمل، وفرق الطور بينهما ~90°،
    فـ A ← B علاقة سببية (A تسبب B).
    """

    def __init__(self):
        self.causal_K = None
        self.is_built = False

    def save(self, path):
        if self.causal_K is None:
            return
        sparse.save_npz(path, self.causal_K)
        self._save_path = path

    def load(self, path, V):
        try:
            self.causal_K = sparse.load_npz(path)
            self.is_built = True
            self._save_path = path
            return True
        except Exception:
            return False

    def build_from_corpus(self, corpus_texts, vocab, window=5):
        """يبني مصفوفة سببية موجهة من الترتيب الخطي للكلمات."""
        V = len(vocab)
        before = sparse.lil_matrix((V, V))
        after = sparse.lil_matrix((V, V))

        for text in corpus_texts:
            for line in text.split('\n'):
                tokens = _tokenize(line)
                ids = [vocab.get(t) for t in tokens if vocab.get(t) is not None]
                for i in range(len(ids)):
                    for j in range(i + 1, min(i + window + 1, len(ids))):
                        before[ids[i], ids[j]] += 1.0
                        after[ids[j], ids[i]] += 1.0

        before = before.tocsr()
        after = after.tocsr()

        # O(nnz) بدلاً من O(V²) — نمر فقط على المداخل غير الصفرية
        before_coo = before.tocoo()
        after_coo = after.tocoo()

        data = []
        rows = []
        cols = []

        # مداخل before
        for i, j, b in zip(before_coo.row, before_coo.col, before_coo.data):
            if i == j:
                continue
            a = after[i, j]
            total = b + a
            if total < 1:
                continue
            strength = (b - a) / total
            if abs(strength) > 0.15:
                data.append(strength)
                rows.append(i)
                cols.append(j)

        # مداخل after التي ليست في before (باستخدام مجموعة)
        b_set = set(zip(before_coo.row, before_coo.col))
        for i, j, a in zip(after_coo.row, after_coo.col, after_coo.data):
            if i == j:
                continue
            if (i, j) in b_set:
                continue
            total = a
            if total < 1:
                continue
            strength = -1.0  # b=0, a>0 → سالب دائماً
            if abs(strength) > 0.15:
                data.append(strength)
                rows.append(i)
                cols.append(j)

        self.causal_K = sparse.csr_matrix((data, (rows, cols)), shape=(V, V))
        self.is_built = True
        return self.causal_K

    def causal_strength(self, cause_id, effect_id):
        """درجة السببية من cause ← effect (0..1)."""
        if not self.is_built:
            return 0.0
        if cause_id >= self.causal_K.shape[0] or effect_id >= self.causal_K.shape[1]:
            return 0.0
        val = self.causal_K[cause_id, effect_id]
        return float(max(val, 0.0))

    def transitive_score(self, word_ids):
        """قياس اتساق العلاقات المتعدية في سلسلة من الكلمات.

        إذا A←B و B←C، فنتوقع A←C. تعيد درجة الاتساق (0..1).
        """
        if not self.is_built or len(word_ids) < 3:
            return 0.0
        score = 0.0
        n = 0
        for i in range(len(word_ids)):
            for j in range(i + 1, len(word_ids)):
                for k in range(j + 1, len(word_ids)):
                    ij = self.causal_strength(word_ids[i], word_ids[j])
                    jk = self.causal_strength(word_ids[j], word_ids[k])
                    ik = self.causal_strength(word_ids[i], word_ids[k])
                    if ij > 0.3 and jk > 0.3:
                        score += min(ik, 1.0)
                        n += 1
        return score / max(n, 1)

    def score_candidate(self, word_id, context_ids):
        """درجة اتساق المرشح سببياً مع السياق.

        إذا كانت كلمات السياق تسبب المرشح باستمرار ← موجب.
        إذا كان المرشح يتعارض مع اتجاه سببي ← سالب.
        """
        if not self.is_built or not context_ids or word_id is None:
            return 0.0
        forward = 0.0
        backward = 0.0
        n_fwd = 0
        n_bwd = 0
        for cid in context_ids[-6:]:
            if cid is None:
                continue
            fwd = self.causal_strength(cid, word_id)
            if fwd > 0.3:
                forward += fwd
                n_fwd += 1
            bwd = self.causal_strength(word_id, cid)
            if bwd > 0.3:
                backward += bwd
                n_bwd += 1
        if n_fwd == 0 and n_bwd == 0:
            return 0.0
        net = (forward - backward * 0.5) / max(n_fwd + n_bwd, 1)
        return float(np.tanh(net * 2.0))
