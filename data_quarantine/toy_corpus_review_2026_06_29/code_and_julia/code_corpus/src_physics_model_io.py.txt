#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
model_io — حفظ وتحميل النموذج كملفات Python ديناميكية قابلة للتوسع.

المبدأ: كل جزء من النموذج يُخزّن كملف Python مستقل:
  - model/vocab.py    : المعجم (مقروء، قابل للتعديل)
  - model/K_sem.py    : مصفوفة K الدلالية (مضغوطة base64+zlib)
  - model/K_syn.py    : مصفوفة K التركيبية
  - model/K_dial.py   : مصفوفة K الحوارية
  - model/syntax.py   : حقل القواعد النحوية

ملفات K تستخدم base64+zlib لأنها كبيرة وغير مقروءة كـ text.
ملف vocab.py مقروء بالكامل — يمكن تعديله يدوياً أو برمجياً.
"""
import os, sys, json, zlib, base64, ast
import numpy as np
from scipy import sparse
from typing import Optional

# ─── دوال مساعدة للترميز ───

def _encode_array(arr: np.ndarray) -> str:
    """ضغط مصفوفة numpy إلى base64+zlib."""
    compressed = zlib.compress(arr.tobytes())
    return base64.b64encode(compressed).decode('ascii')

def _decode_array(b64: str, dtype) -> np.ndarray:
    """فك ضغط base64+zlib إلى مصفوفة numpy."""
    raw = zlib.decompress(base64.b64decode(b64))
    return np.frombuffer(raw, dtype=dtype)

def _k_to_source(K: sparse.csr_matrix, var_name: str = "K_SEM") -> str:
    """توليد كود Python لمصفوفة K من CSR."""
    data_b64 = _encode_array(K.data.astype(np.float64))
    indices_b64 = _encode_array(K.indices.astype(np.int32))
    indptr_b64 = _encode_array(K.indptr.astype(np.int32))
    shape = K.shape
    nnz = K.nnz

    return f'''# -*- coding: utf-8 -*-
\"\"\"مصفوفة {var_name}: {shape[0]}×{shape[1]}, {nnz:,} مدخلاً غير صفري.
تم التوليد بواسطة model_io — يمكن إعادة ضغطها عبر save().
\"\"\"
import numpy as np
from scipy import sparse
import zlib, base64

SHAPE = {shape}
NNZ = {nnz}
DATA = {data_b64!r}
INDICES = {indices_b64!r}
INDPTR = {indptr_b64!r}

def load() -> sparse.csr_matrix:
    \"\"\"إعادة بناء المصفوفة من البيانات المضغوطة.\"\"\"
    data = np.frombuffer(zlib.decompress(base64.b64decode(DATA)), dtype=np.float64)
    indices = np.frombuffer(zlib.decompress(base64.b64decode(INDICES)), dtype=np.int32)
    indptr = np.frombuffer(zlib.decompress(base64.b64decode(INDPTR)), dtype=np.int32)
    return sparse.csr_matrix((data, indices, indptr), shape=SHAPE)

def save(K: sparse.csr_matrix, path: str):
    \"\"\"تحديث ملف Python بمصفوفة جديدة.\"\"\"
    from src.physics.model_io import save_k_py
    save_k_py(K, path, {var_name!r})
'''

def _vocab_to_source(word2id: dict, next_id: int) -> str:
    """توليد كود Python للمعجم."""
    # word2id قد يكون كبيراً — نضعه في سطور متعددة
    items = []
    for w, i in sorted(word2id.items(), key=lambda x: x[1]):
        items.append(f'    {w!r}: {i}')
    words_repr = '{\n' + ',\n'.join(items) + ',\n}'
    
    return f'''# -*- coding: utf-8 -*-
\"\"\"معجم ميران — {len(word2id):,} كلمة.
قابل للتوسع: أضف كلمات جديدة يدوياً أو عبر assimilate.py.
\"\"\"

WORD2ID = {words_repr}

# بناء ID2WORD عكسياً
ID2WORD = {{v: k for k, v in WORD2ID.items()}}

NEXT_ID = {next_id}

def add_word(word: str) -> int:
    \"\"\"إضافة كلمة جديدة ديناميكياً. تعيد ID الكلمة.\"\"\"
    global WORD2ID, ID2WORD, NEXT_ID
    if word in WORD2ID:
        return WORD2ID[word]
    wid = NEXT_ID
    WORD2ID[word] = wid
    ID2WORD[wid] = word
    NEXT_ID += 1
    return wid
'''

def _syntax_to_source(syntax) -> str:
    """توليد كود Python لـ SyntaxField."""
    def _dict_to_str(d, indent=4):
        if not d:
            return '{}'
        items = []
        for k, v in sorted(d.items(), key=lambda x: str(x[0])):
            items.append(f'{" " * indent}{k!r}: {v!r}')
        return '{\n' + ',\n'.join(items) + '\n}'
    
    return f'''# -*- coding: utf-8 -*-
\"\"\"SyntaxField — القواعد النحوية لميران.\"\"\"

BIGRAM_COS_MEAN = {_dict_to_str(syntax.bigram_cos_mean)}
BIGRAM_COUNT = {_dict_to_str(syntax.bigram_count)}
CAT_TRANSITION = {_dict_to_str(syntax.cat_transition)}
CAT_COUNT = {_dict_to_str(syntax.cat_count)}
'''

def _source_to_syntax() -> str:
    """توليد كود دالة load لـ syntax.py."""
    return '''
def load():
    """إعادة بناء SyntaxField من البيانات."""
    from src.physics.grammar_field import SyntaxField
    sf = SyntaxField()
    sf.bigram_cos_mean = {ast.literal_eval(k) if isinstance(k, str) and k.startswith("(") else k: v 
                          for k, v in BIGRAM_COS_MEAN.items()}
    sf.bigram_count = {eval(k) if isinstance(k, str) and k.startswith("(") else k: v 
                       for k, v in BIGRAM_COUNT.items()}
    sf.cat_transition = CAT_TRANSITION
    sf.cat_count = CAT_COUNT
    # إعادة بناء cat_trans_prob من cat_transition
    total_trans = sum(CAT_TRANSITION.values()) or 1
    sf.cat_trans_prob = {}
    for (c1, c2), cnt in CAT_TRANSITION.items():
        key = eval(c1) if isinstance(c1, str) and c1.startswith("(") else c1
        key2 = eval(c2) if isinstance(c2, str) and c2.startswith("(") else c2
        sf.cat_trans_prob[(key, key2)] = cnt / max(total_trans, 1)
    sf.finalized = True
    return sf
'''


# ─── واجهة الحفظ والتحميل الرئيسية ───

def save_model(model_dir: str, vocab, K_sem, syntax, K_syn=None, K_dial=None, K_conc=None):
    """حفظ النموذج — vocab.py (Python) + K matrices (.npz)."""
    from scipy import sparse
    os.makedirs(model_dir, exist_ok=True)

    # 1. المعجم (Python مقروء)
    with open(os.path.join(model_dir, 'vocab.py'), 'w', encoding='utf-8') as f:
        f.write(_vocab_to_source(vocab.word2id, vocab.next_id))

    # 2. مصفوفات K (.npz سريع)
    k_map = {
        'K_sem.npz': K_sem,
        'K_syn.npz': K_syn,
    }
    if K_dial is not None:
        k_map['K_dial.npz'] = K_dial
    if K_conc is not None:
        k_map['K_conc.npz'] = K_conc
    for fname, K in k_map.items():
        if K is not None:
            sparse.save_npz(os.path.join(model_dir, fname), K)

    # 3. SyntaxField (Python)
    with open(os.path.join(model_dir, 'syntax.py'), 'w', encoding='utf-8') as f:
        f.write(_syntax_to_source(syntax))
        f.write(_source_to_syntax())
    
    # 4. __init__.py — دالة load_model()
    _write_init(model_dir, k_map)


def _write_init(model_dir: str, k_map: dict):
    """توليد __init__.py مع load_model()."""
    k_names = [fname.replace('.npz', '') for fname in k_map]
    
    init_code = f'''# -*- coding: utf-8 -*-
"""نموذج ميران — vocab.py ديناميكي + K matrices .npz."""
import importlib, os
from scipy import sparse

_MODEL_DIR = os.path.dirname(__file__)

def load_vocab():
    """تحميل المعجم من vocab.py."""
    mod = importlib.import_module(".vocab", __package__)
    from src.physics.synchronize import Vocabulary
    vocab = Vocabulary()
    vocab.word2id = mod.WORD2ID
    vocab.next_id = mod.NEXT_ID
    for w, i in mod.WORD2ID.items():
        vocab.id2word[i] = w
    return vocab

def load_k(name: str):
    """تحميل مصفوفة K من ملف K_*.npz."""
    path = os.path.join(_MODEL_DIR, f"{{name}}.npz")
    return sparse.load_npz(path)

def load_syntax():
    """تحميل SyntaxField من syntax.py."""
    mod = importlib.import_module(".syntax", __package__)
    return mod.load()

def load_model():
    """تحميل النموذج الكامل."""
    vocab = load_vocab()
    K_sem = load_k("K_sem")
    syntax = load_syntax()
    result = {{'vocab': vocab, 'K_sem': K_sem, 'syntax': syntax}}
    for k_name in {k_names}:
        try:
            result[k_name] = load_k(k_name)
        except Exception:
            result[k_name] = None
    return result

def add_word(word: str):
    """إضافة كلمة إلى المعجم ديناميكياً."""
    mod = importlib.import_module(".vocab", __package__)
    return mod.add_word(word)
'''

    with open(os.path.join(model_dir, '__init__.py'), 'w', encoding='utf-8') as f:
        f.write(init_code)


def convert_npz_to_py(model_dir: str):
    """تحويل ملفات .npz الحالية إلى ملفات Python."""
    from src.physics.synchronize import Vocabulary
    from src.physics.grammar_field import SyntaxField
    
    # تحميل المعجم من JSON
    vocab_path = os.path.join(model_dir, 'vocab.json')
    if os.path.exists(vocab_path):
        with open(vocab_path, encoding='utf-8') as f:
            data = json.load(f)
        vocab = Vocabulary()
        vocab.word2id = data['word2id']
        vocab.next_id = data['next_id']
        for w, i in vocab.word2id.items():
            vocab.id2word[i] = w
    
    # تحميل K matrices
    K_sem = sparse.load_npz(os.path.join(model_dir, 'K.npz')) if os.path.exists(os.path.join(model_dir, 'K.npz')) else None
    K_syn = sparse.load_npz(os.path.join(model_dir, 'K_syn.npz')) if os.path.exists(os.path.join(model_dir, 'K_syn.npz')) else None
    K_dial = sparse.load_npz(os.path.join(model_dir, 'K_dialogue.npz')) if os.path.exists(os.path.join(model_dir, 'K_dialogue.npz')) else None
    
    # تحميل syntax
    syntax_path = os.path.join(model_dir, 'syntax.json')
    syntax = SyntaxField.load(syntax_path) if os.path.exists(syntax_path) else None
    
    # حفظ كملفات Python
    if vocab is not None and K_sem is not None and syntax is not None:
        save_model(model_dir, vocab, K_sem, syntax, K_syn=K_syn, K_dial=K_dial)
        print(f'✓ تم تحويل {model_dir} إلى ملفات Python')
        print(f'  - vocab.py: {len(vocab.word2id):,} كلمة')
        print(f'  - K_sem.py: {K_sem.shape}, {K_sem.nnz:,} مدخلاً')
        if K_syn is not None:
            print(f'  - K_syn.py: {K_syn.shape}, {K_syn.nnz:,} مدخلاً')
        if K_dial is not None:
            print(f'  - K_dial.py: {K_dial.shape}, {K_dial.nnz:,} مدخلاً')
    else:
        print('✗ الملفات المطلوبة غير موجودة في', model_dir)


def assimilate(model_dir: str, new_texts: list, window=5, alpha_blend=0.15):
    """دمج نصوص جديدة في النموذج بشكل تزايدي.
    
    Args:
        model_dir: مجلد النموذج (يحتاج vocab.py, K_*.py, syntax.py)
        new_texts: قائمة نصوص جديدة (list of str)
        window: نافذة الارتباط
        alpha_blend: معامل الدمج (0.0 = لا تغيير, 1.0 = استبدال كامل)
    """
    from src.physics.synchronize import Vocabulary, synchronize, assimilate_text, _tokenize
    
    # تحميل النموذج الحالي
    vocab = load_vocab(model_dir)
    
    # مسح النصوص الجديدة للكلمات غير الموجودة
    new_words = set()
    for text in new_texts:
        for line in text.split('\n'):
            for word in _tokenize(line):
                if word not in vocab.word2id:
                    new_words.add(word)
    
    if new_words:
        print(f'  كلمات جديدة: {len(new_words)}')
        for w in sorted(new_words)[:10]:
            wid = vocab.add_word(w)
            print(f'    + [{wid}] {w}')
    
    # تحميل مصفوفات K الحالية
    K_sem = load_k(model_dir, 'K_sem')
    K_syn = load_k(model_dir, 'K_syn') if os.path.exists(os.path.join(model_dir, 'K_syn.py')) else None
    K_dial = load_k(model_dir, 'K_dial') if os.path.exists(os.path.join(model_dir, 'K_dial.py')) else None
    
    # دمج النصوص الجديدة
    print(f'  دمج K_sem...')
    K_sem = assimilate_text(new_texts, vocab, K_sem, window=window, alpha_blend=alpha_blend)
    
    if K_syn is not None:
        print(f'  دمج K_syn...')
        K_syn = assimilate_text(new_texts, vocab, K_syn, window=2, alpha_blend=alpha_blend)
    
    if K_dial is not None:
        print(f'  دمج K_dial...')
        K_dial = assimilate_text(new_texts, vocab, K_dial, window=2, alpha_blend=alpha_blend)
    
    # تحميل syntax وتحديثه
    from src.physics.grammar_field import SyntaxField
    syntax = load_syntax(model_dir)
    for text in new_texts:
        for line in text.split('\n'):
            tokens = _tokenize(line)
            if len(tokens) >= 2:
                syntax.observe(tokens)
    syntax.finalize(vocab)
    
    # حفظ النموذج المحدث
    save_model(model_dir, vocab, K_sem, syntax, K_syn=K_syn, K_dial=K_dial)
    print(f'✓ تم تحديث النموذج في {model_dir}')


def load_vocab(model_dir: str):
    """تحميل المعجم من vocab.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("vocab", os.path.join(model_dir, "vocab.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from src.physics.synchronize import Vocabulary
    vocab = Vocabulary()
    vocab.word2id = mod.WORD2ID
    vocab.next_id = mod.NEXT_ID
    for w, i in mod.WORD2ID.items():
        vocab.id2word[i] = w
    return vocab


def load_k(model_dir: str, name: str):
    """تحميل مصفوفة K من ملف K_*.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(model_dir, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load()


def load_syntax(model_dir: str):
    """تحميل SyntaxField من syntax.py."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("syntax", os.path.join(model_dir, "syntax.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.load()


if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'convert'
    
    if mode == 'convert':
        model_dir = sys.argv[2] if len(sys.argv) > 2 else 'model'
        convert_npz_to_py(model_dir)
    elif mode == 'assimilate':
        model_dir = sys.argv[2] if len(sys.argv) > 2 else 'model'
        data_path = sys.argv[3] if len(sys.argv) > 3 else None
        if data_path:
            with open(data_path, encoding='utf-8') as f:
                texts = [f.read()]
            assimilate(model_dir, texts)
        else:
            print('الاستخدام: python -m src.physics.model_io assimilate <model_dir> <data_file>')
