# -*- coding: utf-8 -*-
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

def load_benchmark_vocab():
    """تحميل معجم المعايرة المصغر من data/benchmark_vocab.json."""
    import json
    path = os.path.join(os.path.dirname(_MODEL_DIR), "data", "benchmark_vocab.json")
    from src.physics.synchronize import Vocabulary
    vocab = Vocabulary()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            words = json.load(f)
        for i, w in enumerate(words):
            vocab.word2id[w] = i
            vocab.id2word[i] = w
        vocab.next_id = len(words)
    return vocab

def load_k(name: str):
    """تحميل مصفوفة K من ملف K_*.npz."""
    path = os.path.join(_MODEL_DIR, f"{name}.npz")
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
    result = {'vocab': vocab, 'K_sem': K_sem, 'syntax': syntax}
    for k_name in ['K_sem', 'K_syn', 'K_dial']:
        try:
            result[k_name] = load_k(k_name)
        except Exception:
            result[k_name] = None
    return result

def add_word(word: str):
    """إضافة كلمة إلى المعجم ديناميكياً."""
    mod = importlib.import_module(".vocab", __package__)
    return mod.add_word(word)
