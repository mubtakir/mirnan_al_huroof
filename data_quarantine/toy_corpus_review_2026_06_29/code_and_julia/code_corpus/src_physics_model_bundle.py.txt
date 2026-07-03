"""MirnanModelBundle — حزمة نموذج متكاملة للحفظ والتحميل.

تحفظ كل بنى المرنان في مجلد واحد:
  model/
  ├── vocab.json          # Vocabulary
  ├── K.npz               # مصفوفة الاقتران (sparse)
  ├── syntax.json         # SyntaxField
  ├── gss.npz             # GlobalSpectralMemory
  ├── gss_ground.npy      # ground state
  ├── causal_K.npz        # CausalPhaseEngine (optional)
  ├── K_code.npz          # مصفوفة اقتران البرمجة (optional)
  └── code_vocab.json     # CodeVocabulary (optional)
"""

import os
import json
import numpy as np
from scipy import sparse

from src.physics.synchronize import Vocabulary
from src.physics.grammar_field import SyntaxField
from src.physics.spectral_memory import GlobalSpectralMemory
from src.physics.code_engine import CodeVocabulary, TokenType


class MirnanModelBundle:
    """حزمة نموذج المرنان — تحفظ وتحمّل كل البنى الفيزيائية في مجلد واحد."""

    def __init__(self, vocab, K, syntax, gss, causal_K=None, K_code=None, code_vocab=None):
        self.vocab = vocab
        self.K = K
        self.syntax = syntax
        self.gss = gss
        self.causal_K = causal_K
        self.K_code = K_code
        self.code_vocab = code_vocab

    def save(self, path):
        """حفظ الحزمة إلى مجلد.

        Args:
            path: مسار المجلد (سيُخلق إن لم يوجد)
        """
        os.makedirs(path, exist_ok=True)

        # vocab
        with open(os.path.join(path, 'vocab.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'word2id': self.vocab.word2id,
                'next_id': self.vocab.next_id,
            }, f, ensure_ascii=False)

        # K
        sparse.save_npz(os.path.join(path, 'K.npz'), self.K)

        # syntax
        self.syntax.save(os.path.join(path, 'syntax.json'))

        # gss
        self.gss.save(os.path.join(path, 'gss'))

        # causal_K (optional)
        if self.causal_K is not None:
            sparse.save_npz(os.path.join(path, 'causal_K.npz'), self.causal_K)

        # K_code + code_vocab (optional)
        if self.K_code is not None and self.code_vocab is not None:
            sparse.save_npz(os.path.join(path, 'K_code.npz'), self.K_code)
            self.code_vocab.save(os.path.join(path, 'code_vocab.json'))

    @classmethod
    def load(cls, path):
        """تحميل الحزمة من مجلد.

        Args:
            path: مسار المجلد

        Returns:
            MirnanModelBundle
        """
        # vocab
        with open(os.path.join(path, 'vocab.json'), encoding='utf-8') as f:
            vdata = json.load(f)
        vocab = Vocabulary()
        vocab.word2id = vdata['word2id']
        vocab.next_id = vdata['next_id']
        vocab.id2word = {v: k for k, v in vocab.word2id.items()}

        # K
        K = sparse.load_npz(os.path.join(path, 'K.npz'))

        # syntax
        syntax = SyntaxField.load(os.path.join(path, 'syntax.json'))

        # gss
        gss = GlobalSpectralMemory()
        gss.load(os.path.join(path, 'gss'))

        # causal_K (optional)
        causal_K_path = os.path.join(path, 'causal_K.npz')
        causal_K = sparse.load_npz(causal_K_path) if os.path.exists(causal_K_path) else None

        # K_code + code_vocab (optional)
        K_code_path = os.path.join(path, 'K_code.npz')
        code_vocab_path = os.path.join(path, 'code_vocab.json')
        K_code = sparse.load_npz(K_code_path) if os.path.exists(K_code_path) else None
        code_vocab = CodeVocabulary.load(code_vocab_path) if os.path.exists(code_vocab_path) else None

        return cls(vocab, K, syntax, gss, causal_K, K_code, code_vocab)

    def build_generator(self, **kwargs):
        """بناء Generator من الحزمة."""
        from src.physics.generator import Generator
        return Generator(self.vocab, self.K, syntax_field=self.syntax, **kwargs)
