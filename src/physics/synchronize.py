"""المعجم النظيف — مرنان الحروف.

يحتوي على فئة Vocabulary لإدارة الكلمات ومعرّفاتها الرقمية.
"""
from src.physics.word_physics import _normalize_letters

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
        w = _normalize_letters(word)
        if w in self.word2id:
            return self.word2id[w]
        wid = self.next_id
        self.word2id[w] = wid
        self.id2word[wid] = w
        self.next_id += 1
        return wid

    def __len__(self):
        return self.next_id
