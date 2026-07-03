"""CodeEngine — طبقة البرمجة في مرنان.

يحول البرمجة إلى نظام فيزيائي:
- Token البرمجي = جسيم له طور وكتلة
- AST (شجرة التركيب) = حقل نحوي يحدد المسارات المسموحة
- توليد الكود = pathfinding على رسم بياني للـ tokens عبر PGN مع بوابة صارمة

V1 — يدعم Python فقط، ينتج كوداً صحيحاً نحوياً.
"""

import ast
import json
import re
import tokenize
import io
import numpy as np
from scipy import sparse
from enum import Enum, auto
from typing import List, Tuple, Optional, Set


class TokenType(Enum):
    KEYWORD = auto()
    IDENTIFIER = auto()
    OPERATOR = auto()
    LITERAL_NUM = auto()
    LITERAL_STR = auto()
    BRACKET_OPEN = auto()
    BRACKET_CLOSE = auto()
    PAREN_OPEN = auto()
    PAREN_CLOSE = auto()
    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()
    COLON = auto()
    DOT = auto()
    COMMA = auto()
    EQUALS = auto()
    ANNOTATION = auto()


PY_KEYWORDS = {
    'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await',
    'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except',
    'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is',
    'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return',
    'try', 'while', 'with', 'yield',
}

# مصفوفة نحوية — ما هي الـ token types المسموحة بعد كل type
VALID_NEXT = {
    TokenType.KEYWORD: {
        TokenType.IDENTIFIER, TokenType.KEYWORD, TokenType.PAREN_OPEN,
        TokenType.LITERAL_NUM, TokenType.LITERAL_STR, TokenType.COLON,
        TokenType.OPERATOR, TokenType.NEWLINE,
    },
    TokenType.IDENTIFIER: {
        TokenType.OPERATOR, TokenType.PAREN_OPEN, TokenType.PAREN_CLOSE,
        TokenType.NEWLINE, TokenType.COMMA, TokenType.COLON, TokenType.EQUALS,
        TokenType.DOT, TokenType.BRACKET_OPEN,
        TokenType.KEYWORD, TokenType.IDENTIFIER,
    },
    TokenType.OPERATOR: {
        TokenType.IDENTIFIER, TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
        TokenType.PAREN_OPEN, TokenType.OPERATOR,
    },
    TokenType.LITERAL_NUM: {
        TokenType.OPERATOR, TokenType.NEWLINE, TokenType.COMMA,
        TokenType.BRACKET_CLOSE, TokenType.PAREN_CLOSE, TokenType.COLON,
    },
    TokenType.LITERAL_STR: {
        TokenType.OPERATOR, TokenType.NEWLINE, TokenType.COMMA,
        TokenType.BRACKET_CLOSE, TokenType.PAREN_CLOSE,
    },
    TokenType.BRACKET_OPEN: {
        TokenType.IDENTIFIER, TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
        TokenType.BRACKET_CLOSE, TokenType.KEYWORD, TokenType.OPERATOR,
    },
    TokenType.BRACKET_CLOSE: {
        TokenType.OPERATOR, TokenType.NEWLINE, TokenType.COMMA,
        TokenType.BRACKET_CLOSE, TokenType.PAREN_CLOSE, TokenType.COLON,
    },
    TokenType.PAREN_OPEN: {
        TokenType.IDENTIFIER, TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
        TokenType.KEYWORD, TokenType.PAREN_CLOSE, TokenType.OPERATOR,
    },
    TokenType.PAREN_CLOSE: {
        TokenType.OPERATOR, TokenType.NEWLINE, TokenType.COMMA,
        TokenType.COLON, TokenType.PAREN_CLOSE, TokenType.BRACKET_CLOSE,
        TokenType.KEYWORD,
    },
    TokenType.COLON: {TokenType.NEWLINE, TokenType.INDENT},
    TokenType.NEWLINE: {
        TokenType.KEYWORD, TokenType.IDENTIFIER, TokenType.DEDENT, TokenType.INDENT,
        TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
    },
    TokenType.INDENT: {
        TokenType.KEYWORD, TokenType.IDENTIFIER, TokenType.DEDENT,
        TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
    },
    TokenType.DEDENT: {
        TokenType.KEYWORD, TokenType.IDENTIFIER, TokenType.DEDENT, TokenType.NEWLINE,
    },
    TokenType.EQUALS: {
        TokenType.IDENTIFIER, TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
        TokenType.PAREN_OPEN, TokenType.BRACKET_OPEN, TokenType.OPERATOR,
    },
    TokenType.COMMA: {
        TokenType.IDENTIFIER, TokenType.LITERAL_NUM, TokenType.LITERAL_STR,
        TokenType.PAREN_OPEN, TokenType.BRACKET_OPEN,
    },
    TokenType.DOT: {TokenType.IDENTIFIER},
}


def tokenize_code(source: str) -> List[Tuple[str, TokenType]]:
    """تحليل كود Python إلى قائمة (نص, نوع)."""
    tokens = []
    try:
        gen = tokenize.generate_tokens(io.StringIO(source).readline)
        for tok in gen:
            ttype = tok.type
            tstr = tok.string
            if ttype == tokenize.ENDMARKER:
                break
            if ttype == tokenize.NAME:
                tt = TokenType.KEYWORD if tstr in PY_KEYWORDS else TokenType.IDENTIFIER
            elif ttype == tokenize.NUMBER:
                tt = TokenType.LITERAL_NUM
            elif ttype == tokenize.STRING:
                tt = TokenType.LITERAL_STR
            elif ttype == tokenize.OP:
                if tstr == '(':
                    tt = TokenType.PAREN_OPEN
                elif tstr == ')':
                    tt = TokenType.PAREN_CLOSE
                elif tstr == '[':
                    tt = TokenType.BRACKET_OPEN
                elif tstr == ']':
                    tt = TokenType.BRACKET_CLOSE
                elif tstr == ':':
                    tt = TokenType.COLON
                elif tstr == '=':
                    tt = TokenType.EQUALS
                elif tstr == ',':
                    tt = TokenType.COMMA
                elif tstr == '.':
                    tt = TokenType.DOT
                else:
                    tt = TokenType.OPERATOR
            elif ttype == tokenize.INDENT:
                tt = TokenType.INDENT
            elif ttype == tokenize.DEDENT:
                tt = TokenType.DEDENT
            elif ttype == tokenize.NEWLINE:
                tt = TokenType.NEWLINE
            else:
                continue
            tokens.append((tstr, tt))
    except Exception:
        pass
    return tokens


def token_to_key(text: str, ttype: TokenType) -> str:
    """تحويل token إلى مفتاح للمعجم."""
    return f"{ttype.name}:{text}"


class CodeVocabulary:
    """معجم للـ tokens البرمجية — مثل Vocabulary لكن للـ code tokens."""

    def __init__(self):
        self.token2id = {}
        self.id2token = {}
        self.next_id = 0

    def add(self, text: str, ttype: TokenType) -> int:
        key = token_to_key(text, ttype)
        if key not in self.token2id:
            self.token2id[key] = self.next_id
            self.id2token[self.next_id] = (text, ttype)
            self.next_id += 1
        return self.token2id[key]

    def get(self, text: str, ttype: TokenType, default=None):
        return self.token2id.get(token_to_key(text, ttype), default)

    def __len__(self):
        return self.next_id

    def save(self, path: str):
        data = {
            'token2id': self.token2id,
            'id2token': {str(k): [v[0], v[1].name] for k, v in self.id2token.items()},
            'next_id': self.next_id,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path: str):
        obj = cls()
        with open(path, encoding='utf-8') as f:
            data = json.load(f)
        obj.token2id = data['token2id']
        obj.id2token = {int(k): (v[0], TokenType[v[1]]) for k, v in data['id2token'].items()}
        obj.next_id = data['next_id']
        return obj


def build_K_code(corpus_texts, window=5):
    """بناء مصفوفة K للبرمجة من نصوص Python.

    Returns: (code_vocab, K_code)
    """
    vocab = CodeVocabulary()
    all_ids = []

    for text in corpus_texts:
        tokens = tokenize_code(text)
        ids = []
        for tstr, tt in tokens:
            if tt in (TokenType.INDENT, TokenType.DEDENT):
                continue
            wid = vocab.add(tstr, tt)
            ids.append(wid)
        all_ids.append(ids)

    V = len(vocab)
    K = sparse.lil_matrix((V, V), dtype=np.float32)

    for ids in all_ids:
        for i, tid in enumerate(ids):
            for j in range(i + 1, min(i + window + 1, len(ids))):
                nid = ids[j]
                K[tid, nid] += 1.0

    K = K.tocsr()
    return vocab, K


def validate_syntax(tokens: List[Tuple[str, TokenType]]) -> bool:
    """التحقق من صحة التسلسل النحوي للـ tokens."""
    if not tokens:
        return True
    filtered = [(t, tt) for t, tt in tokens if tt not in (TokenType.INDENT, TokenType.DEDENT)]
    for i in range(len(filtered) - 1):
        curr_type = filtered[i][1]
        next_type = filtered[i + 1][1]
        allowed = VALID_NEXT.get(curr_type, set())
        if next_type not in allowed:
            return False
    return True


def compile_check(source: str) -> Tuple[bool, str]:
    """تشغيل compile() على الكود والتأكد من خلوه من الأخطاء النحوية.

    Returns: (is_valid, error_message)
    """
    try:
        compile(source, '<mirnan_code>', 'exec')
        return True, ''
    except SyntaxError as e:
        return False, str(e)


class CodePhaseVector:
    """توليد متجه طوري 64D لـ token برمجي."""

    def __init__(self):
        self.keyword_mass = {kw: 2.0 for kw in PY_KEYWORDS}
        self.keyword_mass.update({
            'def': 5.0, 'class': 5.0, 'return': 4.0,
            'if': 3.0, 'for': 3.0, 'while': 3.0,
            'import': 3.0, 'from': 3.0,
        })

    def get_phase_vector(self, text: str, ttype: TokenType) -> np.ndarray:
        v = np.zeros(64)
        if ttype == TokenType.KEYWORD:
            base = self.keyword_mass.get(text, 1.0)
            idx = hash(text) % 22
            v[idx] = base
            v[22] = base * 0.5
        elif ttype == TokenType.IDENTIFIER:
            mass = max(len(text) * 0.3, 0.5)
            for i, ch in enumerate(text):
                v[i % 22] += ord(ch) / 255.0
            v[22] = mass
        elif ttype == TokenType.LITERAL_NUM:
            try:
                val = float(text)
            except ValueError:
                val = 0.0
            v[0] = val / max(abs(val) + 1, 1)
            v[22] = 0.5
        elif ttype == TokenType.OPERATOR:
            v[23] = 1.0
        elif ttype == TokenType.COLON:
            v[24] = 1.0
        elif ttype == TokenType.NEWLINE:
            v[25] = 1.0
        elif ttype in (TokenType.PAREN_OPEN, TokenType.BRACKET_OPEN):
            v[26] = 1.0
        elif ttype in (TokenType.PAREN_CLOSE, TokenType.BRACKET_CLOSE):
            v[27] = 1.0
        elif ttype == TokenType.INDENT:
            v[28] = 1.0
        elif ttype == TokenType.DEDENT:
            v[29] = 1.0
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-10 else v


class CodeEngine:
    """محرك البرمجة — يدمج التحليل النحوي، التوليد، والتحقق."""

    def __init__(self, vocab=None, K=None, code_vocab=None, K_code=None):
        self.vocab = vocab or {}
        self.K = K
        self.code_vocab = code_vocab or CodeVocabulary()
        self.K_code = K_code
        self.pv_gen = CodePhaseVector()
        self._code_cache = {}

    def tokenize(self, source: str) -> List[Tuple[str, TokenType]]:
        return tokenize_code(source)

    def validate(self, source: str) -> Tuple[bool, str]:
        tokens = self.tokenize(source)
        if not validate_syntax(tokens):
            return False, 'token sequence error'
        ok, err = compile_check(source)
        return ok, err

    def suggest_next(self, partial_tokens: List[Tuple[str, TokenType]]) -> List[Tuple[str, float]]:
        """أقتراح الـ tokens التالية مع درجات رنين من K_code."""

        if not partial_tokens:
            starters = [(kw, 1.0) for kw in ['def', 'import', 'class', 'for', 'if', 'x', 'print']]
            return starters

        last_type = partial_tokens[-1][1]
        last_text = partial_tokens[-1][0]
        allowed_types = VALID_NEXT.get(last_type, set())

        candidates = []

        # 1. من K_code إذا كان متاحاً
        if self.K_code is not None and self.code_vocab is not None:
            last_key = token_to_key(last_text, last_type)
            last_id = self.code_vocab.token2id.get(last_key)
            if last_id is not None and last_id < self.K_code.shape[0]:
                row = self.K_code[last_id].toarray().ravel()
                if row.sum() > 0:
                    probs = row / row.sum()
                    for nid in np.argsort(probs)[::-1][:20]:
                        if probs[nid] > 0.01:
                            ntext, ntype = self.code_vocab.id2token[nid]
                            if ntype in allowed_types:
                                candidates.append((ntext, ntype, float(probs[nid])))

        # 2. تكملة من القواعد النحوية — لكل نوع مسموح به نعطي تمثيلاً
        seen_texts = set(t[0] for t in candidates)
        fallback_scores = {
            TokenType.KEYWORD: 0.1,
            TokenType.IDENTIFIER: 0.3,
            TokenType.LITERAL_NUM: 0.2,
            TokenType.LITERAL_STR: 0.2,
            TokenType.OPERATOR: 0.15,
            TokenType.PAREN_OPEN: 0.25,
            TokenType.PAREN_CLOSE: 0.05,
            TokenType.BRACKET_OPEN: 0.2,
            TokenType.BRACKET_CLOSE: 0.05,
            TokenType.COLON: 0.5,
            TokenType.EQUALS: 0.4,
            TokenType.COMMA: 0.2,
            TokenType.DOT: 0.2,
            TokenType.NEWLINE: 0.3,
        }
        fallback_text = {
            TokenType.KEYWORD: ['pass'],
            TokenType.IDENTIFIER: ['x'],
            TokenType.LITERAL_NUM: ['0'],
            TokenType.LITERAL_STR: ['""'],
            TokenType.OPERATOR: ['+', '-', '*', '==', '<', '>'],
            TokenType.PAREN_OPEN: ['('],
            TokenType.PAREN_CLOSE: [')'],
            TokenType.BRACKET_OPEN: ['['],
            TokenType.BRACKET_CLOSE: [']'],
            TokenType.COLON: [':'],
            TokenType.EQUALS: ['='],
            TokenType.COMMA: [','],
            TokenType.DOT: ['.'],
            TokenType.NEWLINE: ['\n'],
        }
        for ttype in allowed_types:
            if ttype in fallback_text:
                for txt in fallback_text[ttype]:
                    if txt not in seen_texts:
                        candidates.append((txt, ttype, fallback_scores.get(ttype, 0.1)))
                        seen_texts.add(txt)

        candidates.sort(key=lambda x: -x[2])
        return [(t, s) for t, _, s in candidates[:15]]

    def generate_python(self, prompt: str, max_tokens: int = 30) -> str:
        """توليد كود Python من وصف طبيعي — قوالب + رنين K_code."""
        source = prompt.strip()
        templates = {
            'function': self._gen_function,
            'loop': self._gen_loop,
            'class': self._gen_class,
            'condition': self._gen_condition,
            'import': self._gen_import,
        }
        for key, gen_fn in templates.items():
            if key in source.lower():
                return gen_fn(source)
        return self._gen_resonant(prompt, max_tokens)

    def _infer_type(self, text: str) -> TokenType:
        if text in PY_KEYWORDS:
            return TokenType.KEYWORD
        if text in ('(', ')'):
            return TokenType.PAREN_OPEN if text == '(' else TokenType.PAREN_CLOSE
        if text in ('[', ']'):
            return TokenType.BRACKET_OPEN if text == '[' else TokenType.BRACKET_CLOSE
        if text == ':':
            return TokenType.COLON
        if text == '=':
            return TokenType.EQUALS
        if text == ',':
            return TokenType.COMMA
        if text == '.':
            return TokenType.DOT
        if text == '\n':
            return TokenType.NEWLINE
        if text in ('+', '-', '*', '/', '%', '==', '!=', '<', '>', '<=', '>=', 'and', 'or', 'not'):
            return TokenType.OPERATOR
        if text.isdigit() or (text.startswith('"') or text.startswith("'")):
            return TokenType.LITERAL_NUM if text.isdigit() else TokenType.LITERAL_STR
        return TokenType.IDENTIFIER

    def _tokens_to_source(self, tokens: List[Tuple[str, TokenType]]) -> str:
        source = ''
        indent_level = 0
        for tstr, tt in tokens:
            if tt == TokenType.NEWLINE:
                source += '\n' + '    ' * indent_level
            elif tt == TokenType.INDENT:
                indent_level += 1
                source = source.rstrip(' ') + '\n' + '    ' * indent_level
            elif tt == TokenType.DEDENT:
                indent_level = max(0, indent_level - 1)
            elif tt in (TokenType.OPERATOR, TokenType.COMMA, TokenType.DOT, TokenType.COLON):
                source += tstr
            elif tt in (TokenType.PAREN_OPEN, TokenType.BRACKET_OPEN):
                source += tstr
            elif tt in (TokenType.PAREN_CLOSE, TokenType.BRACKET_CLOSE):
                source += tstr
            else:
                if source and not source[-1].isspace() and source[-1] not in ('(', '[', ','):
                    source += ' '
                source += tstr
        return source

    def _gen_resonant(self, prompt: str, max_tokens: int) -> str:
        """توليد سطر واحد صالح — اختيار starter ذكي من الـ prompt + توليد حتى NEWLINE."""
        pl = prompt.lower()
        starter_map = {
            'print': 'print', 'hello': 'print', 'hi': 'print',
            'import': 'import', 'from': 'from',
            'return': 'return', 'yield': 'yield',
            'binary': 'def', 'search': 'def', 'sort': 'def',
            'function': 'def', 'class': 'class',
            'factorial': 'x', 'fib': 'x', 'sum': 'x',
            'max': 'x', 'min': 'x', 'average': 'x',
        }
        starter = 'x'
        for kw, st in starter_map.items():
            if kw in pl:
                starter = st
                break

        if starter == 'def':
            name = 'f'
            for w in prompt.split():
                if w not in ('a', 'an', 'the', 'that', 'function', 'def', 'define', 'new', 'create', 'write', 'make', 'sort', 'search', 'find', 'binary', 'calculate', 'compute'):
                    name = w
                    break
            return f"def {name}():\n    pass\n"

        if starter == 'class':
            name = 'MyClass'
            for w in prompt.split():
                if w not in ('a', 'an', 'the', 'that', 'class', 'define', 'new', 'create', 'write', 'make'):
                    name = w.capitalize()
                    break
            return f"class {name}:\n    pass\n"

        if starter == 'print':
            return 'print("hello")\n'

        if starter in ('import', 'from'):
            return 'import math\n'

        if starter == 'return':
            return 'return x + 1\n'

        # Simple expression: x = ...
        result_tokens = [(starter, self._infer_type(starter))]
        repeat_count = 0
        prev_type = result_tokens[-1][1]
        for _ in range(max_tokens - 1):
            candidates = self.suggest_next(result_tokens)
            if not candidates:
                break
            chosen = None
            for i in range(min(5, len(candidates))):
                text = candidates[i][0]
                ttype = self._infer_type(text)
                trial = result_tokens + [(text, ttype)]
                if validate_syntax(trial):
                    chosen = (text, ttype)
                    break
            if chosen is None:
                break
            text, ttype = chosen
            if ttype == prev_type:
                repeat_count += 1
            else:
                repeat_count = 0
                prev_type = ttype
            if repeat_count > 2:
                break
            result_tokens.append(chosen)
            if ttype == TokenType.NEWLINE:
                break
        source = self._tokens_to_source(result_tokens)
        ok, _ = self.validate(source)
        return source if ok else ''

    def _gen_function(self, prompt: str) -> str:
        name = self._extract_name(prompt) or 'my_func'
        return f"def {name}():\n    pass\n"

    def _gen_loop(self, prompt: str) -> str:
        if 'for' in prompt.lower():
            return "for i in range(10):\n    pass\n"
        return "while True:\n    pass\n"

    def _gen_class(self, prompt: str) -> str:
        name = self._extract_name(prompt) or 'MyClass'
        return f"class {name}:\n    pass\n"

    def _gen_condition(self, prompt: str) -> str:
        return "if x > 0:\n    pass\nelse:\n    pass\n"

    def _gen_import(self, prompt: str) -> str:
        return "import math\n"

    def _extract_name(self, prompt: str) -> Optional[str]:
        m = re.search(r'(?:اسمه|يسمى|named|called|اسم)\s+(\w+)', prompt)
        if m:
            return m.group(1)
        m = re.search(r'`(\w+)`', prompt)
        return m.group(1) if m else None
