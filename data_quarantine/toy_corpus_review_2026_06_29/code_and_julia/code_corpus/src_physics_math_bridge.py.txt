import ast
import re
import numpy as np
from src.physics.word_physics import compute_extended_phase_vector, _normalize_letters
from src.physics.constants import TOTAL_DIM


_SYM_PATTERNS = [
    (r'[\+\-\*\/\^\(\)\=\<\>]', 'math_op'),
    (r'\d+\.?\d*', 'number'),
    (r'[a-zA-Z_]\w*', 'identifier'),
]

_ARITH_RULES = {
    '+': lambda a, b: a + b,
    '-': lambda a, b: a - b,
    '*': lambda a, b: a * b,
    '/': lambda a, b: a / b if b != 0 else float('inf'),
    '**': lambda a, b: a ** b,
}

_ARABIC_DIGITS = str.maketrans('0123456789', '٠١٢٣٤٥٦٧٨٩')


def _tokenize_math(text: str):
    tokens = re.findall(r'[\+\-\*\/\^\(\)\=\<\>]|\d+\.?\d*|[a-zA-Z_]\w*', text)
    return tokens


def _safe_eval(expr: str):
    try:
        tree = ast.parse(expr, mode='eval')
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp,
                                     ast.Num, ast.Constant, ast.Name,
                                     ast.Add, ast.Sub, ast.Mult, ast.Div,
                                     ast.Pow, ast.USub, ast.UAdd)):
                return None
        return eval(compile(tree, '<safe>', 'eval'))
    except Exception:
        return None


def _check_syntax(code: str):
    if not code.strip():
        return None
    try:
        tree = ast.parse(code, mode='exec')
        mean_len = 0.0
        count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.For, ast.While,
                                 ast.If, ast.With, ast.Try)):
                mean_len += node.end_lineno - node.lineno
                count += 1
            if isinstance(node, ast.Name):
                if node.id in ('eval', 'exec', '__import__', 'open',
                                'compile', 'getattr', 'setattr'):
                    return 'dangerous'
        return 'valid' if count == 0 else 'has_control_flow'
    except SyntaxError as e:
        return f'syntax_error: {e.msg} at line {e.lineno}'
    except Exception:
        return 'unknown_error'


def _make_math_phase_vector(expr_value: float):
    pv = np.zeros(TOTAL_DIM)
    val_norm = np.tanh(abs(expr_value))
    sign = 1.0 if expr_value >= 0 else -1.0
    pv[0] = abs(expr_value) / (abs(expr_value) + 1.0)
    pv[1] = sign * 0.3
    pv[2] = val_norm * 0.5
    return pv


class MathBridge:
    def __init__(self):
        self._history = []
        self._last_result = None

    def detect_mode(self, prompt: str):
        math_keywords = {'حساب', 'جمع', 'طرح', 'ضرب', 'قسمة', 'اشتق', 'تكامل',
                         'جذر', 'قوة', 'مجموع', 'ناتج', 'يساوي', 'معادلة',
                         'math', 'calculate', 'compute', 'derivative', 'integral',
                         'equation', 'solve'}
        code_keywords = {'كود', 'برنامج', 'دالة', 'حلقة', 'شرط', 'مصفوفة',
                         'class', 'def', 'function', 'loop', 'array', 'debug',
                         'code', 'program', 'خطأ', 'error', 'bug', 'syntax'}
        
        # الكشف المباشر عن العمليات الحسابية المكتوبة بصورة رقمية
        if re.search(r'\d+\s*[\+\-\*\/]\s*\d+', prompt):
            return 'math'
            
        tokens = prompt.split()
        math_score = sum(1 for t in tokens if t in math_keywords)
        code_score = sum(1 for t in tokens if t in code_keywords)
        if math_score > code_score and math_score > 0:
            return 'math'
        if code_score > 0:
            return 'code'
        return 'text'

    def evaluate_math(self, expression: str):
        tokens = _tokenize_math(expression)
        result = _safe_eval(expression)
        if result is not None:
            pv = _make_math_phase_vector(result)
            self._last_result = result
            self._history.append(('math', expression, result))
            return {'value': result, 'phase_vector': pv, 'valid': True}
        ar_clean = expression.translate(_ARABIC_DIGITS)
        m = re.match(r'[+\-]?\d+(\.\d+)?([+\-*/]\d+(\.\d+)?)+', ar_clean)
        if m:
            eng_expr = m.group().replace('^', '**')
            result = _safe_eval(eng_expr)
            if result is not None:
                pv = _make_math_phase_vector(result)
                self._last_result = result
                self._history.append(('math_ar', expression, result))
                return {'value': result, 'phase_vector': pv, 'valid': True}
        return {'value': None, 'phase_vector': np.zeros(TOTAL_DIM), 'valid': False, 'error': 'cannot_evaluate'}

    def validate_code(self, code_snippet: str):
        result = _check_syntax(code_snippet)
        if result == 'valid':
            return {'status': 'valid', 'score': 1.0, 'phase_align': 0.2}
        if result == 'has_control_flow':
            return {'status': 'valid_with_flow', 'score': 0.8, 'phase_align': 0.15}
        if result and result.startswith('syntax_error'):
            return {'status': 'invalid', 'score': -0.5,
                    'error': result, 'phase_align': -0.3}
        if result == 'dangerous':
            return {'status': 'dangerous', 'score': -1.0, 'phase_align': -0.5}
        return {'status': 'unknown', 'score': 0.0, 'phase_align': 0.0}

    def get_resonance_boost(self, generated_words: list):
        if not generated_words:
            return 0.0, np.zeros(TOTAL_DIM)
        full = ' '.join(generated_words)
        code_checks = []
        for line in full.split('\n'):
            line = line.strip()
            if line and not any(c in line for c in 'اأإبتثجحخدذرزسشصضطظعغفقكلمنهوي'):
                check = self.validate_code(line)
                code_checks.append(check)
        if code_checks:
            avg_score = sum(c['score'] for c in code_checks) / len(code_checks)
            return avg_score * 0.1, _make_math_phase_vector(avg_score)
        math_checks = re.findall(r'[+\-]?\d+(\.\d+)?[+\-*/]\d+(\.\d+)?', full)
        if not math_checks:
            math_checks = re.findall(r'[\+\-]?\d+\.?\d*', full)
        if math_checks:
            return 0.05, _make_math_phase_vector(float(math_checks[0].replace('^', '**')[:10]))
        return 0.0, np.zeros(TOTAL_DIM)

    def reset(self):
        self._history.clear()
        self._last_result = None
