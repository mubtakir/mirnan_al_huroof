"""SymbolicBridge — جسر رمزي-طوري بين القواعد النحوية والحقول الفيزيائية.

6 قواعد مسجلة: نفي، عطف، جر، تفضيل، استفهام، شرط.
كل قاعدة تُقيّم بالتكرار دون تعلم.
"""
import re
import os
import yaml
import numpy as np
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import TOTAL_DIM

_RULES = []

def _register(rule_fn):
    _RULES.append(rule_fn)
    return rule_fn

def load_yaml_rules(yaml_path):
    """تحميل القواعد الرمزية البسيطة من ملف YAML وتوليد دوال لها ديناميكياً"""
    if not os.path.exists(yaml_path):
        return
        
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
        
    if not data or 'rules' not in data:
        return
        
    for rule_def in data['rules']:
        name = rule_def.get('name', 'unnamed_rule')
        lookback = rule_def.get('lookback', 1)
        markers = set(rule_def.get('markers', []))
        targets = set(rule_def.get('targets', []))
        score = rule_def.get('score', 0.0)
        anti_targets = set(rule_def.get('anti_targets', []))
        anti_score = rule_def.get('anti_score', 0.0)
        
        # إنشاء دالة الإغلاق (Closure) التي تمثل القاعدة
        def create_rule_fn(m_set, t_set, s_val, at_set, as_val, lb):
            def generated_rule(word, prev_words, context_ids, vocab, K, _pv_cache=None):
                if not prev_words:
                    return 0.0
                
                # التحقق من وجود العلامة (Marker) في السياق السابق
                has_marker = False
                for pw in prev_words[-lb:]:
                    if pw in m_set:
                        has_marker = True
                        break
                        
                if has_marker:
                    # إذا كان هناك أهداف (Targets) محددة، يجب أن تطابق الكلمة
                    if t_set:
                        if word in t_set:
                            return s_val
                    else:
                        # إذا لم تكن هناك أهداف، نمنح السكور لأي كلمة
                        return s_val
                        
                    # إذا كان هناك أهداف مضادة (Anti Targets)
                    if at_set and word in at_set:
                        return as_val
                        
                return 0.0
            generated_rule.__name__ = name
            return generated_rule
            
        rule_fn = create_rule_fn(markers, targets, score, anti_targets, anti_score, lookback)
        _register(rule_fn)

# تحميل القواعد تلقائياً عند استيراد الملف
default_yaml_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'rules', 'symbolic_rules.yaml')
load_yaml_rules(default_yaml_path)


# ============================================================
# القواعد البرمجية المعقدة (Python-based Rules)
# ============================================================

@_register
def rule_preposition(word, prev_words, context_ids, vocab, K, _pv_cache=None):
    """حروف الجر: في، من، على -> تعزيز الاقتران مع المتبوع باستخدام مصفوفة الترابط K"""
    if not prev_words:
        return 0.0
    prepositions = {'في', 'من', 'على', 'إلى', 'عن', 'مع', 'بين', 'تحت', 'فوق', 'ب', 'ل', 'ك', 'خلال', 'عند', 'لدى', 'حول', 'ضد', 'نحو', 'تجاه'}
    for pw in prev_words[-1:]:
        if pw in prepositions:
            if vocab.get(word) is not None and vocab.get(pw) is not None:
                i = vocab.get(pw)
                j = vocab.get(word)
                if i < K.shape[0] and j < K.shape[1]:
                    k_val = float(K[i, j])
                    if k_val > 0:
                        return min(k_val * 0.01, 0.15)
    return 0.0

@_register
def rule_filament_resonance(word, prev_words, context_ids, vocab, K, _pv_cache=None):
    """الفتيلة الدلالية: الكلمات التي تشارك نفس الفئة الطاقية تتجاذب"""
    if _pv_cache is None:
        _pv_cache = {}
    if not prev_words or len(prev_words) < 2:
        return 0.0
    try:
        if word not in _pv_cache:
            _pv_cache[word] = compute_extended_phase_vector(word)
        word_pv = _pv_cache[word]
        cat_scores = []
        for pw in prev_words[-2:]:
            if pw not in _pv_cache:
                _pv_cache[pw] = compute_extended_phase_vector(pw)
            pw_pv = _pv_cache[pw]
            align = float(np.mean(np.cos(word_pv - pw_pv)))
            cat_scores.append(align)
        avg_cat = sum(cat_scores) / len(cat_scores)
        if avg_cat > 0.85:
            return 0.12
        if avg_cat > 0.75:
            return 0.05
    except Exception as e:
        import logging; logging.getLogger('symbolic_bridge').debug(f'rule_preposition_resonance: {e}')
    return 0.0

@_register
def rule_gravitational_pull(word, prev_words, context_ids, vocab, K, _pv_cache=None):
    """الجاذبية الدلالية: الكلمات عالية الكتلة تجذب الأفعال وتطرد الأسماء"""
    if _pv_cache is None:
        _pv_cache = {}
    if not prev_words or len(prev_words) < 1:
        return 0.0
    try:
        pw = prev_words[-1]
        if pw not in _pv_cache:
            _pv_cache[pw] = compute_extended_phase_vector(pw)
        prev_pv = _pv_cache[pw]
        energy = float(prev_pv[-4])
        op_var = float(prev_pv[-2])
        mass = energy * (1.0 + op_var)
        if mass > 0.6 and len(pw) >= 4:
            verb_like = sum(1 for c in word if c in 'يتسنأف')
            if verb_like >= 2 and len(word) >= 4:
                return min(mass * 0.15, 0.12)
            if word.endswith('ة') and len(word) >= 4:
                return -0.06
    except Exception as e:
        import logging; logging.getLogger('symbolic_bridge').debug(f'rule_gravitational_pull: {e}')
    return 0.0

@_register
def rule_number_sequence(word, prev_words, context_ids, vocab, K, _pv_cache=None):
    """تسلسل رقمي: أرقام وهوية -> جذب للأرقام والوحدات باستخدام Regex"""
    if not prev_words:
        return 0.0
    is_number = bool(re.match(r'^[\+\-]?\d+\.?\d*$', word))
    units = {'سنتيمتر', 'متر', 'كيلومتر', 'غرام', 'كيلو', 'ليتر', 'ثانية', 'دقيقة', 'ساعة'}
    if is_number:
        return 0.15
    if word in units:
        for pw in prev_words[-2:]:
            if bool(re.match(r'^[\+\-]?\d+\.?\d*$', pw)):
                return 0.2
    return 0.0

@_register
def rule_arithmetic_op(word, prev_words, context_ids, vocab, K, _pv_cache=None):
    """عمليات حسابية: + - * / -> تعتمد على الأحرف المكونة للكلمة"""
    arith_ops = {'+', '-', '*', '/', '**', '//', '%', '^'}
    for pw in prev_words[-1:]:
        if pw in arith_ops:
            return 0.1
    for c in word:
        if c in arith_ops:
            return 0.08
    return 0.0

# ============================================================
# Bridge Class
# ============================================================

class SymbolicBridge:
    def __init__(self, vocab=None, coupling_K=None):
        self.vocab = vocab
        self.K = coupling_K
        self._anchor_cache = {}
        self._pv_cache = {}

    def _get_anchor(self, word):
        if word not in self._anchor_cache:
            self._anchor_cache[word] = compute_extended_phase_vector(word)
        return self._anchor_cache[word]

    def _get_pv(self, word):
        if word not in self._pv_cache:
            self._pv_cache[word] = compute_extended_phase_vector(word)
        return self._pv_cache[word]

    def evaluate(self, word, prev_words, context_ids=None):
        score = 0.0
        for rule in _RULES:
            # We pass _pv_cache to avoid computing pv multiple times across rules
            score += rule(word, prev_words, context_ids, self.vocab, self.K, self._pv_cache)
        return min(score, 0.6)   # رفعنا الحد الأقصى من 0.5 إلى 0.6

    @property
    def rule_count(self):
        return len(_RULES)

    def set_vocab(self, vocab, coupling_K):
        self.vocab = vocab
        self.K = coupling_K
