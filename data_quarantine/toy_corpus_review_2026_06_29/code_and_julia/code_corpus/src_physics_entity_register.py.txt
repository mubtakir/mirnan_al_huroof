"""EntityRegister — تتبع الكيانات عبر الحوار في فضاء طوري.

كل كيان (اسم، مكان، شيء) يمر في الحوار يُسجل كمتجه طوري + كتلة.
عند ورود ضمير (هو، هي، هذا، ذلك)، نحلّه إلى أقرب كيان في فضاء الطور.

المبادئ:
- الكيان = (name, phase_vector, mass, عمر, intent_context)
- الضمير = استعلام في فضاء الطور → أقرب كيان حي
- الكيانات تتلاشى مع الزمن (Temporal Decay للكيانات القديمة)
"""
import numpy as np
import logging
from src.physics.constants import PHASE_DIM
from src.physics.word_physics import compute_extended_phase_vector, compute_word_mass
from src.physics.word_spectrum import spectral_resonance

logger = logging.getLogger(__name__)

_PRONOUNS = {'هو', 'هي', 'هم', 'هن', 'هما', 'همو', 'إياه', 'إياها',
             'هذا', 'هذه', 'ذلك', 'تلك', 'ذان', 'تان', 'أولئك',
             'ه', 'ها', 'هم', 'كما', 'كم', 'كن', 'ني', 'نا',
             'الذي', 'التي', 'الذين', 'اللواتي', 'ما', 'من'}


class Entity:
    """كيان واحد في الحوار."""
    def __init__(self, name, pv, mass=1.0, context_intent=""):
        self.name = name
        self.pv = pv
        self.mass = mass
        self.age = 0
        self.mentions = 1
        self.context_intent = context_intent
        self.avg_spectral = 0.0
    
    def reinforce(self, pv, mass):
        self.mentions += 1
        blend = 1.0 / self.mentions
        self.pv = (1 - blend) * self.pv + blend * pv
        nrm = np.linalg.norm(self.pv)
        if nrm > 1e-10:
            self.pv = self.pv / nrm
        self.mass = max(self.mass, mass)


class EntityRegister:
    """سجل الكيانات الطوري.
    
    يتتبع الكيانات عبر الحوار ويحل الضمائر إليها.
    """
    
    def __init__(self, max_entities=30, decay=0.1, merge_threshold=0.82):
        self.max_entities = max_entities
        self.decay = decay
        self.merge_threshold = merge_threshold
        self.entities = []
    
    def _is_entity_like(self, word):
        """هل الكلمة مرشحة لتكون كياناً؟"""
        if word in _PRONOUNS:
            return False
        if len(word) < 3:
            return False
        # كلمات الوظيفة (function words) ليست كيانات
        func_words = {'في', 'من', 'على', 'إلى', 'عن', 'مع', 'كان',
                      'ليس', 'إن', 'أن', 'قد', 'سوف', 'لن', 'لم',
                      'هل', 'أ', 'ما', 'لا', 'إن', 'إذا', 'لو'}
        if word in func_words:
            return False
        return True
    
    def _get_word_pv(self, word):
        try:
            return compute_extended_phase_vector(word)
        except Exception:
            return None
    
    def _get_word_mass(self, word):
        try:
            return compute_word_mass(word)
        except Exception:
            return 0.1
    
    def extract_entities(self, text, context_intent=""):
        """استخراج الكيانات من جملة وتسجيلها."""
        if not text:
            return []
        words = text.split()
        found = []
        for w in words:
            if self._is_entity_like(w):
                self.register(w, context_intent)
                found.append(w)
        self._age_all()
        return found
    
    def register(self, name, context_intent=""):
        """تسجيل كيان جديد أو تعزيز كيان موجود."""
        pv = self._get_word_pv(name)
        if pv is None:
            return
        
        mass = self._get_word_mass(name)
        
        # هل الكيان موجود؟
        for e in self.entities:
            sim = float(np.mean(np.cos(e.pv[:PHASE_DIM] - pv[:PHASE_DIM])))
            if sim > self.merge_threshold:
                e.reinforce(pv, mass)
                return e
        
        # كيان جديد
        entity = Entity(name, pv, mass, context_intent)
        self.entities.append(entity)
        
        if len(self.entities) > self.max_entities:
            # إزالة الأقدم
            self.entities.sort(key=lambda e: e.age)
            self.entities.pop(0)
        
        return entity
    
    def resolve_pronoun(self, pronoun, context_pv=None):
        """حل ضمير إلى أقرب كيان في فضاء الطور.
        
        إذا وُجد context_pv، نرجّح الكيانات المتوافقة مع السياق.
        """
        if pronoun not in _PRONOUNS:
            return None
        
        if not self.entities:
            return None
        
        # نحذف الكيانات الميتة (كبرت جداً)
        alive = [e for e in self.entities 
                 if np.exp(-self.decay * e.age) > 0.05]
        if not alive:
            alive = self.entities[-3:]
        
        best = None
        best_score = -1.0
        
        for e in alive:
            score = e.mass * np.exp(-self.decay * e.age)
            if context_pv is not None:
                context_align = float(np.mean(
                    np.cos(e.pv[:PHASE_DIM] - context_pv[:PHASE_DIM])))
                score *= max(0.0, context_align)
            if e.name in pronoun or pronoun in e.name:
                score *= 1.5
            if score > best_score:
                best_score = score
                best = e
        
        return best
    
    def entity_gravity(self, word, word_pv):
        """جاذبية كيان — هل تشير الكلمة إلى كيان مسجل؟
        
        للضمائر: نعيد أقوى جاذبية من الكيان المحلول.
        للأسماء: نعيد التشابه مع أقرب كيان.
        """
        if not self.entities or word_pv is None:
            return 0.0
        
        if word in _PRONOUNS:
            resolved = self.resolve_pronoun(word)
            if resolved:
                return float(np.mean(
                    np.cos(resolved.pv[:PHASE_DIM] - word_pv[:PHASE_DIM])))
        
        # لأي كلمة: هل تشبه أحد الكيانات المسجلة؟
        best = 0.0
        for e in self.entities:
            sim = float(np.mean(np.cos(e.pv[:PHASE_DIM] - word_pv[:PHASE_DIM])))
            if sim > best:
                best = sim
        return max(0.0, best)
    
    def get_entity_context_pv(self):
        """متجه طوري مرجح لجميع الكيانات الحية."""
        alive = [e for e in self.entities 
                 if np.exp(-self.decay * e.age) > 0.05]
        if not alive:
            return None
        weights = np.array([e.mass * np.exp(-self.decay * e.age) 
                           for e in alive])
        weights = weights / (weights.sum() + 1e-10)
        pvs = np.array([e.pv for e in alive])
        weighted = np.average(pvs, axis=0, weights=weights)
        nrm = np.linalg.norm(weighted)
        if nrm > 1e-10:
            weighted = weighted / nrm
        return weighted
    
    def recent_entities(self, n=5):
        """آخر n كيان (للإبلاغ)."""
        if not self.entities:
            return []
        sorted_e = sorted(self.entities, key=lambda e: e.age)
        return [(e.name, round(e.mass, 2), e.age) for e in sorted_e[:n]]
    
    def _age_all(self):
        for e in self.entities:
            e.age += 1


# ضمائر العربية مع تصنيفها
PRONOUN_GENDERS = {
    'هو': 'm', 'هي': 'f', 'هم': 'mp', 'هن': 'fp', 'هما': 'md',
    'هذا': 'm', 'هذه': 'f', 'ذلك': 'm', 'تلك': 'f',
    'الذي': 'm', 'التي': 'f', 'الذين': 'mp', 'اللواتي': 'fp',
}
