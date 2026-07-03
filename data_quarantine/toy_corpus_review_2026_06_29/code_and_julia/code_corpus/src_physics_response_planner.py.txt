"""ResponsePlanner — تخطيط الرد قبل التوليد.

يحدد هدفاً طورياً للرد كاملاً قبل البدء بالتوليد.
الهدف مشتق من: intent + topic + entities + associative_memory.

ينتج:
- target_pv: متجه طوري 56D يمثل الرد المخطط
- tightening: جدول تشدد لكل خطوة — أول الكلمات حرة، آخرها مقيدة بالهدف
"""
import numpy as np
import logging
from src.physics.constants import TOTAL_DIM, PHASE_DIM
from src.physics.word_physics import compute_extended_phase_vector

logger = logging.getLogger(__name__)


class ResponsePlanner:
    """يخطط للرد بأسره قبل التوليد.
    
    الهدف: بدلاً من التوليد كلمة كلمة بدون توجّه،
    نحدد target_pv للرد كاملاً ونسير باتجاهه.
    """
    
    def __init__(self, intent_detector, associative_memory, entity_register):
        self.intent_detector = intent_detector
        self.assoc_mem = associative_memory
        self.entity_reg = entity_register
        self._current_plan = None
    
    def plan(self, user_text, topic_pv=None, max_words=8):
        """تخطيط الرد بناءً على مدخل المستخدم.
        
        Returns dict:
            target_pv: متجه طوري 56D للرد المخطط
            intent: القصد المكتشف
            confidence: ثقة الرابط الترابطي (0–1)
            tightening: مصفوفة تشدد لكل خطوة (0.0–1.0)
            entities: list of entity names involved
        """
        intent_result = self.intent_detector.detect(user_text)
        intent_name = intent_result['intent']
        
        # 1. استرجاع target من AssociativeMemory
        plan_pv, confidence, matched_intent = self.assoc_mem.retrieve(
            intent_name, topic_pv)
        
        # 2. تعديل الهدف بالكيانات الحية
        entity_pv = self.entity_reg.get_entity_context_pv()
        if entity_pv is not None and plan_pv is not None:
            blend = 0.3
            plan_pv = (1 - blend) * plan_pv + blend * entity_pv
            nrm = np.linalg.norm(plan_pv)
            if nrm > 1e-10:
                plan_pv = plan_pv / nrm
        elif plan_pv is None:
            # Fallback: استخدم الـ intent_pv + entity
            intent_pv = self.intent_detector.get_intent_pv(intent_name)
            if entity_pv is not None and intent_pv is not None:
                plan_pv = 0.6 * intent_pv + 0.4 * entity_pv
                nrm = np.linalg.norm(plan_pv)
                if nrm > 1e-10:
                    plan_pv = plan_pv / nrm
            elif intent_pv is not None:
                plan_pv = intent_pv.copy()
        
        if plan_pv is None:
            return None
        
        # 3. جدول التشدد: كلما تقدّمنا في التوليد، زاد التشدد
        # الخطوة 0: تشدد 0.2 (حرية عالية في اختيار أول كلمة)
        # الخطوة الأخيرة: تشدد 1.0 (يجب أن تطابق الهدف)
        tight_start = 0.2
        tight_end = 1.0
        if max_words <= 1:
            tightening = np.array([1.0])
        else:
            tightening = np.linspace(tight_start, tight_end, max_words)
        
        # الكيانات المستخرجة من مدخل المستخدم
        entities = self.entity_reg.extract_entities(user_text, intent_name)
        
        self._current_plan = {
            'target_pv': plan_pv,
            'intent': intent_name,
            'confidence': confidence,
            'tightening': tightening,
            'entities': entities,
        }
        
        return self._current_plan
    
    @property
    def current_plan(self):
        return self._current_plan
    
    def get_target_for_step(self, step):
        """إرجاع target_pv مع تشدد للخطوة الحالية."""
        if self._current_plan is None:
            return None, 0.0
        tight = self._current_plan['tightening']
        if step < len(tight):
            t = tight[step]
        else:
            t = 1.0
        return self._current_plan['target_pv'], t
    
    def get_intent(self):
        if self._current_plan is None:
            return "STATEMENT"
        return self._current_plan['intent']
