"""AssociativeDialogueMemory — ذاكرة ترابطية طورية للحوار.

تخزن (intent + topic) → response_target كأزواج طورية.
عند ورود جملة جديدة، يتم كشف القصد + topic → البحث عن أقرب مسار →
استرجاع target_phase للرد المخطط.

المبدأ:
- Key = combine(intent_phase, topic_phase) ← متجه 56D
- Value = response_phase المتوقعة
- تخزين وتحديث عبر الاحتكاك الطوري (Phase Hebbian)
"""
import numpy as np
import logging
from collections import deque
from src.physics.constants import PHASE_DIM, TOTAL_DIM
from src.physics.word_physics import compute_extended_phase_vector, compute_word_mass
from src.physics.intent_detector import IntentDetector

logger = logging.getLogger(__name__)


class AssociativeEntry:
    """سجل ترابطي واحد: key → value + تاريخ."""
    
    def __init__(self, key_pv, value_pv, intent_name, topic_text="", age=0):
        self.key_pv = key_pv
        self.value_pv = value_pv
        self.intent = intent_name
        self.topic = topic_text[:40]
        self.age = age
        self.strength = 1.0
        self.hits = 1
    
    def reinforce(self, value_pv):
        """تقوية الرابط عند تكرار (intent, topic)."""
        self.hits += 1
        self.strength = min(5.0, self.strength + 0.5)
        blend = 0.3
        self.value_pv = (1 - blend) * self.value_pv + blend * value_pv
        nrm = np.linalg.norm(self.value_pv)
        if nrm > 1e-10:
            self.value_pv = self.value_pv / nrm


class AssociativeDialogueMemory:
    """ذاكرة ترابطية طورية.
    
    تتعلم: (intent, topic) → response_phase
    تسترجع: أقرب key مطابق → target phase للرد
    """
    
    def __init__(self, max_entries=100, min_similarity=0.6):
        self.max_entries = max_entries
        self.min_similarity = min_similarity
        self.entries = []
        self.intent_detector = IntentDetector()
        self._last_matched = {}  # intent_name → entry (آخر دخول تمت مطابقته)
        self._seed_entries()
    
    def _seed_entries(self):
        """بذر أولي: روابط افتراضية لكل intent.
        
        هذا يعطي النظام توجيهاً أولياً حتى يتعلم من الحوار الحقيقي.
        """
        seeds = {
            "GREETING": "وعليكم السلام ورحمة الله وبركاته",
            "QUESTION": "السؤال جيد، الجواب هو",
            "COMMAND": "سأفعل ذلك حالاً",
            "REQUEST": "بالطبع، تفضل",
            "FAREWELL": "في أمان الله، إلى اللقاء",
            "STATEMENT": "نعم، هذا صحيح",
        }
        for intent, response in seeds.items():
            intent_pv = self.intent_detector.get_intent_pv(intent)
            if intent_pv is None:
                continue
            key_pv = intent_pv.copy()
            response_pv = self._phrase_pv(response)
            if response_pv is None:
                continue
            entry = AssociativeEntry(key_pv, response_pv, intent, topic_text="[seed]")
            self.entries.append(entry)
    
    def _phrase_pv(self, phrase):
        words = phrase.split()
        if not words:
            return None
        pvs = []
        for w in words:
            try:
                pv = compute_extended_phase_vector(w)
                pvs.append(pv)
            except Exception:
                continue
        if not pvs:
            return None
        avg = np.mean(pvs, axis=0)
        nrm = np.linalg.norm(avg)
        if nrm > 1e-10:
            avg = avg / nrm
        return avg
    
    def _make_key(self, intent_pv, topic_pv):
        """دمج intent + topic في مفتاح طوري واحد."""
        if topic_pv is not None:
            combined = 0.6 * intent_pv + 0.4 * topic_pv
        else:
            combined = intent_pv.copy()
        nrm = np.linalg.norm(combined)
        if nrm > 1e-10:
            combined = combined / nrm
        return combined
    
    def store(self, intent_name, topic_pv, response_text):
        """تخزين رابط جديد: (intent, topic) → response."""
        intent_pv = self.intent_detector.get_intent_pv(intent_name)
        if intent_pv is None:
            return
        
        key_pv = self._make_key(intent_pv, topic_pv)
        response_pv = self._phrase_pv(response_text)
        if response_pv is None:
            return
        
        # هل يوجد رابط مشابه؟
        for entry in self.entries:
            sim = float(np.mean(np.cos(key_pv[:PHASE_DIM] - entry.key_pv[:PHASE_DIM])))
            if sim > self.min_similarity:
                entry.reinforce(response_pv)
                return
        
        # رابط جديد
        entry = AssociativeEntry(key_pv, response_pv, intent_name, topic_text="")
        self.entries.append(entry)
        
        if len(self.entries) > self.max_entries:
            # إزالة الأضعف
            self.entries.sort(key=lambda e: e.strength)
            self.entries.pop(0)
    
    def retrieve(self, intent_name, topic_pv):
        """استرجاع target phase للرد بناءً على (intent, topic).
        
        Returns:
            target_pv: متجه طوري هدف للرد
            confidence: 0.0–1.0
            matched_intent: اسم الـ intent المطابق
        """
        intent_pv = self.intent_detector.get_intent_pv(intent_name)
        if intent_pv is None:
            return None, 0.0, None
        
        query_key = self._make_key(intent_pv, topic_pv)
        
        best_entry = None
        best_sim = -1.0
        
        for entry in self.entries:
            sim = float(np.mean(np.cos(query_key[:PHASE_DIM] - entry.key_pv[:PHASE_DIM])))
            weighted = sim * min(entry.strength, 3.0) / 3.0
            if weighted > best_sim:
                best_sim = weighted
                best_entry = entry
        
        if best_entry is None or best_sim < 0.3:
            # fallback: استخدم target الافتراضي من intent_detector
            fallback = self.intent_detector.get_default_response_target(intent_name)
            if fallback is not None:
                best_entry = AssociativeEntry(query_key, fallback, intent_name)
                best_sim = 0.5
        
        if best_entry is None:
            return None, 0.0, None
        
        # تتبع آخر مطابقة لكل intent
        self._last_matched[intent_name] = best_entry
        
        confidence = max(0.0, (best_sim + 1.0) / 2.0)
        return best_entry.value_pv, round(confidence, 4), best_entry.intent
    
    def reinforce_from_generation(self, intent_name, topic_pv, generated_text):
        """تعزيز الذاكرة من التوليد الفعلي (تعلم مستمر)."""
        intent_pv = self.intent_detector.get_intent_pv(intent_name)
        if intent_pv is None or not generated_text:
            return
        key_pv = self._make_key(intent_pv, topic_pv)
        response_pv = self._phrase_pv(generated_text)
        if response_pv is None:
            return
        
        for entry in self.entries:
            sim = float(np.mean(np.cos(key_pv[:PHASE_DIM] - entry.key_pv[:PHASE_DIM])))
            if sim > self.min_similarity:
                entry.reinforce(response_pv)
                return
        
        entry = AssociativeEntry(key_pv, response_pv, intent_name, "")
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.sort(key=lambda e: e.strength)
            self.entries.pop(0)

    def penalize(self, intent_name, factor=0.5):
        """معاقبة آخر رابط تمت مطابقته لـ intent معين.
        
        عند تكرار المستخدم لقصده (repeat)، نخفض strength
        الرابط السابق لدفع النظام لاقتراح رد مختلف.
        """
        entry = self._last_matched.get(intent_name)
        if entry is None:
            return
        entry.strength *= factor
    
    def get_stats(self):
        count_by_intent = {}
        for e in self.entries:
            count_by_intent[e.intent] = count_by_intent.get(e.intent, 0) + 1
        return {
            "entries": len(self.entries),
            "by_intent": count_by_intent,
        }
