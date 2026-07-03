"""DialogueMemory — ذاكرة حوار طورية لمِرنان.

تخزين أدوار الحوار كمسارات طورية في فضاء 56D.
بدلاً من حفظ النصوص حرفياً، تحفظ كل جملة كـ"حالة طورية"
تتفاعل مع الحالات السابقة عبر الرنين الطوري والجاذبية الحوارية.

المبادئ الفيزيائية:
- كل جملة = مسار في فضاء الطور (Phase Trajectory)
- الحوار = حقل متشابك من المسارات (Entangled Trajectory Field)
- القفز بين المواضيع = انهيار إنتروبي ثم إعادة استقرار
- تماسك الحوار = محصلة جاذبية المسارات السابقة
"""
import numpy as np
import logging
from collections import deque
from src.physics.constants import TOTAL_DIM, PHASE_DIM
from src.physics.word_physics import compute_extended_phase_vector, compute_word_mass
from src.physics.word_spectrum import compute_word_spectrum, spectral_resonance, spectral_density

logger = logging.getLogger(__name__)


class DialogueTurn:
    """دور حوار واحد — جملة كاملة محفوظة كحالة فيزيائية."""
    
    def __init__(self, text, pv, spectrum, mass, speaker="user"):
        self.text = text
        self.pv = pv                    # متجه طوري 56D يمثل الجملة
        self.spectrum = spectrum        # متجه طيفي للجملة
        self.mass = mass                # الكتلة الدلالية الكلية للجملة
        self.speaker = speaker
        self.age = 0
        self.entropy = 0.0
    
    def tick(self):
        self.age += 1
    
    def decayed_pv(self, decay=0.05):
        factor = np.exp(-decay * self.age)
        return self.pv * factor if factor > 0.01 else None


class DialogueMemory:
    """ذاكرة حوار تعتمد على الفضاء الطوري والتضاؤل الزمني.
    
    تحافظ على آخر N جملة كحالات طورية.
    كل جملة جديدة تتردد مع الماضي، ويؤثر الرنين الطوري
    على اختيار الكلمات التالية عبر حقل جاذبية حوارية.
    """
    
    def __init__(self, max_turns=20, decay=0.05, topic_threshold=0.3,
                 summarize_every=5):
        self.turns = deque(maxlen=max_turns)
        self.decay = decay
        self.topic_threshold = topic_threshold
        self.current_topic_pv = None      # متجه الطور الحالي للموضوع
        self.current_topic_spectrum = None # متجه الطيف الحالي للموضوع
        self.topic_history = []           # تاريخ المواضيع للكشف عن القفزات
        self.summaries = deque(maxlen=5)  # ملخصات طويلة المدى
        self._summarize_every = summarize_every
        self._turn_count = 0
    
    def compress_turn(self, words):
        """ضغط جملة كاملة إلى متجه طوري واحد 56D وطيف.
        
        العملية:
        1. حساب PV لكل كلمة
        2. المتوسط المرجح بالكتلة (mass-weighted average) كحالة طورية للجملة
        3. تحليل طيفي باستخدام FFT على تسلسل PVs
        """
        if not words:
            return None, None, 0.0
        
        pvs = []
        masses = []
        for w in words:
            try:
                pv = compute_extended_phase_vector(w)
                mass = compute_word_mass(w)
                pvs.append(pv)
                masses.append(mass)
            except Exception:
                continue
        
        if not pvs:
            return None, None, 0.0
        
        pvs_arr = np.array(pvs)
        masses_arr = np.array(masses) + 0.1
        weights = masses_arr / masses_arr.sum()
        
        turn_pv = np.average(pvs_arr, axis=0, weights=weights)
        nrm = np.linalg.norm(turn_pv)
        if nrm > 1e-10:
            turn_pv = turn_pv / nrm
        
        total_mass = float(masses_arr.sum())
        
        # الطيف: نحسب FFT على تسلسل PVs للجملة
        n = max(len(pvs), 3)
        if len(pvs) < n:
            pad = [pvs[-1]] * (n - len(pvs))
            pvs_padded = pvs + pad
        else:
            pvs_padded = pvs
        fft_vals = np.fft.rfft(np.array(pvs_padded), axis=0)
        spectrum = np.concatenate([
            np.mean(np.abs(fft_vals), axis=1),
            np.mean(np.angle(fft_vals), axis=1),
        ])
        
        return turn_pv, spectrum, total_mass
    
    def update(self, text, speaker="system"):
        """إضافة جملة جديدة للذاكرة بعد تحويلها فيزيائياً.
        
        1. تحليل الجملة إلى مكوناتها الفيزيائية
        2. تخزينها كـ DialogueTurn
        3. تحديث حالة الموضوع الحالي
        4. الكشف عن قفزات المواضيع عبر تغير الإنتروبيا
        """
        words = text.split()
        if not words:
            return
        
        turn_pv, spectrum, mass = self.compress_turn(words)
        if turn_pv is None:
            return
        
        turn = DialogueTurn(text, turn_pv, spectrum, mass, speaker)
        
        # قياس تماسك هذا الدور مع الموضوع الحالي
        if self.current_topic_pv is not None and len(self.turns) > 0:
            coherence = float(np.mean(np.cos(turn_pv[:PHASE_DIM] - self.current_topic_pv[:PHASE_DIM])))
            turn.entropy = 1.0 - coherence
            if turn.entropy > self.topic_threshold:
                logger.debug(f"Topic shift detected: entropy={turn.entropy:.3f}")
                self.topic_history.append({
                    'turn': len(self.turns),
                    'text': text[:50],
                    'old_topic_pv': self.current_topic_pv.copy(),
                    'entropy': turn.entropy,
                })
        
        # تحديث الموضوع الحالي (متوسط متحرك)
        if self.current_topic_pv is None:
            self.current_topic_pv = turn_pv.copy()
            self.current_topic_spectrum = spectrum.copy() if spectrum is not None else None
        else:
            decay = 0.7  # وزن الدور الحالي
            self.current_topic_pv = decay * turn_pv + (1 - decay) * self.current_topic_pv
            nrm = np.linalg.norm(self.current_topic_pv)
            if nrm > 1e-10:
                self.current_topic_pv = self.current_topic_pv / nrm
        
        self.turns.append(turn)
        self._turn_count += 1
        
        # تلخيص دوري — كل N دورة
        if self._turn_count % self._summarize_every == 0:
            self._summarize()
        
        # شيخوخة باقي الأدوار
        for t in self.turns:
            if t is not turn:
                t.tick()
        for s in self.summaries:
            s['age'] += 1
    
    def _summarize(self):
        """ضغط أقدم الأدوار في ملخص طوري واحد.
        
        يؤخذ نصف الأدوار الأقدم، يحسب متوسطها الموزون بالكتلة،
        ويخزن كملخص في الذاكرة البعيدة المدى.
        """
        if len(self.turns) < 3:
            return
        
        turns_list = list(self.turns)
        half = len(turns_list) // 2
        old_turns = turns_list[:half]
        
        pvs = [t.pv for t in old_turns if t.pv is not None]
        texts = [t.text for t in old_turns if t.text]
        
        if not pvs:
            return
        
        masses = [t.mass for t in old_turns if t.pv is not None]
        masses = np.array(masses) + 0.1
        weights = masses / masses.sum()
        
        summary_pv = np.average(np.array(pvs), axis=0, weights=weights)
        nrm = np.linalg.norm(summary_pv)
        if nrm > 1e-10:
            summary_pv = summary_pv / nrm
        
        summary_text = ' | '.join(texts[:5])  # أول 5 نصوص فقط
        self.summaries.append({
            'pv': summary_pv,
            'text': summary_text[:100],
            'age': 0,
            'n_turns': len(old_turns),
        })
        logger.debug(f"Summarized {len(old_turns)} turns into dialogue summary")
    
    def get_context_pv(self):
        """الحصول على متجه الطور الموزون للسياق الحواري.
        
        الجمع الموزون لـ:
        - الأدوار الحية (الأخيرة) بوزن أعلى
        - الملخصات (البعيدة المدى) بوزن أقل
        """
        if not self.turns:
            return None
        
        weighted_pv = np.zeros(TOTAL_DIM)
        total_weight = 0.0
        
        for turn in self.turns:
            dpv = turn.decayed_pv(self.decay)
            if dpv is None:
                continue
            weight = float(np.exp(-self.decay * turn.age))
            weighted_pv += weight * dpv
            total_weight += weight
        
        # إضافة الملخصات البعيدة المدى (وزن أقل)
        for s in self.summaries:
            s_pv = s['pv'] * np.exp(-self.decay * s['age'])
            s_weight = 0.3 * np.exp(-self.decay * s['age'])
            weighted_pv += s_weight * s_pv
            total_weight += s_weight
        
        if total_weight > 1e-10:
            weighted_pv = weighted_pv / total_weight
            nrm = np.linalg.norm(weighted_pv)
            if nrm > 1e-10:
                weighted_pv = weighted_pv / nrm
            return weighted_pv
        return None
    
    def dialogue_gravity(self, word, word_pv):
        """حساب جاذبية الحوار باتجاه كلمة معينة.
        
        تقيس الانسجام بين الكلمة المرشحة ومتجه الحوار الموزون.
        قيم عالية ← الكلمة منسجمة مع سياق الحوار.
        """
        ctx_pv = self.get_context_pv()
        if ctx_pv is None or word_pv is None:
            return 0.0
        align = float(np.mean(np.cos(word_pv[:PHASE_DIM] - ctx_pv[:PHASE_DIM])))
        return max(0.0, align)
    
    def spectral_context_resonance(self, word):
        """الرنين الطيفي بين الكلمة والذاكرة الحوارية.
        
        يحسب متوسط spectral_resonance بين الكلمة وآخر 3 أدوار حوار.
        """
        if not self.turns:
            return 0.0
        
        recent = list(self.turns)[-3:]
        resonances = []
        for turn in recent:
            if turn.spectrum is not None:
                r = spectral_resonance(word, turn.text.split()[0] if turn.text.split() else word)
                resonances.append(r)
        
        return float(np.mean(resonances)) if resonances else 0.0
    
    def topic_stability(self):
        """استقرار الموضوع الحالي — قياس تشتت الطيف عبر الأدوار الأخيرة.
        
        كلما انخفضت القيمة ← الموضوع مستقر.
        كلما ارتفعت ← الحوار يقفز بين مواضيع.
        """
        if len(self.turns) < 3:
            return 1.0
        
        recent_text = ' '.join([t.text for t in list(self.turns)[-5:]])
        words = recent_text.split()
        if len(words) < 3:
            return 1.0
        
        density = spectral_density(words)
        # density عالية = تنوع طيفي = موضوع غير مستقر
        return min(1.0, density * 2.0)
    
    def recent_topics(self, n=3):
        """آخر n موضوعات (للجمل التوليدية)."""
        if self.topic_history:
            return self.topic_history[-n:]
        return []
    
    def is_repeat(self, text, threshold=0.75, max_lookback=5):
        """هل النص مكرر لآخر أدوار المستخدم؟
        
        يقارن متجه النص المدخل مع آخر أدوار المستخدم.
        إذا تجاوز التشابه threshold ← repeat محتمل.
        """
        words = text.split()
        if not words:
            return False
        
        pv, _, _ = self.compress_turn(words)
        if pv is None:
            return False
        
        recent_user = [t for t in list(self.turns)[-max_lookback:] 
                       if t.speaker == "user"]
        for turn in recent_user:
            sim = float(np.mean(np.cos(pv[:PHASE_DIM] - turn.pv[:PHASE_DIM])))
            if sim > threshold:
                return True
        
        return False
    
    def get_summary(self):
        """تقرير قصير عن حالة الذاكرة الحوارية."""
        if not self.turns:
            return {"turns": 0, "topic_pv": None, "stability": 1.0}
        
        last_text = self.turns[-1].text[:60] if self.turns[-1].text else ""
        return {
            "turns": len(self.turns),
            "last_turn": last_text,
            "topic_present": self.current_topic_pv is not None,
            "stability": round(self.topic_stability(), 3),
            "shifts": len(self.topic_history),
        }
