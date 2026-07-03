"""PragmaticField — الفضاء القصدي العلائقي (6D).

يكتشف أطر القصد (زمان، مكان، سبب، …) ويبني طاقة ربط
بين الكلمات لضمان التماسك الوظيفي للجملة.
مدعوم بمحرك Mustanbit لاستنتاج النوايا.
"""
import numpy as np
try:
    from src.semantics.mustanbit import AdvancedInferenceEngine
except ImportError:
    AdvancedInferenceEngine = None

class PragmaticBindingEngine:
    """
    طاقة الربط القصدي (Pragmatic Binding Energy):
    محرك يضمن "التماسك العلائقي" للجملة عبر إسقاط الكلمات في فضاء قصدي 6D.
    المرتكزات لا تدرس المعنى بل المقصد (أين، متى، لماذا، كيف).
    """
    def __init__(self):
        # المرتكزات القصدية الأساسية 6D
        # يجب أن تكون غير متعامدة تماماً لضمان انتقالات سلسة
        self.PRAGMATIC_ANCHORS = {
            "locative": np.array([0.0, 0.8, 0.2, 0.0, 0.1, 0.3]),  # أين؟ المكان
            "causal":   np.array([0.3, 0.1, 0.7, 0.0, 0.2, 0.0]),  # لماذا؟ السبب والنتيجة
            "temporal": np.array([0.1, 0.2, 0.1, 0.9, 0.0, 0.4]),  # متى؟ الزمان
            "modal":    np.array([0.0, 0.0, 0.0, 0.1, 0.9, 0.6]),  # كيف/هل؟ الحال والشرط
            "relational": np.array([0.5, 0.5, 0.1, 0.1, 0.8, 0.8]), # مَن/ماذا؟ علاقات وأدوار
            "neutral":  np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),  # مسار محايد (استمرار سردي)
        }
        
        if AdvancedInferenceEngine:
            self.mustanbit = AdvancedInferenceEngine()
        else:
            self.mustanbit = None
        
        # كلمات مفتاحية تفعل الإطارات (بسيطة ومباشرة كمؤشرات قصدية)
        self.triggers = {
            "locative": {"في", "على", "تحت", "فوق", "أين", "حيث", "هنا", "هناك", "نحو", "تجاه", "إلى"},
            "causal": {"لأن", "بسبب", "لذلك", "إذن", "علة", "لماذا", "نتيجة", "من أجل", "حتى"},
            "temporal": {"متى", "حين", "وقت", "أمس", "غداً", "يوم", "شهر", "سنة", "عندما", "بعد", "قبل"},
            "modal": {"كيف", "هل", "أ", "ماذا", "ربما", "قد", "سوف", "لن", "لا", "إن", "إذا"},
        }
        
        self.interrogative_triggers = {"لماذا", "كيف", "أين", "متى", "هل", "ماذا", "من", "أي", "كم", "why", "how", "what", "where", "when"}
        self.answer_attractors = {"بسبب", "لأن", "نتيجة", "عبر", "يعمل", "يقع", "تنهار", "السبب", "تحدث", "يتم", "because", "by", "is", "due"}
        
        # النطاقات في الفضاء الممتد (56 بُعداً)
        # 22 (phase) + 6 (extra) + 6 (syntax) + 16 (semantic) = 50
        # Pragmatic starts at index 50
        self.prag_start = 50
        self.prag_dim = 6
        
        # لضمان عدم تشبع طاقة الربط (L2 norming)
        for k, v in self.PRAGMATIC_ANCHORS.items():
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                self.PRAGMATIC_ANCHORS[k] = v / norm

    def detect_intent_frame(self, prompt_words):
        """
        تستكشف الإطار القصدي المطلوب بناءً على السياق الأخير
        تعيد (intent_frame_vector, alpha_intensity, is_interrogative)
        """
        if not prompt_words:
            return self.PRAGMATIC_ANCHORS["neutral"], 0.0, False
            
        recent = prompt_words[-4:] # ننظر لآخر 4 كلمات
        is_interrogative = any(w.lower() in self.interrogative_triggers for w in prompt_words)
        
        # استخدام المستنبط الذكي إذا كان متاحاً
        if self.mustanbit and len(prompt_words) >= 3:
            sentence_text = " ".join(prompt_words[-10:]) # تحليل آخر 10 كلمات لتقليل العبء
            inferences = self.mustanbit.analyze_sentence(sentence_text)
            if inferences:
                # إذا وجد المستنبط علاقات أو أزمنة، نوجه الفضاء القصدي بقوة
                latest_inf = inferences[-1]
                inf_type = latest_inf.get('type')
                inf_subtype = latest_inf.get('sub_type', '')
                
                if inf_type == 'temporal':
                    return self.PRAGMATIC_ANCHORS["temporal"], 0.9, is_interrogative
                elif inf_subtype == 'located_in':
                    return self.PRAGMATIC_ANCHORS["locative"], 0.9, is_interrogative
                elif inf_type in ('relational', 'role'):
                    return self.PRAGMATIC_ANCHORS["relational"], 0.95, is_interrogative
        
        frame_scores = {k: 0.0 for k in self.PRAGMATIC_ANCHORS}
        
        # الكلمة الأقرب لها وزن أعلى
        for i, word in enumerate(reversed(recent)):
            weight = 1.0 / (i + 1.0)
            for frame, trigger_set in self.triggers.items():
                if word in trigger_set:
                    frame_scores[frame] += weight
                    
        # البحث عن الإطار الغالب
        best_frame = "neutral"
        max_score = 0.0
        for frame, score in frame_scores.items():
            if score > max_score:
                max_score = score
                best_frame = frame
                
        if max_score > 0:
            alpha = min(max_score, 1.0)
            return self.PRAGMATIC_ANCHORS[best_frame], alpha, is_interrogative
            
        return self.PRAGMATIC_ANCHORS["neutral"], 0.1, is_interrogative # طاقة ربط ضعيفة للسرد المستمر

    def extract_relational_phase(self, word_pv):
        """
        تستخرج الأبعاد القصدية الـ 6 من المتجه الكلي للكلمة.
        """
        if len(word_pv) < (self.prag_start + self.prag_dim):
            return np.zeros(self.prag_dim)
        return word_pv[self.prag_start : self.prag_start + self.prag_dim]

    def compute_pragmatic_score(self, word_pv, intent_frame, alpha, word=None, is_interrogative=False):
        """
        تحسب التوافق القصدي (Pragmatic Align).
        المعادلة: pragmatic_align = mean(cos(φ_rel - intent_frame))
        طاقة الربط E_bind ستعكس هذا التوافق. كلما كان عالياً، زادت فرصة الكلمة، 
        حتى لو كان الرنين الدلالي العادي ضعيفاً.
        """
        base_boost = 0.0
        if is_interrogative and word and word in self.answer_attractors:
            base_boost = 2.0

        phi_rel = self.extract_relational_phase(word_pv)
        
        if np.linalg.norm(phi_rel) < 1e-10 or np.linalg.norm(intent_frame) < 1e-10:
            return base_boost
            
        # توافق جيوب التمام (Cosine Similarity) في الفضاء القصدي
        align = float(np.mean(np.cos(phi_rel - intent_frame)))
        
        return alpha * align + base_boost

    def phase_opposition_score(self, word_pv, context_pvs):
        """غرفة رنين قصدي تكتشف التعاكس الطوري (~180°) ككاشف للتناقض المنطقي."""
        if not context_pvs or word_pv is None:
            return 0.0
            
        # توحيد الأبعاد
        min_len = len(word_pv)
        for ctx_pv in context_pvs:
            if len(ctx_pv) < min_len:
                min_len = len(ctx_pv)
                
        # استخراج حالة الأساس (ground_state) من كامل السياق (عالمياً وليس محلياً)
        ctx_matrix = np.array([ctx[:min_len] for ctx in context_pvs])
        ground_state = np.mean(ctx_matrix, axis=0)
        
        # عزل التناقض الحقيقي عن الضجيج عبر طرح حالة الأساس
        word_clean = word_pv[:min_len] - ground_state
        
        scores = []
        for ctx_clean in (ctx_matrix - ground_state):
            # حساب متوسط |cos| للمتجهات المعزولة
            val = np.mean(np.abs(np.cos(word_clean - ctx_clean)))
            scores.append(val)
            
        if not scores:
            return 0.0
            
        avg_score = float(np.mean(scores))
        if avg_score < 0.15:
            return -1.0
        elif avg_score > 0.7:
            return 0.5
            
        return 0.0
