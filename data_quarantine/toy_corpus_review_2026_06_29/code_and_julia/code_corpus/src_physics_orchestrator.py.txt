"""PhysicsOrchestrator — موحّد فيزيائي مركزي.

يدير كل المحركات الفيزيائية ديناميكياً:
- يقرر متى يستخدم أي محرك حسب حالة الطور والإنتروبيا
- يضبط الأوزان في الزمن الحقيقي
- يراقب التوليد ويصدر تقارير فيزيائية موحّدة
- يدير الاسترخاء الرنيني (توليد متعدد الممرات)
- يدير مشهد القصد والذاكرة الهرمية وأثر التفاعل
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class PhysicsState:
    """حالة فيزيائية لحظية للتوليد."""

    def __init__(self):
        self.entropy = 0.0
        self.S_crit = 0.0
        self.k_B = 1.0
        self.beta = 2.0
        self.phase_coherence = 0.0
        self.mass_mean = 0.0
        self.mass_std = 0.0
        self.resonance_mean = 0.0
        self.temperature = 1.0
        self.step = 0
        self.dccf_coupling_strength = 0.0
        self.ppm_field_strength = 0.0
        self.amfs_centrality_mean = 0.0
        self.mode = 'standard'
        self.is_poetic = False
        self.is_math = False
        self.is_code = False
        self.is_english = False
        self.is_creative = False
        self.word_count = 0
        self.cascade_enabled = False
        self.cascade_strength = 0.0
        self.dialogue_mode = False
        self.dialogue_intent = ""
        self.dialogue_confidence = 0.0
        self.coherence = 0.0
        self.trajectory_milestone = ""
        self.interaction_trace_topics = 0
        self.heterodyne_active = 0.0
        self.carrier_active = 0.0
        self.beamform_active = 0.0
        self.refractory_active = 0.0
        self.macro_wave_active = 0.0
        self.oscillator_active = 0.0
        self.gravity_vector = 0.0
        self.thermal_noise_level = 0.0
        self.active_entanglements = 0.0
        self.formed_molecules = 0

    def to_dict(self):
        return {
            'entropy': round(self.entropy, 4),
            'S_crit': round(self.S_crit, 4),
            'k_B': round(self.k_B, 4),
            'beta': round(self.beta, 4),
            'phase_coherence': round(self.phase_coherence, 4),
            'mass_mean': round(self.mass_mean, 4),
            'mass_std': round(self.mass_std, 4),
            'resonance_mean': round(self.resonance_mean, 4),
            'temperature': round(self.temperature, 4),
            'step': self.step,
            'dccf_coupling': round(self.dccf_coupling_strength, 4),
            'ppm_field': round(self.ppm_field_strength, 4),
            'amfs_centrality': round(self.amfs_centrality_mean, 4),
            'cascade_enabled': self.cascade_enabled,
            'cascade_strength': round(self.cascade_strength, 4),
            'dialogue_mode': self.dialogue_mode,
            'dialogue_intent': self.dialogue_intent,
            'dialogue_confidence': round(self.dialogue_confidence, 4),
            'mode': self.mode,
            'is_poetic': self.is_poetic,
            'is_math': self.is_math,
            'is_code': self.is_code,
            'word_count': self.word_count,
            'coherence': round(self.coherence, 4),
            'trajectory_milestone': self.trajectory_milestone,
            'interaction_trace_topics': self.interaction_trace_topics,
            'heterodyne_active': round(self.heterodyne_active, 4),
            'carrier_active': round(self.carrier_active, 4),
            'beamform_active': round(self.beamform_active, 4),
            'refractory_active': round(self.refractory_active, 4),
            'macro_wave_active': round(self.macro_wave_active, 4),
            'oscillator_active': round(self.oscillator_active, 4),
            'gravity_vector': round(self.gravity_vector, 4),
            'thermal_noise_level': round(self.thermal_noise_level, 4),
            'active_entanglements': round(self.active_entanglements, 4),
            'formed_molecules': self.formed_molecules,
        }


class PhysicsOrchestrator:
    """الموحّد الفيزيائي — يلف Generator ويدير كل المحركات."""

    def __init__(self, generator):
        self.gen = generator
        self.state = PhysicsState()
        self._history = []
        self._mode_scores = {}
        self._last_plan = None
        self.multi_pass = None
        self._relaxation_enabled = False
        try:
            from src.physics.multi_pass_generator import MultiPassGenerator
            self.multi_pass = MultiPassGenerator(generator)
        except Exception as e:
            logger.debug(f"MultiPassGenerator غير متاح: {e}")

    def generate(self, prompt, max_words=12, mode='auto', **kwargs):
        """توليد مع اختيار الوضع تلقائياً أو يدوياً.

        Args:
            prompt: str — النص المدخل
            max_words: int — الحد الأقصى للكلمات
            mode: str — 'auto', 'standard', 'quantum', 'multiverse', 'wave', 'poetic'
            **kwargs: poetic_meter, poetic_rhyme, beta, k_B
        """
        self._detect_mode(prompt, mode)
        self._pre_generation(prompt)

        if self._relaxation_enabled and self.multi_pass is not None and self.state.mode == 'standard':
            result_data = self.multi_pass.generate(
                prompt, max_words=max_words, mode='standard',
                base_beta=self.gen.beta)
            result = result_data.get('result', '')
            self._post_generation(prompt, result)
            return result

        mode_map = {
            'quantum': self.gen._quantum_generate,
            'multiverse': self.gen._multiverse_generate,
            'wave': self.gen._wave_generate,
            'poetic': self.gen._poetic_generate,
            'code': self.gen._code_generate,
            'creative': self.gen._creative_generate,
            'math': self.gen._stepwise_generate,
        }

        if self.state.mode in mode_map:
            if self.state.mode == 'poetic':
                meter = kwargs.get('poetic_meter', 'kamil')
                rhyme = kwargs.get('poetic_rhyme', None)
                prompt_tokens = [self.gen.vocab.id2word[self.gen.vocab.get(w)]
                                 for w in prompt.split() if self.gen.vocab.get(w) is not None]
                result = mode_map['poetic'](prompt_tokens, max_words, meter=meter, rhyme=rhyme)
            else:
                result = mode_map[self.state.mode](prompt, max_words)
        else:
            result = self.gen.generate(prompt, max_words=max_words, mode=self.state.mode,
                                       **{k: v for k, v in kwargs.items()
                                          if k in ('beta', 'k_B', 'poetic_meter', 'poetic_rhyme')})

        self._post_generation(prompt, result)
        return result

    def _detect_mode(self, prompt, requested_mode):
        """كشف أفضل وضع فيزيائي للـ prompt."""
        if requested_mode != 'auto':
            self.state.mode = requested_mode
            self.state.is_poetic = (requested_mode == 'poetic')
            self.state.is_math = (requested_mode == 'math')
            self.state.is_code = (requested_mode == 'code')
            return

        math_keywords = {'حساب', 'math', 'calculate', 'solve', '+', '-', '*', '/'}
        code_keywords = {'code', 'function', 'def ', 'class ', 'import', 'print'}
        poetic_keywords = {'شعر', 'قصيدة', 'بيت', 'poem', 'verse'}
        creative_keywords = {'إبداع', 'خيال', 'تخيل', 'ابتكر', 'اكتب', 'creative', 'imagine', 'invent'}

        prompt_lower = prompt.lower()
        words = set(prompt.split())

        if words & creative_keywords:
            self.state.mode = 'creative'
            self.state.is_creative = True
        elif words & math_keywords or any(c in prompt for c in '+-*/='):
            self.state.mode = 'math'
            self.state.is_math = True
        elif words & code_keywords or 'def ' in prompt or 'class ' in prompt:
            self.state.mode = 'code'
            self.state.is_code = True
        elif words & poetic_keywords:
            self.state.mode = 'poetic'
            self.state.is_poetic = True
        elif len(prompt.split()) > 8:
            self.state.mode = 'multiverse'
        else:
            # كشف إن كان الحوار مطلوباً
            if self.gen.dialogue_engine is not None:
                is_dialogue, intent, conf = self.gen.dialogue_engine.detect_need_for_dialogue(prompt)
                if is_dialogue and conf > 0.6:
                    self.state.mode = 'dialogue'
                else:
                    self.state.mode = 'standard'
            else:
                self.state.mode = 'standard'

    def _pre_generation(self, prompt):
        """تحديث الحالة الفيزيائية قبل التوليد."""
        self.state.step = 0
        self.state.is_english = any(c.isascii() and c.isalpha() for c in prompt)

        gen = self.gen
        self.state.k_B = gen.entropy.k_B
        self.state.beta = gen.beta
        self.state.ppm_field_strength = float(np.linalg.norm(gen.prompt_field.field)) \
            if gen.prompt_field.active else 0.0
        self.state.cascade_enabled = gen.cascade_enabled
        self.state.cascade_strength = gen.cascade_layer.lambda_cascade if gen.cascade_enabled else 0.0
        self.state.dialogue_mode = (self.state.mode == 'dialogue')
        if self.state.dialogue_mode:
            result = gen.dialogue_engine.detect_need_for_dialogue(prompt)
            self.state.dialogue_intent = result[1]
            self.state.dialogue_confidence = result[2]
        else:
            self.state.dialogue_intent = ""
            self.state.dialogue_confidence = 0.0

    def _post_generation(self, prompt, result):
        """تحديث بعد التوليد — حفظ الحالة والتغذية الراجعة."""
        if not result:
            return

        gen = self.gen
        self.state.word_count = len(result.split())
        self.state.entropy = float(getattr(gen.entropy, 'S', 0.0))

        # تقييم الترابط عبر CoherenceFeedback
        if hasattr(gen, 'coherence_feedback') and result:
            words = result.split()
            if len(words) >= 3:
                report = gen.coherence_feedback.evaluate(words, gen)
                self.state.coherence = report.get('overall', 0.0)

        # تحديث مسار الطور
        if hasattr(gen, 'trajectory_planner') and gen.trajectory_planner.current_milestones:
            if gen.trajectory_planner.current_milestones:
                current = gen.trajectory_planner.get_milestone_for_step(
                    self.state.word_count, 12)
                self.state.trajectory_milestone = current.name

        # تحديث أثر التفاعل
        if hasattr(gen, 'interaction_trace'):
            self.state.interaction_trace_topics = len(gen.interaction_trace.attractors)

        if hasattr(gen, 'resonant_chain') and gen.resonant_chain:
            masses = []
            for w in result.split():
                try:
                    masses.append(gen._dyn_mass(w))
                except Exception as e:
                    logger.debug(f"تعذر حساب كتلة '{w}': {e}")
            if masses:
                self.state.mass_mean = float(np.mean(masses))
                self.state.mass_std = float(np.std(masses))

        self.state.temperature = 1.0 / max(self.state.beta, 0.1)

        self.state.heterodyne_active = float(gen.W.get('heterodyne', 0.0))
        self.state.carrier_active = float(gen.W.get('carrier', 0.0))
        self.state.beamform_active = float(gen.W.get('beamform', 0.0))
        self.state.refractory_active = float(gen.W.get('refractory', 0.0))
        self.state.macro_wave_active = float(gen.W.get('macro_wave', 0.0))
        self.state.oscillator_active = float(gen.W.get('oscillator', 0.0))
        self.state.gravity_vector = float(gen.W.get('gravity', 0.0))
        
        # استخراج مؤشرات الإبداع والتشابك
        if hasattr(gen, 'thermal_engine'):
            self.state.thermal_noise_level = gen.thermal_engine.base_noise * max(0.0, self.state.temperature - 0.1)
        
        # حساب الجزيئات المتشكلة
        if hasattr(gen, 'molecular_binder') and result:
            words = result.split()
            self.state.formed_molecules = sum(1 for w in words if '_' in w)
        self._history.append({
            'prompt': prompt,
            'result': result,
            'state': self.state.to_dict(),
        })

    def get_report(self):
        """تقرير فيزيائي شامل للحظة الحالية."""
        return self.state.to_dict()

    def get_history(self, n=5):
        """آخر n تقارير."""
        return [h for h in self._history[-n:]]

    def adjust_beta(self, val):
        """ضبط β يدوياً."""
        self.state.beta = max(0.1, min(10.0, val))
        self.gen.beta = self.state.beta

    def adjust_k_B(self, val):
        """ضبط k_B يدوياً."""
        self.state.k_B = max(0.1, min(10.0, val))
        self.gen.entropy.k_B = self.state.k_B

    def set_cascade(self, enabled: bool, lambda_cascade: float = 1.8):
        """تفعيل/تعطيل طبقة الهوي التراكمي (Potential Cascade)."""
        self.gen.set_cascade(enabled, lambda_cascade)

    def set_dialogue(self, enabled: bool):
        """تفعيل/تعطيل وضع الحوار."""
        self.gen.dialogue_mode = enabled

    def reset(self):
        """إعادة تعيين الحالة."""
        self.state = PhysicsState()
        self._history = []
        self.gen.prompt_field.reset()
        if hasattr(self.gen, 'hierarchical_memory'):
            self.gen.hierarchical_memory.reset()
        if hasattr(self.gen, 'trajectory_planner'):
            self.gen.trajectory_planner.reset()

    def set_relaxation(self, enabled: bool, n_passes: int = 3):
        """تفعيل/تعطيل توليد الاسترخاء الرنيني."""
        self._relaxation_enabled = enabled
        if self.multi_pass is not None:
            self.multi_pass.n_passes = n_passes

    def calibrate_weights(self, reference_texts):
        """معايرة أوزان التسجيل ذاتياً عبر تقليل طاقة الرنين."""
        if hasattr(self.gen, 'resonance_calibrator'):
            new_weights = self.gen.resonance_calibrator.calibrate(
                self.gen, reference_texts)
            self.gen.resonance_calibrator.apply_weights(self.gen, new_weights)
            return new_weights
        return None

    def evolve_phase_vectors(self, corpus_texts=None):
        """تشغيل التطور الطوري — تنظيم ذاتي للمتجهات."""
        if hasattr(self.gen, 'phase_evolution'):
            if corpus_texts:
                new_pv, stats = self.gen.phase_evolution.evolve_from_corpus(
                    self.gen._all_pv, corpus_texts, self.gen.vocab)
                self.gen._all_pv = new_pv
                return stats
            else:
                new_pv, stats = self.gen.phase_evolution.evolve(self.gen._all_pv)
                self.gen._all_pv = new_pv
                return stats
        return None
