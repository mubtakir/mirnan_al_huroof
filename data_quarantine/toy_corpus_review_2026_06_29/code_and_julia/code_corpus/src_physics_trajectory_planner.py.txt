# -*- coding: utf-8 -*-
"""مخطط المسار الطوري — تخطيط متعدد المراحل في فضاء الطور.

الفلسفة: كرة تتدحرج في وعاء — تسلك المسار الأقل مقاومة بين
نقطة البداية والهدف. لكن المسار ليس خطاً مستقيماً:
يمر عبر محطات وسيطة (معالم) تمثل بنية الجملة.

الفرق عن chain-of-thought:
- LLM يخطط بالنص (كلمات واضحة) — نحن نخطط بالطور (متجهات هندسية)
- لا نولّد "دعني أفكر" — بل نرسم مساراً في الفضاء 64D
- المعالم ليست كلمات بل مناطق في فضاء الطور

المعالم:
1. opening: منطقة البداية — تتناسب مع قصد المستخدم
2. development: منطقة التطور — توسيع الفكرة
3. elaboration: منطقة التفصيل — أمثلة وتوضيح
4. closing: منطقة الختام — تلخيص أو نتيجة

لكل معلم:
- target_pv: المتجه المستهدف
- tightness: شدة التقريب (0.2 = حرية، 1.0 = تقييد)
- phase: موقع في دورة الجملة (0.0→1.0)
"""
import numpy as np
import logging
from src.physics.word_physics import phase_similarity
from src.physics.constants import TOTAL_DIM

logger = logging.getLogger(__name__)


class TrajectoryMilestone:
    """معلم في مسار الطور."""

    def __init__(self, name, target_pv, tightness, phase_position, description=""):
        self.name = name
        self.target_pv = target_pv
        self.tightness = tightness
        self.phase_position = phase_position
        self.description = description


class TrajectoryPlanner:
    """مخطط المسار الطوري — يخطط مسار التوليد في فضاء 64D."""

    INTENT_MILESTONES = {
        'GREETING': [
            ('opening', 0.4, 'تحية واردة'),
            ('response', 0.7, 'رد التحية'),
            ('warmth', 0.3, 'دفء إضافي'),
        ],
        'QUESTION': [
            ('opening', 0.3, 'مقدمة مباشرة'),
            ('elaboration', 0.5, 'توضيح'),
            ('detail', 0.6, 'تفصيل'),
            ('closing', 0.4, 'خلاصة'),
        ],
        'COMMAND': [
            ('acknowledgment', 0.6, 'إقرار بالطلب'),
            ('execution', 0.7, 'تنفيذ'),
            ('result', 0.5, 'نتيجة'),
        ],
        'STATEMENT': [
            ('acknowledgment', 0.4, 'إقرار'),
            ('expansion', 0.5, 'توسيع'),
            ('insight', 0.3, 'بصيرة'),
        ],
        'OPINION': [
            ('engagement', 0.4, 'تفاعل'),
            ('nuance', 0.5, 'تفصيل ورأي'),
            ('synthesis', 0.6, 'تركيب'),
        ],
        'DEFAULT': [
            ('opening', 0.3, 'بداية'),
            ('development', 0.5, 'تطور'),
            ('closing', 0.4, 'ختام'),
        ],
    }

    def __init__(self, config=None):
        cfg = config or {}
        self.min_milestones = cfg.get('trajectory_min_milestones', 2)
        self.max_milestones = cfg.get('trajectory_max_milestones', 5)
        self.opening_tightness = cfg.get('trajectory_opening_tightness', 0.2)
        self.closing_tightness = cfg.get('trajectory_closing_tightness', 1.0)
        self.current_milestones = []
        self.current_step = 0

    def plan_trajectory(self, prompt_pv, intent='DEFAULT', max_words=12, dialogue_memory=None):
        """تخطيط مسار الطور من البداية إلى النهاية.

        Args:
            prompt_pv: متجه أو متجهات المدخل
            intent: القصد المكتشف
            max_words: عدد الكلمات المتوقع
            dialogue_memory: ذاكرة الحوار (اختياري)

        Returns:
            list[TrajectoryMilestone]: معالم المسار
        """
        if isinstance(prompt_pv, list):
            if len(prompt_pv) > 0:
                start_pv = np.mean([p[:TOTAL_DIM] for p in prompt_pv], axis=0)
            else:
                start_pv = np.zeros(TOTAL_DIM)
        else:
            start_pv = prompt_pv[:TOTAL_DIM].copy()

        milestone_defs = self.INTENT_MILESTONES.get(intent, self.INTENT_MILESTONES['DEFAULT'])
        n_milestones = min(len(milestone_defs), max(2, max_words // 4))
        milestone_defs = milestone_defs[:n_milestones]

        milestones = []

        for i, (name, tightness, desc) in enumerate(milestone_defs):
            phase_pos = (i + 1) / len(milestone_defs)

            if i == 0:
                target_pv = start_pv.copy()
                t = self.opening_tightness
            elif i == len(milestone_defs) - 1:
                rng = np.random.RandomState(42 + i)
                target_pv = start_pv * (0.3 + 0.2 * i) + rng.randn(TOTAL_DIM) * 0.1
                t = self.closing_tightness
            else:
                rng = np.random.RandomState(42 + i)
                blend = 0.4 + 0.15 * i
                target_pv = start_pv * (1 - blend) + rng.randn(TOTAL_DIM) * 0.15 * blend
                t = tightness

            pv_norm = np.linalg.norm(target_pv)
            if pv_norm > 1e-10:
                target_pv /= pv_norm

            milestones.append(TrajectoryMilestone(
                name=name,
                target_pv=target_pv,
                tightness=t,
                phase_position=phase_pos,
                description=desc,
            ))

        if dialogue_memory and hasattr(dialogue_memory, 'get_context_pv'):
            context_pv = dialogue_memory.get_context_pv()
            if context_pv is not None and len(milestones) > 1:
                context_norm = np.linalg.norm(context_pv[:TOTAL_DIM])
                if context_norm > 1e-10:
                    for m in milestones[1:]:
                        blend = 0.3
                        m.target_pv = m.target_pv * (1 - blend) + context_pv[:TOTAL_DIM] * blend
                        pv_norm = np.linalg.norm(m.target_pv)
                        if pv_norm > 1e-10:
                            m.target_pv /= pv_norm

        self.current_milestones = milestones
        self.current_step = 0
        return milestones

    def get_milestone_for_step(self, step, max_steps):
        """الحصول على المعلم المناسب لخطوة التوليد الحالية.

        Args:
            step: رقم الخطوة (0-based)
            max_steps: إجمالي عدد الخطوات

        Returns:
            TrajectoryMilestone: المعلم الحالي
        """
        if not self.current_milestones:
            default = TrajectoryMilestone(
                name='default',
                target_pv=np.zeros(TOTAL_DIM),
                tightness=0.5,
                phase_position=0.5,
            )
            return default

        progress = step / max(1, max_steps)

        milestone_idx = min(
            int(progress * len(self.current_milestones)),
            len(self.current_milestones) - 1
        )

        if milestone_idx < len(self.current_milestones) - 1:
            next_m = self.current_milestones[milestone_idx + 1]
            curr_m = self.current_milestones[milestone_idx]
            local_progress = (progress * len(self.current_milestones)) - milestone_idx

            blended_pv = curr_m.target_pv * (1 - local_progress) + next_m.target_pv * local_progress
            blended_tightness = curr_m.tightness * (1 - local_progress) + next_m.tightness * local_progress

            return TrajectoryMilestone(
                name=f"{curr_m.name}→{next_m.name}",
                target_pv=blended_pv,
                tightness=blended_tightness,
                phase_position=progress,
                description=f"انتقال من {curr_m.description} إلى {next_m.description}",
            )

        return self.current_milestones[milestone_idx]

    def compute_trajectory_score(self, word_pv, step, max_steps):
        """تقييم مدى توافق كلمة مع المسار في الخطوة الحالية.

        Args:
            word_pv: متجه الكلمة المرشحة
            step: رقم الخطوة
            max_steps: إجمالي الخطوات

        Returns:
            float: درجة التوافق مع المسار
        """
        milestone = self.get_milestone_for_step(step, max_steps)

        if milestone.target_pv is None or np.linalg.norm(milestone.target_pv) < 1e-10:
            return 0.0

        align = phase_similarity(word_pv[:22], milestone.target_pv[:22])
        score = align * milestone.tightness

        return score

    def reset(self):
        """إعادة تعيين المسار."""
        self.current_milestones = []
        self.current_step = 0