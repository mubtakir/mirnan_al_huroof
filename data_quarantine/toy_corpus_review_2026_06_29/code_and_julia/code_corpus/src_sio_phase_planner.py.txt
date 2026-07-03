# -*- coding: utf-8 -*-
"""PhasePlanner — مخطط المراحل طويل المدى بمسار طوري فيزيائي.

يستخدم مسار الطور (phase trajectory) للحفاظ على تماسك الهدف
عبر مئات الخطوات دون انحراف. كل مرحلة = معلم (milestone) على المسار.

عند اكتمال مرحلة، يُسجَّل متجهها الطوري. المرحلة التالية تُقارَن
بالمتجه التراكمي للتأكد من أنها تخدم الهدف لا تنحرف عنه.

التغاير الترددي يُستخدم لكشف التبعيات بين المراحل:
  - إذا كانت مرحلة B تتناغم ترددياً مع مخرجات المرحلة A،
    فهي جاهزة للبدء بعد A.

آلية "سلاسل بناء تعيد تكوين نفسها":
  - إذا فشلت مرحلة، يُعاد تخطيطها بمعامل تصحيح طوري.
  - التصحيح = فرق الطور بين المطلوب والمتحقق.
"""

import numpy as np


class PhasePlanner:
    """يخطط وينسق تنفيذ المراحل باستخدام فيزياء الطور."""

    def __init__(self, gen=None):
        self.gen = gen
        self.phases = []
        self.current_index = 0
        self.completed_phase_vectors = []
        self.retry_counts = {}
        self.max_retries = 3

    def set_plan(self, phases: list):
        self.phases = phases
        self.current_index = 0
        self.completed_phase_vectors = []
        self.retry_counts = {p["name"]: 0 for p in phases}

    def current_phase(self) -> dict:
        if self.current_index < len(self.phases):
            return self.phases[self.current_index]
        return None

    def next_phase(self) -> dict:
        self.current_index += 1
        return self.current_phase()

    def is_complete(self) -> bool:
        return self.current_index >= len(self.phases)

    def progress(self) -> float:
        if not self.phases:
            return 1.0
        return self.current_index / len(self.phases)

    def mark_completed(self, phase_name: str, output_text: str):
        """تسجيل اكتمال مرحلة مع بصمتها الطورية."""
        pv = self._compute_output_pv(output_text)
        self.completed_phase_vectors.append({
            "phase": phase_name,
            "pv": pv,
            "text": output_text,
        })
        self.retry_counts[phase_name] = 0

    def mark_failed(self, phase_name: str, error: str) -> dict:
        """معالجة فشل مرحلة — إعادة تكوين ذاتية."""
        count = self.retry_counts.get(phase_name, 0) + 1
        self.retry_counts[phase_name] = count

        correction = {
            "retry": True,
            "attempt": count,
            "max_retries": self.max_retries,
            "phase_name": phase_name,
            "error": error,
        }

        if count >= self.max_retries:
            correction["retry"] = False
            correction["action"] = "skip"
            correction["reason"] = f"تجاوز الحد الأقصى للمحاولات ({self.max_retries})"

        return correction

    def compute_goal_coherence(self, current_output: str) -> float:
        if not self.completed_phase_vectors:
            return 1.0

        current_pv = self._compute_output_pv(current_output)
        if current_pv is None:
            return 0.5

        prev_pvs = [c["pv"] for c in self.completed_phase_vectors if c["pv"] is not None]
        if not prev_pvs:
            return 1.0

        goal_pv = np.mean(prev_pvs, axis=0)
        goal_norm = np.linalg.norm(goal_pv)
        cur_norm = np.linalg.norm(current_pv)

        if goal_norm < 1e-10 or cur_norm < 1e-10:
            return 0.5

        coherence = float(np.dot(goal_pv, current_pv) / (goal_norm * cur_norm))
        return max(-1.0, min(1.0, coherence))

    def compute_heterodyne_dependency(self, phase_a_output: str, phase_b_name: str) -> float:
        if self.gen is None or not hasattr(self.gen, 'heterodyne'):
            return 0.0
        try:
            pattern_keywords = {
                "plan": ["structure", "design", "plan", "architecture", "requirements"],
                "build": ["code", "implement", "function", "class", "build", "module"],
                "test": ["test", "verify", "assert", "check", "validate"],
                "deploy": ["deploy", "server", "config", "run", "publish", "launch"],
                "integrate": ["connect", "integrate", "wire", "combine", "merge"],
                "monitor": ["monitor", "log", "track", "measure", "observe"],
                "document": ["document", "readme", "guide", "explain", "manual"],
            }
            keywords = pattern_keywords.get(phase_b_name, [phase_b_name])
            output_words = phase_a_output.split()[:30]
            max_resonance = 0.0
            for kw in keywords:
                r = self.gen.heterodyne.score_candidate(kw, output_words)
                max_resonance = max(max_resonance, r)
            return max_resonance
        except Exception:
            return 0.0

    def _compute_output_pv(self, text: str):
        if self.gen is None or not text:
            return None
        try:
            words = text.split()[:20]
            pvs = []
            for w in words:
                pv = self.gen._get_pv_fast(w)
                if pv is not None:
                    pvs.append(pv)
            if pvs:
                return np.mean([p[:22] for p in pvs], axis=0)
        except Exception:
            pass
        return None

    def get_plan_summary(self) -> dict:
        return {
            "total_phases": len(self.phases),
            "current": self.current_index,
            "completed": len(self.completed_phase_vectors),
            "progress": self.progress(),
            "phases": [
                {
                    "name": p["name"],
                    "desc": p["desc"],
                    "tasks": p["tasks"],
                    "retries": self.retry_counts.get(p["name"], 0),
                }
                for p in self.phases
            ],
        }
