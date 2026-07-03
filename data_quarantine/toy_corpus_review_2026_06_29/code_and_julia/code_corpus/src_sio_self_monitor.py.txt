# -*- coding: utf-8 -*-
"""SelfMonitor — راصد ذاتي: يكتشف الانحراف عن الهدف ويعيد التوجيه.

آلية الرصد:
  1. لكل مرحلة منفذة، يُحسب متجهها الطوري.
  2. يُقارَن بالمتجه التراكمي للهدف.
  3. إذا انخفض التماسك تحت العتبة ← إنذار انحراف.
  4. يُعاد توجيه المرحلة الحالية بسياق مصحح.

لا هلوسة:
  - كل مخرج يُفحص ضد قواعد التحقق.
  - المخرجات غير الصالحة تُرفض قبل أن تصل للمستخدم.
  - الحلقة: generate → validate → fail? → diagnose → regenerate.
"""

import numpy as np


class SelfMonitor:
    """يرصد جودة المخرجات وتماسكها مع الهدف."""

    def __init__(self, gen=None, drift_threshold: float = 0.2, coherence_threshold: float = 0.4):
        self.gen = gen
        self.drift_threshold = drift_threshold
        self.coherence_threshold = coherence_threshold
        self.alerts = []
        self.corrections_applied = 0
        self.total_validations = 0

    def validate(self, output: str, phase_name: str, planner) -> dict:
        self.total_validations += 1
        result = {"accepted": True, "coherence": 1.0, "alerts": [], "corrections": []}

        if not output or len(output) < 10:
            result["accepted"] = False
            result["alerts"].append("مخرج فارغ أو قصير جداً")
            return result

        coherence = 1.0
        if planner is not None and planner.completed_phase_vectors:
            coherence = planner.compute_goal_coherence(output)

        result["coherence"] = coherence

        if coherence < self.coherence_threshold:
            result["alerts"].append(f"انخفاض التماسك مع الهدف ({coherence:.2f})")
            if coherence < self.drift_threshold:
                result["accepted"] = False
                result["alerts"].append("انحراف شديد — المخرج مرفوض")

        quality_issues = self._check_quality(output, phase_name)
        if quality_issues:
            result["alerts"].extend(quality_issues)
            if len(quality_issues) >= 3:
                result["accepted"] = False

        if not result["accepted"]:
            corrections = self._suggest_corrections(output, phase_name, planner)
            result["corrections"] = corrections
            self.corrections_applied += 1

        if result["alerts"]:
            self.alerts.append({
                "phase": phase_name,
                "alerts": result["alerts"],
                "coherence": coherence,
            })

        return result

    def compute_phase_quality(self, output: str, planner) -> float:
        if self.gen is None or planner is None:
            return 0.5
        try:
            pv = planner._compute_output_pv(output)
            if pv is None:
                return 0.5
            norm = np.linalg.norm(pv)
            if norm < 1e-10:
                return 0.0
            variance = float(np.var(pv))
            quality = 1.0 / (1.0 + variance * 10.0)
            return quality
        except Exception:
            return 0.5

        coherence = 1.0
        if planner is not None and planner.completed_phase_vectors:
            coherence = planner.compute_goal_coherence(output)

        result["coherence"] = coherence

        if coherence < self.coherence_threshold:
            result["alerts"].append(f"انخفاض التماسك مع الهدف ({coherence:.2f})")
            if coherence < self.drift_threshold:
                result["accepted"] = False
                result["alerts"].append("انحراف شديد — المخرج مرفوض")

        quality_issues = self._check_quality(output, phase_name)
        if quality_issues:
            result["alerts"].extend(quality_issues)
            if len(quality_issues) >= 3:
                result["accepted"] = False

        if not result["accepted"]:
            corrections = self._suggest_corrections(output, phase_name, planner)
            result["corrections"] = corrections
            self.corrections_applied += 1

        if result["alerts"]:
            self.alerts.append({
                "phase": phase_name,
                "alerts": result["alerts"],
                "coherence": coherence,
            })

        return result

    def _check_quality(self, output: str, phase_name: str) -> list:
        issues = []

        if phase_name in ("build", "code"):
            if "TODO" in output or "FIXME" in output:
                issues.append("يحتوي على علامات TODO/FIXME")
            if output.count("pass") > 3:
                issues.append("يحتوي على دوال فارغة (pass) بشكل مفرط")
            if "print(" in output and phase_name not in ("test", "debug"):
                pass

        if phase_name == "plan":
            if len(output) < 200:
                issues.append("الخطة مختصرة جداً")

        if phase_name == "document":
            if "README" not in output and "readme" not in output.lower():
                issues.append("لا يحتوي على تعليمات README")

        return issues

    def _suggest_corrections(self, output: str, phase_name: str, planner) -> list:
        corrections = []
        previous_outputs = []

        if planner is not None:
            for c in planner.completed_phase_vectors:
                previous_outputs.append(c.get("text", ""))

        if phase_name in ("build", "code"):
            corrections.append("أعد كتابة الكود مع تعريفات كاملة للدوال والفئات")
            corrections.append("تأكد من أن كل import له استخدام فعلي")

        if phase_name == "plan":
            corrections.append("أضف تفاصيل أكثر: هيكل الملفات، المكونات، نقاط الاتصال")

        if phase_name in ("document",):
            corrections.append("اكتب README بهيكل: تثبيت، استخدام، API، أمثلة")

        return corrections

    def get_status(self) -> dict:
        return {
            "total_validations": self.total_validations,
            "corrections_applied": self.corrections_applied,
            "alert_count": len(self.alerts),
            "last_alerts": self.alerts[-3:] if self.alerts else [],
            "correction_rate": self.corrections_applied / max(self.total_validations, 1),
        }
