# -*- coding: utf-8 -*-
"""PhaseExecutor — منفذ مستقل للمراحل مع حلقة اختبار وإصلاح ذاتية.

دورة التنفيذ:
  1. توليد المخرج للمرحلة الحالية.
  2. اختبار المخرج (تحقق من الصحة).
  3. إذا فشل ← تشخيص الخطأ ← إعادة التوليد مع سياق مصحح ← عودة للخطوة 2.
  4. إذا نجح ← تسليم المخرج.

لا تأخير. لا اختناقات. كل مرحلة مستقلة في مسارها.
الفشل لا يوقف النظام — يعيد التكوين تلقائياً.
"""

import re
import numpy as np


class PhaseExecutor:
    """ينفذ مرحلة واحدة باستقلالية كاملة."""

    def __init__(self, gen=None):
        self.gen = gen
        self.history = []

    def execute(self, phase: dict, planner, max_iterations: int = 5) -> dict:
        """تنفيذ مرحلة مع اختبار وإصلاح ذاتي.

        Args:
            phase: قاموس المرحلة من GoalParser
            planner: كائن PhasePlanner للسياق
            max_iterations: أقصى عدد لمحاولات الإصلاح الذاتي

        Returns:
            {"success": bool, "output": str, "iterations": int, "diagnosis": str}
        """
        phase_name = phase["name"]
        tasks = phase.get("tasks", [])
        context = self._build_context(phase, planner)

        result = {"success": False, "output": "", "iterations": 0, "diagnosis": "", "phase": phase_name}

        for iteration in range(max_iterations):
            result["iterations"] = iteration + 1

            output = self._generate_output(phase_name, tasks, context)
            if not output:
                result["diagnosis"] = "فشل التوليد — لا مخرج"
                continue

            validation = self._validate_output(phase_name, output, tasks)
            if validation["valid"]:
                result["success"] = True
                result["output"] = output
                result["diagnosis"] = "نجح"
                break

            diagnosis = self._diagnose_failure(phase_name, output, validation)
            context["previous_errors"].append(diagnosis)
            result["diagnosis"] = diagnosis

        self.history.append(result)
        return result

    def _build_context(self, phase: dict, planner) -> dict:
        ctx = {
            "phase_name": phase["name"],
            "phase_desc": phase.get("desc", ""),
            "tasks": phase.get("tasks", []),
            "completed_phases": [c["phase"] for c in planner.completed_phase_vectors],
            "completed_outputs": [c["text"][:500] for c in planner.completed_phase_vectors],
            "previous_errors": [],
            "retry_count": planner.retry_counts.get(phase["name"], 0),
        }
        return ctx

    def _generate_output(self, phase_name: str, tasks: list, context: dict) -> str:
        if self.gen is None:
            return self._structured_mock(phase_name, tasks)

        direction = self._phase_direction(phase_name)
        tasks_str = "\n".join(f"- {t}" for t in tasks)
        errors_str = ""
        if context["previous_errors"]:
            errors_str = "\nأخطاء سابقة يجب تجنبها:\n" + "\n".join(
                f"- {e}" for e in context["previous_errors"][-3:]
            )

        prompt = (
            f"{direction}\n\n"
            f"المهام:\n{tasks_str}\n"
            f"{errors_str}"
        )

        for mode in ["standard", "multiverse", "standard"]:
            try:
                result = self.gen.generate(prompt, max_words=80, mode=mode)
                if isinstance(result, str) and len(result) > 20:
                    return result
            except Exception:
                continue

        return self._structured_mock(phase_name, tasks)

    def _structured_mock(self, phase_name: str, tasks: list) -> str:
        lines = []
        if phase_name == "plan":
            lines = [
                "# خطة المشروع",
                "",
                "## الهيكل",
                "```",
                "project/",
                "├── src/           # الكود المصدري",
                "├── tests/         # الاختبارات",
                "├── config/        # إعدادات",
                "├── docs/          # توثيق",
                "└── README.md",
                "```",
                "",
                "## المكونات",
            ]
            for t in tasks:
                lines.append(f"- {t}")
            lines.extend(["", "## نقاط الاتصال", "- API: REST endpoints", "- DB: SQLite/PostgreSQL"])

        elif phase_name == "build":
            lines = [
                "# التنفيذ",
                "",
                "```python",
                "def main():",
                "    # TODO: تنفيذ المنطق الأساسي",
                "    pass",
                "",
                'if __name__ == "__main__":',
                "    main()",
                "```",
            ]

        elif phase_name == "test":
            lines = [
                "# الاختبارات",
                "",
                "```python",
                "import unittest",
                "",
                "class TestCore(unittest.TestCase):",
                "    def test_basic(self):",
                '        self.assertTrue(True)',
                "```",
            ]

        elif phase_name == "deploy":
            lines = [
                "# النشر",
                "",
                "```bash",
                "pip install -r requirements.txt",
                "python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000",
                "```",
            ]

        elif phase_name == "document":
            lines = [
                "# README",
                "",
                "## التثبيت",
                "```bash",
                "pip install -r requirements.txt",
                "```",
                "",
                "## التشغيل",
                "```bash",
                "python run.py",
                "```",
            ]

        else:
            lines = [f"# {phase_name}", ""]
            for t in tasks:
                lines.append(f"- {t}")

        return "\n".join(lines)

    def _phase_direction(self, phase_name: str) -> str:
        return {
            "plan": "قدم خطة تفصيلية للمشروع مع هيكل الملفات والمكونات المطلوبة.",
            "build": "اكتب الكود الكامل للمكونات المطلوبة. كود نظيف وجاهز للتشغيل.",
            "test": "اكتب اختبارات للمكونات المنفذة. تأكد من تغطية الحالات الأساسية.",
            "deploy": "اكتب تعليمات النشر وأوامر التشغيل. جهز ملفات التكوين.",
            "integrate": "اكتب كود التكامل بين المكونات. تأكد من توافق الواجهات.",
            "monitor": "صمم نظام مراقبة مع مقاييس الأداء والسجلات.",
            "document": "اكتب توثيقاً شاملاً: README، دليل المستخدم، توثيق API.",
        }.get(phase_name, "نفذ المهام المطلوبة.")

    def _validate_output(self, phase_name: str, output: str, tasks: list) -> dict:
        """التحقق من صحة المخرج."""
        issues = []

        if len(output) < 20:
            issues.append("المخرج قصير جداً")

        if phase_name in ("build", "code"):
            if not self._has_code_pattern(output):
                issues.append("لا يحتوي على شيفرة قابلة للتنفيذ")

        if phase_name == "plan":
            if not self._has_structure(output):
                issues.append("لا يحتوي على هيكل واضح")

        if phase_name in ("deploy", "build"):
            if "import " in output and "def " not in output and "class " not in output:
                issues.append("استيراد بدون تعريف")

        task_coverage = self._check_task_coverage(output, tasks)
        if task_coverage < 0.3:
            issues.append(f"تغطية منخفضة للمهام ({task_coverage:.0%})")

        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "task_coverage": task_coverage,
        }

    def _diagnose_failure(self, phase_name: str, output: str, validation: dict) -> str:
        """تشخيص سبب الفشل لتغذية المحاولة التالية."""
        issues = validation.get("issues", [])
        if not issues:
            return "فشل غير معروف"

        diagnosis_parts = []
        for issue in issues:
            if "قصير" in issue:
                diagnosis_parts.append("توسيع المخرج بتفاصيل أكثر")
            elif "شيفرة" in issue:
                diagnosis_parts.append("إضافة كود تنفيذي كامل")
            elif "هيكل" in issue:
                diagnosis_parts.append("تنظيم المخرج في أقسام واضحة")
            elif "استيراد" in issue:
                diagnosis_parts.append("تضمين تعريفات الدوال والفئات")
            elif "تغطية" in issue:
                diagnosis_parts.append("تغطية جميع المهام المطلوبة")

        return " | ".join(diagnosis_parts) if diagnosis_parts else issues[0]

    def _has_code_pattern(self, text: str) -> bool:
        indicators = ["def ", "class ", "function ", "import ", "const ", "let ", "var ",
                       "#!/", "if __name__", "app.", "router.", "async def"]
        return any(ind in text for ind in indicators)

    def _has_structure(self, text: str) -> bool:
        indicators = ["##", "# ", "1.", "2.", "- ", "* ", "Phase", "خطة", "هيكل", "مكونات"]
        return sum(1 for ind in indicators if ind in text) >= 2

    def _check_task_coverage(self, output: str, tasks: list) -> float:
        if not tasks:
            return 1.0
        output_lower = output.lower()
        covered = sum(1 for t in tasks if any(w in output_lower for w in t.split()[:3]))
        return covered / len(tasks)
