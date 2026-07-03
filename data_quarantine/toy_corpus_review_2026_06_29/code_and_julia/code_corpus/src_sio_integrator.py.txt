# -*- coding: utf-8 -*-
"""Integrator — دامج المخرجات: يجمع مخرجات كل المراحل في منتج نهائي واحد.

دمج المنطق والواجهة والنشر:
  - يجمع مخرجات plan + build + test + deploy في وحدة واحدة.
  - يحل التعارضات بين المراحل.
  - ينتج حزمة تسليم نهائية جاهزة.
"""

import re


class Integrator:
    """يدمج مخرجات المراحل في منتج نهائي متكامل."""

    def integrate(self, phase_outputs: list, goal: str = "") -> dict:
        """دمج مخرجات جميع المراحل.

        Args:
            phase_outputs: [{"phase": "plan", "output": "..."}, ...]
            goal: الهدف الأصلي

        Returns:
            {"deliverable": str, "structure": dict, "summary": str}
        """
        outputs_by_phase = {}
        for po in phase_outputs:
            outputs_by_phase[po.get("phase", "?")] = po.get("output", "")

        plan = outputs_by_phase.get("plan", "")
        build = outputs_by_phase.get("build", "")
        test = outputs_by_phase.get("test", "")
        deploy = outputs_by_phase.get("deploy", "")
        doc = outputs_by_phase.get("document", "")
        integ = outputs_by_phase.get("integrate", "")
        monitor = outputs_by_phase.get("monitor", "")

        sections = []

        if goal:
            sections.append(f"# المشروع: {goal}\n")

        if plan:
            sections.append("## الخطة\n")
            sections.append(plan)
            sections.append("")

        if build:
            sections.append("## التنفيذ\n")
            sections.append(self._extract_code_blocks(build) if self._has_code(build) else build)
            sections.append("")

        if integ:
            sections.append("## التكامل\n")
            sections.append(integ)
            sections.append("")

        if test:
            sections.append("## الاختبارات\n")
            sections.append(self._extract_test_code(test) if self._has_code(test) else test)
            sections.append("")

        if deploy:
            sections.append("## النشر\n")
            sections.append(self._format_deployment(deploy))
            sections.append("")

        if monitor:
            sections.append("## المراقبة\n")
            sections.append(monitor)
            sections.append("")

        if doc:
            sections.append("## التوثيق\n")
            sections.append(doc)
            sections.append("")

        if not sections:
            all_text = "\n\n".join(outputs_by_phase.values())
            sections.append(all_text)

        deliverable = "\n".join(sections)
        structure = self._infer_structure(outputs_by_phase, goal)

        return {
            "deliverable": deliverable,
            "structure": structure,
            "summary": self._generate_summary(outputs_by_phase, goal),
        }

    def _extract_code_blocks(self, text: str) -> str:
        """استخراج كتل الشيفرة من النص."""
        blocks = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
        if blocks:
            return "\n\n".join(f"```\n{b.strip()}\n```" for b in blocks)
        return text

    def _extract_test_code(self, text: str) -> str:
        blocks = re.findall(r'```[\w]*\n(.*?)```', text, re.DOTALL)
        if blocks:
            return "\n\n".join(f"```\n{b.strip()}\n```" for b in blocks)
        return text

    def _has_code(self, text: str) -> bool:
        return "```" in text or "def " in text or "class " in text or "import " in text

    def _format_deployment(self, text: str) -> str:
        commands = re.findall(r'(`[^`]+`)', text)
        if commands:
            return "أوامر النشر:\n" + "\n".join(f"  {c}" for c in commands)
        return text

    def _infer_structure(self, outputs: dict, goal: str) -> dict:
        structure = {"files": [], "directories": []}

        for phase, output in outputs.items():
            if phase in ("build", "code"):
                filenames = re.findall(r'(?:file|ملف|path)[:\s]+`?([\w./\\-]+\.\w+)`?', output, re.IGNORECASE)
                structure["files"].extend(filenames)
            if phase == "plan":
                dirs = re.findall(r'(?:directory|مجلد|dir)[:\s]+`?([\w./\\-]+)`?', output, re.IGNORECASE)
                structure["directories"].extend(dirs)

        structure["files"] = list(set(structure["files"]))
        structure["directories"] = list(set(structure["directories"]))
        return structure

    def _generate_summary(self, outputs: dict, goal: str) -> str:
        phases_completed = list(outputs.keys())
        total_size = sum(len(v) for v in outputs.values())

        return (
            f"الهدف: {goal}\n"
            f"المراحل المكتملة: {', '.join(phases_completed)}\n"
            f"عدد المراحل: {len(phases_completed)}\n"
            f"الحجم الكلي: {total_size} حرف"
        )
