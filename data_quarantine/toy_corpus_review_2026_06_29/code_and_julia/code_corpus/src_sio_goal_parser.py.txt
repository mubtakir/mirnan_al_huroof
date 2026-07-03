# -*- coding: utf-8 -*-
"""GoalParser — محلل الأهداف: يحول الهدف الطبيعي إلى مراحل منظمة.

المبدأ الفيزيائي:
  الهدف = متجه طوري في فضاء المعنى.
  تحليله = إسقاطه على محاور المراحل (plan, build, test, deploy, monitor).
  كل مرحلة = تحت-متجه يُستخرج من الهدف عبر التغاير مع أنماط المراحل.

لا يستخدم تعليمات ثابتة — يستخدم رنيناً طورياً بين كلمات الهدف
وأنماط المراحل المخزنة لتحديد ما يحتاجه كل طور.
"""

import re
import numpy as np
from typing import List, Dict, Optional

PHASE_PATTERNS = {
    "plan": {
        "ar": ["خطط", "صمم", "حلل", "ادرس", "هيكل", "نظم", "رتب", "حدد", "اختر"],
        "en": ["plan", "design", "analyze", "architect", "structure", "define", "choose", "decide", "spec", "requirements"],
        "desc": "تخطيط وتحليل المتطلبات",
    },
    "build": {
        "ar": ["ابن", "اكتب", "نفذ", "شيد", "برمج", "طور", "أنشئ", "كون", "اصنع"],
        "en": ["build", "write", "implement", "code", "develop", "create", "construct", "make", "generate"],
        "desc": "بناء وتنفيذ المكونات",
    },
    "test": {
        "ar": ["اختبر", "تأكد", "تحقق", "افحص", "جرب", "دقق", "راجع"],
        "en": ["test", "verify", "validate", "check", "try", "review", "audit", "inspect"],
        "desc": "اختبار والتحقق من الصحة",
    },
    "deploy": {
        "ar": ["انشر", "شغل", "أطلق", "ثبت", "رفع", "نزّل", "وزع"],
        "en": ["deploy", "launch", "release", "install", "publish", "ship", "distribute", "run"],
        "desc": "نشر وتشغيل النظام",
    },
    "integrate": {
        "ar": ["ادمج", "وصل", "اربط", "نسق", "وحد", "جمع"],
        "en": ["integrate", "connect", "link", "combine", "merge", "unify", "wire"],
        "desc": "دمج وربط المكونات",
    },
    "monitor": {
        "ar": ["راقب", "تابع", "لاحظ", "سجل", "قِس"],
        "en": ["monitor", "observe", "track", "log", "measure", "watch"],
        "desc": "مراقبة ومتابعة الأداء",
    },
    "document": {
        "ar": ["وثق", "اشرح", "اكتب دليل", "سجل"],
        "en": ["document", "explain", "readme", "manual", "guide"],
        "desc": "توثيق وكتابة أدلة",
    },
}


class GoalParser:
    """يحلل الهدف ويكتشف المراحل المطلوبة."""

    def __init__(self, gen=None):
        self.gen = gen
        self._phase_cache = {}

    def parse(self, goal: str) -> List[Dict]:
        """تحليل الهدف إلى مراحل.

        Returns:
            قائمة مراحل: [{"name": "plan", "desc": "...", "tasks": [...], "priority": 0, "resonance": 0.85}, ...]
        """
        phases = []
        words = goal.split()
        goal_lower = goal.lower()

        for phase_name, pattern in PHASE_PATTERNS.items():
            ar_matches = sum(1 for w in pattern["ar"] if w in goal)
            en_matches = sum(1 for kw in pattern["en"] if kw in goal_lower)
            total_matches = ar_matches + en_matches

            resonance = 0.0
            if self.gen is not None and hasattr(self.gen, 'heterodyne'):
                for kw in pattern["ar"] + pattern["en"]:
                    try:
                        r = self.gen.heterodyne.score_candidate(kw, words)
                        resonance = max(resonance, r)
                    except Exception:
                        pass

            total_matches = ar_matches + en_matches
            if total_matches > 0 or resonance > 0.05:
                tasks = self._extract_tasks(goal, phase_name, pattern)
                phases.append({
                    "name": phase_name,
                    "desc": pattern["desc"],
                    "tasks": tasks,
                    "priority": self._phase_priority(phase_name),
                    "resonance": max(total_matches * 0.3, resonance),
                    "match_count": total_matches,
                })

        if not phases:
            phases = self._default_phases(goal)

        phases.sort(key=lambda p: p["priority"])
        return phases

    def _extract_tasks(self, goal: str, phase_name: str, pattern: Dict) -> List[str]:
        """استخراج مهام محددة من الهدف لكل مرحلة."""
        tasks = []

        if phase_name == "plan":
            if "api" in goal.lower() or "واجهة" in goal:
                tasks.append("تحديد نقاط النهاية (endpoints)")
            if "قاعدة" in goal or "database" in goal.lower() or "بيانات" in goal:
                tasks.append("تصميم هيكل قاعدة البيانات")
            if "ui" in goal.lower() or "واجهة" in goal:
                tasks.append("تخطيط هيكل الواجهة")
            if not tasks:
                tasks.append("تحليل المتطلبات الأساسية")

        elif phase_name == "build":
            if "api" in goal.lower() or "واجهة" in goal:
                tasks.append("بناء طبقة API")
            if "قاعدة" in goal or "database" in goal.lower() or "بيانات" in goal:
                tasks.append("إنشاء نماذج البيانات")
            if "ui" in goal.lower() or "واجهة" in goal or "web" in goal.lower():
                tasks.append("بناء مكونات الواجهة")
            if not tasks:
                tasks.append("تنفيذ المنطق الأساسي")

        elif phase_name == "test":
            tasks.append("اختبار الوحدات (unit tests)")
            tasks.append("اختبار التكامل (integration tests)")
            if "api" in goal.lower():
                tasks.append("اختبار نقاط النهاية")

        elif phase_name == "deploy":
            tasks.append("إعداد بيئة النشر")
            tasks.append("توليد أوامر النشر")

        elif phase_name == "integrate":
            tasks.append("ربط المكونات")
            tasks.append("اختبار التكامل النهائي")

        elif phase_name == "monitor":
            tasks.append("إعداد سجلات المراقبة")
            tasks.append("تعريف مقاييس الأداء")

        elif phase_name == "document":
            tasks.append("كتابة README")
            tasks.append("توثيق API")

        return tasks

    def _phase_priority(self, phase_name: str) -> int:
        return {
            "plan": 0,
            "build": 1,
            "test": 2,
            "integrate": 3,
            "deploy": 4,
            "monitor": 5,
            "document": 6,
        }.get(phase_name, 5)

    def _default_phases(self, goal: str) -> List[Dict]:
        return [
            {"name": "plan", "desc": "تخطيط", "tasks": ["تحليل الهدف"], "priority": 0, "resonance": 0.0, "match_count": 0},
            {"name": "build", "desc": "بناء", "tasks": ["تنفيذ الحل"], "priority": 1, "resonance": 0.0, "match_count": 0},
            {"name": "test", "desc": "اختبار", "tasks": ["التحقق من الصحة"], "priority": 2, "resonance": 0.0, "match_count": 0},
            {"name": "deploy", "desc": "نشر", "tasks": ["تجهيز النشر"], "priority": 3, "resonance": 0.0, "match_count": 0},
        ]
