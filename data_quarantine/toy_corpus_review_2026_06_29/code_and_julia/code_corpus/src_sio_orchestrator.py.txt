# -*- coding: utf-8 -*-
"""SIOOrchestrator — منسق الذكاء التوليفي: الطبقة العليا.

يحول فكرة إلى منتج نهائي في عملية واحدة متصلة:

  فكرة ← GoalParser ← PhasePlanner ← Executor ← SelfMonitor ← Integrator ← منتج

في أي لحظة:
  - إذا انحرف المخرج ← SelfMonitor يعيد التوجيه.
  - إذا فشلت مرحلة ← Executor يعيد التوليد بسياق مصحح.
  - إذا اكتملت كل المراحل ← Integrator يدمج في منتج واحد.

لا تأخير. لا انتظار. النظام يتكيف في الوقت الفعلي.
"""

import time
import traceback
from typing import Optional, Dict, Any


class SIOOrchestrator:
    """المنسق الأعلى للذكاء التوليفي."""

    def __init__(self, gen=None):
        from src.sio.goal_parser import GoalParser
        from src.sio.phase_planner import PhasePlanner
        from src.sio.executor import PhaseExecutor
        from src.sio.self_monitor import SelfMonitor
        from src.sio.integrator import Integrator

        self.gen = gen
        self.parser = GoalParser(gen=gen)
        self.planner = PhasePlanner(gen=gen)
        self.executor = PhaseExecutor(gen=gen)
        self.monitor = SelfMonitor(gen=gen)
        self.integrator = Integrator()

        self.current_goal = ""
        self.phase_outputs = []
        self.start_time = 0.0
        self.status = "idle"

    def synthesize(self, goal: str, max_phase_retries: int = 5) -> Dict[str, Any]:
        """تحويل هدف إلى منتج نهائي.

        هذه هي دورة الذكاء التوليفي الكاملة:
          1. تحليل الهدف إلى مراحل.
          2. لكل مرحلة: نفذ ← تحقق ← أصلح ← كرر حتى النجاح أو الاستسلام.
          3. ادمج كل المخرجات في منتج نهائي.

        Args:
            goal: الهدف باللغة الطبيعية
            max_phase_retries: أقصى محاولات لكل مرحلة

        Returns:
            تقرير شامل بالمنتج النهائي والمراحل والتحليلات
        """
        self.current_goal = goal
        self.phase_outputs = []
        self.start_time = time.time()
        self.status = "parsing"

        phases = self.parser.parse(goal)
        self.planner.set_plan(phases)

        self.status = "executing"
        skipped_phases = []
        failed_phases = []

        while not self.planner.is_complete():
            phase = self.planner.current_phase()
            if phase is None:
                break

            phase_name = phase["name"]
            self.status = f"executing:{phase_name}"

            result = self.executor.execute(phase, self.planner, max_phase_retries)

            if result["success"]:
                validated = self.monitor.validate(
                    result["output"], phase_name, self.planner
                )

                if validated["accepted"]:
                    self.planner.mark_completed(phase_name, result["output"])
                    self.phase_outputs.append({
                        "phase": phase_name,
                        "output": result["output"],
                        "iterations": result["iterations"],
                        "diagnosis": result["diagnosis"],
                        "coherence": validated["coherence"],
                    })
                    self.planner.next_phase()
                else:
                    correction = self.planner.mark_failed(phase_name, str(validated["alerts"]))
                    if not correction.get("retry", False):
                        failed_phases.append(phase_name)
                        self.planner.next_phase()
            else:
                correction = self.planner.mark_failed(phase_name, result["diagnosis"])
                if not correction.get("retry", False):
                    failed_phases.append(phase_name)
                    self.planner.next_phase()

        self.status = "integrating"

        integrated = self.integrator.integrate(self.phase_outputs, goal)

        self.status = "complete"
        elapsed = time.time() - self.start_time

        return {
            "goal": goal,
            "deliverable": integrated["deliverable"],
            "summary": integrated["summary"],
            "structure": integrated["structure"],
            "phases_completed": [p["phase"] for p in self.phase_outputs],
            "phases_failed": failed_phases,
            "phases_skipped": skipped_phases,
            "total_phases": len(phases),
            "phase_details": [
                {
                    "name": p["phase"],
                    "iterations": p["iterations"],
                    "diagnosis": p["diagnosis"],
                    "coherence": p.get("coherence", 0.0),
                }
                for p in self.phase_outputs
            ],
            "monitor": self.monitor.get_status(),
            "plan": self.planner.get_plan_summary(),
            "time_elapsed": round(elapsed, 2),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "goal": self.current_goal,
            "phases_completed": len(self.phase_outputs),
            "total_phases": len(self.planner.phases) if self.planner else 0,
            "progress": self.planner.progress() if self.planner else 0.0,
            "monitor": self.monitor.get_status(),
        }
