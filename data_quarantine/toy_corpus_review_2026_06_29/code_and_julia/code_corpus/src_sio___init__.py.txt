# -*- coding: utf-8 -*-
"""Synthetic Intelligence Orchestrator — طبقة الذكاء التوليفي فوق مرنان.

المبادئ:
  1. لا استعلامات ثابتة — أهداف تتحلل إلى مسارات تنفيذ مستقلة.
  2. سلاسل بناء تعيد تكوين نفسها — عند الفشل، يعاد التخطيط والتوليد تلقائياً.
  3. دمج المنطق والواجهة والنشر — في عملية واحدة متصلة.
  4. تكيف في الوقت الفعلي — الرصد المستمر وإعادة التوجيه.
  5. لا هلوسة — كل مخرج يُختبر قبل القبول.

الطبقات:
  GoalParser  → PhasePlanner → Executor → SelfMonitor → Integrator
"""

from src.sio.goal_parser import GoalParser
from src.sio.phase_planner import PhasePlanner
from src.sio.executor import PhaseExecutor
from src.sio.self_monitor import SelfMonitor
from src.sio.integrator import Integrator
from src.sio.orchestrator import SIOOrchestrator

__all__ = [
    'GoalParser', 'PhasePlanner', 'PhaseExecutor',
    'SelfMonitor', 'Integrator', 'SIOOrchestrator',
]
