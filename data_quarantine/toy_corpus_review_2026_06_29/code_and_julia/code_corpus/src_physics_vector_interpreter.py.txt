# -*- coding: utf-8 -*-
"""مفسر المتجهات — ترجمة الأبعاد الرقمية الـ 22D إلى نصوص مفهومة.

كل بُعد فيزيائي للحرف له دلالة. هذا الملف يفسر قيم المتجه:
  - القيم الموجبة (> 0.3): تنشيط المعنى في هذا الاتجاه
  - القيم السالبة (< -0.3): تنشيط الضد
  - القيم القريبة من الصفر: حيادية
"""

import numpy as np

DIM_INTERPRETATIONS = [
    {
        "name": "التركيز",
        "positive": "تركيز وانتباه",
        "negative": "تشتت وانفراط",
        "index": 0,
    },
    {
        "name": "الداخل والخارج",
        "positive": "باطن وداخل",
        "negative": "ظاهر وخارج",
        "index": 1,
    },
    {
        "name": "الاستقرار والحركة",
        "positive": "استقرار وثبات",
        "negative": "حركة واضطراب",
        "index": 2,
    },
    {
        "name": "الكثافة",
        "positive": "كثافة وتراص",
        "negative": "خلخلة وتباعد",
        "index": 3,
    },
    {
        "name": "الحرارة",
        "positive": "حرارة ونشاط",
        "negative": "برودة وخمود",
        "index": 4,
    },
    {
        "name": "تراكم الزمن",
        "positive": "تراكم وامتداد زمني",
        "negative": "آنية ولحظية",
        "index": 5,
    },
    {
        "name": "ذروة الزمن",
        "positive": "ذروة وحدّة زمنية",
        "negative": "انبساط زمني",
        "index": 6,
    },
    {
        "name": "تفريغ الزمن",
        "positive": "تفريغ وتلاشٍ",
        "negative": "امتلاء زمني",
        "index": 7,
    },
    {
        "name": "الحركة الخطية",
        "positive": "حركة مستقيمة للأمام",
        "negative": "تراجع وتردد",
        "index": 8,
    },
    {
        "name": "الحركة الدورانية",
        "positive": "دوران والتفاف",
        "negative": "جمود دوراني",
        "index": 9,
    },
    {
        "name": "الحركة النبضية",
        "positive": "نبض واندفاع",
        "negative": "سكون وانقطاع",
        "index": 10,
    },
    {
        "name": "الحركة المطاطة",
        "positive": "تمدد وامتداد",
        "negative": "انكماش وانقباض",
        "index": 11,
    },
    {
        "name": "الحركة الانزلاقية",
        "positive": "انزلاق وانسياب",
        "negative": "احتكاك ومقاومة",
        "index": 12,
    },
    {
        "name": "الحركة الهوائية",
        "positive": "انطلاق هوائي",
        "negative": "سكون هوائي",
        "index": 13,
    },
    {
        "name": "المحور الرأسي",
        "positive": "ارتفاع وعلو",
        "negative": "انخفاض وهبوط",
        "index": 14,
    },
    {
        "name": "الكتلة",
        "positive": "ثقل ووزن",
        "negative": "خفة وانعدام وزن",
        "index": 15,
    },
    {
        "name": "الصلابة",
        "positive": "صلابة وشدة",
        "negative": "ليونة ورخاوة",
        "index": 16,
    },
    {
        "name": "الاختراق",
        "positive": "نفاذ واختراق",
        "negative": "منع وصد",
        "index": 17,
    },
    {
        "name": "الشحنة",
        "positive": "شحنة وجاذبية",
        "negative": "تنافر وتباعد",
        "index": 18,
    },
    {
        "name": "المرجعية الذاتية",
        "positive": "ذاتي واستقلالي",
        "negative": "غيري وتابع",
        "index": 19,
    },
    {
        "name": "الامتداد المكاني",
        "positive": "امتداد واتساع",
        "negative": "انحسار وضيق",
        "index": 20,
    },
    {
        "name": "السببية الزمنية",
        "positive": "سببية وتأثير",
        "negative": "انفعال وتأثر",
        "index": 21,
    },
]


def interpret_vector(vector: np.ndarray, threshold: float = 0.3) -> dict:
    """تفسير متجه 22D إلى وصف مفهوم.

    Returns:
        dict with:
          - dominant: list of (dim_name, description) for strong dimensions
          - opposite: list of (dim_name, description) for strong negative dimensions
          - neutral_count: number of neutral dimensions
          - summary: one-line Arabic summary
    """
    dominant = []
    opposite = []
    for dim in DIM_INTERPRETATIONS:
        idx = dim["index"]
        if idx >= len(vector):
            continue
        val = float(vector[idx])
        if val > threshold:
            dominant.append((dim["name"], dim["positive"], round(val, 2)))
        elif val < -threshold:
            opposite.append((dim["name"], dim["negative"], round(val, 2)))

    neutral = len(DIM_INTERPRETATIONS) - len(dominant) - len(opposite)

    parts = []
    if dominant:
        dom_str = "، ".join(f"{name} ({desc})" for name, desc, _ in dominant)
        parts.append(f"ينشط في: {dom_str}")
    if opposite:
        opp_str = "، ".join(f"{name} ({desc})" for name, desc, _ in opposite)
        parts.append(f"ينخفض في: {opp_str}")
    if not parts:
        parts.append("متجه متوازن بلا تنشيطات بارزة")

    summary = " | ".join(parts)

    return {
        "dominant": dominant,
        "opposite": opposite,
        "neutral_count": neutral,
        "summary": summary,
    }


def interpret_letter(letter: str, db=None) -> dict:
    """تفسير متجه حرف كامل (22D + معانيه)."""
    if db is None:
        from src.physics.letter_db import LetterDB
        db = LetterDB()

    from src.semantics.letter_meanings import LETTER_RICH_MEANINGS

    vec = db.get_vector(letter)
    vec_interp = interpret_vector(vec)
    rich = LETTER_RICH_MEANINGS.get(letter, {})

    return {
        "letter": letter,
        "vector_values": vec.tolist() if hasattr(vec, "tolist") else list(vec),
        "vector_interpretation": vec_interp,
        "operator": db.get_operator(letter),
        "omega_0": round(db.get_omega_0(letter), 3),
        "core_meaning": rich.get("core", "?"),
        "branches": rich.get("branches", []),
        "opposite": rich.get("opposite", "?"),
        "standard_of": rich.get("standard_of", "?"),
    }
