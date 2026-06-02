# -*- coding: utf-8 -*-
"""
مشروع بحثي: جدول الضرب الدلالي للحروف العربية
===========================================
يقوم بحساب الجداء الهندسي (جداء كليفورد) لكل حرفين عربيين (28 × 28)
لاستخلاص المعنى الناشئ (Emergent Meaning) والجوهر المشترك (Scalar Essence)
وحفظ النتائج في ملف JSON لدعم نظرية "كيمياء الحروف".
"""

import os
import json
import numpy as np
from src.physics.clifford_math import Multivector22
from src.physics.word_fusion_engine import decompose_fusion
from src.physics.letter_db import LetterDB

ARABIC_LETTERS = [
    "ء", "ا", "ب", "ت", "ث", "ج", "ح", "خ", "د", "ذ",
    "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ",
    "ف", "ق", "ك", "ل", "م", "ن", "ه", "و", "ي"
]


def build_letter_multiplication_table(save_path: str = None):
    db = LetterDB()
    table = {}
    
    print("="*70)
    print("  بدء حساب جدول الضرب الدلالي الهندسي للحروف العربية (29 × 29)")
    print("="*70)
    
    top_emergent = []
    top_scalar = []
    
    for ch1 in ARABIC_LETTERS:
        if not db.has(ch1):
            continue
        v1 = db.get_vector(ch1)
        mv1 = Multivector22.from_vector(v1).normalize()
        
        table[ch1] = {}
        
        for ch2 in ARABIC_LETTERS:
            if not db.has(ch2):
                continue
            v2 = db.get_vector(ch2)
            mv2 = Multivector22.from_vector(v2).normalize()
            
            # الجداء الهندسي: A * B
            prod_mv = mv1 * mv2
            decomp = decompose_fusion(prod_mv)
            
            # تخزين البيانات الأساسية
            table[ch1][ch2] = {
                "scalar_essence": decomp["shared_essence"],
                "emergent_intensity": decomp["emergent_intensity"],
                "vector_remnant": decomp["vector_remnant"],
                "dominant_pairs": decomp["bivector_dominant_pairs"][:3]  # أفضل 3 تفاعلات ثنائية
            }
            
            # ترتيب للملخص العلمي
            if ch1 != ch2:  # تجنب ضرب الحرف في نفسه
                top_emergent.append((ch1, ch2, decomp["emergent_intensity"], decomp["bivector_dominant_pairs"][:2]))
                top_scalar.append((ch1, ch2, decomp["shared_essence"]))
                
    # حفظ الجدول
    if save_path is None:
        save_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "letter_multiplication_table.json")
        
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=4)
        
    print(f"[✓] تم حفظ جدول الضرب الدلالي في: {save_path}")
    
    # عرض النتائج البحثية
    top_emergent.sort(key=lambda x: x[2], reverse=True)
    top_scalar.sort(key=lambda x: x[2], reverse=True)
    
    print("\n" + "="*60)
    print("  أقوى التفاعلات الناشئة (أعلى شدة Bivector - Emergent Intensity)")
    print("="*60)
    for i, (c1, c2, val, pairs) in enumerate(top_emergent[:7]):
        pairs_str = " | ".join([f"{p[0]}×{p[1]}({p[2]})" for p in pairs])
        print(f"  {i+1}. {c1} × {c2} ➔ شدة: {val:.4f} | تفاعلات مهيمنة: [{pairs_str}]")
        
    print("\n" + "="*60)
    print("  أعلى جوهر مشترك (أعلى قيمة Scalar - Shared Essence)")
    print("="*60)
    for i, (c1, c2, val) in enumerate(top_scalar[:7]):
        print(f"  {i+1}. {c1} × {c2} ➔ تداخل: {val:.4f}")
        
    print("="*60)
    return table


if __name__ == "__main__":
    build_letter_multiplication_table()
