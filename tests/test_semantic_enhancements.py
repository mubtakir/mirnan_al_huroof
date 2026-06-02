# -*- coding: utf-8 -*-
"""
ملف اختبار ومعايرة التحسينات الدلالية الجديدة
=================================================
1. اختبار مؤثر التضاد العام (الخطية، القسمة، كليفورد، الدوران) عبر التحقق المتقاطع (Leave-One-Out).
2. قياس مدى تباعد الأضداد باستخدام المتجهات المركبة (Remnant vs Projection vs Linear).
"""

import numpy as np
import pytest
from src.physics.clifford_math import Multivector22
from src.physics.semantic_arithmetic import (
    build_general_antinomy_operator,
    apply_antinomy_operator,
    predict_antonym,
    compute_compound_word_vector,
    _get_word_linear,
    _get_word_mv,
    build_field_specific_antinomy_operators,
    predict_antonym_field_specific,
    _get_word_vector,
    apply_field_specific_antinomy_operator
)
from src.physics.word_physics import phase_similarity

ANTONYM_PAIRS = [
    ("قوة", "ضعف"),
    ("نور", "ظلام"),
    ("علم", "جهل"),
    ("كبر", "صغر"),
    ("صدق", "كذب"),
    ("حر", "برد"),
    ("حياة", "موت"),
    ("حق", "باطل")
]


def test_antonym_operator_leave_one_out():
    """
    اختبار مؤثر التضاد العام باستخدام Leave-One-Out Cross-Validation.
    لكل زوج متضادين، نقوم ببناء المؤثر من الـ 7 أزواج الأخرى، ثم نحاول التنبؤ بالضد للكلمة الثامنة.
    """
    runs = [
        ("linear", "linear"),
        ("linear", "remnant"),
        ("linear", "projection"),
        ("division", "linear"),
        ("division", "remnant"),
        ("division", "projection"),
        ("clifford", "linear"),  # vector_type is ignored
        ("rotation", "linear"),
        ("rotation", "remnant"),
        ("rotation", "projection"),
    ]
    results_summary = []

    print("\n" + "="*80)
    print("  بنشمارك مؤشر التضاد العام عبر Leave-One-Out Cross-Validation")
    print("="*80)

    for method, vector_type in runs:
        top1_hits = 0
        top5_hits = 0
        all_ranks = []
        
        print(f"\n[الطريقة: {method.upper()} | نوع المتجه: {vector_type.upper()}]")
        print("-" * 55)
        
        for i, (pos, neg) in enumerate(ANTONYM_PAIRS):
            # إعداد بيانات التدريب والاختبار
            train_pairs = ANTONYM_PAIRS[:i] + ANTONYM_PAIRS[i+1:]
            
            # بناء المؤثر وتطبيقه
            operator = build_general_antinomy_operator(train_pairs, method=method, vector_type=vector_type)
            candidates = predict_antonym(pos, operator, method=method, vector_type=vector_type, top_k=10)
            
            # إيجاد رتبة الكلمة الصحيحة المتوقعة (neg)
            rank = -1
            for r, (cand, sim) in enumerate(candidates):
                if cand == neg:
                    rank = r + 1
                    break
            
            all_ranks.append(rank)
            
            if rank == 1:
                top1_hits += 1
                top5_hits += 1
            elif 1 < rank <= 5:
                top5_hits += 1
                
            rank_str = f"Rank {rank}" if rank != -1 else "Not in top 10"
            cand_str = ", ".join([f"{w}({s:.2f})" for w, s in candidates[:3]])
            print(f"  {pos} ➔ {neg} | المتوقع الصحيح: {rank_str} | أفضل المرشحين: [{cand_str}]")

        n = len(ANTONYM_PAIRS)
        results_summary.append({
            "method": method,
            "vector_type": vector_type if method != "clifford" else "N/A",
            "top1_acc": top1_hits / n,
            "top5_acc": top5_hits / n,
        })
        
    print("\n" + "="*70)
    print("  ملخص دقة التنبؤ بالتضاد:")
    print("="*70)
    print(f"{'Method':<12} | {'Vector Type':<12} | {'Top-1 Accuracy':<15} | {'Top-5 Accuracy':<15}")
    print("-" * 70)
    for res in results_summary:
        print(f"{res['method']:<12} | {res['vector_type']:<12} | {res['top1_acc']:<15.2%} | {res['top5_acc']:<15.2%}")
    print("="*70)

    # نتحقق من أن التنفيذ تم بنجاح بدون أخطاء برمجية
    assert len(results_summary) == len(runs)


def test_antonym_distinction_with_compounds():
    """
    بنشمارك تمايز الأضداد باستخدام المتجهات المركبة (Remnant vs Projection vs Linear).
    يقيس التشابه بين كل زوج ضدين للتأكد من انخفاض التشابه مقارنة بالخطي.
    """
    print("\n" + "="*80)
    print("  بنشمارك تمايز الأضداد عبر المتجهات المركبة (Cosine Similarity)")
    print("="*80)
    print(f"{'الأضداد':<15} | {'Linear':<10} | {'Remnant':<10} | {'Projection':<10}")
    print("-" * 60)

    linear_sims = []
    remnant_sims = []
    projection_sims = []

    for pos, neg in ANTONYM_PAIRS:
        # 1. تشابه خطي
        v_pos_lin = _get_word_linear(pos)
        v_neg_lin = _get_word_linear(neg)
        sim_lin = phase_similarity(v_pos_lin, v_neg_lin)
        linear_sims.append(sim_lin)

        # 2. تشابه الجزء المتبقي (Remnant)
        v_pos_rem = compute_compound_word_vector(pos, method="remnant")
        v_neg_rem = compute_compound_word_vector(neg, method="remnant")
        sim_rem = float(np.dot(v_pos_rem, v_neg_rem))
        remnant_sims.append(sim_rem)

        # 3. تشابه الإسقاط (Projection)
        v_pos_proj = compute_compound_word_vector(pos, method="projection")
        v_neg_proj = compute_compound_word_vector(neg, method="projection")
        sim_proj = float(np.dot(v_pos_proj, v_neg_proj))
        projection_sims.append(sim_proj)

        print(f"{pos} ↔ {neg:<10} | {sim_lin:<10.4f} | {sim_rem:<10.4f} | {sim_proj:<10.4f}")

    print("-" * 60)
    avg_lin = np.mean(linear_sims)
    avg_rem = np.mean(remnant_sims)
    avg_proj = np.mean(projection_sims)
    print(f"{'المتوسط':<15} | {avg_lin:<10.4f} | {avg_rem:<10.4f} | {avg_proj:<10.4f}")
    print("="*80)

    # نتحقق من أن المتجهات المركبة (Remnant أو Projection) تعطي فصلاً أفضل (أقل تشابهاً) للأضداد من الخطي
    assert avg_rem < avg_lin or avg_proj < avg_lin, "Compound vectors should reduce similarity between antonyms compared to linear"


ROOT_PAIRS = [
    ("قوي", "ضعف"),
    ("نور", "ظلم"),
    ("علم", "جهل"),
    ("كبر", "صغر"),
    ("صدق", "كذب"),
    ("حرر", "برد"),
    ("حيي", "موت"),
    ("حقق", "بطل")
]


def test_root_level_analysis():
    """بنشمارك تمايز الأضداد على مستوى الجذور الثلاثية المجردة."""
    print("\n" + "="*80)
    print("  بنشمارك تمايز الأضداد على مستوى الجذور الثلاثية (Cosine Similarity)")
    print("="*80)
    print(f"{'الجذور الأضداد':<15} | {'Linear':<10} | {'Remnant':<10} | {'Projection':<10}")
    print("-" * 60)

    linear_sims = []
    remnant_sims = []
    projection_sims = []

    for pos, neg in ROOT_PAIRS:
        v_pos_lin = _get_word_linear(pos)
        v_neg_lin = _get_word_linear(neg)
        sim_lin = phase_similarity(v_pos_lin, v_neg_lin)
        linear_sims.append(sim_lin)

        v_pos_rem = compute_compound_word_vector(pos, method="remnant")
        v_neg_rem = compute_compound_word_vector(neg, method="remnant")
        sim_rem = float(np.dot(v_pos_rem, v_neg_rem))
        remnant_sims.append(sim_rem)

        v_pos_proj = compute_compound_word_vector(pos, method="projection")
        v_neg_proj = compute_compound_word_vector(neg, method="projection")
        sim_proj = float(np.dot(v_pos_proj, v_neg_proj))
        projection_sims.append(sim_proj)

        print(f"{pos} ↔ {neg:<10} | {sim_lin:<10.4f} | {sim_rem:<10.4f} | {sim_proj:<10.4f}")

    print("-" * 60)
    avg_lin = np.mean(linear_sims)
    avg_rem = np.mean(remnant_sims)
    avg_proj = np.mean(projection_sims)
    print(f"{'المتوسط':<15} | {avg_lin:<10.4f} | {avg_rem:<10.4f} | {avg_proj:<10.4f}")
    print("="*80)

    assert avg_rem < avg_lin or avg_proj < avg_lin, "Compound root vectors should separate antonyms better than linear"


def test_harakat_modulation_distinction():
    """اختبار تمييز الحركات بين الصيغ الصرفية والوظائف النحوية المختلفة (الفاعل، المفعول، المصدر)."""
    w_active = "عَلِمَ"
    w_passive = "عُلِمَ"
    w_noun = "عِلْمٌ"
    
    # 1. التشابه الخطي
    v_active_lin = _get_word_linear(w_active)
    v_passive_lin = _get_word_linear(w_passive)
    v_noun_lin = _get_word_linear(w_noun)
    
    sim_active_passive_lin = float(np.dot(v_active_lin, v_passive_lin))
    sim_active_noun_lin = float(np.dot(v_active_lin, v_noun_lin))
    
    # 2. التشابه الهندسي (Remnant)
    v_active_rem = compute_compound_word_vector(w_active, method="remnant")
    v_passive_rem = compute_compound_word_vector(w_passive, method="remnant")
    v_noun_rem = compute_compound_word_vector(w_noun, method="remnant")
    
    sim_active_passive_rem = float(np.dot(v_active_rem, v_passive_rem))
    sim_active_noun_rem = float(np.dot(v_active_rem, v_noun_rem))
    
    # 3. التشابه الهندسي (Projection)
    v_active_proj = compute_compound_word_vector(w_active, method="projection")
    v_passive_proj = compute_compound_word_vector(w_passive, method="projection")
    v_noun_proj = compute_compound_word_vector(w_noun, method="projection")
    
    sim_active_passive_proj = float(np.dot(v_active_proj, v_passive_proj))
    sim_active_noun_proj = float(np.dot(v_active_proj, v_noun_proj))
    
    print("\n" + "="*80)
    print("  بنشمارك تمييز الصيغ والحركات (علم/عَلِمَ/عُلِمَ/عِلْمٌ) في الفضاء 22D")
    print("="*80)
    print(f"  عَلِمَ ↔ عُلِمَ (فاعل/مفعول)   | الخطي: {sim_active_passive_lin:.4f} | الباقي: {sim_active_passive_rem:.4f} | الإسقاط: {sim_active_passive_proj:.4f}")
    print(f"  عَلِمَ ↔ عِلْمٌ  (فعل/مصدر)    | الخطي: {sim_active_noun_lin:.4f} | الباقي: {sim_active_noun_rem:.4f} | الإسقاط: {sim_active_noun_proj:.4f}")
    print("="*80)
    
    # التحقق من أن التناظر تم كسر الصفر فيه (التشابه ليس 1.0)
    assert sim_active_passive_lin < 0.999
    assert sim_active_noun_lin < 0.999
    assert sim_active_noun_rem < 0.99
    
    
def test_field_specific_antonyms_on_roots():
    """بنشمارك مؤثرات التضاد المتخصصة بالحقول (Field-Specific Operators) عبر التحقق المتقاطع."""
    methods = ["linear", "division", "rotation"]
    vector_types = ["linear", "remnant", "projection"]
    
    all_roots_list = []
    for pos, neg in ROOT_PAIRS:
        all_roots_list.extend([pos, neg])
        
    print("\n" + "="*80)
    print("  بنشمارك مؤثر التضاد المتخصص بالحقول (Field-Specific) داخل فضاء الجذور الـ 16")
    print("="*80)

    results_summary = []
    
    for method in methods:
        for vt in vector_types:
            correct = 0
            total = len(ROOT_PAIRS)
            
            for i, (pos, neg) in enumerate(ROOT_PAIRS):
                train_pairs = ROOT_PAIRS[:i] + ROOT_PAIRS[i+1:]
                operators = build_field_specific_antinomy_operators(train_pairs, method=method, vector_type=vt)
                
                target_v = apply_field_specific_antinomy_operator(pos, operators, method, vt)
                target_v_norm = np.linalg.norm(target_v)
                
                best_cand = "?"
                best_sim = -1.0
                if target_v_norm > 1e-10:
                    for cand in all_roots_list:
                        if cand == pos:
                            continue
                        wv = _get_word_vector(cand, vt)
                        sim = float(np.dot(target_v, wv) / (target_v_norm * np.linalg.norm(wv) + 1e-10))
                        if sim > best_sim:
                            best_sim = sim
                            best_cand = cand
                            
                if best_cand == neg:
                    correct += 1
                    
            acc = correct / total
            results_summary.append((method, vt, acc))
            
    print("\n  ملخص دقة التنبؤ بمؤثرات الحقول المتخصصة (Field-Specific):")
    print("-" * 65)
    print(f"{'Method':<12} | {'Vector Type':<12} | {'Accuracy':<15}")
    print("-" * 65)
    for m, vt, acc in results_summary:
        print(f"{m:<12} | {vt:<12} | {acc:<15.2%}")
    print("="*80)
    
    assert len(results_summary) == len(methods) * len(vector_types)


if __name__ == "__main__":
    test_antonym_operator_leave_one_out()
    test_antonym_distinction_with_compounds()
    test_root_level_analysis()
    test_harakat_modulation_distinction()
    test_field_specific_antonyms_on_roots()

