#!/usr/bin/env python3
"""مرنان النظيف — واجهة الأوامر التفاعلية لحقل الجذب والتنافر.

الاستخدام:
  python cli.py [الكلمة] [--space combined|physical|philosophical] [--k 15] [--interactive]
"""

import argparse
import sys
import os

# إضافة مسار المشروع
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure') and "pytest" not in sys.modules:
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import model
from src.physics import FieldEngine
from src.physics.vector_interpreter import interpret_letter, interpret_vector
from src.physics.word_physics import compute_word_phase_vector, get_letter_db
from src.semantics.arabic_semantics import decompose_word_definition, CharacterSemanticEmbedding
from src.semantics.letter_meanings import LETTER_RICH_MEANINGS

def print_results(word, space_type, res):
    """طباعة نتائج التجاذب والتنافر بتنسيق منظم في الطرفية."""
    print("\n" + "=" * 60)
    print(f"  تحليل حقل الكلمة: {word}")
    print(f"  الفضاء الطوري:  {space_type}")
    print("=" * 60)
    
    print("\n🟢 الكلمات الأكثر تجاذباً (تقارب دلالي وفيزيائي):")
    print("-" * 60)
    if not res["attracted"]:
        print("  [لا توجد نتائج]")
    for w, s in res["attracted"]:
        # رسم شريط صغير يمثل شدة التجاذب (مع تحديد الحد الأقصى)
        bar_len = int(min(1.0, max(0.0, s)) * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        print(f"  {w:18s} | {s:+.3f} | {bar}")
        
    print("\n🔴 الكلمات الأكثر تنافراً (تباعد دلالي وفيزيائي):")
    print("-" * 60)
    if not res["repelled"]:
        print("  [لا توجد نتائج]")
    for w, s in res["repelled"]:
        # للتنافر: القيمة سالبة، وكلما كانت سالبة أكثر كان التنافر أشد
        bar_len = int(min(1.0, abs(min(0.0, s))) * 15)
        bar = "█" * bar_len + "░" * (15 - bar_len)
        print(f"  {w:18s} | {s:+.3f} | {bar}")
    print("=" * 60 + "\n")


def print_definition(word):
    """طباعة تفكيك الكلمة إلى حروفها ودلالاتها المتفرعة."""
    parts = decompose_word_definition(word)
    if not parts:
        print(f"لم يتم التعرف على حروف عربية في: {word}")
        return
    print("\n" + "=" * 60)
    print(f"  تفكيك دلالي لكلمة: {word}")
    print("=" * 60)
    core_meanings = []
    for ch, _ in parts:
        rich = LETTER_RICH_MEANINGS.get(ch, {})
        core = rich.get("core", "?")
        branches = rich.get("branches", [])
        opp = rich.get("opposite", "?")
        standard = rich.get("standard_of", "?")
        core_meanings.append(core)
        print(f"  [{ch}] ← {core}")
        print(f"       التفرعات : {' > '.join(branches)}")
        print(f"       الضد     : {opp}")
        print(f"       المعيار   : {standard}")
    print(f"  {'-' * 56}")
    print(f"  خلاصة دلالية: {' + '.join(core_meanings)}")
    print("=" * 60 + "\n")


def print_letter_info(letter, engine=None):
    """عرض معلومات شاملة عن حرف: متجهه 22D + تفسيره + معانيه المتفرعة."""
    from src.physics.vector_interpreter import interpret_letter
    info = interpret_letter(letter)
    vec_interp = info["vector_interpretation"]

    print("\n" + "=" * 60)
    print(f"  تحليل الحرف: {letter}")
    print("=" * 60)

    # المعاني
    print(f"\n  المعنى العام  : {info['core_meaning']}")
    print(f"  التفرعات      : {' > '.join(info['branches'])}")
    print(f"  الضد          : {info['opposite']}")
    print(f"  المعيار        : {info['standard_of']}")
    print(f"  المؤثر        : {info['operator']}")
    print(f"  التردد الذاتي  : {info['omega_0']}")

    # تفسير المتجه
    print(f"\n  {'─' * 56}")
    print(f"  تفسير المتجه 22D:")
    print(f"  {vec_interp['summary']}")

    # عرض الأبعاد النشطة
    if vec_interp["dominant"]:
        print(f"\n  الأبعاد النشطة إيجاباً (+):")
        for name, desc, val in vec_interp["dominant"]:
            bar_len = int(val * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            print(f"    {name:20s} {val:+.2f} |{bar}| {desc}")

    if vec_interp["opposite"]:
        print(f"\n  الأبعاد النشطة سلباً (-):")
        for name, desc, val in vec_interp["opposite"]:
            bar_len = int(abs(val) * 15)
            bar = "█" * bar_len + "░" * (15 - bar_len)
            print(f"    {name:20s} {val:+.2f} |{bar}| {desc}")

    # قيم المتجه الكاملة
    vals = info["vector_values"]
    print(f"\n  {'─' * 56}")
    print(f"  قيم المتجه 22D كاملة:")
    for i in range(0, 22, 6):
        chunk = vals[i:i+6]
        labels = [f"D{i+j:02d}" for j in range(len(chunk))]
        vals_str = "  ".join(f"{l}={v:+.2f}" for l, v in zip(labels, chunk))
        print(f"    {vals_str}")

    # اختبار تأثير الحرف في كلمات
    if engine is not None:
        print(f"\n  {'─' * 56}")
        print(f"  كلمات تحتوي على هذا الحرف في صدرها (أقوى تجاذب):")
        try:
            res = engine.find_attraction_repulsion(letter, space_type="physical", top_k=5)
            for w, s in res["attracted"]:
                print(f"    {w:18s} | {s:+.3f}")
        except Exception:
            pass

    print("=" * 60 + "\n")

def interactive_loop(engine, space_type, top_k):
    print("\n" + "*" * 60)
    print("      مرنان الفيزيائي — حلقة التحليل التفاعلي لحقل الحروف")
    print("  اكتب الكلمة للتحليل، أو 'exit' للخروج.")
    print("  لتغيير الفضاء: /space [combined|physical|philosophical]")
    print("  لتغيير العدد: /k [5-30]")
    print("  لتفكيك دلالي: /define [كلمة]")
    print("  لتحليل حرف  : /letter [حرف]")
    print("  لتعديل بعد  : /letter [حرف] set [0-21] [قيمة]")
    print("  لحفظ التعديلات: /letter [حرف] save")
    print("*" * 60)
    
    while True:
        try:
            prompt = input("\nأدخل كلمة > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
            
        if not prompt:
            continue
        if prompt.lower() in ('exit', 'quit', 'خروج'):
            break
            
        # معالجة الأوامر الداخلية
        if prompt.startswith('/space'):
            parts = prompt.split()
            if len(parts) > 1 and parts[1] in ('combined', 'physical', 'philosophical'):
                space_type = parts[1]
                print(f"✓ تم تحويل الفضاء إلى: {space_type}")
            else:
                print(f"الفضاء الحالي: {space_type}")
            continue
            
        if prompt.startswith('/k'):
            parts = prompt.split()
            try:
                val = int(parts[1])
                if 5 <= val <= 50:
                    top_k = val
                    print(f"✓ تم تعديل عدد النتائج إلى: {top_k}")
                else:
                    print("الرجاء إدخال قيمة بين 5 و 50")
            except (IndexError, ValueError):
                print(f"العدد الحالي: {top_k}")
            continue

        if prompt.startswith('/define'):
            parts = prompt.split(maxsplit=1)
            if len(parts) > 1 and parts[1].strip():
                print_definition(parts[1].strip())
            else:
                print("الاستخدام: /define [كلمة]")
            continue

        if prompt.startswith('/letter'):
            parts = prompt.split(maxsplit=3)
            if len(parts) < 2:
                print("الاستخدام:")
                print("  /letter حرف             عرض تحليل الحرف")
                print("  /letter حرف set N VAL   تعديل البعد N إلى VAL")
                print("  /letter حرف save        حفظ التعديلات إلى الملف")
                continue

            letter = parts[1]
            db = get_letter_db()
            if len(letter) != 1 or not db.has(letter):
                print(f"الحرف '{letter}' غير موجود. اختر حرفاً عربياً واحداً.")
                continue

            if len(parts) >= 4 and parts[2] == "set":
                try:
                    dim_idx = int(parts[3])
                    if len(parts) < 5:
                        print("الرجاء تحديد القيمة: /letter حرف set N VAL")
                        continue
                    val = float(parts[4])
                    if dim_idx < 0 or dim_idx >= 22:
                        print(f"البعد يجب أن يكون بين 0 و 21")
                        continue
                    vec = db.get_vector(letter).copy()
                    vec[dim_idx] = val
                    db.set_vector(letter, vec)
                    engine.rebuild()
                    print(f"✓ تم تعديل {letter}[D{dim_idx:02d}] = {val:+.2f}")
                    print_letter_info(letter, engine)
                except ValueError:
                    print("الرجاء إدخال أرقام صحيحة.")
                continue

            if len(parts) >= 3 and parts[2] == "save":
                db.save()
                print(f"✓ تم حفظ التعديلات إلى {db.path}")
                continue

            print_letter_info(letter, engine)
            continue
            
        if len(prompt.split()) > 1:
            print("الرجاء إدخال كلمة واحدة فقط بدون مسافات.")
            continue
            
        # التحليل والعرض
        try:
            res = engine.find_attraction_repulsion(prompt, space_type=space_type, top_k=top_k)
            print_results(prompt, space_type, res)
        except Exception as e:
            print(f"حدث خطأ أثناء التحليل: {e}")


def main():
    parser = argparse.ArgumentParser(description="مرنان النظيف — تحليل حقل الجذب والتنافر للكلمات")
    parser.add_argument('word', nargs='?', help="الكلمة المراد تحليل حقلها")
    parser.add_argument('--space', default='combined', choices=['combined', 'physical', 'philosophical'],
                        help="الفضاء الطوري للحساب (combined, physical, philosophical)")
    parser.add_argument('--k', type=int, default=15, help="عدد الكلمات في حقل التجاذب وحقل التنافر (افتراضي 15)")
    parser.add_argument('--interactive', '-i', action='store_true', help="تشغيل الوضع التفاعلي المباشر")
    parser.add_argument('--define', '-d', action='store_true', help="تفكيك الكلمة إلى حروفها ودلالاتها بدلاً من تحليل الحقل")
    parser.add_argument('--letter', '-l', type=str, default=None, help="تحليل حرف واحد: عرض متجهه 22D + تفسيره + معانيه المتفرعة")
    
    args = parser.parse_args()
    
    print("جاري تحميل المعجم وبناء مصفوفة المتجهات الطورية (38D)...", end='', flush=True)
    vocab = model.load_vocab()
    engine = FieldEngine(vocab)
    
    if args.letter:
        print_letter_info(args.letter, engine)
        return

    if args.interactive or not args.word:
        interactive_loop(engine, args.space, args.k)
        return

    if args.define:
        print_definition(args.word)
        return

    # تحليل كلمة واحدة مباشرة
    res = engine.find_attraction_repulsion(args.word, space_type=args.space, top_k=args.k)
    print_results(args.word, args.space, res)


if __name__ == '__main__':
    main()
