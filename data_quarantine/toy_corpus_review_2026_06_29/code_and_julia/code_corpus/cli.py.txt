#!/usr/bin/env python3
"""مرنان — واجهة أوامر تفاعلية فيزيائية.

الاستخدام:
  python cli.py [--mode auto|standard|quantum|multiverse|wave|poetic]
                [--beta 2.0] [--k_B 1.0] [--meter kamil] [--rhyme ر]
                [--report] [--interactive]
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if hasattr(sys.stdout, 'reconfigure') and "pytest" not in sys.modules:
    sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import model
from src.physics.orchestrator import PhysicsOrchestrator
from src.physics.generator import Generator


def load_generator():
    """تحميل النموذج من ملفات Python الديناميكية (model/)."""
    print("جاري تحميل مرنان...", end=' ', flush=True)
    data = model.load_model()
    # تحميل الأطياف السياقية (اختبار تجريبي)
    model_dir_path = os.path.dirname(model.__file__)
    spectra = None
    dial_spectra = None
    try:
        sp = os.path.join(model_dir_path, 'contextual_spectra.npy')
        if os.path.exists(sp):
            spectra = np.load(sp)
    except Exception:
        pass
    try:
        dsp = os.path.join(model_dir_path, 'dialogue_spectra.npy')
        if os.path.exists(dsp):
            dial_spectra = np.load(dsp)
    except Exception:
        pass
    gen = Generator(
        data['vocab'], data['K_sem'],
        syntax_field=data['syntax'],
        K_syn=data.get('K_syn'),
        K_dialogue=data.get('K_dial'),
        contextual_spectra=spectra,
    )
    gen.dialogue_spectra = dial_spectra
    print(f"✓ ({len(data['vocab']):,} كلمة{' + طيف' if spectra is not None else ''})")
    return gen


def print_report(report):
    """طباعة تقرير فيزيائي منسّق."""
    print("\n" + "=" * 50)
    print("         التقرير الفيزيائي")
    print("=" * 50)
    print(f"  الوضع:         {report['mode']}")
    print(f"  الإنتروبيا:    {report['entropy']}")
    print(f"  S_crit:        {report['S_crit']}")
    print(f"  k_B:           {report['k_B']}")
    print(f"  β:             {report['beta']}")
    print(f"  الحرارة:       {report['temperature']}")
    print(f"  الالتحام الطوري:{report['phase_coherence']}")
    if report['mass_mean']:
        print(f"  متوسط الكتلة:  {report['mass_mean']}")
        print(f"  انحراف الكتلة: {report['mass_std']}")
    print(f"  DCCF اقتران:   {report['dccf_coupling']}")
    print(f"  PPM حقل:       {report['ppm_field']}")
    print(f"  AMFS مركزية:   {report['amfs_centrality']}")
    print(f"  كاسكيد:        {'ON' if report.get('cascade_enabled') else 'OFF'} ({report.get('cascade_strength', 0.0)})")
    if report.get('dialogue_mode'):
        print(f"  حوار:          ON ({report.get('dialogue_intent', '')} {report.get('dialogue_confidence', 0.0)})")
    print(f"  عدد الكلمات:   {report['word_count']}")
    print("=" * 50)


def interactive_loop(orch):
    """حلقة تفاعلية مع تحكم فيزيائي حي."""
    print("\nمرنان — الوضع التفاعلي (اكتب 'exit' للخروج)")
    print("أوامر: /beta N, /k_B N, /mode NAME, /cascade ON|OFF [strength], /dialogue ON|OFF, /report, /reset, /history, /field WORD")
    print("-" * 50)

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not prompt:
            continue
        if prompt.lower() == 'exit':
            break

        # الأوامر
        if prompt.startswith('/beta'):
            try:
                val = float(prompt.split()[1])
                orch.adjust_beta(val)
                print(f"β = {orch.state.beta:.2f}")
            except (IndexError, ValueError):
                print(f"β = {orch.state.beta:.2f}")
            continue
        if prompt.startswith('/k_B'):
            try:
                val = float(prompt.split()[1])
                orch.adjust_k_B(val)
                print(f"k_B = {orch.state.k_B:.2f}")
            except (IndexError, ValueError):
                print(f"k_B = {orch.state.k_B:.2f}")
            continue
        if prompt.startswith('/mode'):
            try:
                orch.state.mode = prompt.split()[1]
                print(f"الوضع = {orch.state.mode}")
            except IndexError:
                print(f"الوضع = {orch.state.mode}")
            continue
        if prompt == '/report':
            r = orch.get_report()
            print_report(r)
            continue
        if prompt.startswith('/cascade'):
            parts = prompt.split()
            enabled = parts[1].lower() in ('1', 'true', 'on', 'yes') if len(parts) > 1 else True
            strength = float(parts[2]) if len(parts) > 2 else 3.0
            orch.set_cascade(enabled, strength)
            print(f"الكاسكيد = {'ON' if enabled else 'OFF'} (شدة {strength})")
            continue
        if prompt.startswith('/dialogue'):
            parts = prompt.split()
            enabled = parts[1].lower() in ('1', 'true', 'on', 'yes') if len(parts) > 1 else True
            orch.set_dialogue(enabled)
            print(f"الحوار = {'ON' if enabled else 'OFF'}")
            continue
        if prompt == '/reset':
            orch.reset()
            print("إعادة تعيين الحالة الفيزيائية ✓")
            continue
        if prompt == '/history':
            for h in orch.get_history(10):
                print(f"  [{h['state']['mode']}] {h['prompt'][:40]:40s} → {h['result'][:40]}")
            continue

        # توليد
        result = orch.generate(prompt, mode=orch.state.mode)
        if result:
            if result.startswith("###"):
                print(result)
            else:
                print(f"  ↳ {result}")
        else:
            print("  ↳ [توليد فارغ]")


def main():
    parser = argparse.ArgumentParser(description="مرنان — مولد نصوص فيزيائي")
    parser.add_argument('prompt', nargs='*', help="النص المدخل")
    parser.add_argument('--mode', default='auto',
                        choices=['auto', 'standard', 'quantum', 'multiverse', 'wave', 'poetic', 'creative', 'dialogue', 'code', 'math', 'attract', 'field'])
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--k_B', type=float, default=None)
    parser.add_argument('--meter', default=None, help="البحر الشعري")
    parser.add_argument('--rhyme', default=None, help="حرف القافية")
    parser.add_argument('--cascade', action='store_true', help="تفعيل طبقة الهوي التراكمي (Potential Cascade)")
    parser.add_argument('--cascade-strength', type=float, default=None, help="شدة الكاسكيد (1.0-3.0, افتراضي 1.8)")
    parser.add_argument('--report', action='store_true', help="اطبع تقريراً فيزيائياً")
    parser.add_argument('--interactive', '-i', action='store_true', help="وضع تفاعلي")
    parser.add_argument('--max-words', type=int, default=4, help="أقصى عدد كلمات (افتراضي 4 للحوار)")

    args = parser.parse_args()

    gen = load_generator()
    orch = PhysicsOrchestrator(gen)

    if args.beta is not None:
        orch.adjust_beta(args.beta)
    if args.k_B is not None:
        orch.adjust_k_B(args.k_B)
    if args.cascade:
        strength = args.cascade_strength if args.cascade_strength is not None else 1.8
        orch.set_cascade(True, strength)

    if args.interactive:
        interactive_loop(orch)
        return

    prompt = ' '.join(args.prompt)
    if not prompt:
        print("يرجى إدخال نص. استخدم --interactive للوضع التفاعلي.")
        print("مثال: python cli.py السلام عليكم --mode poetic --meter kamil")
        return

    kwargs = {}
    if args.meter:
        kwargs['poetic_meter'] = args.meter
    if args.rhyme:
        kwargs['poetic_rhyme'] = args.rhyme

    result = orch.generate(prompt, max_words=args.max_words, mode=args.mode, **kwargs)
    if result:
        print(result)
    else:
        print("[توليد فارغ]")

    if args.report:
        r = orch.get_report()
        print_report(r)


if __name__ == '__main__':
    main()
