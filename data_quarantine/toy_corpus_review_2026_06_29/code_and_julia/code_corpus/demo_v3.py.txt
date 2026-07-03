#!/usr/bin/env python3
"""mirnan V3.0 — Phase-Symbolic Resonance Architecture — Interactive CLI"""

import sys
import io
import time
import re as _re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

# ANSI colors (works in PowerShell 5.1+)
C = {
    "RESET": "\033[0m",
    "BOLD": "\033[1m",
    "DIM": "\033[2m",
    "GREEN": "\033[92m",
    "CYAN": "\033[96m",
    "YELLOW": "\033[93m",
    "RED": "\033[91m",
    "MAGENTA": "\033[95m",
    "BLUE": "\033[94m",
    "WHITE": "\033[97m",
    "BGGREEN": "\033[42m",
    "BGBLUE": "\033[44m",
}

BANNER = f"""{C['CYAN']}{C['BOLD']}
  __  __ ___ _   _ _   _   ___ _   _  ___ 
 |  \/  |_ _| \ | | \ | | |_ _| \ | |/ __|
 | |\/| || ||  \| |  \| |  | ||  \| | (__ 
 | |  | || || |\  | |\  |  | || |\  |\___|
 |_|  |_|___|_| \_|_| \_| |___|_| \_||___/
{C['RESET']}
 {C['GREEN']}Phase-Symbolic Resonance Architecture{C['RESET']}
 {C['DIM']}Physics-inspired Arabic NLG — No Neural Networks{C['RESET']}
"""


def cprint(text, color="", end="\n"):
    print(f"{color}{text}{C['RESET']}", end=end, flush=True)


def print_score(name, value, weight):
    bar_len = 20
    fill = int(abs(value * weight) * bar_len * 4)
    fill = min(fill, bar_len)
    bar = "█" * fill + "░" * (bar_len - fill)
    sign = "+" if value >= 0 else ""
    cprint(f"    {name:20s} │{bar}│ {sign}{value:.3f} × {weight:.2f} = {value * weight:.4f}", C['DIM'])


def print_step(step, word, all_pv, prompt_pv, beam_words):
    cprint(f"\n  {C['BOLD']}Step {step}:{C['RESET']} selecting next word...", C['WHITE'])
    cprint(f"    Current: {' '.join(beam_words)}", C['YELLOW'])


def print_result(prompt, result, gen, gen_time):
    cprint(f"\n  {C['BOLD']}Prompt:{C['RESET']} {prompt}", C['GREEN'])
    cprint(f"  {C['BOLD']}Output:{C['RESET']} {result}", C['WHITE'])
    tokens = result.split()
    words_info = []
    for i, w in enumerate(tokens):
        wid = gen.vocab.get(w)
        wpv = gen._get_pv(w)
        wlen = len(w)
        a = gen.morpho.analyze(w)
        morph_tag = ""
        if a['has_morph'] and a['root']:
            morph_tag = f"{a['root']}/{a['weight']}"
        words_info.append(f"      {C['BOLD']}{i+1}.{C['RESET']} {w:15s}  root={morph_tag or '—':20s}  dim={wpv.shape[0]}")

    cprint(f"  {C['BOLD']}Word Analysis:{C['RESET']}", C['CYAN'])
    for wi in words_info:
        print(f"  {wi}")

    cprint(f"\n  {C['BOLD']}System:{C['RESET']} ⏱ {gen_time:.3f}s  🧠 RAM={gen.ram.size}  🔥 Entropy corrections={gen.entropy.corrections_applied}", C['DIM'])


def run_demo():
    print(BANNER)

    cprint("  Loading corpus and training...", C['YELLOW'])
    t0 = time.time()
    from src.physics.synchronize import synchronize

    with open('data/corpus.txt', encoding='utf-8') as f:
        c1 = f.read()
    with open('data/dialogue_corpus.txt', encoding='utf-8') as f:
        c2 = f.read()

    # نظرية الفتائل — مصدر تدريبي ثالث
    import os as _os
    filaments_path = 'data/filaments_corpus.txt'
    c3 = ''
    if _os.path.exists(filaments_path):
        with open(filaments_path, encoding='utf-8') as f:
            c3 = f.read()
        cprint(f"  ✅ نظرية الفتائل: {len(c3.splitlines())} جملة مُحمَّلة", C['CYAN'])

    corpus_list = [c1, c2] + ([c3] if c3 else [])
    # المبادئ الفيزيائية والرياضية التأسيسية
    physics_path = 'data/physics_foundations_corpus.txt'
    if _os.path.exists(physics_path):
        with open(physics_path, encoding='utf-8') as f:
            c4 = f.read()
        corpus_list.append(c4)
        cprint(f"  ✅ الفيزياء التأسيسية: {len(c4.splitlines())} جملة مُحمَّلة", C['CYAN'])
    # دروس الرياضيات والبرمجة
    math_path = 'data/math_code_lessons.txt'
    if _os.path.exists(math_path):
        with open(math_path, encoding='utf-8') as f:
            c5 = f.read()
        corpus_list.append(c5)
        cprint(f"  ✅ الرياضيات والبرمجة: {len(c5.splitlines())} جملة مُحمَّلة", C['CYAN'])

    # البرمجة باللغة الإنجليزية
    eng_code_path = 'data/english_code_corpus.txt'
    if _os.path.exists(eng_code_path):
        with open(eng_code_path, encoding='utf-8') as f:
            c_eng = f.read()
        corpus_list.append(c_eng)
        cprint(f"  ✅ الإنجليزية البرمجية: {len(c_eng.splitlines())} جملة مُحمَّلة", C['CYAN'])

    quran_path = 'data/quran.txt'
    if _os.path.exists(quran_path):
        with open(quran_path, encoding='utf-8') as f:
            c6 = f.read()
        if c6:
            corpus_list.append(c6)
            cprint(f"  ✅ Quran: {len(c6.splitlines())} lines", C['CYAN'])

    vocab, K, syntax = synchronize(corpus_list, window=5)
    train_time = time.time() - t0

    from src.physics.generator import Generator
    gen = Generator(vocab, K, beam_width=3, top_k=250,
                    syntax_field=syntax, beta=2.0)

    cprint(f"\n  {'─' * 50}", C['DIM'])
    cprint(f"  ✅ Training: {len(vocab)} words, {K.nnz} couplings in {train_time:.2f}s", C['GREEN'])
    cprint(f"  ✅ PV matrix: {gen._all_pv.shape} precomputed", C['GREEN'])
    cprint(f"  ✅ RAM: {gen.ram.size} attractors | Entropy S_crit={gen.entropy.S_crit}", C['GREEN'])
    cprint(f"  ✅ Bridge: {gen.bridge.rule_count} symbolic rules", C['GREEN'])
    m = gen.morpho
    root_count = len([1 for w in vocab.id2word.values() if m.analyze(w)['has_morph'] and m.analyze(w)['root']])
    cprint(f"  ✅ Morph: {root_count}/{len(vocab)} words with known roots", C['GREEN'])
    cprint(f"  ✅ MathBridge: auto-detects math/code contexts", C['GREEN'])
    # Load RAM if saved
    ram_path = 'data/ram_state.json'
    if _os.path.exists(ram_path) and gen.ram.size == 0:
        try:
            gen.ram = AttractorMemory.load(ram_path)
            cprint(f"  ✅ RAM loaded: {gen.ram.size} attractors from {ram_path}", C['GREEN'])
        except Exception:
            pass
    cprint(f"  {'─' * 50}", C['DIM'])

    history = []

    while True:
        try:
            prompt = input(f"\n{C['CYAN']}{C['BOLD']}  >>>{C['RESET']} ")
        except (EOFError, KeyboardInterrupt):
            gen.ram.save('data/ram_state.json')
            cprint(f"\n  {'─' * 50}", C['DIM'])
            cprint(f"  Session ended. {len(history)} exchanges.", C['GREEN'])
            cprint(f"  RAM saved: {gen.ram.size} attractors.", C['DIM'])
            break

        prompt = prompt.strip()
        if not prompt:
            continue
        if prompt.lower() in ('/q', 'quit', 'exit', 'خروج'):
            gen.ram.save('data/ram_state.json')
            cprint(f"  RAM saved: {gen.ram.size} attractors", C['DIM'])
            break
        if prompt == '/reset':
            gen.ram.clear()
            gen._pv_cache.clear()
            gen.entropy.reset()
            cprint("  RAM and cache cleared.", C['YELLOW'])
            continue
        if prompt == '/save':
            ram_path = 'data/ram_state.json'
            gen.ram.save(ram_path)
            cprint(f"  RAM saved: {gen.ram.size} attractors to {ram_path}", C['GREEN'])
            continue
        if prompt == '/status':
            cprint(f"  RAM: {gen.ram.size} attractors", C['DIM'])
            cprint(f"  Entropy corrections: {gen.entropy.corrections_applied}", C['DIM'])
            cprint(f"  Vocab: {len(gen.vocab)} words", C['DIM'])
            cprint(f"  Top-K: {gen.top_k}", C['DIM'])
            continue

        t0 = time.time()
        try:
            result = gen.generate(prompt, max_words=20, mode='multiverse')
        except Exception as e:
            cprint(f"  ❌ Error: {e}", C['RED'])
            continue
        gen_time = time.time() - t0

        if result:
            history.append((prompt, result))
            print_result(prompt, result, gen, gen_time)
        else:
            cprint("  ⚠️  No output generated.", C['YELLOW'])


if __name__ == "__main__":
    run_demo()
