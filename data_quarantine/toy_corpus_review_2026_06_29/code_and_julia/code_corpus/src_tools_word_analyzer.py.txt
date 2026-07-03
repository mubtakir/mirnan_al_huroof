"""Word Analyzer — تحليل فيزيائي ومعنوي لأي كلمة إنجليزية أو عربية.

الاستخدام:
    python -m src.tools.word_analyzer book
    python -m src.tools.word_analyzer كتاب
    python -m src.tools.word_analyzer water fire sky

يعرض لكل حرف: ω₀, المخرج, المعنى, + فيزياء الكلمة الكلية.
"""
import sys
import numpy as np
from src.physics.word_physics import (
    compute_word_frequency, compute_word_mass, compute_word_energy,
    compute_word_phase_vector, _normalize_letters,
)
from src.physics.letter_db import LetterDB, DIM_NAMES

_db = None
def get_db():
    global _db
    if _db is None:
        _db = LetterDB()
    return _db


def analyze_letter(letter):
    """تحليل حرف واحد: فيزياء + معنى + مخرج."""
    db = get_db()
    if not db.has(letter):
        return None
    
    info = db.data.get(letter, {})
    v = info.get('vector', np.zeros(22))
    omega_0 = db.get_omega_0(letter)
    operator = info.get('operator', '0')
    activation = info.get('activation', 0.0)
    spin = info.get('spin', 0.0)
    articulation = info.get('articulation', '?')
    manner = info.get('manner', '?')
    meaning = info.get('meaning', '')
    
    # أقوى 3 أبعاد في المتجه
    top_dims = sorted(
        [(DIM_NAMES[i], v[i]) for i in range(len(v))],
        key=lambda x: -abs(x[1] - 0.5)
    )[:3]
    
    return {
        'letter': letter,
        'omega_0': round(omega_0, 3),
        'operator': operator,
        'activation': round(activation, 3),
        'spin': round(spin, 3),
        'articulation': articulation,
        'manner': manner,
        'meaning': meaning,
        'top_dimensions': [(name, round(val, 3)) for name, val in top_dims],
        'vector_norm': round(np.linalg.norm(v), 3),
    }


def analyze_word(word):
    """تحليل كلمة كاملة: فيزياء + معاني الحروف."""
    db = get_db()
    norm = _normalize_letters(word)
    letters_info = []
    
    for ch in norm:
        info = analyze_letter(ch)
        if info:
            letters_info.append(info)
    
    freq = compute_word_frequency(word)
    mass = compute_word_mass(word)
    energy = compute_word_energy(word)
    pv = compute_word_phase_vector(word)
    pv_norm = np.linalg.norm(pv)
    
    # هل الكلمة عربية أم إنجليزية؟
    has_arabic = any('\u0600' <= c <= '\u06ff' for c in word)
    has_english = any(c.isascii() and c.isalpha() for c in word)
    lang = 'العربية' if has_arabic else 'English' if has_english else 'mixed'
    
    return {
        'word': word,
        'normalized': norm,
        'language': lang,
        'frequency': round(freq, 3),
        'mass': round(mass, 6),
        'energy': round(energy, 6),
        'phase_vector_norm': round(pv_norm, 3),
        'phase_vector_sum': round(float(pv.sum()), 3),
        'num_letters': len(norm),
        'letters': letters_info,
    }


def print_analysis(result):
    """طباعة التحليل بشكل منسق."""
    w = result['word']
    lang = result['language']
    print(f"\n{'='*60}")
    print(f"  {w}  ({lang}, normalized: {result['normalized']})")
    print(f"{'='*60}")
    print(f"  frequency: {result['frequency']}")
    print(f"  mass:      {result['mass']}")
    print(f"  energy:    {result['energy']}")
    print(f"  |pv|:      {result['phase_vector_norm']}")
    print(f"  ∑pv:       {result['phase_vector_sum']}")
    print(f"  letters:   {result['num_letters']}")
    print(f"{'-'*60}")
    
    for li in result['letters']:
        meaning = li['meaning'][:55] if li['meaning'] else '—'
        top = ', '.join(f"{n}={v}" for n, v in li['top_dimensions'])
        print(f"  {li['letter']}: ω₀={li['omega_0']:.2f}  "
              f"{li['articulation']:12s} {li['manner']:18s}  "
              f"{li['activation']:.2f}")
        if li['meaning']:
            print(f"    ↳ {meaning}")
        print(f"    ↳ {top}")
    
    # إن وجد التفسير التركيبي (word as miniature narrative)
    print()


def main():
    words = sys.argv[1:] if len(sys.argv) > 1 else ['hello', 'world', 'water', 'fire', 'sky', 'book']
    for w in words:
        try:
            result = analyze_word(w)
            print_analysis(result)
        except Exception as e:
            print(f"  {w}: error — {e}")


if __name__ == '__main__':
    main()
