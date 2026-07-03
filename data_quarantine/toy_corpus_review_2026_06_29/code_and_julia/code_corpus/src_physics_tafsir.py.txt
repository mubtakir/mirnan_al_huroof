"""Machine Tafsīr — تفكيك معنى الكلمة عبر التحليل الطيفي لكل حرف + الجذر الصرفي + التطابق الطيفي.

يجمع:
- تحليل كل حرف على حدة (متجه 22D، operator، معنى)
- الجذر الصرفي المستخرج
- الطيف الترددي (FFT)
- الموجة الطيفية والرنين
- الكتلة والطاقة والطور
"""

import numpy as np
from src.physics.letter_db import LetterDB, DIM_NAMES as TAFSIR_DIMS
from src.physics.word_physics import (
    compute_word_phase_vector, compute_word_mass, compute_word_energy,
    compute_word_frequency, compute_extended_phase_vector, _extract_root_light,
)
from src.physics.word_spectrum import compute_word_spectrum, spectral_resonance

_letter_db = LetterDB()


def _describe_operator(op):
    return {'+1': 'بنائي', '-1': 'تدميري', '0': 'محايد', '+': 'بنائي', '-': 'تدميري', 'i': 'تخيلي', 'r': 'رنيني'}.get(op, f'?({op})')


def _letter_meaning_summary(letter):
    """تلخيص معنى الحرف من قاعدة البيانات."""
    info = _letter_db.data.get(letter, {})
    parts = []
    op = info.get('operator', '0')
    parts.append(f"operator={_describe_operator(op)}")
    meaning = info.get('meaning', '')
    if meaning:
        parts.append(f"معنى: {meaning}")
    articulation = info.get('articulation', '')
    if articulation:
        parts.append(f"مخرج: {articulation}")
    return ', '.join(parts)


def _dominant_dims(vector, top_n=3):
    """أقوى 3 أبعاد في المتجه."""
    indices = np.argsort(np.abs(vector))[::-1][:top_n]
    result = []
    for idx in indices:
        if abs(vector[idx]) > 0.01:
            name = TAFSIR_DIMS[idx] if idx < len(TAFSIR_DIMS) else f"dim_{idx}"
            result.append((name, float(vector[idx])))
    return result


def _describe_spectrum(spectrum):
    """ترجمة الطيف الترددي إلى وصف دلالي."""
    if spectrum is None:
        return "لا يمكن حساب الطيف (كلمة قصيرة جداً)"
    freqs = spectrum['frequencies']
    amps = spectrum['amplitudes']
    if not freqs:
        return "طيف فارغ"
    avg_freq = np.mean(freqs)
    max_amp_idx = int(np.argmax(amps))
    dom_freq = freqs[max_amp_idx] if max_amp_idx < len(freqs) else 0
    lines = []
    if avg_freq > 0.3:
        lines.append(f"ترددات عالية ({avg_freq:.2f}) ← معنى تجريدي / فلسفي")
    elif avg_freq > 0.15:
        lines.append(f"ترددات متوسطة ({avg_freq:.2f}) ← معنى ملموس / وظيفي")
    else:
        lines.append(f"ترددات منخفضة ({avg_freq:.2f}) ← معنى حسي / مادي")
    lines.append(f"التردد المسيطر: {dom_freq:.3f}")
    return ' ; '.join(lines)


def _describe_root(root):
    """وصف الجذر الصرفي."""
    if not root:
        return "لم يتم استخراج جذر"
    return f"الجذر: {''.join(root)} (3 أحرف — وزن صرفي ثابت)"


def analyze_word(word, context_words=None):
    """تحليل شامل لكلمة من المنظور الفيزيائي.

    Returns:
        dict يحتوي على كل التحليلات
    """
    result = {'word': word, 'letters': [], 'root': None, 'phase': None, 'mass': 0, 'energy': 0, 'frequency': 0, 'spectrum': None}

    # 1. تحليل كل حرف
    for ch in word:
        vec = _letter_db.get_vector(ch) if _letter_db.has(ch) else np.zeros(22)
        op = _letter_db.get_operator(ch) if _letter_db.has(ch) else '0'
        dominant = _dominant_dims(vec)
        result['letters'].append({
            'char': ch,
            'vector_sample': vec[:6].tolist(),
            'operator': op,
            'operator_desc': _describe_operator(op),
            'dominant_dims': dominant,
            'summary': _letter_meaning_summary(ch),
        })

    # 2. الجذر الصرفي
    root = _extract_root_light(word)
    result['root'] = {
        'letters': root,
        'description': _describe_root(root),
    }

    # 3. الكميات الفيزيائية الأساسية
    pv = compute_word_phase_vector(word)
    result['phase'] = {
        'vector_sample': pv[:6].tolist(),
        'norm': float(np.linalg.norm(pv)),
        'mean_angle': float(np.mean(np.arctan2(np.imag(pv) if np.iscomplexobj(pv) else np.zeros_like(pv), np.real(pv) if np.iscomplexobj(pv) else pv))),
    }
    result['mass'] = float(compute_word_mass(word))
    result['energy'] = float(compute_word_energy(word))
    result['frequency'] = float(compute_word_frequency(word))

    # 4. الطيف الترددي (FFT)
    spectrum = compute_word_spectrum(word)
    if spectrum:
        result['spectrum'] = {
            'frequencies': spectrum['frequencies'],
            'amplitudes': spectrum['amplitudes'],
            'phases': spectrum['phases'],
            'description': _describe_spectrum(spectrum),
        }

    # 5. رنين مع السياق
    if context_words:
        resonances = []
        for cw in context_words:
            r = spectral_resonance(word, cw)
            if r > 0.1:
                resonances.append({'word': cw, 'resonance': r})
        result['context_resonance'] = sorted(resonances, key=lambda x: -x['resonance'])

    return result


def explain(word, context_words=None):
    """توليد شرح طبيعي للكلمة من التحليل الفيزيائي.

    Returns:
        str: شرح عربي طبيعي
    """
    a = analyze_word(word, context_words)
    lines = [f"=== تحليل كلمة: {a['word']} ==="]

    # الكميات الأساسية
    lines.append(f"\n【الكميات الفيزيائية】")
    lines.append(f"الكتلة: {a['mass']:.3f}")
    lines.append(f"الطاقة: {a['energy']:.3f}")
    lines.append(f"التردد الذاتي: {a['frequency']:.3f}")

    # الحروف
    lines.append(f"\n【تحليل الحروف ({len(a['letters'])} حروف)】")
    for li, l in enumerate(a['letters']):
        dims_str = ', '.join([f"{d[0]}={d[1]:+.2f}" for d in l['dominant_dims'][:2]])
        lines.append(f"  {l['char']}: operator={l['operator_desc']} | {dims_str}")

    # الجذر
    if a['root'] and a['root']['letters']:
        lines.append(f"\n【الجذر الصرفي】")
        lines.append(f"  {a['root']['description']}")

    # الطيف
    if a.get('spectrum'):
        lines.append(f"\n【الطيف الترددي】")
        lines.append(f"  {a['spectrum']['description']}")

    # الرنين السياقي
    if a.get('context_resonance'):
        lines.append(f"\n【الرنين مع السياق】")
        for cr in a['context_resonance'][:3]:
            lines.append(f"  مع '{cr['word']}': {cr['resonance']:.2f}")

    return '\n'.join(lines)


def tafsir(phrase, context_words=None):
    """تفصيل كامل — يحلل كل كلمة في العبارة ويقارن الأطياف."""
    words = phrase.split()
    analyses = [analyze_word(w, context_words) for w in words]
    return {
        'words': analyses,
        'resonance_matrix': [
            [spectral_resonance(w1['word'], w2['word']) for w2 in analyses]
            for w1 in analyses
        ],
    }
