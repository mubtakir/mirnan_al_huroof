#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
assimilate.py — تدريب تزايدي لميران.

الاستخدام:
  python assimilate.py --data new_data.txt          # دمج ملف
  python assimilate.py --data data/*.txt             # دمج عدة ملفات
  python assimilate.py --data new_data.txt --model model/  # مجلد مخصص

المبدأ: بدلاً من إعادة التدريب الكامل، نأخذ النموذج الحالي (vocab.py, K_*.py, syntax.py)
ونضيف بيانات جديدة عليه: كلمات جديدة → توسع المعجم + مصفوفات K.
"""
import os, sys, glob, time
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, 'reconfigure') and "pytest" not in sys.modules:
    sys.stdout.reconfigure(encoding='utf-8')


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='تدريب تزايدي لميران')
    parser.add_argument('--data', nargs='+', required=True,
                        help='ملفات البيانات الجديدة')
    parser.add_argument('--model', default='model',
                        help='مجلد النموذج (افتراضي: model/)')
    parser.add_argument('--window', type=int, default=5,
                        help='نافذة الارتباط لـ K_sem (افتراضي: 5)')
    parser.add_argument('--alpha', type=float, default=0.15,
                        help='معامل الدمج — 0.0=لا تغيير, 1.0=استبدال (افتراضي: 0.15)')
    args = parser.parse_args()
    
    model_dir = args.model
    
    if not os.path.exists(os.path.join(model_dir, 'vocab.py')):
        print(f'✗ لا يوجد نموذج في {model_dir}. شغّل train.py أولاً.')
        sys.exit(1)
    
    # تحميل النصوص الجديدة
    texts = []
    for pattern in args.data:
        for path in glob.glob(pattern):
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                texts.append(f.read())
                print(f'  📖 {path}: {len(texts[-1]):,} حرف')
    
    if not texts:
        print('✗ لا توجد بيانات للتدريب')
        return
    
    from src.physics.synchronize import Vocabulary, _tokenize
    from src.physics.model_io import load_vocab, load_k, load_syntax, save_model
    
    print(f'\n─── الاستيعاب التزايدي ───')
    
    # 1. تحميل المعجم الحالي
    print('  تحميل المعجم...')
    vocab = load_vocab(model_dir)
    start_size = len(vocab)
    print(f'    حجم المعجم الحالي: {start_size:,} كلمة')
    
    # 2. مسح الكلمات الجديدة
    new_words = set()
    for text in texts:
        for line in text.split('\n'):
            for word in _tokenize(line):
                if word not in vocab.word2id:
                    new_words.add(word)
    
    if new_words:
        print(f'  كلمات جديدة: {len(new_words)}')
        for w in sorted(new_words)[:5]:
            wid = vocab.add_word(w)
            print(f'    + [{wid}] {w}')
        if len(new_words) > 5:
            print(f'    ... و {len(new_words) - 5} أخرى')
    else:
        print('  لا توجد كلمات جديدة')
    
    # 3. تحميل مصفوفات K الحالية
    print('  تحميل مصفوفات K...')
    K_sem = load_k(model_dir, 'K_sem')
    K_syn = load_k(model_dir, 'K_syn')
    K_dial = load_k(model_dir, 'K_dial') if os.path.exists(os.path.join(model_dir, 'K_dial.py')) else None
    
    # 4. دمج النصوص الجديدة
    from src.physics.synchronize import assimilate_text
    
    print('  دمج K_sem...')
    t0 = time.time()
    K_sem = assimilate_text(texts, vocab, K_sem, window=args.window, alpha_blend=args.alpha)
    print(f'    ✓ {K_sem.shape}, nnz={K_sem.nnz:,} ({time.time()-t0:.1f}ث)')
    
    print('  دمج K_syn...')
    t0 = time.time()
    K_syn = assimilate_text(texts, vocab, K_syn, window=2, alpha_blend=args.alpha)
    print(f'    ✓ {K_syn.shape}, nnz={K_syn.nnz:,} ({time.time()-t0:.1f}ث)')
    
    if K_dial is not None:
        print('  دمج K_dial...')
        t0 = time.time()
        K_dial = assimilate_text(texts, vocab, K_dial, window=2, alpha_blend=args.alpha)
        print(f'    ✓ {K_dial.shape}, nnz={K_dial.nnz:,} ({time.time()-t0:.1f}ث)')
    
    # 5. تحديث SyntaxField
    print('  تحديث SyntaxField...')
    from src.physics.grammar_field import SyntaxField
    syntax = load_syntax(model_dir)
    for text in texts:
        for line in text.split('\n'):
            tokens = _tokenize(line)
            if len(tokens) >= 2:
                syntax.observe(tokens)
    syntax.finalize(vocab)
    print(f'    ✓ {len(syntax.bigram_count)} bigrams')
    
    # 6. حفظ النموذج المحدث
    print('  حفظ النموذج...')
    t0 = time.time()
    save_model(model_dir, vocab, K_sem, syntax, K_syn=K_syn, K_dial=K_dial)
    print(f'    ✓ {time.time()-t0:.1f}ث')
    
    print(f'\n{"═" * 50}')
    print(f'  ✓ اكتمل الاستيعاب التزايدي!')
    print(f'  • المعجم: {start_size:,} ← {len(vocab):,} كلمة (+{len(vocab) - start_size})')
    print(f'  • K_sem: {K_sem.shape}')
    print(f'  • النموذج في: {model_dir}')
    print(f'{"═" * 50}')


if __name__ == '__main__':
    main()
