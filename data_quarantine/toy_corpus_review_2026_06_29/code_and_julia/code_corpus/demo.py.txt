import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.physics.synchronize import synchronize, _tokenize

with open('data/corpus.txt', encoding='utf-8') as f:
    corpus = f.read()

print(f"Corpus: {len(corpus)} chars, {len(_tokenize(corpus))} tokens")

vocab, K, syntax = synchronize([corpus], window=5, alpha=0.25)
print(f"Vocabulary: {len(vocab)} words, K non-zero: {K.nnz}")

from src.physics.generator import Generator
gen = Generator(vocab, K, syntax_field=syntax)

while True:
    try:
        prompt = input("\n>>> ")
    except (EOFError, KeyboardInterrupt):
        break
    if not prompt:
        continue
    response = gen.generate(prompt, max_words=12)
    print(f"  {response}")
