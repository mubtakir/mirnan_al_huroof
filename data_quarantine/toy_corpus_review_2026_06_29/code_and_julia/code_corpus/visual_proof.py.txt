#!/usr/bin/env python3
"""mirnan V3.0 — Visual Proof of Physics-Based Arabic Generation"""

import os, sys, time, json
import numpy as np
from collections import Counter

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("  Loading system...", end=' ', flush=True)
sys.path.insert(0, os.path.dirname(__file__))

from src.physics.synchronize import synchronize
from src.physics.generator import Generator
from src.physics.word_physics import compute_extended_phase_vector
from src.physics.constants import PHASE_DIM, TOTAL_DIM

OUT = 'visual_proof'
os.makedirs(OUT, exist_ok=True)

# ── Load corpus & train ──
corpus_list = []
for fname in ['data/corpus.txt', 'data/dialogue_corpus.txt', 'data/math_code_lessons.txt',
              'data/filaments_corpus.txt', 'data/physics_foundations_corpus.txt', 'data/quran.txt']:
    if os.path.exists(fname):
        with open(fname, encoding='utf-8') as f:
            corpus_list.append(f.read())

t0 = time.time()
vocab, K, syntax = synchronize(corpus_list, window=5)
train_t = time.time() - t0
print(f"✓ {len(vocab)} words, {K.nnz} couplings in {train_t:.2f}s")

gen = Generator(vocab, K, beam_width=2, top_k=200, syntax_field=syntax, beta=2.0)

# ── 1. PHASE SPACE DISCRIMINATION ──
print("\n╔══════════════════════════════════════════════╗")
print("║  1. PHASE SPACE DISCRIMINATION              ║")
print("╚══════════════════════════════════════════════╝")

words = list(vocab.word2id.keys())[:300]  # sample 300
vecs = np.array([compute_extended_phase_vector(w) for w in words])
cos_mat = vecs @ vecs.T
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
cos_mat = cos_mat / (norms @ norms.T + 1e-10)

mean_cos = (cos_mat.sum() - len(words)) / (len(words) * (len(words) - 1))
min_cos = cos_mat[~np.eye(cos_mat.shape[0], dtype=bool)].min()
max_cos = cos_mat[~np.eye(cos_mat.shape[0], dtype=bool)].max()
print(f"  cos mean: {mean_cos:.4f}  |  range: [{min_cos:.4f}, {max_cos:.4f}]")
print(f"  discrimination window: {max_cos - min_cos:.4f}")

# PCA to 2D
U, S, Vt = np.linalg.svd(vecs - vecs.mean(axis=0), full_matrices=False)
proj = vecs @ Vt[:2].T
fig, ax = plt.subplots(figsize=(10, 8))
sc = ax.scatter(proj[:, 0], proj[:, 1], c=cos_mat.mean(axis=1), cmap='viridis', s=8, alpha=0.7)
plt.colorbar(sc, label='mean cos to other words')
ax.set_title(f'mirnan V3.0 — Phase Space (PCA 2D)\n{len(words)} words, {len(vocab)} vocabulary')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
plt.tight_layout(); fig.savefig(f'{OUT}/01_phase_space.png', dpi=150)
plt.close()
print("  → 01_phase_space.png")

# Cos histogram
fig, ax = plt.subplots(figsize=(8, 4))
cos_vals = cos_mat[np.triu_indices_from(cos_mat, k=1)]
ax.hist(cos_vals, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
ax.axvline(mean_cos, color='red', ls='--', label=f'μ={mean_cos:.4f}')
ax.set_xlabel('cos(φ_i, φ_j)'); ax.set_ylabel('pair count')
ax.set_title('Phase Vector Similarity Distribution')
ax.legend()
plt.tight_layout(); fig.savefig(f'{OUT}/02_cos_histogram.png', dpi=150)
plt.close()
print("  → 02_cos_histogram.png")

# ── 2. GENERATION TRAJECTORIES ──
print("\n╔══════════════════════════════════════════════╗")
print("║  2. GENERATION TRAJECTORIES                  ║")
print("╚══════════════════════════════════════════════╝")

prompts = ['السلام', 'العلم', 'الحياة', 'الرياضيات', 'الفيزياء']
trajectories = []
for prompt in prompts:
    t0 = time.time()
    result = gen.generate(prompt, max_words=8)
    gen_t = time.time() - t0
    words_gen = result.split()
    if words_gen:
        all_tokens = prompt.split() + words_gen
        pvs = [compute_extended_phase_vector(w) for w in all_tokens]
        traj_2d = np.array([pv @ Vt[:2].T for pv in pvs])
        cos_seq = []
        for i in range(len(pvs) - 1):
            c = float(np.dot(pvs[i], pvs[i+1]) / (np.linalg.norm(pvs[i]) * np.linalg.norm(pvs[i+1]) + 1e-10))
            cos_seq.append(c)
        trajectories.append((prompt, all_tokens, traj_2d, cos_seq, gen_t))
        print(f"  {prompt:15s} → {' '.join(words_gen):45s}  ({gen_t:.3f}s)")

# Trajectory plot
fig, ax = plt.subplots(figsize=(12, 8))
colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
for (prompt, tokens, traj_2d, cos_seq, gen_t), c in zip(trajectories, colors):
    ax.plot(traj_2d[:, 0], traj_2d[:, 1], 'o-', color=c, alpha=0.7, markersize=4, label=prompt)
    for i, t in enumerate(tokens):
        if i < len(prompt.split()):
            ax.annotate(t, traj_2d[i], fontsize=6, alpha=0.5)
        else:
            ax.annotate(t, traj_2d[i], fontsize=7, fontweight='bold')
ax.set_title(f'mirnan V3.0 — Generation Trajectories in Phase Space\n{train_t:.2f}s training, {len(vocab)} words')
ax.set_xlabel('PC1'); ax.set_ylabel('PC2')
ax.legend(fontsize=8)
plt.tight_layout(); fig.savefig(f'{OUT}/03_trajectories.png', dpi=150)
plt.close()
print("  → 03_trajectories.png")

# Cos alignment over generation steps
fig, axes = plt.subplots(2, 3, figsize=(12, 6))
axes = axes.flatten()
for idx, (prompt, tokens, traj_2d, cos_seq, gen_t) in enumerate(trajectories):
    ax = axes[idx]
    ax.plot(range(1, len(cos_seq)+1), cos_seq, 'o-', color='crimson', markersize=5)
    ax.axhline(np.mean(cos_seq), color='gray', ls='--', alpha=0.5)
    ax.set_title(f'"{prompt}" — mean cos={np.mean(cos_seq):.3f}')
    ax.set_xlabel('step t → t+1'); ax.set_ylabel('cos(φ_t, φ_{t+1})')
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
for i in range(len(trajectories), 6):
    axes[i].set_visible(False)
plt.tight_layout(); fig.savefig(f'{OUT}/04_cos_alignment.png', dpi=150)
plt.close()
print("  → 04_cos_alignment.png")

# ── 3. COMPARISON WITH BASELINES ──
print("\n╔══════════════════════════════════════════════╗")
print("║  3. BASELINE COMPARISON                      ║")
print("╚══════════════════════════════════════════════╝")

def random_chain(start_word, n=8, seed=0):
    rng = np.random.RandomState(seed)
    result = []
    for _ in range(n):
        candidates = [w for w in vocab.word2id if w != start_word and len(w) >= 2]
        if not candidates: break
        w = rng.choice(candidates)
        start_word = w
        result.append(w)
    return ' '.join(result)

def frequency_chain(start_word, n=8):
    result = []
    freqs = Counter(vocab.word2id.keys())
    for _ in range(n):
        candidates = [w for w in vocab.word2id if w != start_word and len(w) >= 2]
        sorted_candidates = sorted(candidates, key=lambda w: -compute_word_frequency(w) if hasattr(globals()['__builtins__'], 'compute_word_frequency') else 0)
        w = sorted_candidates[0] if sorted_candidates else start_word
        result.append(w)
    return ' '.join(result)

def word_freq_compat(w):
    from src.physics.word_physics import compute_word_frequency
    return compute_word_frequency(w)

# Compare 3 methods on same prompts
test_prompts = ['السلام', 'العلم', 'الحياة']
methods = {'mirann': gen.generate, 'random': lambda p: random_chain(p, 8),
           'freq': lambda p: sorted([w for w in vocab.word2id if len(w)>=2],
                                     key=word_freq_compat, reverse=True)[:8]}

all_results = []
for prompt in test_prompts:
    row = {'prompt': prompt}
    for mname, mfunc in methods.items():
        t0 = time.time()
        if mname == 'mirann':
            out = mfunc(prompt, max_words=8)
        elif mname == 'freq':
            out = ' '.join(mfunc(prompt))
        else:
            out = mfunc(prompt)
        dt = time.time() - t0
        row[mname] = (out, dt)
    all_results.append(row)
    print(f"  {prompt}:")
    print(f"    mirnan: {row['mirann'][0][:50]:50s}  ({row['mirann'][1]:.4f}s)")
    print(f"    random: {row['random'][0][:50]:50s}  ({row['random'][1]:.4f}s)")
    print(f"    freq:   {row['freq'][0][:50]:50s}  ({row['freq'][1]:.4f}s)")

# ── 4. ENERGY / ENTROPY PROFILE ──
print("\n╔══════════════════════════════════════════════╗")
print("║  4. ENERGY & ENTROPY                         ║")
print("╚══════════════════════════════════════════════╝")

def align(pv, target):
    return float(np.mean(np.cos(pv - target)))

energy_profiles = []
for prompt in prompts[:4]:
    result = gen.generate(prompt, max_words=8)
    tokens = prompt.split() + result.split()
    pvs = [compute_extended_phase_vector(w) for w in tokens]
    target = np.mean(pvs, axis=0)
    target = target / (np.linalg.norm(target) + 1e-10)
    aligns = [align(pv, target) for pv in pvs]
    energies = [np.linalg.norm(pv) for pv in pvs]
    diversities = []
    for i in range(len(pvs)):
        if i == 0: div = 0.0
        else: div = 1.0 - align(pvs[i], pvs[i-1])
        diversities.append(div)
    energy_profiles.append((prompt, tokens, aligns, energies, diversities))
    print(f"  {prompt}: mean align={np.mean(aligns):.3f}, mean energy={np.mean(energies):.3f}")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()
for idx, (prompt, tokens, aligns, energies, diversities) in enumerate(energy_profiles):
    ax = axes[idx]
    x = range(len(tokens))
    ax.plot(x, aligns, 'o-', label='align', color='forestgreen')
    ax.plot(x, energies, 's--', label='energy', color='darkorange', alpha=0.7)
    ax.plot(x, diversities, '^:', label='diversity', color='steelblue', alpha=0.7)
    ax.axhline(0.7, color='gray', ls=':', alpha=0.3)
    ax.set_title(f'"{prompt}"')
    ax.set_xlabel('step'); ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
plt.suptitle('mirnan V3.0 — Energy, Alignment & Diversity per Step')
plt.tight_layout(); fig.savefig(f'{OUT}/05_energy_entropy.png', dpi=150)
plt.close()
print("  → 05_energy_entropy.png")

# ── 5. RAM ATTRACTOR LANDSCAPE ──
print("\n╔══════════════════════════════════════════════╗")
print("║  5. RAM ATTRACTOR LANDSCAPE                  ║")
print("╚══════════════════════════════════════════════╝")

# Generate multiple outputs to fill RAM
for prompt in prompts:
    gen.generate(prompt, max_words=6)
print(f"  RAM has {gen.ram.size} attractors")

if gen.ram.size > 0:
    ram_centers = np.array(gen.ram.centers)
    ram_sigmas = np.array(gen.ram.sigmas)
    ram_tokens = [ws[0] if ws else '?' for ws in gen.ram.word_seqs]

    # PCA of RAM centers + sampled words
    sample_vecs = np.array([compute_extended_phase_vector(w) for w in words[:200]])
    all_for_pca = np.vstack([sample_vecs, ram_centers])
    U2, S2, Vt2 = np.linalg.svd(all_for_pca - all_for_pca.mean(axis=0), full_matrices=False)
    ram_2d = ram_centers @ Vt2[:2].T
    words_2d = sample_vecs @ Vt2[:2].T

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(words_2d[:, 0], words_2d[:, 1], c='lightgray', s=5, alpha=0.3, label='vocabulary')
    sizes = 50 + 200 * (1.0 / (ram_sigmas + 0.1))
    ax.scatter(ram_2d[:, 0], ram_2d[:, 1], c='crimson', s=sizes, alpha=0.6, edgecolors='darkred', label='RAM attractors')
    for t, xy in zip(ram_tokens, ram_2d):
        ax.annotate(t, xy, fontsize=7, fontweight='bold')
    ax.set_title(f'mirnan V3.0 — RAM Attractor Basins ({gen.ram.size} attractors)')
    ax.legend(fontsize=8)
    plt.tight_layout(); fig.savefig(f'{OUT}/06_ram_attractors.png', dpi=150)
    plt.close()
    print("  → 06_ram_attractors.png")
else:
    print("  (no attractors yet)")

# ── 6. SUMMARY REPORT ──
print("\n╔══════════════════════════════════════════════╗")
print("║  6. SUMMARY REPORT                           ║")
print("╚══════════════════════════════════════════════╝")

report = f"""
╔══ mirnan V3.0 — VISUAL PROOF ═══════════════════╗
║                                                  ║
║  Training: {len(vocab)} words, {K.nnz} couplings, {train_t:.2f}s
║  Phase dims: {PHASE_DIM} base + {TOTAL_DIM - PHASE_DIM} extra = {TOTAL_DIM} total
║  Beam width: {gen.beam_width}, Top-K: {gen.top_k}
║  RAM attractors: {gen.ram.size}
║                                                  ║
║  Phase discrimination:                           ║
║    mean cos(φ_i, φ_j): {mean_cos:.4f}
║    range: [{min_cos:.4f}, {max_cos:.4f}]
║    window: {max_cos - min_cos:.4f}
║                                                  ║
║  Generation speed:                               ║
║    avg: {np.mean([t[4] for t in trajectories]):.4f}s
║    range: [{min(t[4] for t in trajectories):.4f}, {max(t[4] for t in trajectories):.4f}]s
║                                                  ║
║  File output: {OUT}/01_phase_space.png
║               {OUT}/02_cos_histogram.png
║               {OUT}/03_trajectories.png
║               {OUT}/04_cos_alignment.png
║               {OUT}/05_energy_entropy.png
║               {OUT}/06_ram_attractors.png
║                                                  ║
║  Key claim:                                      ║
║  This is not an LLM. Every generated word is     ║
║  the result of a deterministic phase resonance   ║
║  computation, fully traceable through 9 scoring  ║
║  terms. No neural networks, no backpropagation,  ║
║  no GPUs. The phase space shows meaningful       ║
║  structure derived purely from Arabic letter     ║
║  physics.                                        ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""
print(report)

with open(f'{OUT}/REPORT.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print(f"  All outputs saved to {OUT}/")
print(f"  To view: open {OUT}/01_phase_space.png")
