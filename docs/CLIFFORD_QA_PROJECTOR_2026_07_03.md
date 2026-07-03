# Clifford QA Projector Layer (Semantic Calculus V2)
**Date**: July 3, 2026

## Overview

The **Clifford QA Projector Layer** is a zero-shot semantic retrieval and guidance engine implemented in **Mirnan V9**. By representing sentences as high-dimensional (10,000D) phase vectors and utilizing Clifford algebraic transformations, the projector learns the geometric shift (translation rotor) from question form to statement/answer form. This allows Mirnan to answer queries on unseen topics by projecting the question into statement space and retrieving matching facts directly from the corpus.

---

## Mathematical Foundation

In the semantic universe of Mirnan, words and sentences are represented as normalized phase vectors $\vec{v} \in \mathbb{R}^{10000}$. 

### 1. Translation Vector Calculation
Given a set of pre-trained question-answer pairs $(Q_i, A_i)$ belonging to a relation category $C$, the projector extracts the semantic translation vector $\vec{d}_i$:
$$\vec{d}_i = \vec{v}_{A_i} - \vec{v}_{Q_i}$$

Since both $\vec{v}_{A_i}$ and $\vec{v}_{Q_i}$ are normalized unit vectors, $\vec{d}_i$ represents the transition from the question formulation to the statement formulation.

### 2. Relation-Specific Average Shift
The projector averages these translation vectors for each question category (such as `yes_no`, `method`, `reason`, `definition`, `time`, `place`, `general`):
$$\vec{\Delta}_C = \text{Normalize}\left( \frac{1}{N} \sum_{i=1}^N \vec{d}_i \right)$$

This averaged shift vector $\vec{\Delta}_C$ captures the pure grammatical and structural transition from a question of type $C$ to its corresponding statement form, while topic-specific terms (which appear on both sides) cancel out.

### 3. Zero-Shot Geometric Projection
When the user asks a new question $Q_{\text{new}}$ on an unseen topic, the projector detects its question type $C$ and projects the question vector into statement space:
$$\vec{v}_{\text{projected}} = \text{Normalize}\left( \vec{v}_{Q_{\text{new}}} + \vec{\Delta}_C \right)$$

The resulting vector $\vec{v}_{\text{projected}}$ points directly towards the factual statement form of the answer.

---

## Architecture & Code Map

The implementation is modular and integrates seamlessly into the existing wave-based generation pipeline:

### 1. Core Engine: [clifford_qa_projector.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/src/physics/engines/clifford_qa_projector.jl)
- **`QAProjectorMemory`**: Stores the average transition vectors `shifts` and learning frequencies `counts` for each question type.
- **`get_sentence_vector(text)`**: Computes the 10,000-dimensional normalized sum of the constituent word phase vectors.
- **`learn_qa_shift!(mem, question, answer)`**: Updates the running average shift vector for the corresponding question type.
- **`project_question(mem, question)`**: Computes the projected target vector.
- **`retrieve_answer_facts(mem, question, corpus_sentences)`**: Scores candidate sentences using cosine similarity against the projected target vector.

### 2. Registration & Exports
- **[arabic_group.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/src/physics/groups/arabic_group.jl)**: Includes `clifford_qa_projector.jl` and exports `QAProjectorMemory`, `learn_qa_shift!`, `project_question`, and `retrieve_answer_facts`.
- **[MirnanNew.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/src/MirnanNew.jl)**: Re-exports these functions at the package root level.

### 3. Integration with Generator: [generator.jl](file:///C:/Users/allmy/Desktop/aaa/basil/majnon/models/mirnan/src/physics/engines/generator.jl)
- Added `qa_projector::QAProjectorMemory` as a core field in `MirnanGenerator`.
- **Constructor Harvesting**: During generator initialization, the projector automatically trains itself on all Q&A examples stored inside `hisban.records`.
- **Scoring Bias Injection**: In `_hisban_prompt_guidance`, the projector retrieves the top matching facts from the corpus using `retrieve_answer_facts` and appends their constituent words to `guidance["target_terms"]`. This provides a significant born-rule probability boost (up to `0.35 * confidence`) to words in the projected fact, guiding the beam search to generate the correct answer.

---

## Verification Results

The zero-shot capabilities were verified using a test script:
1. **Training**: The projector learned shifts from 4 Q&A pairs (unrelated to "ibn sina" or "bukhara").
2. **Fact Database**: A corpus of 6 random facts was defined, including `"ولد ابن سينا في مدينة بخارى القديمة"`.
3. **Query**: `"من هو المولود في بخارى؟"` (Unseen question/topic).
4. **Retrieval**:
   * Rank 1 (Similarity: **0.8660**): `"ولد ابن سينا في مدينة بخارى القديمة"` (CORRECT)
   * Rank 2 (Similarity: **0.7818**): `"الماء يغلي عند مئة درجة مئوية تحت الضغط"`
   * Rank 3 (Similarity: **0.7720)**: `"سافر الطالب إلى المدينة المنورة أمس"`
