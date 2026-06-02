"""مرنان الحروف النظيف — التجاذب والتنافر الفيزيائي + الحساب الدلالي.
"""
from src.physics.constants import PHASE_DIM
from src.physics.word_physics import (
    compute_word_phase_vector,
    phase_similarity,
    get_letter_db,
)
from src.physics.synchronize import Vocabulary
from src.physics.field_engine import FieldEngine
from src.physics.clifford_math import Multivector22
from src.physics.word_fusion_engine import (
    geometric_word_fusion,
    decompose_fusion,
    full_word_analysis,
    compare_antonyms,
)
from src.physics.semantic_arithmetic import (
    letter_distance_matrix,
    find_fundamental_semantic_units,
    semantic_difference,
    solve_analogy,
    blend_words,
    semantic_midpoint,
)
