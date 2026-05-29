"""مرنان النظيف — فقط التجاذب والتنافر الفيزيائي للحروف.
"""
from src.physics.constants import PHASE_DIM
from src.physics.word_physics import (
    compute_word_phase_vector,
    phase_similarity,
    get_letter_db,
)
from src.physics.synchronize import Vocabulary
from src.physics.field_engine import FieldEngine
