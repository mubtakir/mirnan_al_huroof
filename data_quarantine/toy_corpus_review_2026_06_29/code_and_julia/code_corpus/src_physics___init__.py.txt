"""مرنان — محرك توليد نصوص فيزيائي (Physics-Inspired Language Engine).

V8 — Partonic Resonance
========================
4 تحسينات فيزيائية محورية:
- QuantumDensityMatrix: بديل np.average للسياق — يحافظ على العلاقات البينية
- CausalFlowField: حقل تدفق سببي مستمر للاستدلال المنطقي
- Partonic Separation: فصل النواة الحرفية عن الغلاف الدلالي
- Hierarchical Modulation: تعديل موجي AM/FM عبر 4 مستويات

المبادئ:
- الحرف = مذبذب بتردد ذاتي ω₀
- الكلمة = حزمة موجية (wave packet) من تداخل المذبذبات
- المعنى = رنين طوري + اقتران ديناميكي (DCCF)
- التعلم = تعزيز طوري ذاتي (PhaseReinforcement) — لا backprop
- السياق = مصفوفة كثافة كمومية (ρ = Σ pᵢ|ψᵢ⟩⟨ψᵢ|)
- المنطق = حقل تدفق سببي (J(pv) = Σ Cᵢ·(pv_target - pv))
"""

from src.physics.constants import (
    PLANCK_H, LIGHT_SPEED_C, GRAVITY_G, BOLTZMANN_KB,
    PHASE_DIM, ROOT_DIMS, EXTRA_DIMS, SYNTAX_DIMS, TOTAL_DIM,
)
from src.physics.word_physics import (
    compute_word_frequency,
    compute_word_energy,
    compute_word_mass,
    compute_word_phase_vector,
    compute_extended_phase_vector,
    _extract_root_light,
    _compute_root_dims,
    get_letter_db,
    dress_phase_vector,
    dress_extended_phase_vector,
    phase_similarity,
)
from src.physics.ram_core import AttractorMemory
from src.physics.entropy_gate import EntropyGate
from src.physics.symbolic_bridge import SymbolicBridge
from src.physics.morpho_phasic import MorphoPhasicEngine
from src.physics.synchronize import synchronize, Vocabulary
from src.physics.generator import Generator
from src.physics.gravity import gravitational_force
from src.physics.grammar_field import SyntaxField
from src.physics.oscillator import OscillatorEngine
from src.physics.math_bridge import MathBridge
from src.physics.syntax_field import SyntaxFieldCache, compute_syntax_vector, expected_syntax
from src.physics.syntax_monitor import SyntaxMonitor
from src.physics.weight_resonance import WeightResonanceEngine
from src.physics.word_spectrum import compute_word_spectrum, spectral_resonance, spectral_density
from src.physics.code_engine import CodeEngine, tokenize_code, validate_syntax, compile_check, CodePhaseVector, CodeVocabulary, build_K_code, PY_KEYWORDS
from src.physics.spectral_memory import GlobalSpectralMemory, build_contexts_map
from src.physics.resonant_chain import ResonantChain
from src.physics.concept_matrix import ConceptMatrix, build_multi_k
from src.physics.causal_engine import CausalPhaseEngine
from src.physics.local_thermo_gate import local_entropy, local_temperature, compute_theta
from src.physics.dccf import DCCF
from src.physics.ppm import PromptField
from src.physics.amfs import adapt_word
from src.physics.phase_reinforcement import PhaseReinforcement
from src.physics.potential_cascade import PotentialCascadeLayer
from src.physics.dialogue_engine import DialogueEngine
from src.physics.orchestrator import PhysicsOrchestrator, PhysicsState
from src.physics.dialogue_memory import DialogueMemory, DialogueTurn
from src.physics.intent_detector import IntentDetector
from src.physics.associative_memory import AssociativeDialogueMemory, AssociativeEntry
from src.physics.entity_register import EntityRegister
from src.physics.response_planner import ResponsePlanner
from src.physics.sentiment_polarity import compute_word_polarity, compute_sentence_polarity, sentiment_fidelity
from src.physics.rotating_anchor import RotatingAnchor
from src.physics.model_bundle import MirnanModelBundle
from src.physics.tafsir import analyze_word, explain, tafsir
from src.physics.spectral_wave_engine import (
    SpectralCouplingMatrix,
    PhysicsGenerativeNetwork,
    PhysicsTranslator,
    PhaseLogicalInference,
    compute_word_wave,
    wave_at,
    adaptive_context_filter,
    wave_signature,
)
from src.physics.density_matrix import QuantumDensityMatrix
from src.physics.causal_flow import CausalFlowField
