"""
الثوابت الفيزيائية للمزنان الجديد.

الثوابت مُحسَّنة بناءً على الاختبارات السابقة.
"""

module Constants

export PLANCK_H, LIGHT_SPEED_C, GRAVITY_G, BOLTZMANN_KB,
       PHASE_DIM, ROOT_DIMS, EXTRA_DIMS, SYNTAX_DIMS, SEMANTIC_DIMS,
       PRAGMATIC_DIMS, TOTAL_DIM, RICH_SEMANTIC_DIMS, TOTAL_RICH_DIM,
       POSITION_WEIGHTS, EXPONENTIAL_ALPHA, ENHANCED_DIM

# ═══════════════════════════════════════════════════════
# الثوابت الفيزيائية
# ═══════════════════════════════════════════════════════

# Natural units for the semantic universe: h=c=G=1, kB=0.1 prevents numerical underflow in Float64
const PLANCK_H = 1.0          # ثابت بلانك
const LIGHT_SPEED_C = 1.0     # سرعة الضوء
const GRAVITY_G = 1.0         # ثابت الجاذبية
const BOLTZMANN_KB = 0.1      # ثابت بولتزمان

# ═══════════════════════════════════════════════════════
# ثوابت النموذج
# ═══════════════════════════════════════════════════════

const TOTAL_VEC_DIM = 10000   # البعد الإجمالي الموحد لذاكرة المتجه
const ROOT_DIMS = 8
const EXTRA_DIMS = 6
const SYNTAX_DIMS = 6
const SEMANTIC_DIMS = 16
const PRAGMATIC_DIMS = 6
const STRUCTURAL_DIMS = ROOT_DIMS + EXTRA_DIMS + SYNTAX_DIMS + SEMANTIC_DIMS + PRAGMATIC_DIMS  # الأبعاد الهيكلية الإجمالية (42 بُعداً)
const PHASE_DIM = TOTAL_VEC_DIM - STRUCTURAL_DIMS  # الأبعاد المخصصة لأطوار الحروف (9958 بُعداً)

const TOTAL_DIM = TOTAL_VEC_DIM

const RICH_SEMANTIC_DIMS = 29
const TOTAL_RICH_DIM = PHASE_DIM + ROOT_DIMS + EXTRA_DIMS + SYNTAX_DIMS + RICH_SEMANTIC_DIMS + PRAGMATIC_DIMS

const POSITION_WEIGHTS = [3.5, 2.5, 2.0, 1.5, 1.2, 1.0, 0.8, 0.6, 0.4, 0.3]

# ═══════════════════════════════════════════════════════
# العامل الأسي
# ═══════════════════════════════════════════════════════

const EXPONENTIAL_ALPHA = 1.0  # معامل العامل الأسي لتكبير التباعد
const ENHANCED_DIM = 27        # بعد المتجه المحسّن (المعاملات الخام بعد العامل الأسي)

end # module Constants
