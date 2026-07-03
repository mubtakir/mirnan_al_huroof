"""Letter Database — قاعدة بيانات الحروف الفيزيائية.

30 حرفاً عربياً + 26 حرفاً إنجليزياً، لكل حرف:
- ω₀ (تردد ذاتي)، operator (±1, 0),
- متجه فلسفي 16D، متجه طوري 22D.
"""
import json
import os
import numpy as np

DIM_NAMES = [
    "concentration", "internal_external", "stability_motion",
    "density", "temperature", "time_accumulation", "time_peak",
    "time_discharge", "motion_linear", "motion_rotary",
    "motion_pulse", "motion_stretch", "motion_slip", "motion_air",
    "axis_v", "mass", "hardness_solid", "penetration", "charge",
    "reference_self", "space_extensionality", "time_causality",
]

class LetterDB:
    def __init__(self, path=None):
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "..", "..",
                                "data", "letter_physics_matrix.json")
        self.data = {}
        self.dim_names = DIM_NAMES
        self.dim = len(self.dim_names)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)
            for ch, info in raw.get("letters", {}).items():
                v = info.get("v", [0]*self.dim)
                if len(v) < self.dim:
                    v = list(v) + [0.0] * (self.dim - len(v))
                elif len(v) > self.dim:
                    v = list(v)[:self.dim]
                v = np.array(v, dtype=np.float64)
                self.data[ch] = {
                    "vector": v,
                    "operator": info.get("operator", "0"),
                    "activation": info.get("a", 0.0),
                    "spin": info.get("s", 0.0),
                    "articulation": info.get("articulation", ""),
                    "manner": info.get("manner", ""),
                    "meaning": info.get("meaning", ""),
                }

    def get_vector(self, letter):
        return self.data.get(letter, {}).get("vector", np.zeros(self.dim))

    def get_operator(self, letter):
        return self.data.get(letter, {}).get("operator", "0")

    def get_omega_0(self, letter):
        v = self.get_vector(letter)
        return 0.5 + 2.0 * np.linalg.norm(v) / np.sqrt(self.dim)

    def get_raw_norm(self, letter):
        """إرجاع القيمة الأصلية (المقدار) لمتجه الحرف قبل أي تطبيع."""
        return float(np.linalg.norm(self.get_vector(letter)))

    def has(self, letter):
        return letter in self.data
