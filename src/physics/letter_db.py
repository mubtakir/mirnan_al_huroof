"""Letter Database — قاعدة بيانات الحروف الفيزيائية.

تحتوي على متجهات الحروف (22 بُعداً) وتدعم التحديث والحفظ التفاعلي للمعايرة.
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
        self.path = path
        self.data = {}
        self.dim_names = DIM_NAMES
        self.dim = len(self.dim_names)
        self.raw_data = {"letters": {}, "dim_names": DIM_NAMES}
        
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                self.raw_data = json.load(f)
            for ch, info in self.raw_data.get("letters", {}).items():
                v = np.array(info.get("v", [0]*self.dim), dtype=np.float64)
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

    def set_vector(self, letter, vector):
        """تعديل متجه حرف معين في الذاكرة."""
        if letter in self.data:
            self.data[letter]["vector"] = np.array(vector, dtype=np.float64)
            # تحديث البيانات الخام استعداداً للحفظ
            if "letters" in self.raw_data and letter in self.raw_data["letters"]:
                # تحويل القيم إلى أرقام صحيحة أو عشرية مناسبة
                self.raw_data["letters"][letter]["v"] = [int(x) if x.is_integer() else float(round(x, 4)) for x in vector]

    def get_operator(self, letter):
        return self.data.get(letter, {}).get("operator", "0")

    def get_omega_0(self, letter):
        v = self.get_vector(letter)
        return 0.5 + 2.0 * np.linalg.norm(v) / np.sqrt(self.dim)

    def has(self, letter):
        return letter in self.data

    def save(self):
        """حفظ التعديلات الحالية إلى الملف المصدري JSON."""
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self.raw_data, f, ensure_ascii=False, indent=4)
            print(f"[LetterDB] Saved successfully to {self.path}")
            return True
        except Exception as e:
            print(f"[LetterDB] Error saving matrix: {e}")
            return False
