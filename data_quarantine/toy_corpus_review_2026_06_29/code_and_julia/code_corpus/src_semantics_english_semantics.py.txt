# -*- coding: utf-8 -*-
"""
English Semantics - Adapted for Mirnan
======================================

استخراج الجذع (Lemma/Stem) للغة الإنجليزية لتوحيد المتجهات الطورية.
"""

import re
import math
import numpy as np
from typing import Tuple, Optional

class EnglishRootExtractor:
    """
    أداة لاستخراج الأصل أو الجذع للكلمات الإنجليزية.
    لا تستخدم مكتبات خارجية (Zero Dependency).
    """
    
    def __init__(self):
        # A lightweight dictionary of common irregular verbs, especially for programming context.
        self.irregular_verbs = {
            "wrote": "write", "written": "write",
            "ran": "run",
            "built": "build",
            "did": "do", "done": "do",
            "went": "go", "gone": "go",
            "saw": "see", "seen": "see",
            "took": "take", "taken": "take",
            "gave": "give", "given": "give",
            "made": "make",
            "had": "have", "has": "have",
            "was": "be", "were": "be", "is": "be", "am": "be", "are": "be", "been": "be", "being": "be",
            "knew": "know", "known": "know",
            "thought": "think",
            "found": "find",
            "caught": "catch",
            "taught": "teach",
            "bought": "buy",
            "brought": "bring",
            "read": "read", # spelling is same, but keeping for completeness
            "spoke": "speak", "spoken": "speak",
            "chose": "choose", "chosen": "choose",
            "drew": "draw", "drawn": "draw",
            "grew": "grow", "grown": "grow",
            "threw": "throw", "thrown": "throw",
            "flew": "fly", "flown": "fly",
            "swam": "swim", "swum": "swim",
            "began": "begin", "begun": "begin",
            "drank": "drink", "drunk": "drink",
            "rang": "ring", "rung": "ring",
            "sang": "sing", "sung": "sing",
            "sank": "sink", "sunk": "sink",
            "stank": "stink", "stunk": "stink",
            "broke": "break", "broken": "break",
            "tore": "tear", "torn": "tear",
            "wore": "wear", "worn": "wear",
            "swore": "swear", "sworn": "swear",
            "bore": "bear", "born": "bear", "borne": "bear",
            "bit": "bite", "bitten": "bite",
            "hid": "hide", "hidden": "hide",
            "rode": "ride", "ridden": "ride",
            "drove": "drive", "driven": "drive",
            "strove": "strive", "striven": "strive",
            "rose": "rise", "risen": "rise",
            "arose": "arise", "arisen": "arise",
            "wrote": "write", "written": "write",
            "smote": "smite", "smitten": "smite",
            "shook": "shake", "shaken": "shake",
            "forsook": "forsake", "forsaken": "forsake",
            "mistook": "mistake", "mistaken": "mistake",
            "understood": "understand",
            "withstood": "withstand",
            "stood": "stand",
            "sat": "sit",
            "spat": "spit",
            "led": "lead",
            "fed": "feed",
            "bled": "bleed",
            "fled": "flee",
            "held": "hold",
            "beheld": "behold",
            "told": "tell",
            "sold": "sell",
            "heard": "hear",
            "said": "say",
            "paid": "pay",
            "laid": "lay",
            "slept": "sleep",
            "kept": "keep",
            "wept": "weep",
            "crept": "creep",
            "swept": "sweep",
            "leapt": "leap",
            "left": "leave",
            "felt": "feel",
            "knelt": "kneel",
            "dealt": "deal",
            "meant": "mean",
            "dreamt": "dream",
            "burnt": "burn",
            "learnt": "learn",
            "spilt": "spill",
            "spoilt": "spoil",
            "lost": "lose",
            "shot": "shoot",
            "got": "get", "gotten": "get",
            "forgot": "forget", "forgotten": "forget",
            "lit": "light",
            "met": "meet",
            "won": "win",
            "spun": "spin",
            "clung": "cling",
            "flung": "fling",
            "slung": "sling",
            "stung": "sting",
            "strung": "string",
            "swung": "swing",
            "wrung": "wring",
            "hung": "hang",
            "dug": "dig",
            "stuck": "stick",
            "struck": "strike",
            "bound": "bind",
            "found": "find",
            "ground": "grind",
            "wound": "wind",
            "fought": "fight",
            "sought": "seek",
            "besought": "beseech",
            "caught": "catch",
            "taught": "teach",
            "thought": "think",
            "bought": "buy",
            "brought": "bring",
            "put": "put",
            "cut": "cut",
            "hit": "hit",
            "let": "let",
            "set": "set",
            "upset": "upset",
            "shut": "shut",
            "quit": "quit",
            "cost": "cost",
            "hurt": "hurt",
            "spread": "spread",
            "shed": "shed",
            "burst": "burst",
            "cast": "cast",
            "broadcast": "broadcast",
            "forecast": "forecast",
            "thrust": "thrust",
            "split": "split",
            "slit": "slit",
            "spit": "spit",
            "beat": "beat", "beaten": "beat",
            # Common -ing forms where 'e' is dropped
            "writing": "write",
            "making": "make",
            "taking": "take",
            "giving": "give",
            "having": "have",
            "using": "use",
            "creating": "create",
            "moving": "move",
            "changing": "change",
            "saving": "save",
            "closing": "close",
            "coding": "code",
            "typing": "type",
            "parsing": "parse",
            "compiling": "compile",
            "executing": "execute",
            "resolving": "resolve",
            "defining": "define",
            "declaring": "declare",
            "invoking": "invoke",
            "computing": "compute",
        }
        
    def extract(self, word: str) -> Tuple[str, float]:
        """
        يستخرج الجذع (Lemma/Stem) للكلمة الإنجليزية ويعيد (الجذع، مستوى الثقة).
        """
        w = word.lower().strip()
        
        if not w:
            return "", 0.0
            
        # 1. تحقق من الأفعال الشاذة
        if w in self.irregular_verbs:
            return self.irregular_verbs[w], 1.0
            
        # 2. إزالة اللواحق البسيطة (Suffix Stripping)
        # قاعدة -ing
        if w.endswith("ing") and len(w) > 4:
            stem = w[:-3]
            # معالجة تضعيف الحرف الأخير (running -> run)
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in ['l', 's', 'z']:
                stem = stem[:-1]
            return stem, 0.8
            
        # قاعدة -ed
        if w.endswith("ed") and len(w) > 3:
            stem = w[:-2]
            if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in ['l', 's', 'z']:
                stem = stem[:-1]
            return stem, 0.8
            
        # قاعدة -es و -s
        if w.endswith("es") and len(w) > 4:
            if w.endswith("sses") or w.endswith("shes") or w.endswith("ches") or w.endswith("xes") or w.endswith("zes"):
                return w[:-2], 0.8
            elif w.endswith("ies"):
                return w[:-3] + "y", 0.8
            
        if w.endswith("s") and len(w) > 3 and not w.endswith("ss") and not w.endswith("is") and not w.endswith("us"):
            return w[:-1], 0.8
            
        # قاعدة -ly
        if w.endswith("ly") and len(w) > 4:
            if w.endswith("ily"):
                return w[:-3] + "y", 0.8
            return w[:-2], 0.8
            
        # إذا لم يتم تعديل الكلمة
        return w, 0.5


class EnglishCharacterEmbedding:
    """
    تضمين فيزيائي دلالي للحروف الإنجليزية بناءً على الملاحظات الفلسفية (Letters That Speak).
    كل حرف يعكس طاقة أو حركة فيزيائية محددة.
    """
    def __init__(self, dim=16):
        self.dim = dim
        self.semantic_categories = {
            'a': 0, 'A': 0, # الارتفاع، الهيمنة، القمة، الاحتواء
            'b': 1, 'B': 1, # الاصطدام، الامتلاء، الحمل
            'c': 2, 'C': 2, # الاحتواء، الانحناء
            'd': 3, 'D': 3, # الدك، التأثير المادي، الهدم والبناء
            'e': 4, 'E': 4, # العمق النفسي، التركيز الداخلي
            'f': 5, 'F': 5, # الدفع، التدفق الهوائي
            'g': 6, 'G': 6, # التجميع، الدوران المفتوح
            'h': 7, 'H': 7, # التنفس، الارتفاع
            'i': 8, 'I': 8, # التمركز، الاستقامة، الأداة
            'j': 9, 'J': 9, # الانطلاق، القفز
            'k': 10, 'K': 10, # القدرة، الاحتواء الأقصى، العطاء
            'l': 11, 'L': 11, # الامتداد الخطي
            'm': 12, 'M': 12, # المادية، التعددية، الفوضى
            'n': 13, 'N': 13, # الحالة، اللاشيء، الاتصال
            'o': 14, 'O': 14, # الحركة الدائرية، التقدم، العجلة
            'p': 15, 'P': 15, # الدفع الخفيف، النقطة
            'q': 16, 'Q': 16, # البحث، المركز
            'r': 17, 'R': 17, # التموج، الجريان، المقاومة، التكرار
            's': 18, 'S': 18, # الانسياب، التموج المستمر
            't': 19, 'T': 19, # الأداة، المطرقة، الثبات، القطع
            'u': 20, 'U': 20, # الاحتواء المفتوح، التجويف
            'v': 21, 'V': 21, # التجمع بالنقطة، الاحتكاك
            'w': 22, 'W': 22, # التموج العريض، النسيج، التباعد
            'x': 23, 'X': 23, # التقاطع، الانعكاس
            'y': 24, 'Y': 24, # التشعب، التفرع
            'z': 25, 'Z': 25, # الاهتزاز، الانزلاق، الكثافة
        }
        self.num_categories = 26
        self._cache = {}

    def _generate_phase_vector(self, category_idx: int, is_upper: bool) -> np.ndarray:
        vec = np.zeros(self.dim)
        for i in range(self.dim):
            freq = (category_idx + 1) * math.pi / self.num_categories
            # إزاحة طورية إضافية للحروف الكبيرة لتعكس القوة أو التأسيس
            phase_shift = math.pi / 4 if is_upper else 0.0
            
            if i % 2 == 0:
                vec[i] = math.sin(freq * (i + 1) + phase_shift)
            else:
                vec[i] = math.cos(freq * (i + 1) + phase_shift)
        
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec = vec / norm
        return vec

    def get_character_vector(self, char: str) -> np.ndarray:
        if char in self._cache:
            return self._cache[char]
            
        cat_idx = self.semantic_categories.get(char.lower(), -1)
        if cat_idx == -1:
            vec = np.zeros(self.dim)
        else:
            is_upper = char.isupper()
            vec = self._generate_phase_vector(cat_idx, is_upper)
            
        self._cache[char] = vec
        return vec

    def get_word_semantic_vector(self, word: str) -> np.ndarray:
        vectors = []
        weights = []
        
        has_ing = word.lower().endswith("ing")
        
        for i, char in enumerate(word):
            vec = self.get_character_vector(char)
            if np.any(vec):
                vectors.append(vec)
                weight = 1.0 / (i + 1)
                
                # تعزيز طاقة الرنين لحرفي n و g إذا كانت الكلمة تنتهي بـ ing
                if has_ing and i >= len(word) - 3 and char.lower() in ['n', 'g']:
                    weight *= 1.5 
                    
                weights.append(weight)
                
        if not vectors:
            return np.zeros(self.dim)
            
        vectors = np.array(vectors)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        semantic_vec = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(semantic_vec)
        if norm > 1e-10:
            semantic_vec = semantic_vec / norm
            
        return semantic_vec
