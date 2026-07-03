import re
from src.grammar.irab_engine import IrabEngine

class ArabicGrammar:
    VERB_TENSES = ["الماضي", "المضارع", "الأمر"]
    SENTENCE_TYPES = ["اسمية", "فعلية", "ظرفية"]

    WAZN_RULES = {
        "فَعَلَ": {
            "الماضي": {"مذكر": "{f1}َ{f2}َ{f3}َ", "مؤنث": "{f1}َ{f2}َ{f3}َتْ"},
            "المضارع": {"مذكر": "ي{f1}{f2}َ{f3}", "مؤنث": "ت{f1}{f2}َ{f3}"}
        },
        "أَفْعَلَ": {
            "الماضي": {"مذكر": "أ{f1}{f2}َ{f3}َ", "مؤنث": "أ{f1}{f2}َ{f3}َتْ"},
            "المضارع": {"مذكر": "يُ{f1}{f2}ِ{f3}", "مؤنث": "تُ{f1}{f2}ِ{f3}"}
        }
    }

    LEXICON = {
        ("سار", "المضارع", "مذكر"): "يسير",
        ("جاد", "المضارع", "مذكر"): "يجود",
        ("فاض", "المضارع", "مؤنث"): "تفيض",
        ("انجلى", "المضارع", "مؤنث"): "تنجلي",
        ("أضاء", "المضارع", "مؤنث"): "تضيء",
        ("أضاء", "المضارع", "مذكر"): "يضيء",
        ("بزغ", "المضارع", "مذكر"): "يبزغ",
        ("أشرق", "المضارع", "مؤنث"): "تشرق",
        ("أشرق", "المضارع", "مذكر"): "يشرق",
    }

    @staticmethod
    def conjugate_verb(root_verb, tense, gender="مذكر", number="مفرد"):
        if (root_verb, tense, gender) in ArabicGrammar.LEXICON:
            return ArabicGrammar.LEXICON[(root_verb, tense, gender)]
        wazn = "أَفْعَلَ" if len(root_verb) == 4 or root_verb.startswith("أ") else "فَعَلَ"
        clean_root = root_verb[1:] if wazn == "أَفْعَلَ" and root_verb.startswith("أ") else root_verb
        if len(clean_root) < 3:
            return root_verb
        f1, f2, f3 = clean_root[0], clean_root[1], clean_root[2]
        try:
            rule = ArabicGrammar.WAZN_RULES[wazn][tense][gender]
            return rule.format(f1=f1, f2=f2, f3=f3)
        except KeyError:
            return root_verb

    @staticmethod
    def get_diacritic(word, prev_word="", role=""):
        token = {"word": word, "pos": "اسم", "role": role, "definite": word.startswith("ال"),
                 "number": "مفرد", "is_five_nouns": False}
        tokens = []
        if prev_word:
            tokens.append({"word": prev_word, "pos": "حرف", "role": "أداة", "definite": False})
        tokens.append(token)
        diacritized = IrabEngine.apply_irab(tokens)
        if prev_word and diacritized.startswith(prev_word):
            return diacritized[len(prev_word):].strip()
        return diacritized

    @staticmethod
    def derive_comparative(trait_value):
        mapping = {
            "شديد": "أشدُّ", "واسع": "أوسعُ", "عالية": "أعلى", "تام": "أتمُّ",
            "شاسع": "أوسعُ", "مستيقظ": "أكثرُ استيقاظاً", "قاحلة": "أشدُّ قحطاً", "غالب": "أغلبُ"
        }
        return mapping.get(trait_value, f"أكثرُ {trait_value}")

    @staticmethod
    def is_verb_like(word):
        verb_chars = sum(1 for c in word if c in 'يتسنأف')
        return verb_chars >= 2 and len(word) >= 4 and not word.startswith('ال')

    @staticmethod
    def is_noun_like(word):
        return word.startswith('ال') or word.endswith('ة') or word.endswith('ات') or word.endswith('ون')

    @staticmethod
    def detect_sentence_type(words):
        if not words:
            return "غير معروف"
        first = words[0]
        if ArabicGrammar.is_verb_like(first):
            return "فعلية"
        if first in {"هل", "أ", "ماذا", "كيف", "لماذا", "متى", "أين"}:
            return "استفهامية"
        if first in {"إن", "قد", "لقد", "س", "سوف", "ما", "لا", "لن", "لم"}:
            return "فعلية"
        if first.startswith("ال"):
            return "اسمية"
        return "اسمية"
