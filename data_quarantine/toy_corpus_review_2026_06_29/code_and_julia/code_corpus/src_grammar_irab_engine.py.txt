import re
from typing import List, Dict

class IrabEngine:
    JAR_PREPS = {"في", "من", "إلى", "عن", "على", "بـ", "لـ", "كـ", "حتى", "رب", "خلا", "عدا", "حاشا", "مع"}
    NASB_NOUNS = {"إن", "أن", "لكن", "ليت", "لعل", "كأن", "إلا"}
    JZM_VERB_PARTICLES = {"لم", "لما", "لا", "لام الأمر"}
    NASB_VERB_PARTICLES = {"أن", "لن", "كي", "إذن", "حتى", "فاء السببية", "واو المعية"}
    INDECLINABLES = {"هذا", "هذه", "هؤلاء", "الذي", "التي", "الذين", "من", "ما", "متى", "أين", "كيف", "نحن"}

    @staticmethod
    def apply_irab(tokens: List[Dict[str, str]]) -> str:
        output = []
        for i, tok in enumerate(tokens):
            word = tok.get("word", "")
            pos = tok.get("pos", "اسم")
            role = tok.get("role", "")
            is_definite = tok.get("definite", False)
            number = tok.get("number", "مفرد")
            is_five_nouns = tok.get("is_five_nouns", False)
            if pos in ("حرف", "أداة") or word in IrabEngine.INDECLINABLES:
                output.append(word)
                continue
            diacritic = IrabEngine._resolve_case(i, tokens, word, pos, role)
            output.append(IrabEngine._apply_diacritic(word, diacritic, is_definite, number, is_five_nouns))
        return " ".join(output)

    @staticmethod
    def _resolve_case(index: int, tokens: List[Dict], word: str, pos: str, role: str) -> str:
        prev_tok = tokens[index - 1] if index > 0 else {}
        prev_word = prev_tok.get("word", "")
        prev_pos = prev_tok.get("pos", "")
        if prev_word in IrabEngine.JAR_PREPS or prev_pos == "حرف جر":
            return "kasra"
        if prev_word in IrabEngine.NASB_NOUNS:
            return "fatha"
        if role in ("مفعول", "خبر منصوب", "اسم إن"):
            return "fatha"
        if role in ("مضاف إليه", "مجرور"):
            return "kasra"
        if pos == "فعل":
            if prev_word in IrabEngine.JZM_VERB_PARTICLES:
                return "sukun"
            if prev_word in IrabEngine.NASB_VERB_PARTICLES:
                return "fatha"
            return "default"
        return "damma"

    @staticmethod
    def _apply_diacritic(word: str, case: str, is_definite: bool, number: str = "مفرد", is_five_nouns: bool = False) -> str:
        clean = re.sub(r'[ًٌٍَُِّْ]', '', word)
        if is_five_nouns:
            root = clean[:-1] if clean.endswith(("و", "ا", "ي")) else clean
            if case == "damma": return root + "و"
            if case == "fatha": return root + "ا"
            if case == "kasra": return root + "ي"
        if number == "مثنى":
            root = clean[:-2] if clean.endswith(("ان", "ين")) else clean
            if case == "damma": return root + "انِ"
            return root + "ينِ"
        if number == "جمع_مذكر":
            root = clean[:-2] if clean.endswith(("ون", "ين")) else clean
            if case == "damma": return root + "ونَ"
            return root + "ينَ"
        if clean.endswith(("ا", "ى", "و", "ي")):
            return clean
        tanween = ""
        if not is_definite and clean[-1] != "ة" and case != "default":
            if case == "damma": tanween = "ٌ"
            elif case == "fatha": tanween = "ً"
            elif case == "kasra": tanween = "ٍ"
        if tanween:
            if case == "fatha" and not clean.endswith(("ة", "اء")):
                return f"{clean}اً"
            return f"{clean}{tanween}"
        if case == "damma": return f"{clean}ُ"
        if case == "fatha": return f"{clean}َ"
        if case == "kasra": return f"{clean}ِ"
        if case == "sukun": return f"{clean}ْ"
        return clean

    @staticmethod
    def score_sentence(words: list) -> float:
        if len(words) < 3:
            return 0.0
        score = 0.0
        i_tokens = IrabEngine._infer_tokens(words)
        try:
            diacritized = IrabEngine.apply_irab(i_tokens)
            score += 0.3
        except Exception:
            return 0.0
        for i, tok in enumerate(i_tokens):
            word = tok["word"]
            if word in IrabEngine.JAR_PREPS and i + 1 < len(i_tokens):
                next_word = i_tokens[i + 1]["word"]
                if next_word.startswith("ال") and len(next_word) > 3:
                    score += 0.15
            if word in {"كان", "كانت", "ليس", "ليست", "أصبح", "صار"} and i + 1 < len(i_tokens):
                score += 0.1
            if i > 0 and i_tokens[i - 1]["word"] in IrabEngine.NASB_NOUNS and word.startswith("ال"):
                score += 0.1
            if i > 0 and i_tokens[i - 1]["word"] in IrabEngine.JZM_VERB_PARTICLES:
                score += 0.1
        return min(score, 1.0)

    @staticmethod
    def _infer_tokens(words: list) -> list:
        tokens = []
        for w in words:
            pos = "اسم"
            role = ""
            is_def = w.startswith("ال")
            if w in IrabEngine.INDECLINABLES or w in IrabEngine.JAR_PREPS:
                pos = "حرف"
            for suffix in ["ت", "ا", "وا", "ن", "ي"]:
                if len(w) >= 4 and w.endswith(suffix) and not w.startswith("ال"):
                    if w[-1] in "تاي":
                        pos = "فعل"
                    break
            tokens.append({
                "word": w, "pos": pos, "role": role,
                "definite": is_def, "number": "مفرد",
                "is_five_nouns": False
            })
        return tokens
