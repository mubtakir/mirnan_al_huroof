import os
import json
import re

# Paths
NOUNS_TXT = r"C:\Users\allmy\Desktop\aaa\mirnan_julia\dictionaries_source\0 all_nouns_clean.txt"
VERBS_TXT = r"C:\Users\allmy\Desktop\aaa\mirnan_julia\dictionaries_source\4 verbs_clean.txt"

NOUNS_JSON = r"C:\Users\allmy\Desktop\aaa\mirnan_julia\data\nouns.json"
VERBS_JSON = r"C:\Users\allmy\Desktop\aaa\mirnan_julia\data\verbs.json"

# Normalization mapping (matching WordPhysics in Julia)
NORM_MAP = {
    'آ': 'ا', 'أ': 'ا', 'إ': 'ا',
    'ؤ': 'و', 'ئ': 'ي', 'ة': 'ه', 'ى': 'ي'
}

DIACRITICS = set(['ً', 'ٌ', 'ٍ', 'َ', 'ُ', 'ِ', 'ّ', 'ْ', 'ـ'])

def strip_diacritics(word):
    return "".join(c for c in word if c not in DIACRITICS)

def normalize_word(word):
    # Normalize letters using the mapping
    return "".join(NORM_MAP.get(c, c) for c in word).lower()

def clean_and_parse_file(filepath):
    unique_words = set()
    print(f"Parsing: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"Error: {filepath} does not exist!")
        return []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            # If the line contains "جذرها", split it
            if "جذرها" in line:
                word_part = line.split("جذرها")[0].strip()
            else:
                word_part = line
            
            # Remove any unwanted punctuation or indicators
            # We want to skip lines containing "/", punctuation or numbers
            if any(c in word_part for c in ['/', '(', ')', '[', ']', '.', ',', '?', '!', '*', '-', '+', '=', '_']):
                continue
                
            # If the word is entirely whitespace or digits, skip it
            if not word_part or word_part.isdigit():
                continue
            
            # Clean and split into individual tokens if there are spaces, but typically it is one word
            tokens = word_part.split()
            for token in tokens:
                token = token.strip()
                if not token:
                    continue
                
                # Check if it has any non-Arabic characters (e.g. English, symbols)
                # Arabic unicode range: 0600 - 06FF
                if any(ord(c) < 0x0600 or ord(c) > 0x06FF for c in token):
                    continue
                
                # Strip diacritics
                plain = strip_diacritics(token)
                # Normalize
                norm = normalize_word(plain)
                
                if plain:
                    unique_words.add(plain)
                if norm:
                    unique_words.add(norm)
                if token:
                    unique_words.add(token)

    return sorted(list(unique_words))

def main():
    nouns = clean_and_parse_file(NOUNS_TXT)
    verbs = clean_and_parse_file(VERBS_TXT)
    
    print(f"Total unique nouns found: {len(nouns)}")
    print(f"Total unique verbs found: {len(verbs)}")
    
    # Save nouns
    with open(NOUNS_JSON, 'w', encoding='utf-8') as f:
        json.dump(nouns, f, ensure_ascii=False, indent=2)
    print(f"Saved nouns to {NOUNS_JSON}")
        
    # Save verbs
    with open(VERBS_JSON, 'w', encoding='utf-8') as f:
        json.dump(verbs, f, ensure_ascii=False, indent=2)
    print(f"Saved verbs to {VERBS_JSON}")

if __name__ == "__main__":
    main()
