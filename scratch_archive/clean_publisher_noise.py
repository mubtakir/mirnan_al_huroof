import os
import re
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

data_dir = r"c:\Users\allmy\Desktop\aaa\mirnan_julia\data"

TERMS = [
    # Phrases (longer first)
    "جميع الحقوق محفوظة",
    "حقوق الطبع والنشر",
    "حقوق الطبع",
    "حقوق النشر",
    "دار الكتب المصرية",
    "دار إحياء الكتب العربية",
    "دار الكتاب العربي",
    "دار المأمون للتراث",
    "مؤسسة الرسالة",
    "مكتبة المعارف",
    "عالم الكتب",
    "المكتب الاسلامي",
    "المكتبة الاسلامية",
    "منشورات جامعة",
    "الهيئة المصرية العامة",
    "المطبعة المنيرية",
    "إدارة الطباعة",
    "مطبعة العاني",
    "مطبعة الحلبي",
    "مطبعة السعادة",
    "دار الناشر",
    "دار النشر",
    "دار الفكر",
    "دار الكتب",
    "دار العلم",
    "دار المعرفة",
    "دار القلم",
    "دار إحياء",
    "دار الاصلاح",
    "الطبعة الاولى",
    "الطبعة الثانية",
    "الطبعة الثالثة",
    "الطبعة الرابعة",
    "الطبعة الخامسة",
    "الطبعة السادسة",
    
    # Words
    "الناشر",
    "للنشر",
    "والتوزيع",
    "تحقيق",
    "حققه",
    "مطبعة",
    "مؤسسة",
    "طبعة",
    "الطبعة",
    "مكتبة",
    "بيروت",
    "القاهرة",
]

# Function to construct a regex pattern that handles Arabic variations
def make_arabic_regex(phrase):
    pattern = ""
    for char in phrase:
        if char == ' ':
            pattern += r"\s+"
        elif char in ['أ', 'إ', 'آ', 'ا']:
            pattern += r"[أإآا]"
        elif char in ['ة', 'ه']:
            pattern += r"[ةه]"
        elif char in ['ى', 'ي']:
            pattern += r"[ىي]"
        else:
            pattern += re.escape(char)
    # Surround with lookarounds for word boundaries in Arabic (exclude Arabic block U+0600 to U+06FF and word characters)
    return rf"(?<![\u0600-\u06FF\w]){pattern}(?![\u0600-\u06FF\w])"

# Compile all patterns
regex_patterns = []
for term in TERMS:
    pat = make_arabic_regex(term)
    regex_patterns.append((term, re.compile(pat)))

print(f"Loaded {len(regex_patterns)} metadata cleaning patterns.")

files_changed = 0
total_replacements = 0

for root, dirs, files in os.walk(data_dir):
    # Skip system/git dirs
    dirs[:] = [d for d in dirs if d not in ["rules", ".git", ".DS_Store"]]
    for fname in files:
        if fname.endswith(".txt") or fname.endswith(".md"):
            path = os.path.join(root, fname)
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                original_content = content
                # Apply each pattern
                for term, regex in regex_patterns:
                    content, count = regex.subn('\t', content)
                    total_replacements += count
                
                if content != original_content:
                    with open(path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    files_changed += 1
            except Exception as e:
                print(f"Error processing {fname}: {e}")

print(f"Corpus cleaning completed!")
print(f"Total files updated: {files_changed}")
print(f"Total metadata tokens removed/replaced: {total_replacements}")
