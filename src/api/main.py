"""mirnan Clean API — Dedicated Field Engine Backend.

سيرفر خفيف لخدمة خاصية التجاذب والتنافر الفيزيائي للحروف.
"""

import os
import sys
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# تأكد من مسار المشروع
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import model as mirnan_model
from src.physics import FieldEngine
from src.physics.vector_interpreter import interpret_letter
from src.physics.word_physics import get_letter_db
from src.semantics.arabic_semantics import decompose_word_definition, LETTER_DEFINITIONS
from src.semantics.letter_meanings import LETTER_RICH_MEANINGS

app = FastAPI(title="Mirnan Clean API", description="Letter-Physics Attraction & Repulsion Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = None
_loading = False

def get_engine():
    global engine, _loading
    if engine is not None:
        return engine
    if _loading:
        raise HTTPException(status_code=503, detail="Model is loading, please retry in a moment")
    _loading = True
    try:
        print("Loading mirnan vocabulary and building FieldEngine...")
        vocab = mirnan_model.load_vocab()
        engine = FieldEngine(vocab)
        return engine
    finally:
        _loading = False


class FieldRequest(BaseModel):
    word: str
    space_type: str = "combined"  # physical, philosophical, combined
    top_k: int = 15
    use_hebbian: bool = True
    vocab_type: str = "full"  # full or benchmark


class WordScore(BaseModel):
    word: str
    score: float


class FieldResponse(BaseModel):
    word: str
    space_type: str
    attracted: List[WordScore]
    repelled: List[WordScore]
    time_taken: float


# We keep /api/chat path to match script.js fetch if needed, but let's implement /api/field
@app.post("/api/field", response_model=FieldResponse)
async def analyze_field(request: FieldRequest):
    eng = get_engine()
    w = request.word.strip()
    if not w:
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    
    t0 = time.time()
    try:
        res = eng.find_attraction_repulsion(w, space_type=request.space_type, top_k=request.top_k, use_hebbian=request.use_hebbian, vocab_type=request.vocab_type)
        time_taken = time.time() - t0
        
        attracted_list = [WordScore(word=word, score=score) for word, score in res["attracted"]]
        repelled_list = [WordScore(word=word, score=score) for word, score in res["repelled"]]
        
        return FieldResponse(
            word=w,
            space_type=request.space_type,
            attracted=attracted_list,
            repelled=repelled_list,
            time_taken=time_taken
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# support old /api/chat endpoint to return a beautiful Markdown representation of the report
@app.post("/api/chat")
async def generate_chat(request: Dict[str, Any]):
    prompt = request.get("prompt", "").strip()
    space_type = request.get("space_type", "combined")
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    eng = get_engine()
    t0 = time.time()
    try:
        # If user passes space_type in the old request format
        mode = request.get("mode", "combined")
        if mode in ("physical", "philosophical", "combined"):
            space_type = mode
        elif mode == "attract" or mode == "field":
            space_type = "combined"
            
        res = eng.find_attraction_repulsion(prompt, space_type=space_type, top_k=15)
        time_taken = time.time() - t0
        
        # Build Markdown response
        md = f"### تحليل حقل الجذب والتنافر للكلمة: **{prompt}**\n"
        md += f"*الفضاء الطوري المستخدم: `{space_type}`*\n\n"
        
        md += "#### 🟢 الكلمات الأكثر تجاذباً (تقارب دلالي وفيزيائي):\n"
        for w, s in res["attracted"]:
            md += f"- **{w}** (درجة التجاذب: `{s:.2f}`)\n"
            
        md += "\n#### 🔴 الكلمات الأكثر تنافراً (تباعد دلالي وفيزيائي):\n"
        for w, s in res["repelled"]:
            md += f"- **{w}** (درجة التنافر: `{s:.2f}`)\n"
            
        return {
            "result": md,
            "physics_report": {
                "entropy": "0.00",
                "beta": "1.00",
                "k_B": "1.00",
                "mode": space_type,
                "word_count": len(res["attracted"]) + len(res["repelled"])
            },
            "time_taken": time_taken
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpdateLetterRequest(BaseModel):
    letter: str
    v: List[float]


@app.get("/api/letters")
async def get_letters():
    from src.physics.word_physics import get_letter_db
    db = get_letter_db()
    # تحويل متجهات numpy إلى قوائم عادية لـ JSON
    letters_dict = {}
    for ch, info in db.data.items():
        letters_dict[ch] = {
            "operator": info["operator"],
            "v": info["vector"].tolist()
        }
    return {
        "letters": letters_dict,
        "dim_names": db.dim_names
    }


@app.post("/api/letters/update")
async def update_letter(request: UpdateLetterRequest):
    from src.physics.word_physics import get_letter_db
    db = get_letter_db()
    
    if not db.has(request.letter):
        raise HTTPException(status_code=404, detail="Letter not found in database")
        
    if len(request.v) != db.dim:
        raise HTTPException(status_code=400, detail=f"Vector length must be exactly {db.dim}")
        
    try:
        # تحديث الحرف في الذاكرة
        db.set_vector(request.letter, request.v)
        # إعادة بناء المصفوفة الطورية للكلمات
        eng = get_engine()
        eng.rebuild()
        return {"message": f"Successfully updated letter '{request.letter}' and rebuilt wave field matrix."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/letters/save")
async def save_letters():
    from src.physics.word_physics import get_letter_db
    db = get_letter_db()
    success = db.save()
    if success:
        return {"message": "Letter physics matrix saved successfully to JSON."}
    else:
        raise HTTPException(status_code=500, detail="Failed to save letter physics matrix.")


@app.get("/api/benchmark")
async def get_benchmark_words():
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "benchmark_vocab.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            words = json.load(f)
        return words
    return []


class BenchmarkUpdateRequest(BaseModel):
    words: List[str]


class DefinitionRequest(BaseModel):
    word: str


class LetterDef(BaseModel):
    letter: str
    meaning: str
    branches: List[str]
    opposite: str
    standard_of: str


class DefinitionResponse(BaseModel):
    word: str
    letters: List[LetterDef]
    definition: str


@app.post("/api/define", response_model=DefinitionResponse)
async def define_word(request: DefinitionRequest):
    w = request.word.strip()
    if not w:
        raise HTTPException(status_code=400, detail="Word cannot be empty")
    parts = decompose_word_definition(w)
    letters_list = []
    for ch, _ in parts:
        rich = LETTER_RICH_MEANINGS.get(ch, {})
        letters_list.append(LetterDef(
            letter=ch,
            meaning=rich.get("core", "?"),
            branches=rich.get("branches", []),
            opposite=rich.get("opposite", "?"),
            standard_of=rich.get("standard_of", "?")
        ))
    def_str = " + ".join(rich.get("core", "?") for rich in LETTER_RICH_MEANINGS.values() 
                         if rich.get("core") != "?")[:100]
    core_meanings = [LETTER_RICH_MEANINGS.get(ch, {}).get("core", "?") for ch, _ in parts]
    def_str = " + ".join(core_meanings)
    return DefinitionResponse(word=w, letters=letters_list, definition=def_str)


@app.get("/api/letters/meanings")
async def get_letter_meanings():
    return LETTER_DEFINITIONS


@app.get("/api/letters/meanings/rich")
async def get_letter_rich_meanings():
    return LETTER_RICH_MEANINGS


@app.get("/api/letter/{letter}")
async def get_letter_info(letter: str):
    if len(letter) != 1:
        raise HTTPException(status_code=400, detail="Provide a single Arabic letter")
    db = get_letter_db()
    if not db.has(letter):
        raise HTTPException(status_code=404, detail=f"Letter '{letter}' not found")
    info = interpret_letter(letter, db)
    return info


class SetDimRequest(BaseModel):
    dim_index: int
    value: float


@app.post("/api/letter/{letter}/set")
async def set_letter_dim(letter: str, request: SetDimRequest):
    if len(letter) != 1:
        raise HTTPException(status_code=400, detail="Provide a single Arabic letter")
    db = get_letter_db()
    if not db.has(letter):
        raise HTTPException(status_code=404, detail=f"Letter '{letter}' not found")
    if request.dim_index < 0 or request.dim_index >= 22:
        raise HTTPException(status_code=400, detail="Dimension index must be 0-21")

    vec = db.get_vector(letter).copy()
    vec[request.dim_index] = request.value
    db.set_vector(letter, vec)
    eng = get_engine()
    eng.rebuild()

    info = interpret_letter(letter, db)
    return {"message": f"Set {letter}[D{request.dim_index:02d}] = {request.value:+.2f}", "letter_info": info}


@app.post("/api/letter/{letter}/save")
async def save_letter(letter: str):
    if len(letter) != 1:
        raise HTTPException(status_code=400, detail="Provide a single Arabic letter")
    db = get_letter_db()
    if not db.has(letter):
        raise HTTPException(status_code=404, detail=f"Letter '{letter}' not found")
    success = db.save()
    if success:
        return {"message": f"Saved letter '{letter}' changes to JSON."}
    raise HTTPException(status_code=500, detail="Failed to save")


@app.post("/api/benchmark/update")
async def update_benchmark_words(request: BenchmarkUpdateRequest):
    import json
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "benchmark_vocab.json")
    try:
        # 1. حفظ الكلمات في ملف JSON
        with open(path, "w", encoding="utf-8") as f:
            json.dump(request.words, f, ensure_ascii=False, indent=2)
            
        # 2. تحديث كائن معجم المعايرة المصغر في الـ engine
        eng = get_engine()
        import model as mirnan_model
        eng.benchmark_vocab = mirnan_model.load_benchmark_vocab()
        
        # 3. إعادة بناء المصفوفات الطورية
        eng.rebuild()
        
        return {"message": "Benchmark vocabulary updated and wave matrices rebuilt successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Mount UI
ui_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "ui")
app.mount("/static", StaticFiles(directory=ui_dir), name="static")

@app.get("/")
async def root():
    index_path = os.path.join(ui_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Mirnan Clean Field API is running. UI index.html not found."}
