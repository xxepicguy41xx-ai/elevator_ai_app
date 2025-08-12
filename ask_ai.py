import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Optional, Dict

# ========= Load data once =========
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
sources = json.load(open("sources.json", "r", encoding="utf-8"))

index = faiss.read_index("faiss_index.index")

# Lazy encoder load (speeds up cold start)
_model = None
def get_encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("hkunlp/instructor-base")
        # For speed, you can switch to: "all-MiniLM-L6-v2"
    return _model

# Optional reranker (only instantiated if requested)
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ========= Brand/type inference =========
BRANDS = ["otis","kone","thyssen","thyssenkrupp","tke","schindler","dover","mce","smartrise","gal","nidec","magnetek"]
TYPES  = ["hydraulic","traction","machine_room_less","mrl","gearless","geared"]

def infer_brand(name: str):
    low = name.lower()
    for b in BRANDS:
        if b in low:
            return "tke" if b in ["thyssenkrupp","tke","thyssen"] else b
    return "unknown"

def infer_type(name: str):
    low = name.lower().replace("-", "_").replace(" ", "_")
    for t in TYPES:
        if t in low:
            return "mrl" if t == "machine_room_less" else t
    if "hyd" in low: return "hydraulic"
    if "trl" in low or "mrl" in low: return "mrl"
    return "unknown"

for meta in sources:
    m = meta["manual"]
    meta["brand"] = meta.get("brand") or infer_brand(m)
    meta["etype"] = meta.get("etype") or infer_type(m)

ALL_BRANDS = sorted({s["brand"] for s in sources})
ALL_TYPES  = sorted({s["etype"] for s in sources})

def _filter_mask(indices: np.ndarray, brand: Optional[str], etype: Optional[str]) -> np.ndarray:
    if not (brand or etype):
        return np.ones(len(indices), dtype=bool)
    mask = []
    for i in indices:
        meta = sources[int(i)]
        ok = True
        if brand and meta["brand"] != brand: ok = False
        if etype and meta["etype"] != etype: ok = False
        mask.append(ok)
    return np.array(mask, dtype=bool)

def search_chunks(
    query: str,
    k: int = 3,
    pool: int = 30,
    brand: Optional[str] = None,
    etype: Optional[str] = None,
    use_reranker: bool = False
) -> List[int]:
    enc = get_encoder()
    qv = enc.encode([query])
    D, I = index.search(np.array(qv).reshape(1, -1), pool)
    cand = I[0]

    # cheap filter first
    if brand or etype:
        mask = _filter_mask(cand, brand, etype)
        filtered = cand[mask]
        if len(filtered) >= max(k, 5):
            cand = filtered

    # rerank if needed
    if use_reranker:
        reranker = get_reranker()
        pairs = [(query, chunks[int(i)]) for i in cand]
        scores = reranker.predict(pairs)
        order = np.argsort(-scores)
        cand = cand[order]

    picked = []
    seen_manuals = set()
    for i in cand:
        meta = sources[int(i)]
        if brand and meta["brand"] != brand: 
            continue
        if etype and meta["etype"] != etype:
            continue
        key = meta["manual"]
        if key in seen_manuals and len(seen_manuals) < 2:
            pass
        picked.append(int(i))
        seen_manuals.add(key)
        if len(picked) == k:
            break

    # backfill if strict filter too narrow
    if len(picked) < k:
        for i in cand:
            ii = int(i)
            if ii not in picked:
                picked.append(ii)
            if len(picked) == k:
                break

    return picked

# Optional base URL to make manual links clickable (e.g., a public S3/SharePoint dir)
BASE_DOC_URL = os.environ.get("BASE_DOC_URL", "").rstrip("/")

def format_source(meta: Dict) -> str:
    # If you have hosted PDFs, make them clickable
    manual = meta["manual"]
    page = meta.get("page", "?")
    if BASE_DOC_URL and manual.lower().endswith(".pdf"):
        link = f"{BASE_DOC_URL}/{manual}#page={page}"
        return f"[{manual} — p.{page}]({link})"
    return f"{manual} — p.{page}"

# ======== Groq client ========
from groq import Groq
def get_client():
    api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

def ask_groq(
    query: str,
    brand: Optional[str] = None,
    etype: Optional[str] = None,
    k: int = 3,
    pool: int = 30,
    use_reranker: bool = False,
    temperature: float = 0.1
) -> Tuple[str, List[Dict]]:
    picked_indices = search_chunks(query, k=k, pool=pool, brand=brand, etype=etype, use_reranker=use_reranker)

    # Build compact context
    context_lines = []
    used = []
    for i in picked_indices:
        text = chunks[i]
        meta = sources[i]
        used.append(meta)
        context_lines.append(f"{text}\n(Source: {meta['manual']} - Page {meta['page']})")

    scope = []
    if brand: scope.append(f"Brand: {brand.upper()}")
    if etype: scope.append(f"Type: {etype.upper()}")
    scope_line = " | ".join(scope) if scope else "Generic scope"

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert elevator mechanic assistant. "
                "Answer ONLY using the provided context. Do NOT mix brands. "
                "If brand-specific context is insufficient, say so briefly and cite the closest relevant source."
            )
        },
        {
            "role": "user",
            "content": f"{scope_line}\n\n" +
                       "\n\n".join(context_lines) +
                       f"\n\nQuestion: {query}"
        }
    ]

    client = get_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    answer = resp.choices[0].message.content.strip()
    return answer, used
