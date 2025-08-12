import os
import json
import numpy as np
import faiss
from typing import List, Tuple, Optional

# ========= Fast loads (module level = loaded once per process) =========
# Preload data
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
sources = json.load(open("sources.json", "r", encoding="utf-8"))

# Preload FAISS index
index = faiss.read_index("faiss_index.index")

# Preload encoder (SentenceTransformer can be slow to import; load lazily)
# We import here, but instantiate on first use to avoid slow cold starts.
_model = None
def get_encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        # TIP: switch to "all-MiniLM-L6-v2" for much faster queries with small hit to recall.
        _model = SentenceTransformer("hkunlp/instructor-base")
    return _model

# Optional Cross-Encoder reranker (instantiated only if used)
_reranker = None
def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker

# ========= Brand/type inference (lightweight) =========
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
        if brand  and meta["brand"] != brand: ok = False
        if etype  and meta["etype"] != etype: ok = False
        mask.append(ok)
    return np.array(mask, dtype=bool)

# ========= Fast search with optional rerank =========
def search_chunks(
    query: str,
    k: int = 3,
    pool: int = 30,
    brand: Optional[str] = None,
    etype: Optional[str] = None,
    use_reranker: bool = False
) -> List[Tuple[str, dict]]:
    enc = get_encoder()
    qv = enc.encode([query])
    D, I = index.search(np.array(qv).reshape(1, -1), pool)
    cand = I[0]

    # filter by metadata first (cheap)
    if brand or etype:
        mask = _filter_mask(cand, brand, etype)
        filtered = cand[mask]
        if len(filtered) >= max(k, 5):  # keep some headroom
            cand = filtered

    # optional rerank (only over the small candidate list)
    if use_reranker:
        reranker = get_reranker()
        pairs = [(query, chunks[int(i)]) for i in cand]
        scores = reranker.predict(pairs)
        order = np.argsort(-scores)
        cand = cand[order]

    # pick top-k, prefer not to overspread manuals if brand/type chosen
    picked = []
    seen = set()
    for i in cand:
        meta = sources[int(i)]
        if brand and meta["brand"] != brand: 
            continue
        if etype and meta["etype"] != etype:
            continue
        key = meta["manual"]
        if key in seen and len(seen) < 2:
            pass
        picked.append(int(i))
        seen.add(key)
        if len(picked) == k:
            break

    # backfill if strict filter yields too few
    if len(picked) < k:
        for i in cand:
            ii = int(i)
            if ii not in picked:
                picked.append(ii)
            if len(picked) == k:
                break

    return [(chunks[i], sources[i]) for i in picked]

# ========= Answer with Groq =========
from groq import Groq
def get_client():
    # Prefer Streamlit secrets if available, else env
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
) -> str:
    context_chunks = search_chunks(
        query, k=k, pool=pool, brand=brand, etype=etype, use_reranker=use_reranker
    )

    scope = []
    if brand: scope.append(f"Brand scope: {brand.upper()}")
    if etype: scope.append(f"Type: {etype.upper()}")
    scope_line = " | ".join(scope) if scope else "Generic scope"

    context = "\n\n".join([
        f"{text}\n(Source: {meta['manual']} - Page {meta['page']})"
        for text, meta in context_chunks
    ])

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
            "content": f"{scope_line}\n\n{context}\n\nQuestion: {query}"
        }
    ]

    client = get_client()
    resp = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=temperature,
        max_tokens=500
    )
    return resp.choices[0].message.content
