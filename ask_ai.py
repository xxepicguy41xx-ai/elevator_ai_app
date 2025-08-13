import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# === Load data ===
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
sources = json.load(open("sources.json", "r", encoding="utf-8"))
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")

# === Load models ===
model = SentenceTransformer("hkunlp/instructor-base")
RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# === Brand/type inference ===
BRANDS = ["otis", "kone", "thyssen", "thyssenkrupp", "tke", "schindler", "dover", "mce", "smartrise", "gal", "nidec", "magnetek"]
TYPES = ["hydraulic", "traction", "machine_room_less", "mrl", "gearless", "geared"]

def infer_brand(name: str):
    low = name.lower()
    for b in BRANDS:
        if b in low:
            return "tke" if b in ["thyssenkrupp", "tke", "thyssen"] else b
    return "unknown"

def infer_type(name: str):
    low = name.lower()
    for t in TYPES:
        if t in low.replace("-", "_").replace(" ", "_"):
            return "mrl" if t == "machine_room_less" else t
    if "hyd" in low: return "hydraulic"
    if "trl" in low or "mrl" in low: return "mrl"
    return "unknown"

for meta in sources:
    m = meta["manual"]
    meta["brand"] = infer_brand(m)
    meta["etype"] = infer_type(m)

ALL_BRANDS = sorted({s["brand"] for s in sources})
ALL_TYPES = sorted({s["etype"] for s in sources})

# === Helper: filter mask ===
def _filter_mask(indices, brand, etype):
    mask = []
    for i in indices:
        meta = sources[i]
        ok = True
        if brand and meta["brand"] != brand:
            ok = False
        if etype and meta["etype"] != etype:
            ok = False
        mask.append(ok)
    return np.array(mask, dtype=bool)

# === Search with brand/type filter + rerank ===
def search_chunks(query, k=5, pool=50, brand=None, etype=None):
    qv = model.encode([query])
    D, I = index.search(np.array(qv).reshape(1, -1), pool)
    cand_indices = I[0]

    if brand or etype:
        mask = _filter_mask(cand_indices, brand, etype)
        filtered = cand_indices[mask]
        if len(filtered) >= k:
            cand_indices = filtered

    cand_pairs = [(query, chunks[i]) for i in cand_indices]
    scores = RERANKER.predict(cand_pairs)
    order = np.argsort(-scores)

    picked = []
    seen_manuals = set()
    for idx in order:
        i = cand_indices[idx]
        meta = sources[i]
        if brand and meta["brand"] != brand:
            continue
        if etype and meta["etype"] != etype:
            continue
        key = meta["manual"]
        if key in seen_manuals and len(seen_manuals) < 2:
            pass
        picked.append(i)
        seen_manuals.add(key)
        if len(picked) == k: break

    if len(picked) < k:
        for idx in order:
            i = cand_indices[idx]
            if i not in picked:
                picked.append(i)
            if len(picked) == k: break

    return [(chunks[i], sources[i]) for i in picked]

# === Groq client ===
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# === Ask Groq with scope ===
def ask_groq(query, brand=None, etype=None):
    context_chunks = search_chunks(query, k=5, pool=60, brand=brand, etype=etype)

    scope = []
    if brand: scope.append(f"Brand scope: {brand.upper()}")
    if etype: scope.append(f"Elevator type: {etype.upper()}")
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
                "Answer ONLY using the provided context. "
                "Do NOT mix brands; if sources conflict or span multiple brands, prefer the selected brand. "
                "If insufficient brand-scoped context exists, say so and return the closest relevant match WITH citation."
            )
        },
        {
            "role": "user",
            "content": (
                f"{scope_line}\n\n"
                "Use the following manual excerpts to answer. "
                "Cite the manual file and page.\n\n"
                f"{context}\n\nQuestion: {query}"
            )
        }
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.1,
        max_tokens=600
    )
    return response.choices[0].message.content
