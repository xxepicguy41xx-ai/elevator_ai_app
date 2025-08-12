import os
import json
import re
from typing import List, Tuple, Optional, Dict

import numpy as np
import faiss
from groq import Groq

# ========= Load data once =========
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
sources = json.load(open("sources.json", "r", encoding="utf-8"))
index = faiss.read_index("faiss_index.index")

# ========= Lazy model loaders =========
_model = None
def get_encoder():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("hkunlp/instructor-base")
    return _model

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
            return "TKE" if b in ["thyssenkrupp","tke","thyssen"] else b.title()
    return "Unknown"

def infer_type(name: str):
    low = name.lower().replace("-", "_").replace(" ", "_")
    for t in TYPES:
        if t in low:
            return "MRL" if t == "machine_room_less" else t.replace("_", " ").title()
    if "hyd" in low:
        return "Hydraulic"
    if "trl" in low or "mrl" in low:
        return "MRL"
    return "Unknown"

for meta in sources:
    m = meta["manual"]
    meta["brand"] = meta.get("brand") or infer_brand(m)
    meta["etype"] = meta.get("etype") or infer_type(m)

ALL_BRANDS = sorted({s["brand"].title() for s in sources})
ALL_TYPES  = sorted({s["etype"].replace("_", " ").title() for s in sources})

# ========= Retrieval =========
def _filter_mask(indices: np.ndarray, brand: Optional[str], etype: Optional[str]) -> np.ndarray:
    if not (brand or etype):
        return np.ones(len(indices), dtype=bool)
    mask = []
    for i in indices:
        meta = sources[int(i)]
        ok = True
        if brand and meta["brand"].title() != brand: ok = False
        if etype and meta["etype"].replace("_", " ").title() != etype: ok = False
        mask.append(ok)
    return np.array(mask, dtype=bool)

def search_chunks(query: str, k: int = 3, pool: int = 30,
                  brand: Optional[str] = None, etype: Optional[str] = None,
                  use_reranker: bool = False) -> List[int]:
    enc = get_encoder()
    qv = enc.encode([query])
    D, I = index.search(np.array(qv).reshape(1, -1), pool)
    cand = I[0]

    if brand or etype:
        mask = _filter_mask(cand, brand, etype)
        filtered = cand[mask]
        if len(filtered) >= max(k, 5):
            cand = filtered

    if use_reranker:
        reranker = get_reranker()
        pairs = [(query, chunks[int(i)]) for i in cand]
        scores = reranker.predict(pairs)
        order = np.argsort(-scores)
        cand = cand[order]

    picked, seen = [], set()
    for i in cand:
        meta = sources[int(i)]
        if brand and meta["brand"].title() != brand: 
            continue
        if etype and meta["etype"].replace("_", " ").title() != etype:
            continue
        key = meta["manual"]
        if key in seen and len(seen) < 2:
            pass
        picked.append(int(i))
        seen.add(key)
        if len(picked) == k:
            break

    if len(picked) < k:
        for i in cand:
            ii = int(i)
            if ii not in picked:
                picked.append(ii)
            if len(picked) == k:
                break

    return picked

# ========= Output cleaning =========
CHART_FENCES = (
    r"```mermaid.*?```", r"```chart.*?```", r"```vega.*?```",
    r"```vegalite.*?```", r"```plotly.*?```", r"```echarts.*?```"
)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE = re.compile(r"^\s*\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\s*$")

def strip_charts_and_tables(text: str) -> str:
    for pat in CHART_FENCES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    lines = []
    for line in text.splitlines():
        if TABLE_RULE.match(line):
            continue
        if TABLE_ROW.match(line):
            clean = re.sub(r"^\s*\|\s*", "", line.strip())
            clean = re.sub(r"\s*\|\s*$", "", clean)
            parts = [p.strip() for p in clean.split("|")]
            lines.append(" • " + " — ".join(parts))
        else:
            lines.append(line)
    return "\n".join(lines).strip()

def ensure_tldr(answer_md: str) -> Tuple[str, str]:
    if re.search(r"(?im)^###\s*tl;?dr", answer_md):
        after = re.split(r"(?im)^###\s*tl;?dr\s*", answer_md, maxsplit=1)[-1].strip()
        first_para = after.split("\n\n")[0].strip()
        return first_para, answer_md
    text = re.sub(r"\s+", " ", re.sub(r"[#>*`_]", " ", answer_md)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    bullets = []
    for s in sentences:
        if len(s) < 8: continue
        bullets.append("• " + s.strip())
        if len(bullets) == 2:
            break
    tldr = "\n".join(bullets) if bullets else "• See answer below."
    full = "### TL;DR\n" + tldr + "\n\n" + answer_md
    return tldr, full

BASE_DOC_URL = os.environ.get("BASE_DOC_URL", "").rstrip("/")
def format_source(meta: Dict) -> str:
    manual = meta["manual"]
    page = meta.get("page", "?")
    brand = meta.get("brand", "").title()
    etype = meta.get("etype", "").replace("_", " ").title()
    if BASE_DOC_URL and manual.lower().endswith(".pdf"):
        link = f"{BASE_DOC_URL}/{manual}#page={page}"
        return f"[{manual} — p.{page}]({link}) ({brand}, {etype})"
    return f"{manual} — p.{page} ({brand}, {etype})"

# ========= Groq client =========
def get_client():
    return Groq(api_key=os.getenv("GROQ_API_KEY"))

# ========= Public API =========
def ask_groq(query: str, brand: Optional[str] = None, etype: Optional[str] = None,
             k: int = 3, pool: int = 30, use_reranker: bool = False,
             temperature: float = 0.1) -> Tuple[str, str, List[Dict]]:
    picked = search_chunks(query, k=k, pool=pool, brand=brand, etype=etype, use_reranker=use_reranker)
    context_lines, used = [], []
    for i in picked:
        text = chunks[i]
        meta = sources[i]
        used.append(meta)
        context_lines.append(f"{text}\n(Source: {meta['manual']} - Page {meta['page']})")

    scope = []
    if brand: scope.append(f"Brand: {brand}")
    if etype: scope.append(f"Type: {etype}")
    scope_line = " | ".join(scope) if scope else "Generic scope"

    messages = [
        {"role": "system", "content": (
            "You are an expert elevator mechanic assistant. "
            "Answer ONLY using the provided context. Do NOT mix brands. "
            "Do NOT include charts, code blocks, images, or markdown tables."
        )},
        {"role": "user", "content": (
            f"{scope_line}\n\nUse ONLY the excerpts below:\n\n" +
            "\n\n".join(context_lines) + f"\n\nQuestion: {query}"
        )}
    ]
    resp = get_client().chat.completions.create(
        model="openai/gpt-oss-120b", messages=messages,
        temperature=temperature, max_tokens=600
    )
    raw = resp.choices[0].message.content.strip()
    clean = strip_charts_and_tables(raw)
    tldr, with_tldr = ensure_tldr(clean)
    return tldr, with_tldr, used
