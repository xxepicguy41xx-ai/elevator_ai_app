# core_ai.py
import os
import json
import logging
from typing import List, Optional, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------- Files ----------------
CHUNKS_PATH = "chunks.json"
SOURCES_PATH = "sources.json"
EMBEDDINGS_PATH = "embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.index"

# ---------------- Models ----------------
EMBEDDING_MODEL = "hkunlp/instructor-base"   # keeps compatibility with your existing embeddings
GROQ_MODEL = "openai/gpt-oss-120b"           # richer answers via Groq

# ---------------- Brand aliases ----------------
BRAND_MAP = {
    "thyssenkrupp": "tke", "thyssen": "tke", "tke": "tke",
    "otis": "otis", "kone": "kone", "schindler": "schindler",
    "dover": "dover", "mce": "mce", "smartrise": "smartrise",
    "gal": "gal", "nidec": "nidec", "magnetek": "magnetek",
}

def infer_brand_from_name(name: str) -> str:
    low = name.lower()
    for k, v in BRAND_MAP.items():
        if k in low:
            return v
    return "unknown"

def get_groq_api_key() -> Optional[str]:
    """
    Prefer Streamlit secrets if available; fall back to environment variable.
    Works on Streamlit Cloud and local dev.
    """
    # Try Streamlit secrets
    try:
        import streamlit as st
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass
    # Fallback to env var
    return os.getenv("GROQ_API_KEY")

class ElevatorAIPipeline:
    """
    Simple RAG pipeline with brand-only filtering and a rich prompt that
    allows tables, lists, and detailed step-by-step guidance.
    """

    def __init__(self):
        # --- Load data ---
        if not os.path.exists(CHUNKS_PATH) or not os.path.exists(SOURCES_PATH):
            raise RuntimeError("Missing chunks.json or sources.json")
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(FAISS_INDEX_PATH):
            raise RuntimeError("Missing embeddings.npy or faiss_index.index")

        self.chunks: List[str] = json.load(open(CHUNKS_PATH, "r", encoding="utf-8"))
        self.sources: List[Dict] = json.load(open(SOURCES_PATH, "r", encoding="utf-8"))

        emb = np.load(EMBEDDINGS_PATH)
        if emb.dtype != np.float32:
            emb = emb.astype("float32")
        self.embeddings: np.ndarray = emb

        self.index = faiss.read_index(FAISS_INDEX_PATH)
        if self.index.d != self.embeddings.shape[1]:
            raise RuntimeError(f"FAISS dim {self.index.d} != embeddings dim {self.embeddings.shape[1]}")

        # --- Enrich metadata with brand only ---
        for m in self.sources:
            if "brand" not in m or m["brand"] in (None, "", "unknown"):
                m["brand"] = infer_brand_from_name(m.get("manual", ""))

        self.ALL_BRANDS = sorted({m.get("brand", "unknown") for m in self.sources})

        # --- Encoder & Groq ---
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        api_key = get_groq_api_key()
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found. Add it to .streamlit/secrets.toml or set an environment variable.")
        self.client = Groq(api_key=api_key)

        logging.info("ElevatorAIPipeline initialized.")

    # --------- Retrieval (brand-only filter) ---------
    def search(self, query: str, brand: Optional[str] = None, k: int = 5, pool: int = 60) -> List[int]:
        qv = self.encoder.encode([query]).astype("float32")
        _, I = self.index.search(qv.reshape(1, -1), pool)
        cand = I[0]

        if brand:
            filtered = [i for i in cand if self.sources[int(i)].get("brand") == brand]
            if filtered:
                cand = np.array(filtered, dtype=int)

        return [int(i) for i in cand[:k]]

    # --------- Ask LLM with a richer, less strict prompt ---------
    def ask(self, query: str, brand: Optional[str] = None, k: int = 5, pool: int = 60) -> str:
        idxs = self.search(query, brand=brand, k=k, pool=pool)
        if not idxs:
            return "No matching excerpts found. Try removing the brand filter or rephrasing the question."

        excerpts = []
        for i in idxs:
            meta = self.sources[i]
            text = self.chunks[i]
            excerpts.append(
                f"Excerpt — {meta.get('manual','?')} (p.{meta.get('page','?')}, brand={meta.get('brand','?').upper()}):\n{text}"
            )
        context = "\n\n---\n\n".join(excerpts)

        brand_line = f"Selected brand: {brand.upper()}" if brand else "Brand not specified."

        system = (
            "You are an expert elevator mechanic assistant. Use the provided excerpts primarily, "
            "but you may add general best-practice reasoning to fill gaps as long as you clearly separate it.\n"
            "You may use bullet lists, numbered steps, and **Markdown tables** to organize the answer."
        )

        user = (
            f"{brand_line}\n\n"
            "Using the excerpts below, produce a comprehensive troubleshooting and fix guide. Include:\n"
            "1) TL;DR (2 bullets)\n"
            "2) Safety notes (if applicable)\n"
            "3) Quick checklist as a Markdown table with columns: 'Likely Cause' | 'How to Check' | 'Fix'\n"
            "4) Step-by-step diagnostics (numbered)\n"
            "5) Final fix/adjustments (concrete actions)\n"
            "6) If unresolved (next steps)\n"
            "7) Sources: list manuals and page numbers referenced\n\n"
            "If a detail is not present in the excerpts, label it as 'General guidance'.\n\n"
            "----- BEGIN EXCERPTS -----\n"
            f"{context}\n"
            "----- END EXCERPTS -----\n\n"
            f"Question: {query}"
        )

        resp = self.client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            temperature=0.3,
            max_tokens=1400,
        )
        return resp.choices[0].message.content

# Singleton for Streamlit import
try:
    ai_pipeline = ElevatorAIPipeline()
    ALL_BRANDS = ai_pipeline.ALL_BRANDS
except Exception as e:
    logging.critical(f"Failed to initialize AI core: {e}")
    ai_pipeline = None
    ALL_BRANDS = []
