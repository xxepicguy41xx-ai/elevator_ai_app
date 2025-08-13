import os
import re
import json
import logging
from typing import List, Tuple, Optional, Dict

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------- Config -----------------
CHUNKS_PATH = "chunks.json"
SOURCES_PATH = "sources.json"
EMBEDDINGS_PATH = "embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.index"

EMBEDDING_MODEL = "hkunlp/instructor-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama3-70b-8192"   # strong Groq model

# Brand / Type normalization
BRAND_MAP = {
    "thyssenkrupp": "tke", "thyssen": "tke", "tke": "tke",
    "otis": "otis", "kone": "kone", "schindler": "schindler",
    "dover": "dover", "mce": "mce", "smartrise": "smartrise",
    "gal": "gal", "nidec": "nidec", "magnetek": "magnetek"
}
TYPE_MAP = {
    "machine_room_less": "mrl", "mrl": "mrl", "trl": "mrl",
    "hydraulic": "hydraulic", "hyd": "hydraulic",
    "traction": "traction", "gearless": "gearless", "geared": "geared"
}

# ---------------- Helpers: sanitize output & TL;DR ----------------
CHART_FENCES = (
    r"```mermaid.*?```", r"```chart.*?```", r"```vega.*?```",
    r"```vegalite.*?```", r"```plotly.*?```", r"```echarts.*?```"
)
TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
TABLE_RULE = re.compile(r"^\s*\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\s*$")

def strip_charts_and_tables(text: str) -> str:
    # Remove fenced chart/code blocks
    for pat in CHART_FENCES:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    # Convert markdown tables to bullet lines
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
    # If TL;DR heading exists, extract its first paragraph
    if re.search(r"(?im)^###\s*tl;?dr", answer_md):
        after = re.split(r"(?im)^###\s*tl;?dr\s*", answer_md, maxsplit=1)[-1].strip()
        first_para = after.split("\n\n")[0].strip()
        return first_para, answer_md
    # Otherwise synthesize from first sentences
    text = re.sub(r"\s+", " ", re.sub(r"[#>*`_]", " ", answer_md)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    bullets = []
    for s in sentences:
        if len(s) < 8:
            continue
        bullets.append("• " + s.strip())
        if len(bullets) == 2:
            break
    tldr = "\n".join(bullets) if bullets else "• See detailed steps and sources below."
    full = "### TL;DR\n" + tldr + "\n\n" + answer_md
    return tldr, full

# Optional: base URL for hosted PDFs to make links clickable
BASE_DOC_URL = os.environ.get("BASE_DOC_URL", "").rstrip("/")

def format_source(meta: Dict) -> str:
    manual = meta.get("manual", "")
    page = meta.get("page", "?")
    brand = meta.get("brand", "unknown").upper()
    etype = meta.get("etype", "unknown").upper()
    if BASE_DOC_URL and manual.lower().endswith(".pdf"):
        link = f"{BASE_DOC_URL}/{manual}#page={page}"
        return f"[{manual} — p.{page}]({link}) ({brand}, {etype})"
    return f"{manual} — p.{page} ({brand}, {etype})"

# ---------------- Core Pipeline ----------------
class ElevatorAIPipeline:
    """
    End-to-end RAG pipeline:
    - load data and indices
    - brand/type enrichment
    - vector search + metadata filter + rerank
    - build strict, comprehensive repair/troubleshooting prompts
    - call Groq LLM
    - sanitize + ensure TL;DR
    """

    def __init__(self):
        logging.info("Initializing ElevatorAIPipeline...")
        # Load data
        self.chunks = self._load_json(CHUNKS_PATH, "Chunks")
        self.sources = self._load_json(SOURCES_PATH, "Sources")
        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.index = faiss.read_index(FAISS_INDEX_PATH)

        # Models
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        # Groq client
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable not set.")
        self.groq_client = Groq(api_key=api_key)

        # Enrich metadata
        self._enrich_source_metadata()
        self.all_brands = sorted({s.get("brand", "unknown") for s in self.sources})
        self.all_types = sorted({s.get("etype", "unknown") for s in self.sources})

        logging.info("Pipeline initialized successfully.")

    # ---------- IO ----------
    def _load_json(self, path, name):
        logging.info(f"Loading {name} from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- Metadata ----------
    def _infer_metadata(self, name, mapping):
        lowered = name.lower().replace("-", "_").replace(" ", "_")
        for k, v in mapping.items():
            if k in lowered:
                return v
        return "unknown"

    def _enrich_source_metadata(self):
        logging.info("Enriching source metadata with brand and type…")
        for meta in self.sources:
            fname = meta.get("manual", "")
            meta["brand"] = meta.get("brand") or self._infer_metadata(fname, BRAND_MAP)
            meta["etype"] = meta.get("etype") or self._infer_metadata(fname, TYPE_MAP)

    # ---------- Retrieval ----------
    def search_and_rerank(
        self, query: str, brand: str = None, etype: str = None, k: int = 5, pool: int = 50
    ) -> List[int]:
        """
        1) FAISS vector search to get a pool
        2) filter by brand/type
        3) rerank with cross-encoder
        4) return top-k indices
        """
        qv = self.embedding_model.encode([query])
        _, I = self.index.search(qv, pool)
        candidates = I[0]

        # Filter by metadata if set
        if brand or etype:
            filtered = []
            for i in candidates:
                meta = self.sources[int(i)]
                if brand and meta.get("brand") != brand:
                    continue
                if etype and meta.get("etype") != etype:
                    continue
                filtered.append(int(i))
            if filtered:
                candidates = np.array(filtered, dtype=int)

        if len(candidates) == 0:
            logging.warning("No documents matched the filter criteria.")
            return []

        # Rerank
        pairs = [(query, self.chunks[int(i)]) for i in candidates]
        scores = self.reranker.predict(pairs)
        order = np.argsort(-scores)
        ranked = candidates[order]

        # Cap 1–2 per manual to avoid dilution
        picked = []
        seen_manuals = {}
        for i in ranked:
            meta = self.sources[int(i)]
            m = meta["manual"]
            seen_manuals[m] = seen_manuals.get(m, 0)
            if brand and meta.get("brand") != brand:
                continue
            if etype and meta.get("etype") != etype:
                continue
            if seen_manuals[m] >= 2:
                continue
            picked.append(int(i))
            seen_manuals[m] += 1
            if len(picked) == k:
                break

        if len(picked) < k:
            for i in ranked:
                ii = int(i)
                if ii not in picked:
                    picked.append(ii)
                if len(picked) == k:
                    break

        return picked

    # ---------- Context builder ----------
    def _build_context(self, indices: List[int]) -> Tuple[str, List[Dict]]:
        """
        Groups chunks by manual and formats compact, readable excerpts.
        """
        by_manual: Dict[str, List[Tuple[str, Dict]]] = {}
        used: List[Dict] = []
        for i in indices:
            t = self.chunks[i]
            m = self.sources[i]
            used.append(m)
            by_manual.setdefault(m["manual"], []).append((t, m))

        parts = []
        for manual, entries in by_manual.items():
            header = f"=== {manual} ==="
            body_lines = []
            for text, meta in entries:
                body_lines.append(f"(Page {meta.get('page', '?')} | Brand {meta.get('brand','?').upper()} | Type {meta.get('etype','?').upper()})")
                body_lines.append(text)
            parts.append(header + "\n" + "\n".join(body_lines))
        return "\n\n".join(parts), used

    # ---------- Public ask ----------
    def ask(self, query: str, brand: str = None, etype: str = None) -> str:
        """
        Returns a comprehensive troubleshooting/fix guide with TL;DR and explicit steps.
        """
        logging.info(f"Query: '{query}' | Brand: {brand} | Type: {etype}")
        idxs = self.search_and_rerank(query, brand=brand, etype=etype, k=5, pool=60)
        if not idxs:
            return (
                "### TL;DR\n"
                "• I couldn't find brand/type-matching content for that question.\n\n"
                "### Answer\n"
                "Please widen the brand/type filters or rephrase your question.\n"
            )

        context_str, used = self._build_context(idxs)

        scope_bits = []
        if brand: scope_bits.append(f"BRAND={brand.upper()}")
        if etype: scope_bits.append(f"TYPE={etype.upper()}")
        scope_line = " | ".join(scope_bits) if scope_bits else "GENERIC SCOPE"

        # **STRICT** system prompt for comprehensive guides (no charts/tables)
        system_prompt = (
            "You are an expert elevator mechanic assistant. Use ONLY the provided manual excerpts. "
            "Do NOT mix brands; prefer the selected brand if there is any conflict. "
            "OUTPUT FORMAT (use these exact headings):\n"
            "### TL;DR\n"
            "- 1–2 bullet summary of the most likely fix.\n\n"
            "### Safety First\n"
            "- Lockout/Tagout, secure car, keep doors guarded, PPE, and other safety warnings from the context if present.\n\n"
            "### Tools Needed\n"
            "- Short list of tools/meters from the context. If not explicitly listed, say 'Standard hand tools and multimeter'.\n\n"
            "### Likely Causes\n"
            "- Bullet list of the most plausible root causes grounded in the excerpts.\n\n"
            "### Step-by-Step Diagnostics\n"
            "1. Ordered checks with what to measure/observe and expected results.\n"
            "2. Reference page lines like (Source: manual.pdf p.12) where applicable.\n\n"
            "### Fix / Adjustments\n"
            "- Concrete actions to restore operation. Include setpoints, dip switches, valve adjustments, resets, etc., ONLY if present in the excerpts; otherwise state 'See Source'.\n\n"
            "### If Still Not Resolved\n"
            "- The next best steps grounded in the excerpts.\n\n"
            "### Sources\n"
            "- List all manuals and pages used, like: manual.pdf p.12; manual2.pdf p.7.\n\n"
            "RULES:\n"
            "- Do NOT output charts, images, or markdown tables.\n"
            "- If a needed detail is NOT in the excerpts, do NOT invent it; instead write 'Not in provided context'."
        )

        user_prompt = (
            f"{scope_line}\n\n"
            "Use ONLY the following manual excerpts:\n\n"
            f"{context_str}\n\n"
            f"Question: {query}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        try:
            logging.info("Calling Groq…")
            resp = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Groq API error: {e}")
            return (
                "### TL;DR\n• Error contacting the AI engine.\n\n"
                "### Answer\nPlease try again in a moment."
            )

        # Sanitize & enforce TL;DR
        clean = strip_charts_and_tables(raw)
        tldr, with_tldr = ensure_tldr(clean)
        return with_tldr

# ------------- Singleton for app import -------------
try:
    ai_pipeline = ElevatorAIPipeline()
    ALL_BRANDS = sorted({ai_pipeline._infer_metadata(b, BRAND_MAP) for b in BRAND_MAP.values()})  # just placeholders
    ALL_BRANDS = sorted({s.get("brand", "unknown").upper() for s in ai_pipeline.sources})
    ALL_TYPES = sorted({s.get("etype", "unknown").upper() for s in ai_pipeline.sources})
except Exception as e:
    ai_pipeline = None
    ALL_BRANDS, ALL_TYPES = [], []
    logging.critical(f"CRITICAL: AI Pipeline failed to initialize. Error: {e}")
