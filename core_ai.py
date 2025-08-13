# core_ai.py
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
# File paths
CHUNKS_PATH = "chunks.json"
SOURCES_PATH = "sources.json"
EMBEDDINGS_PATH = "embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.index"

# Model identifiers
EMBEDDING_MODEL = "hkunlp/instructor-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama3-70b-8192"  # powerful Groq model

# Brand/Type Inference Maps (alias handling)
BRAND_MAP = {
    "thyssenkrupp": "tke",
    "thyssen": "tke",
    "tke": "tke",
    "otis": "otis",
    "kone": "kone",
    "schindler": "schindler",
    "dover": "dover",
    "mce": "mce",
    "smartrise": "smartrise",
    "gal": "gal",
    "nidec": "nidec",
    "magnetek": "magnetek",
}
TYPE_MAP = {
    "machine_room_less": "mrl",
    "mrl": "mrl",
    "trl": "mrl",
    "hydraulic": "hydraulic",
    "hyd": "hydraulic",
    "traction": "traction",
    "gearless": "gearless",
    "geared": "geared",
}

# ---------------- Output Sanitizers (no charts/tables) ----------------
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
    # Convert markdown tables to simple bullets
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
    """
    Returns (tldr_text, full_answer_with_tldr_heading).
    If TL;DR section missing, synthesize 1–2 bullets from first sentences.
    """
    if re.search(r"(?im)^###\s*tl;?dr", answer_md):
        after = re.split(r"(?im)^###\s*tl;?dr\s*", answer_md, maxsplit=1)[-1].strip()
        first_para = after.split("\n\n")[0].strip()
        return first_para, answer_md

    # Synthesize TL;DR from first 1–2 sentences
    text = re.sub(r"\s+", " ", re.sub(r"[#>*`_]", " ", answer_md)).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    bullets = []
    for s in sentences:
        if len(s) < 8:
            continue
        bullets.append("• " + s.strip())
        if len(bullets) == 2:
            break
    tldr = "\n".join(bullets) if bullets else "• See steps and sources below."
    full = "### TL;DR\n" + tldr + "\n\n" + answer_md
    return tldr, full

# Optional: base URL for hosted PDFs to make links clickable (manual.pdf#page=12)
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

# ---------------- Core RAG Pipeline ----------------
class ElevatorAIPipeline:
    """
    RAG pipeline: loads data/models, searches + reranks, and asks the LLM.
    Produces a comprehensive troubleshooting/fix guide (not just references).
    """

    def __init__(self):
        logging.info("Initializing ElevatorAIPipeline...")
        try:
            # --- Load Data ---
            self.chunks = self._load_json(CHUNKS_PATH, "Chunks")
            self.sources = self._load_json(SOURCES_PATH, "Sources")
            self.embeddings = np.load(EMBEDDINGS_PATH)
            if self.embeddings.dtype != np.float32:
                self.embeddings = self.embeddings.astype(np.float32)
            self.index = faiss.read_index(FAISS_INDEX_PATH)
            if self.index.d != self.embeddings.shape[1]:
                raise ValueError(f"FAISS dim {self.index.d} != embeddings dim {self.embeddings.shape[1]}")

            # --- Load Models ---
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
            self.reranker = CrossEncoder(RERANKER_MODEL)

            # --- Configure Groq Client ---
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            self.groq_client = Groq(api_key=api_key)

            # --- Enrich Metadata (brand/type) ---
            self._enrich_source_metadata()
            self.all_brands = sorted({s.get("brand", "unknown") for s in self.sources})
            self.all_types = sorted({s.get("etype", "unknown") for s in self.sources})

            logging.info("Pipeline initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
            raise

    # ---------- IO ----------
    def _load_json(self, path, name):
        logging.info(f"Loading {name} from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    # ---------- Metadata ----------
    def _infer_metadata(self, name, mapping):
        lowered_name = name.lower().replace("-", "_").replace(" ", "_")
        for key, value in mapping.items():
            if key in lowered_name:
                return value
        return "unknown"

    def _enrich_source_metadata(self):
        logging.info("Enriching source metadata with brand and type...")
        for meta in self.sources:
            manual_name = meta.get("manual", "")
            meta["brand"] = self._infer_metadata(manual_name, BRAND_MAP)
            meta["etype"] = self._infer_metadata(manual_name, TYPE_MAP)

    # ---------- Retrieval + Rerank ----------
    def search_and_rerank(
        self,
        query: str,
        brand: str = None,
        etype: str = None,
        k: int = 6,
        pool: int = 60
    ):
        """
        1) Retrieve with FAISS
        2) Optional brand/type filter
        3) Cross-encoder rerank
        4) Return top k [(text, meta)...]
        """
        # 1. Initial Retrieval
        query_vector = self.embedding_model.encode([query])
        _, initial_indices = self.index.search(query_vector, pool)
        candidate_indices = initial_indices[0]

        # 2. Filter by Metadata
        if brand or etype:
            filtered = []
            for i in candidate_indices:
                meta = self.sources[i]
                if brand and meta.get("brand") != brand:
                    continue
                if etype and meta.get("etype") != etype:
                    continue
                filtered.append(i)
            candidate_indices = np.array(filtered, dtype=int) if filtered else np.array([], dtype=int)

        if len(candidate_indices) == 0:
            logging.warning("No documents matched the filter criteria.")
            return []

        # 3. Rerank (pairwise cross-encoder)
        rerank_pairs = [(query, self.chunks[i]) for i in candidate_indices]
        scores = self.reranker.predict(rerank_pairs)
        order = np.argsort(-scores)
        ranked = candidate_indices[order]

        # 4. Select Top K with light manual spread control (max 2 per manual)
        picked = []
        per_manual = {}
        for i in ranked:
            meta = self.sources[int(i)]
            mname = meta["manual"]
            per_manual[mname] = per_manual.get(mname, 0)
            if per_manual[mname] >= 2:
                continue
            picked.append(int(i))
            per_manual[mname] += 1
            if len(picked) == k:
                break

        # Fallback fill if too few after spread control
        if len(picked) < k:
            for i in ranked:
                ii = int(i)
                if ii not in picked:
                    picked.append(ii)
                if len(picked) == k:
                    break

        return [(self.chunks[i], self.sources[i]) for i in picked]

    # ---------- Public API ----------
    def ask(self, query: str, brand: str = None, etype: str = None):
        """
        Returns a comprehensive, field-ready troubleshooting/fix guide.
        Not just references—structured steps, safety, tools, causes, diagnostics,
        fix actions, next steps, and sources. Grounded ONLY in provided excerpts.
        """
        logging.info(f"Received query: '{query}' with filters Brand='{brand}', Type='{etype}'")

        # Retrieve context (bigger pool/k for robustness by default)
        context_chunks = self.search_and_rerank(query, brand=brand, etype=etype, k=6, pool=80)

        if not context_chunks:
            return (
                "### TL;DR\n"
                "• No matching excerpts found for that brand/type.\n\n"
                "### Answer\n"
                "Try removing filters or rephrasing your question so I can pull better instructions from the manuals."
            )

        # Group snippets by manual for readability
        by_manual: Dict[str, List[Tuple[str, Dict]]] = {}
        for text, meta in context_chunks:
            by_manual.setdefault(meta["manual"], []).append((text, meta))

        parts = []
        for manual, entries in by_manual.items():
            parts.append(f"=== {manual} ===")
            for t, m in entries:
                parts.append(f"(Page {m.get('page','?')} | Brand {m.get('brand','?').upper()} | Type {m.get('etype','?').upper()})")
                parts.append(t)

        context_block = "\n".join(parts)

        # Scope text seen by the model
        scope_bits = []
        if brand: scope_bits.append(f"BRAND={brand.upper()}")
        if etype: scope_bits.append(f"TYPE={etype.upper()}")
        scope_line = " | ".join(scope_bits) if scope_bits else "GENERIC SCOPE"

        # --- Strict system prompt for comprehensive guides ---
        system_prompt = (
            "You are an expert elevator mechanic assistant. Use ONLY the provided manual excerpts. "
            "Do NOT mix brands; if there is any conflict, prefer the selected brand. "
            "Your answer must be a comprehensive troubleshooting and fix guide, not just a quick reference.\n\n"
            "OUTPUT FORMAT (use exactly these headings):\n"
            "### TL;DR\n"
            "- 1–2 bullets with the most likely fix.\n\n"
            "### Safety First\n"
            "- Lockout/Tagout, secure car, barricade doors, PPE, and any manual-specific warnings if present.\n\n"
            "### Tools Needed\n"
            "- List tools/meters from the excerpts; if not listed, state 'Standard hand tools and multimeter'.\n\n"
            "### Likely Causes\n"
            "- Bullet list of plausible root causes grounded in the excerpts.\n\n"
            "### Step-by-Step Diagnostics\n"
            "1. Ordered checks with what to look for or measure and expected results. "
            "Cite inline as (Source: manual.pdf p.12) when specific.\n\n"
            "### Fix / Adjustments\n"
            "- Concrete actions (valve/parameter adjustments, resets, switch positions) ONLY if present in the excerpts; "
            "otherwise write 'Not in provided context'.\n\n"
            "### If Still Not Resolved\n"
            "- The next best steps grounded in the excerpts.\n\n"
            "### Sources\n"
            "- List manuals and pages used, like: manual.pdf p.12; manual2.pdf p.7.\n\n"
            "RULES:\n"
            "- Do NOT output charts, images, mermaid/vega/plotly code, or markdown tables.\n"
            "- Do NOT invent details or numbers that are not in the excerpts."
        )

        user_prompt = (
            f"{scope_line}\n\n"
            "Use ONLY the following manual excerpts:\n\n"
            f"{context_block}\n\n"
            f"Question: {query}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call the Groq API
        try:
            logging.info("Sending request to Groq API...")
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                temperature=0.1,      # tighter/safer, less blending
                max_tokens=1400
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            return (
                "### TL;DR\n• Error contacting the AI engine.\n\n"
                "### Answer\nPlease try again shortly."
            )

        # Sanitize and ensure TL;DR
        clean = strip_charts_and_tables(raw)
        _tldr, with_tldr = ensure_tldr(clean)
        return with_tldr

# --- Singleton Instance for the app to import ---
try:
    ai_pipeline = ElevatorAIPipeline()
    ALL_BRANDS = ai_pipeline.all_brands
    ALL_TYPES = ai_pipeline.all_types
except Exception as e:
    ai_pipeline = None
    ALL_BRANDS = []
    ALL_TYPES = []
    logging.critical(f"CRITICAL: AI Pipeline failed to initialize. The app will not be functional. Error: {e}")
