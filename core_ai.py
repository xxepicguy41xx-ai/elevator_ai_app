# core_ai.py
import os
import json
import logging
from typing import List, Tuple, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Constants for Configuration ---
# File paths
CHUNKS_PATH = "chunks.json"
SOURCES_PATH = "sources.json"
EMBEDDINGS_PATH = "embeddings.npy"
FAISS_INDEX_PATH = "faiss_index.index"

# Model identifiers
EMBEDDING_MODEL = "hkunlp/instructor-base"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GROQ_MODEL = "llama3-70b-8192"  # Using a powerful, standard Groq model

# Brand/Type Inference Mappings
# This makes it easier to manage aliases (e.g., different spellings for Thyssenkrupp)
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

class ElevatorAIPipeline:
    """
    Encapsulates the entire RAG pipeline for the Elevator AI assistant.
    This includes loading data, models, searching, and interacting with the LLM.
    """

    def __init__(self):
        """
        Initializes the pipeline by loading all necessary models and data.
        This is a heavy operation and should only be done once.
        """
        logging.info("Initializing ElevatorAIPipeline...")
        try:
            # --- Load Data ---
            self.chunks = self._load_json(CHUNKS_PATH, "Chunks")
            self.sources = self._load_json(SOURCES_PATH, "Sources")

            self.embeddings = np.load(EMBEDDINGS_PATH)
            if self.embeddings.dtype != np.float32:
                logging.warning(f"embeddings dtype was {self.embeddings.dtype}; casting to float32 for FAISS.")
                self.embeddings = self.embeddings.astype(np.float32)

            self.index = faiss.read_index(FAISS_INDEX_PATH)
            if self.index.d != self.embeddings.shape[1]:
                raise ValueError(
                    f"FAISS index dim {self.index.d} != embeddings dim {self.embeddings.shape[1]}"
                )

            # --- Load Models ---
            self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

            # Make reranker optional so init never hard-fails on download issues
            try:
                self.reranker = CrossEncoder(RERANKER_MODEL)
                logging.info("CrossEncoder reranker loaded successfully.")
            except Exception as e:
                logging.warning(f"CrossEncoder not available ({e}). Proceeding WITHOUT rerank.")
                self.reranker = None

            # --- Configure Groq Client ---
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable not set.")
            self.groq_client = Groq(api_key=api_key)

            # --- Pre-process Metadata ---
            # This adds 'brand' and 'etype' to each source for efficient filtering.
            # In a production environment, this should be a one-time offline process.
            self._enrich_source_metadata()
            self.all_brands = sorted({s.get("brand", "unknown") for s in self.sources})
            self.all_types = sorted({s.get("etype", "unknown") for s in self.sources})

            logging.info("Pipeline initialized successfully.")

        except Exception as e:
            logging.error(f"Failed to initialize pipeline: {e}")
            raise

    def _load_json(self, path, name):
        """Helper to load a JSON file with proper error handling."""
        logging.info(f"Loading {name} from {path}...")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _infer_metadata(self, name, mapping):
        """Generic function to infer a category from a name using a mapping."""
        lowered_name = name.lower().replace("-", "_").replace(" ", "_")
        for key, value in mapping.items():
            if key in lowered_name:
                return value
        return "unknown"

    def _enrich_source_metadata(self):
        """
        Adds 'brand' and 'etype' keys to each source document in-memory.
        """
        logging.info("Enriching source metadata with brand and type...")
        for meta in self.sources:
            manual_name = meta.get("manual", "")
            meta["brand"] = self._infer_metadata(manual_name, BRAND_MAP)
            meta["etype"] = self._infer_metadata(manual_name, TYPE_MAP)

    def search_and_rerank(
        self,
        query: str,
        brand: str = None,
        etype: str = None,
        k: int = 5,
        pool: int = 50
    ):
        """
        Performs a streamlined search and reranking process.
        1. Retrieves an initial pool of candidates using FAISS.
        2. Filters candidates by brand and/or type.
        3. Reranks the filtered candidates using a CrossEncoder (if available).
        4. Returns the top 'k' results as [(text, meta), ...].
        """
        # 1. Initial Retrieval (Vector Search)
        query_vector = self.embedding_model.encode([query])
        _, initial_indices = self.index.search(query_vector, pool)
        candidate_indices = initial_indices[0]

        # 2. Filter by Metadata
        if brand or etype:
            filtered_indices = []
            for i in candidate_indices:
                meta = self.sources[i]
                brand_match = not brand or meta.get("brand") == brand
                etype_match = not etype or meta.get("etype") == etype
                if brand_match and etype_match:
                    filtered_indices.append(i)
            candidate_indices = np.array(filtered_indices, dtype=int) if filtered_indices else np.array([], dtype=int)

        if len(candidate_indices) == 0:
            logging.warning("No documents matched the filter criteria.")
            return []

        # 3. Rerank the Filtered Candidates (optional)
        rerank_pairs = [(query, self.chunks[i]) for i in candidate_indices]
        if self.reranker is not None:
            try:
                scores = self.reranker.predict(rerank_pairs)
                reranked_results = sorted(zip(scores, candidate_indices), key=lambda x: x[0], reverse=True)
                final_indices = [idx for score, idx in reranked_results]
            except Exception as e:
                logging.warning(f"Reranker prediction failed ({e}). Using vector search order.")
                final_indices = list(candidate_indices)
        else:
            logging.info("Skipping rerank (CrossEncoder unavailable). Using vector search order.")
            final_indices = list(candidate_indices)

        # 4. Select Top K Results
        top_k_indices = final_indices[:k]
        return [(self.chunks[i], self.sources[i]) for i in top_k_indices]

    def ask(self, query: str, brand: str = None, etype: str = None):
        """
        The main public method to ask a question to the AI assistant.
        """
        logging.info(f"Received query: '{query}' with filters Brand='{brand}', Type='{etype}'")

        # Retrieve context from manuals
        context_chunks = self.search_and_rerank(query, brand=brand, etype=etype)

        if not context_chunks:
            return (
                "I couldn't find any relevant information in the manuals for your specific query and filters. "
                "Please try rephrasing your question or broadening the scope."
            )

        # Build the context string for the LLM
        context = "\n\n".join([
            f"Excerpt from '{meta['manual']}' (Page {meta['page']}):\n> {text}"
            for text, meta in context_chunks
        ])

        # Construct the prompt for the LLM
        scope_parts = []
        if brand: scope_parts.append(f"elevator brand: {brand.upper()}")
        if etype: scope_parts.append(f"elevator type: {etype.upper()}")
        scope_line = (
            f"You are answering within the following scope: {', '.join(scope_parts)}."
            if scope_parts else
            "You are answering a general question."
        )

        system_prompt = (
            "You are an expert elevator mechanic AI assistant. Your sole purpose is to answer questions based "
            "*only* on the provided excerpts from technical manuals. Be concise, clear, and direct. Present your "
            "answer in a way that a mechanic in the field can quickly understand and use. If the provided excerpts "
            "are insufficient to answer the question, clearly state that the information is not available in the "
            "provided context. You must cite the source manual and page number for the information you use, like this: "
            "(Source: [manual_name], Page: [page_number])."
        )

        user_prompt = (
            f"{scope_line}\n\n"
            "Please use the following manual excerpts to answer the question.\n\n"
            f"---BEGIN MANUAL EXCERPTS---\n{context}\n---END MANUAL EXCERPTS---\n\n"
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
                temperature=0.1,
                max_tokens=1024
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"Error calling Groq API: {e}")
            return "Sorry, I encountered an error while trying to generate an answer. Please try again later."

# --- Singleton Instance ---
# This creates a single instance of the pipeline that the Streamlit app can import.
# Streamlit's architecture will cache this module, so the heavy initialization
# runs only once when the app starts.
try:
    ai_pipeline = ElevatorAIPipeline()
    ALL_BRANDS = ai_pipeline.all_brands
    ALL_TYPES = ai_pipeline.all_types
except Exception as e:
    # If initialization fails, set placeholders to allow the app to load and show an error.
    ai_pipeline = None
    ALL_BRANDS = []
    ALL_TYPES = []
    logging.critical(
        f"CRITICAL: AI Pipeline failed to initialize. The app will not be functional. Error: {e}"
    )
