import os
import json
import numpy as np
import faiss
import streamlit as st

# Import your pipeline
from core_ai import ai_pipeline, ALL_BRANDS, ALL_TYPES

st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

# ---------- Quick diagnostics helper ----------
def run_diagnostics():
    st.subheader("🔎 Quick Diagnostics")
    # 1) Secrets / env
    groq_ok = bool(os.getenv("GROQ_API_KEY"))
    st.write("GROQ_API_KEY present:", "✅" if groq_ok else "❌")

    # 2) Files present + size
    files = {
        "chunks.json": "text chunks",
        "sources.json": "chunk metadata",
        "embeddings.npy": "vector embeddings",
        "faiss_index.index": "FAISS index",
    }
    missing = False
    for f, desc in files.items():
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        st.write(f"- **{f}** ({desc}):", "✅" if exists else "❌", f"({size/1024/1024:.2f} MB)")
        if not exists or size == 0:
            missing = True

    if not missing:
        # 3) JSON lengths
        try:
            chunks = json.load(open("chunks.json","r",encoding="utf-8"))
            sources = json.load(open("sources.json","r",encoding="utf-8"))
            st.write(f"chunks: {len(chunks)}  |  sources: {len(sources)}", "✅" if len(chunks)==len(sources) else "⚠️ lengths differ")
        except Exception as e:
            st.error(f"JSON read error: {e}")

        # 4) Embeddings + FAISS shape check
        try:
            emb = np.load("embeddings.npy")
            st.write("embeddings.npy dtype:", str(emb.dtype))
            st.write("embeddings.npy shape:", str(emb.shape))
            try:
                idx = faiss.read_index("faiss_index.index")
                st.write("faiss index dim:", idx.d)
                if emb.shape[1] != idx.d:
                    st.error(f"Dimension mismatch: embeddings dim {emb.shape[1]} vs index dim {idx.d}")
                else:
                    st.write("Dimension check:", "✅")
            except Exception as e:
                st.error(f"FAISS index read error: {e}")
        except Exception as e:
            st.error(f"Embeddings load error: {e}")

# ---------- Main ----------
if not ai_pipeline:
    st.error(
        "**Fatal Error:** The AI core failed to initialize. "
        "This is likely due to a missing model, data file, or API key."
    )
    if st.button("Run quick diagnostics"):
        run_diagnostics()
else:
    st.markdown(
        "Ask a technical question about an elevator system. "
        "If you know the **Brand** or **Type**, select them to get more accurate results."
    )

    col1, col2 = st.columns(2)
    brand_options = ["any"] + [b for b in ALL_BRANDS if b != "unknown"]
    type_options  = ["any"] + [t for t in ALL_TYPES if t != "unknown"]
    brand = col1.selectbox("Select Brand (optional)", brand_options)
    etype = col2.selectbox("Select Type (optional)", type_options)

    query = st.text_input("Enter your question here:", placeholder="e.g., What is the procedure for a brake test?")

    if query:
        brand_filter = None if brand == "any" else brand
        etype_filter = None if etype == "any" else etype

        with st.spinner("Searching manuals and consulting the AI..."):
            try:
                answer = ai_pipeline.ask(query, brand=brand_filter, etype=etype_filter)
                st.divider()
                st.subheader("AI Assistant's Answer")
                st.caption(f"**Scope:** `{brand}` / `{etype}`")
                st.markdown(answer)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
