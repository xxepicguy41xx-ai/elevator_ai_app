import os
import json
import numpy as np
import faiss
import importlib
import streamlit as st
import core_ai  # import the module, not just the names

from core_ai import ai_pipeline, ALL_BRANDS, ALL_TYPES

st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

def run_diagnostics():
    st.subheader("🔎 Quick Diagnostics")
    st.write("GROQ_API_KEY present:", "✅" if bool(os.getenv("GROQ_API_KEY")) else "❌")
    files = {
        "chunks.json": "text chunks",
        "sources.json": "chunk metadata",
        "embeddings.npy": "vector embeddings",
        "faiss_index.index": "FAISS index",
    }
    for f, desc in files.items():
        exists = os.path.exists(f)
        size = os.path.getsize(f) if exists else 0
        st.write(f"- **{f}** ({desc}):", "✅" if exists else "❌", f"({size/1024/1024:.2f} MB)")
    try:
        chunks = json.load(open("chunks.json","r",encoding="utf-8"))
        sources = json.load(open("sources.json","r",encoding="utf-8"))
        st.write(f"chunks: {len(chunks)} | sources: {len(sources)}",
                 "✅" if len(chunks)==len(sources) else "⚠️ mismatch")
    except Exception as e:
        st.error(f"JSON read error: {e}")
    try:
        emb = np.load("embeddings.npy")
        st.write("embeddings.npy dtype:", str(emb.dtype))
        st.write("embeddings.npy shape:", str(emb.shape))
        idx = faiss.read_index("faiss_index.index")
        st.write("faiss index dim:", idx.d)
        st.write("Dimension check:", "✅" if emb.shape[1]==idx.d else f"❌ ({emb.shape[1]} vs {idx.d})")
    except Exception as e:
        st.error(f"Embedding/Index check error: {e}")

# ---------- If core failed ----------
if not ai_pipeline:
    st.error(
        "**Fatal Error:** The AI core failed to initialize. "
        "This is usually a model download issue (like the CrossEncoder)."
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("Run quick diagnostics"):
            run_diagnostics()
    with colB:
        if st.button("🔄 Reload AI core"):
            importlib.reload(core_ai)
            from core_ai import ai_pipeline as _ai, ALL_BRANDS as _B, ALL_TYPES as _T
            # overwrite globals so the rest of the app sees the fresh core
            globals()["ai_pipeline"] = _ai
            globals()["ALL_BRANDS"] = _B
            globals()["ALL_TYPES"] = _T
            st.rerun()
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
