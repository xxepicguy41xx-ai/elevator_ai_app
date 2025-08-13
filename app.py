# app.py
import streamlit as st
from core_ai import ai_pipeline, ALL_BRANDS

st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

if not ai_pipeline:
    st.error("**Fatal Error:** The AI core failed to initialize. Check GROQ_API_KEY, data files, and FAISS index.")
else:
    st.markdown(
        "Ask a repair or troubleshooting question. "
        "Select a **Brand** to focus results. Answers include tables, steps, and sources."
    )

    brand = st.selectbox("Brand (optional)", ["any"] + [b for b in ALL_BRANDS if b != "unknown"])
    query = st.text_input("Your question", placeholder="e.g., Doors won’t close on inspection mode. What should I check?")

    if st.button("Get answer", type="primary") and query:
        with st.spinner("Consulting the manuals and AI…"):
            answer = ai_pipeline.ask(query, brand=None if brand == "any" else brand, k=5, pool=60)
        st.divider()
        st.subheader("Answer")
        st.markdown(answer)
