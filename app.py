import streamlit as st
from ask_ai import ask_groq, ALL_BRANDS, ALL_TYPES

st.set_page_config(page_title="Elevator AI", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

col1, col2 = st.columns(2)
brand = col1.selectbox("Brand (optional)", ["any"] + [b for b in ALL_BRANDS if b != "unknown"])
etype = col2.selectbox("Type (optional)", ["any"] + [t for t in ALL_TYPES if t != "unknown"])

query = st.text_input("Ask a question:")
if query:
    with st.spinner("Searching manuals and thinking..."):
        answer = ask_groq(
            query,
            brand=None if brand == "any" else brand,
            etype=None if etype == "any" else etype
        )
    if brand or etype:
        st.caption(f"Scope: {brand or 'any brand'} • {etype or 'any type'}")
    st.markdown(answer)
