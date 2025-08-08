import streamlit as st
from ask_ai import ask_groq

st.set_page_config(page_title="Elevator AI Assistant", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

query = st.text_input("Ask a question about elevator repair:")

if query:
    with st.spinner("Thinking..."):
        answer = ask_groq(query)
        st.markdown("### 📘 Answer:")
        st.markdown(answer)
