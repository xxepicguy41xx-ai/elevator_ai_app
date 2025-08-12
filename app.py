import streamlit as st
from ask_ai import ask_groq, ALL_BRANDS, ALL_TYPES, format_source

st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Filter by Brand & Type")
    brand = st.selectbox("Brand", ["Any"] + [b for b in ALL_BRANDS if b != "Unknown"])
    etype = st.selectbox("Elevator Type", ["Any"] + [t for t in ALL_TYPES if t != "Unknown"])

    st.header("Answer Speed / Accuracy")
    mode = st.radio(
        "Choose how you want answers:",
        ["Quick (Faster, might skip details)", "Accurate (Slower, best detail)"],
        horizontal=False
    )
    num_pages = st.slider("Number of matching pages to show", 2, 6, 3)
    search_scope = st.slider("How many manuals to search through", 10, 80, 30, step=5)
    use_reranker = ("Accurate" in mode)
    temperature = 0.1 if "Accurate" in mode else 0.2

    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.experimental_rerun()

for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.success(turn["tldr"])
        st.markdown(turn["a"])
        with st.expander("Sources", expanded=False):
            for meta in turn["sources"]:
                st.markdown(f"- {format_source(meta)}")

prompt = st.chat_input("Describe the problem or ask a repair question…")
if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Looking through manuals…"):
            tldr, answer_md, used = ask_groq(
                prompt,
                brand=None if brand == "Any" else brand,
                etype=None if etype == "Any" else etype,
                k=num_pages,
                pool=search_scope,
                use_reranker=use_reranker,
                temperature=temperature
            )
        st.success(tldr)
        st.markdown(answer_md)
        with st.expander("Sources", expanded=False):
            for meta in used:
                st.markdown(f"- {format_source(meta)}")

    st.session_state.history.append({"q": prompt, "tldr": tldr, "a": answer_md, "sources": used})
    st.session_state.history = st.session_state.history[-20:]
