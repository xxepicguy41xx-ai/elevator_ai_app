import streamlit as st
from ask_ai import ask_groq, ALL_BRANDS, ALL_TYPES, format_source

# ---------- Page setup ----------
st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

# ---------- Session state ----------
if "history" not in st.session_state:
    st.session_state.history = []   # list of dicts: {"q": ..., "a": ..., "sources": [...]}

# ---------- Sidebar: filters & performance ----------
with st.sidebar:
    st.header("Filters")
    brand = st.selectbox("Brand", ["any"] + [b for b in ALL_BRANDS if b != "unknown"])
    etype = st.selectbox("Elevator Type", ["any"] + [t for t in ALL_TYPES if t != "unknown"])

    st.header("Performance")
    mode = st.radio("Mode", ["Fast", "Accurate"], horizontal=True)
    k = st.slider("Top-K (context)", 2, 6, 3)
    pool = st.slider("Candidate pool", 10, 80, 30, step=5)
    use_reranker = (mode == "Accurate")
    temperature = 0.1 if mode == "Accurate" else 0.2

    st.divider()
    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.experimental_rerun()

# ---------- Chat history display ----------
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.markdown(turn["a"])
        # Show sources in a neat expandable box
        with st.expander("Sources", expanded=False):
            for meta in turn["sources"]:
                st.markdown(f"- {format_source(meta)}")

# ---------- Input (chat-style) ----------
prompt = st.chat_input("Ask a question about repair, faults, procedures…")
if prompt:
    # Echo user
    with st.chat_message("user"):
        st.markdown(prompt)

    # Compute answer
    with st.chat_message("assistant"):
        with st.spinner("Searching manuals and thinking…"):
            ans, used = ask_groq(
                prompt,
                brand=None if brand == "any" else brand,
                etype=None if etype == "any" else etype,
                k=k,
                pool=pool,
                use_reranker=use_reranker,
                temperature=temperature
            )
        st.markdown(ans)

        # Sources panel
        with st.expander("Sources", expanded=False):
            for meta in used:
                st.markdown(f"- {format_source(meta)}")

        # Quick actions row
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("⬇️ Download answer", data=ans, file_name="answer.txt", mime="text/plain")
        with col2:
            # Simple copy helper (uses a read-only text area users can Ctrl/Cmd+C)
            st.text_area("Copy answer", value=ans, height=100, label_visibility="collapsed")

    # Save to history (keep last 20)
    st.session_state.history.append({"q": prompt, "a": ans, "sources": used})
    st.session_state.history = st.session_state.history[-20:]
