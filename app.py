import streamlit as st
from ask_ai import ask_groq, ALL_BRANDS, ALL_TYPES, format_source

st.set_page_config(page_title="Elevator AI Assistant", page_icon="🛗", layout="centered")
st.title("🛗 Elevator Mechanic AI Assistant")

# ------------ session state ------------
if "history" not in st.session_state:
    st.session_state.history = []   # list of {"q","tldr","a","sources"}

def _safe_rerun():
    # Streamlit renamed experimental_rerun() to rerun() in newer versions
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass  # last resort: no-op

# ------------ sidebar controls ------------
with st.sidebar:
    st.header("Filter by Brand & Type")
    brand = st.selectbox("Brand", ["Any"] + [b for b in ALL_BRANDS if b != "Unknown"], key="brand_sel")
    etype = st.selectbox("Elevator Type", ["Any"] + [t for t in ALL_TYPES if t != "Unknown"], key="type_sel")

    st.header("Answer Speed / Accuracy")
    mode = st.radio(
        "Choose how you want answers:",
        ["Quick (Faster, might skip details)", "Accurate (Slower, best detail)"],
        horizontal=False,
        key="mode_sel"
    )
    num_pages = st.slider("Number of matching pages to show", 2, 6, 3, key="k_slider")
    search_scope = st.slider("How many manuals to search through", 10, 80, 30, step=5, key="pool_slider")
    use_reranker = ("Accurate" in mode)
    temperature = 0.1 if "Accurate" in mode else 0.2

    st.divider()
    # IMPORTANT: don't mutate state used in the current render; set a flag then rerun
    if st.button("🧹 Clear conversation", use_container_width=True, key="clear_btn"):
        st.session_state.history = []
        _safe_rerun()

# ------------ history display ------------
for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(turn["q"])
    with st.chat_message("assistant"):
        st.success(turn["tldr"])
        st.markdown(turn["a"])
        with st.expander("Sources", expanded=False):
            for meta in turn["sources"]:
                st.markdown(f"- {format_source(meta)}")

# ------------ input ------------
prompt = st.chat_input("Describe the problem or ask a repair question…", key="chat_input_main")
if prompt:
    # echo user
    with st.chat_message("user"):
        st.markdown(prompt)

    # compute answer
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
        st.success(tldr)      # TL;DR box
        st.markdown(answer_md)
        with st.expander("Sources", expanded=False):
            for meta in used:
                st.markdown(f"- {format_source(meta)}")

    # update history then rerender
    st.session_state.history.append({"q": prompt, "tldr": tldr, "a": answer_md, "sources": used})
