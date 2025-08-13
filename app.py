import streamlit as st
# We import the initialized pipeline instance and constants from our core AI module.
# Renaming ask_ai.py to core_ai.py is recommended for clarity.
from core_ai import ai_pipeline, ALL_BRANDS, ALL_TYPES

# --- Page Configuration ---
st.set_page_config(
    page_title="Elevator AI Assistant",
    page_icon="🛗",
    layout="centered"
)

# --- Main UI ---
st.title("🛗 Elevator Mechanic AI Assistant")

# Check if the AI pipeline loaded correctly.
if not ai_pipeline:
    st.error(
        "**Fatal Error:** The AI core failed to initialize. "
        "This is likely due to a missing model, data file, or API key. "
        "Please check the server logs for more details."
    )
else:
    st.markdown(
        "Ask a technical question about an elevator system. "
        "If you know the **Brand** or **Type**, select them to get more accurate results."
    )

    # --- Filter Selection ---
    col1, col2 = st.columns(2)
    
    # Create lists for dropdowns, ensuring 'any' is the default and 'unknown' is excluded.
    brand_options = ["any"] + [b for b in ALL_BRANDS if b != "unknown"]
    type_options = ["any"] + [t for t in ALL_TYPES if t != "unknown"]

    brand = col1.selectbox("Select Brand (optional)", brand_options)
    etype = col2.selectbox("Select Type (optional)", type_options)

    # --- User Query Input ---
    query = st.text_input("Enter your question here:", placeholder="e.g., What is the procedure for a brake test?")

    if query:
        # Convert 'any' from the UI to None for the backend logic.
        brand_filter = None if brand == "any" else brand
        etype_filter = None if etype == "any" else etype

        with st.spinner("Searching manuals and consulting the AI..."):
            try:
                # Call the pipeline's main method.
                answer = ai_pipeline.ask(
                    query,
                    brand=brand_filter,
                    etype=etype_filter
                )

                # --- Display Results ---
                st.divider()
                st.subheader("AI Assistant's Answer")

                # Display the scope that was used for the search.
                scope_text = f"**Scope:** `{brand}` / `{etype}`"
                st.caption(scope_text)

                st.markdown(answer)

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

