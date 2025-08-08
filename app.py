{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eb0bcf32-98b8-4188-a5f6-3164c527a35b",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "from ask_ai import ask_groq\n",
    "\n",
    "st.set_page_config(page_title=\"Elevator AI Assistant\", layout=\"centered\")\n",
    "st.title(\"🛗 Elevator Mechanic AI Assistant\")\n",
    "\n",
    "query = st.text_input(\"Ask a question about elevator repair...\")\n",
    "\n",
    "if query:\n",
    "    with st.spinner(\"Searching manuals and thinking...\"):\n",
    "        answer = ask_groq(query)\n",
    "    st.markdown(\"### 📘 Answer:\")\n",
    "    st.markdown(answer)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5cafeed3-4337-4a74-8a80-4028c79dd787",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
