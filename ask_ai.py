import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from groq import Groq
import streamlit as st

# Load data
chunks = json.load(open("chunks.json", "r", encoding="utf-8"))
sources = json.load(open("sources.json", "r", encoding="utf-8"))
embeddings = np.load("embeddings.npy")
index = faiss.read_index("faiss_index.index")

model = SentenceTransformer("hkunlp/instructor-base")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def search_chunks(query, k=5):
    query_vec = model.encode([query])
    D, I = index.search(np.array(query_vec).reshape(1, -1), k)
    return [(chunks[i], sources[i]) for i in I[0]]

def ask_groq(query):
    context_chunks = search_chunks(query)
    context = "\n\n".join([
        f"{text}\n(Source: {meta['manual']} - Page {meta['page']})"
        for text, meta in context_chunks
    ])

    messages = [
        {"role": "system", "content": "You are an expert assistant for elevator mechanics. Use the following manual context to answer clearly."},
        {"role": "user", "content": f"{context}\n\nQuestion: {query}"}
    ]

    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content
