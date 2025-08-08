{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "f99c0ddd-5aa9-4444-8efb-8c9c239a1744",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-08-08 10:15:32.180 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\ajgil\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    },
    {
     "ename": "FileNotFoundError",
     "evalue": "No secrets files found. Valid paths for a secrets.toml file are: C:\\Users\\ajgil\\.streamlit\\secrets.toml, C:\\Users\\ajgil\\Elevator_Project\\.streamlit\\secrets.toml",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[1], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mstreamlit\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mst\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m api_key \u001b[38;5;241m=\u001b[39m st\u001b[38;5;241m.\u001b[39msecrets[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mGROQ_API_KEY\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\streamlit\\runtime\\secrets.py:305\u001b[0m, in \u001b[0;36mSecrets.__getitem__\u001b[1;34m(self, key)\u001b[0m\n\u001b[0;32m    299\u001b[0m \u001b[38;5;250m\u001b[39m\u001b[38;5;124;03m\"\"\"Return the value with the given key. If no such key\u001b[39;00m\n\u001b[0;32m    300\u001b[0m \u001b[38;5;124;03mexists, raise a KeyError.\u001b[39;00m\n\u001b[0;32m    301\u001b[0m \n\u001b[0;32m    302\u001b[0m \u001b[38;5;124;03mThread-safe.\u001b[39;00m\n\u001b[0;32m    303\u001b[0m \u001b[38;5;124;03m\"\"\"\u001b[39;00m\n\u001b[0;32m    304\u001b[0m \u001b[38;5;28;01mtry\u001b[39;00m:\n\u001b[1;32m--> 305\u001b[0m     value \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_parse(\u001b[38;5;28;01mTrue\u001b[39;00m)[key]\n\u001b[0;32m    306\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(value, Mapping):\n\u001b[0;32m    307\u001b[0m         \u001b[38;5;28;01mreturn\u001b[39;00m value\n",
      "File \u001b[1;32m~\\anaconda3\\Lib\\site-packages\\streamlit\\runtime\\secrets.py:214\u001b[0m, in \u001b[0;36mSecrets._parse\u001b[1;34m(self, print_exceptions)\u001b[0m\n\u001b[0;32m    212\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m print_exceptions:\n\u001b[0;32m    213\u001b[0m         st\u001b[38;5;241m.\u001b[39merror(err_msg)\n\u001b[1;32m--> 214\u001b[0m     \u001b[38;5;28;01mraise\u001b[39;00m \u001b[38;5;167;01mFileNotFoundError\u001b[39;00m(err_msg)\n\u001b[0;32m    216\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m([p \u001b[38;5;28;01mfor\u001b[39;00m p \u001b[38;5;129;01min\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_file_paths \u001b[38;5;28;01mif\u001b[39;00m os\u001b[38;5;241m.\u001b[39mpath\u001b[38;5;241m.\u001b[39mexists(p)]) \u001b[38;5;241m>\u001b[39m \u001b[38;5;241m1\u001b[39m:\n\u001b[0;32m    217\u001b[0m     _LOGGER\u001b[38;5;241m.\u001b[39minfo(\n\u001b[0;32m    218\u001b[0m         \u001b[38;5;124mf\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSecrets found in multiple locations: \u001b[39m\u001b[38;5;132;01m{\u001b[39;00m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124m, \u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;241m.\u001b[39mjoin(\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_file_paths)\u001b[38;5;132;01m}\u001b[39;00m\u001b[38;5;124m. \u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    219\u001b[0m         \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhen multiple secret.toml files exist, local secrets will take precedence over global secrets.\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[0;32m    220\u001b[0m     )\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: No secrets files found. Valid paths for a secrets.toml file are: C:\\Users\\ajgil\\.streamlit\\secrets.toml, C:\\Users\\ajgil\\Elevator_Project\\.streamlit\\secrets.toml"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "api_key = st.secrets[\"GROQ_API_KEY\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e7734fcf-e91a-4c64-aee2-6bf2bf457a5e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "import json\n",
    "import numpy as np\n",
    "import faiss\n",
    "from sentence_transformers import SentenceTransformer\n",
    "from groq import Groq\n",
    "from dotenv import load_dotenv\n",
    "\n",
    "load_dotenv()\n",
    "client = Groq(api_key=os.getenv(\"GROQ_API_KEY\"))\n",
    "\n",
    "# Load data\n",
    "chunks = json.load(open(\"chunks.json\", \"r\", encoding=\"utf-8\"))\n",
    "sources = json.load(open(\"sources.json\", \"r\", encoding=\"utf-8\"))\n",
    "embeddings = np.load(\"embeddings.npy\")\n",
    "index = faiss.read_index(\"faiss_index.index\")\n",
    "\n",
    "model = SentenceTransformer(\"hkunlp/instructor-base\")\n",
    "\n",
    "def search_chunks(query, k=5):\n",
    "    query_vec = model.encode([query])\n",
    "    D, I = index.search(np.array(query_vec).reshape(1, -1), k)\n",
    "    return [(chunks[i], sources[i]) for i in I[0]]\n",
    "\n",
    "def ask_groq(query):\n",
    "    results = search_chunks(query)\n",
    "    context = \"\\n\\n\".join([\n",
    "        f\"{text} (Source: {meta['manual']} - Page {meta['page']})\"\n",
    "        for text, meta in results\n",
    "    ])\n",
    "\n",
    "    messages = [\n",
    "        {\"role\": \"system\", \"content\": \"You are a helpful elevator repair assistant. Use the provided manual context to answer clearly.\"},\n",
    "        {\"role\": \"user\", \"content\": f\"{context}\\n\\nQuestion: {query}\"}\n",
    "    ]\n",
    "\n",
    "    response = client.chat.completions.create(\n",
    "        model=\"openai/gpt-oss-120b\",\n",
    "        messages=messages,\n",
    "        temperature=0.3\n",
    "    )\n",
    "\n",
    "    return response.choices[0].message.content\n"
   ]
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
