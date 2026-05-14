import streamlit as st
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import textwrap

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

st.set_page_config(page_title="AI Thesis Generator (Local)", layout="wide")
st.title("📈 AI Investment Thesis Generator")


def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            t = page.extract_text()
            if t:
                text += t + "\n\n"
    return text

def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    st.success("PDF uploaded!")

    with st.spinner("Extracting text..."):
        raw_text = extract_text_from_pdf(uploaded_file)

    st.text_area("Text Preview", raw_text[:1000], height=200)

    with st.spinner("Embedding and indexing..."):
        chunks = chunk_text(raw_text)
        embedder = SentenceTransformer("all-mpnet-base-v2")
        embeddings = embedder.encode(chunks, show_progress_bar=False)

        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(np.array(embeddings).astype("float32"))


    st.subheader("Generate Thesis (Local Model)")

    company = st.text_input("Company Name", "Company X")

    if st.button("Generate Thesis"):
        with st.spinner("Retrieving evidence..."):
            q_emb = embedder.encode([f"investment thesis {company}"])[0]
            D, I = index.search(np.array([q_emb]).astype("float32"), 4)
            retrieved = [chunks[i] for i in I[0]]

        evidence_block = "\n\n".join(
            [f"[EVIDENCE {i+1}] {textwrap.shorten(r, width=450)}"
             for i, r in enumerate(retrieved)]
        )

        prompt = f"""
Write a structured investment thesis for {company} using ONLY this evidence:

{evidence_block}

Follow this format:
1. Summary
2. Key Growth Drivers
3. Risks
4. Competitive Positioning
5. 2-line Conviction Statement
"""

        with st.spinner("Running local model... (1–2 seconds)"):

            tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
            model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")


            nlp = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=512
            )

            result = nlp(prompt)[0]["generated_text"]

        st.subheader("📄 Investment Thesis")
        st.write(result)
