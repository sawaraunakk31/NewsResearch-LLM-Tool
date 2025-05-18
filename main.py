import os
import streamlit as st
import pickle
import time
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

st.title("News Research Tool ")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_pipeline.pkl"

main_placeholder = st.empty()
model_name = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

text_generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# --- Process URLs ---
if process_url_clicked:
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Loading the Data...")
    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", ","],
        chunk_size=1000
    )
    main_placeholder.text("Splitting Text...")
    docs = text_splitter.split_documents(data)

    # Embeddings to Faiss Index
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Building Embeddings...")
    time.sleep(2)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

# --- Question / Answer Section ---
query = main_placeholder.text_input("Ask a Question:")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever()
        relevant_docs = retriever.get_relevant_documents(query)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Build a prompt for the model
        prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"

        # Generate the answer
        generated_texts = text_generator(
            prompt,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            do_sample=True,
            num_return_sequences=1
        )

        generated_answer = generated_texts[0]["generated_text"].split("Answer:")[-1].strip()

        if generated_answer and not generated_answer[0].isupper():
            generated_answer = generated_answer[0].upper() + generated_answer[1:]

        st.header("Answer")
        st.write(generated_answer)

        # Extract and show sources
        st.header("Sources")
        source_urls = []
        for doc in relevant_docs:
            source = doc.metadata.get("source", None)
            if source and source not in source_urls:
                source_urls.append(source)

        for url in source_urls:
            st.write(f"({url})")
