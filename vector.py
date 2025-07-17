from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Path where your PDF files are stored
pdf_folder_path = "spm"

# Directory to persist the Chroma DB
db_location = "./chrome_langchain_db"

# Check if DB already exists — skip re-indexing if so
add_documents = not os.path.exists(db_location)

# Embedding model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

if add_documents:
    documents = []
    ids = []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    for filename in os.listdir(pdf_folder_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_folder_path, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(file_path)
            pages = loader.load_and_split()
            chunks = text_splitter.split_documents(pages)

            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                chunk.metadata["source_file"] = filename
                chunk.metadata["chunk"] = i
                chunk.metadata["id"] = doc_id
                documents.append(chunk)
                ids.append(doc_id)

    # Initialize and add documents to vector store
    vector_store = Chroma(
        collection_name="spm_pdfs",
        persist_directory=db_location,
        embedding_function=embeddings
    )
    vector_store.add_documents(documents=documents, ids=ids)

else:
    print("Vector store already exists. Skipping document addition.")
    vector_store = Chroma(
        collection_name="spm_pdfs",
        persist_directory=db_location,
        embedding_function=embeddings
    )

# Final retriever used in main.py
retriever = vector_store.as_retriever(
    search_type="mmr",  # better match diversity
    search_kwargs={"k": 6}
)
