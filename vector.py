from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

# Embedding model (can be global as it's reusable)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

def get_vector_store(chat_id: str, pdf_files: list = None):
    """
    Initializes or retrieves a Chroma vector store for a given chat_id.
    If pdf_files are provided, it will load and add documents to the store.
    """
    db_location_for_chat = f"./chroma_db/{chat_id}" # Unique directory for each chat

    # Flag to check if the directory for this chat's DB already exists
    db_exists = os.path.exists(db_location_for_chat)

    # Instantiate the text splitter with adjusted parameters for better context capture
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,  # Increased from 1000 to capture more context
        chunk_overlap=300 # Increased from 200 for smoother transitions
    )

    documents_to_add = []
    ids_to_add = []

    if pdf_files: # Only process PDFs if provided
        for file_path in pdf_files:
            if file_path.endswith(".pdf") and os.path.exists(file_path):
                filename = os.path.basename(file_path)
                print(f"[{chat_id}] Loading {filename} for chunking...")
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                chunks = text_splitter.split_documents(pages)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{filename}_{i}"
                    chunk.metadata["source_file"] = filename
                    chunk.metadata["chunk"] = i
                    chunk.metadata["id"] = doc_id
                    documents_to_add.append(chunk)
                    ids_to_add.append(doc_id)
            else:
                print(f"[{chat_id}] Skipping invalid file: {file_path}")

    # Initialize or load the Chroma vector store
    vector_store = Chroma(
        collection_name=f"chat_{chat_id}_pdfs",
        persist_directory=db_location_for_chat,
        embedding_function=embeddings
    )

    if not db_exists:
        if documents_to_add:
            vector_store.add_documents(documents=documents_to_add, ids=ids_to_add)
            print(f"[{chat_id}] New vector store created and {len(documents_to_add)} documents added.")
        else:
            print(f"[{chat_id}] New vector store initialized (no documents added yet).")
    elif documents_to_add: # DB exists, and we have new documents to add
        vector_store.add_documents(documents=documents_to_add, ids=ids_to_add)
        print(f"[{chat_id}] {len(documents_to_add)} new documents added to existing vector store.")
    else: # DB exists, and no new documents were provided in this call
        print(f"[{chat_id}] Vector store already exists. Loading existing store.")
        # No need to call add_documents here, as no new docs were provided, and it's already loaded by Chroma() constructor

    return vector_store

def get_retriever(chat_id: str, pdf_files: list = None):
    """
    Returns a retriever for a given chat_id, creating the vector store if necessary.
    """
    # Call get_vector_store which now handles logging more precisely
    vector_store = get_vector_store(chat_id, pdf_files)
    retriever = vector_store.as_retriever(
        search_type="mmr",  # better match diversity
        search_kwargs={"k": 8}
    )
    return retriever