from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os

embeddings = OllamaEmbeddings(model="mxbai-embed-large") # Initializes the Ollama embeddings model.

def get_vector_store(chat_id: str, pdf_files: list = None):
    """Initializes or retrieves a Chroma vector store for a given chat_id, loading and adding documents if provided."""
    db_location_for_chat = f"./chroma_db/{chat_id}" # Defines the unique directory path for the chat's vector store.

    db_exists = os.path.exists(db_location_for_chat) # Checks if the vector store directory already exists for the chat.

    text_splitter = RecursiveCharacterTextSplitter( # Instantiates the text splitter for document chunking.
        chunk_size=1500,
        chunk_overlap=300
    )

    documents_to_add = [] # Initializes a list to hold documents prepared for addition.
    ids_to_add = [] # Initializes a list to hold unique IDs for the documents.

    if pdf_files: # Processes PDF files if a list of file paths is provided.
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

    vector_store = Chroma( # Initializes or loads the Chroma vector store instance.
        collection_name=f"chat_{chat_id}_pdfs",
        persist_directory=db_location_for_chat,
        embedding_function=embeddings
    )

    if not db_exists: # Handles the case where the vector store is being created for the first time.
        if documents_to_add:
            vector_store.add_documents(documents=documents_to_add, ids=ids_to_add)
            print(f"[{chat_id}] New vector store created and {len(documents_to_add)} documents added.")
        else:
            print(f"[{chat_id}] New vector store initialized (no documents added yet).")
    elif documents_to_add: # Handles adding new documents to an existing vector store.
        vector_store.add_documents(documents=documents_to_add, ids=ids_to_add)
        print(f"[{chat_id}] {len(documents_to_add)} new documents added to existing vector store.")
    else: # Handles loading an existing vector store when no new documents are provided.
        print(f"[{chat_id}] Vector store already exists. Loading existing store.")

    return vector_store

def get_retriever(chat_id: str, pdf_files: list = None):
    """Returns a configured retriever for a given chat_id, creating the vector store if necessary."""
    vector_store = get_vector_store(chat_id, pdf_files)
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8}
    )
    return retriever