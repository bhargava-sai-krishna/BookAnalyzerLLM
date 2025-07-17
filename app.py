from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from vector import get_retriever, get_vector_store
import os
import uuid
import shutil # For deleting directories

app = Flask(__name__)

# In a real application, you'd use a database for chat history.
# For this example, we'll use a simple in-memory dictionary.
chat_histories = {} # Stores history per chat_id: {chat_id: [HumanMessage, AIMessage, ...]}
retrievers = {} # Stores active retrievers per chat_id: {chat_id: retriever_instance}

# Initialize the LLM model globally as it's likely reusable across chats
model = OllamaLLM(model="llama3.2")

# Template for the LLM
template = """
You are a helpful assistant.
Your primary goal is to answer questions ONLY based on the provided context.
Do NOT make up information or introduce external knowledge.
If the answer cannot be found in the provided context, state "I cannot answer this question based on the provided documents."

Here are some relevant excerpts from official sources:
{context}

Here is the conversation history:
{chat_history}

Based on the provided context and conversation history, answer the following question.
Provide a detailed, comprehensive, and thorough answer, elaborating on all relevant points found in the documents.
Organize your response in clear, well-structured paragraphs or bullet points to ensure it is informative.
Unless a specific word count is mentioned in the question, aim for an answer length of approximately 500 words.
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs_with_sources(docs):
    output = []
    for doc in docs:
        source = doc.metadata.get("source_file", "unknown.pdf")
        chunk = doc.metadata.get("chunk", "n/a")
        content = doc.page_content.strip()
        output.append(f"[Source: {source}, Chunk: {chunk}]\n{content}")
    return "\n\n".join(output)

def format_chat_history(history):
    formatted_history = []
    for message in history:
        if isinstance(message, HumanMessage):
            formatted_history.append(f"Human: {message.content}")
        elif isinstance(message, AIMessage):
            formatted_history.append(f"AI: {message.content}")
    return "\n".join(formatted_history)

# --- HTML Template Routes ---
@app.route('/')
def index():
    """Serves the main landing page to create a new chat."""
    return render_template('index.html')

@app.route('/chat_page/<chat_id>')
def chat_page(chat_id):
    """Serves the chat interface for a specific chat_id."""
    # The JavaScript on chat.html will extract the chat_id from the URL.
    return render_template('chat.html')
# --- End HTML Template Routes ---


@app.route('/create_chat', methods=['POST'])
def create_chat():
    """
    API endpoint to create a new chat session.
    Generates a unique chat_id and initializes an empty chat history and vector store.
    """
    chat_id = str(uuid.uuid4()) # Generate a unique ID for the new chat
    chat_histories[chat_id] = []
    # Initialize an empty vector store for the new chat. PDFs can be uploaded later.
    get_vector_store(chat_id) # Call with no pdf_files to just create the directory
    print(f"[{chat_id}] New chat created.")
    return jsonify({"chat_id": chat_id}), 201

@app.route('/upload_pdfs/<chat_id>', methods=['POST'])
def upload_pdfs(chat_id):
    """
    API endpoint to upload PDF files for a given chat_id.
    Files are saved and then indexed into the chat's specific Chroma DB.
    """
    if chat_id not in chat_histories:
        return jsonify({"error": "Chat ID not found."}), 404

    if 'pdfs' not in request.files:
        return jsonify({"error": "No PDF files provided."}), 400

    uploaded_files = request.files.getlist('pdfs')
    if not uploaded_files:
        return jsonify({"error": "No PDF files selected."}), 400

    pdf_paths = []
    upload_folder = f"uploaded_pdfs/{chat_id}"
    os.makedirs(upload_folder, exist_ok=True)

    for pdf_file in uploaded_files:
        if pdf_file.filename == '':
            continue
        filename = pdf_file.filename
        file_path = os.path.join(upload_folder, filename)
        pdf_file.save(file_path)
        pdf_paths.append(file_path)
        print(f"[{chat_id}] Saved {filename} to {file_path}")

    try:
        # Call get_vector_store to potentially add new documents
        # The internal logic of get_vector_store will determine if new documents
        # are added or if it just loads the existing store (and adds new docs to it)
        vector_store_instance = get_vector_store(chat_id, pdf_files=pdf_paths)

        # Now, explicitly get the retriever from this instance
        retrievers[chat_id] = vector_store_instance.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 6}
        )
        print(f"[{chat_id}] PDFs processed and retriever updated.")
        return jsonify({"message": f"PDFs uploaded and indexed for chat {chat_id}."}), 200
    except Exception as e:
        print(f"[{chat_id}] Error processing PDFs: {e}")
        return jsonify({"error": f"Failed to process PDFs: {str(e)}"}), 500

@app.route('/chat/<chat_id>', methods=['POST'])
def chat_with_llm(chat_id):
    """
    API endpoint to handle chat interactions for a given chat_id.
    Retrieves documents, constructs prompt with history, and gets LLM response.
    """
    if chat_id not in chat_histories:
        return jsonify({"error": "Chat ID not found."}), 404

    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    current_retriever = retrievers.get(chat_id)
    if not current_retriever:
        # If retriever isn't in memory (e.g., server restart), try to load it
        try:
            current_retriever = get_retriever(chat_id)
            retrievers[chat_id] = current_retriever
            print(f"[{chat_id}] Retriever loaded from disk.")
        except Exception as e:
            print(f"[{chat_id}] Error loading retriever: {e}")
            return jsonify({"error": f"No documents indexed for this chat yet, please upload PDFs. {str(e)}"}), 400

    print(f"[{chat_id}] Question received: {question}")

    # Retrieve relevant documents
    docs = current_retriever.invoke(question)
    context = format_docs_with_sources(docs)
    print(f"[{chat_id}] Retrieved {len(docs)} documents.")

    # Get chat history for the current chat
    current_chat_history = chat_histories[chat_id]
    formatted_history = format_chat_history(current_chat_history)

    # Create the chain for the current interaction
    chain = prompt | model

    # Invoke the chain with context, question, and chat history
    result = chain.invoke({"context": context, "chat_history": formatted_history, "question": question})

    # Update chat history
    chat_histories[chat_id].append(HumanMessage(content=question))
    chat_histories[chat_id].append(AIMessage(content=result))
    print(f"[{chat_id}] Answer generated and history updated.")

    return jsonify({"answer": result, "sources": [doc.metadata for doc in docs]}), 200

@app.route('/clear_all_data', methods=['POST'])
def clear_all_data():
    """
    API endpoint to clear all chat histories, uploaded files, and Chroma DBs.
    USE WITH CAUTION: This will permanently delete all data.
    """
    global chat_histories, retrievers
    chat_histories = {}
    retrievers = {}

    chroma_db_dir = "./chroma_db"
    uploaded_pdfs_dir = "./uploaded_pdfs"

    try:
        if os.path.exists(chroma_db_dir):
            shutil.rmtree(chroma_db_dir)
            print(f"Deleted directory: {chroma_db_dir}")
        if os.path.exists(uploaded_pdfs_dir):
            shutil.rmtree(uploaded_pdfs_dir)
            print(f"Deleted directory: {uploaded_pdfs_dir}")

        os.makedirs(chroma_db_dir, exist_ok=True)
        os.makedirs(uploaded_pdfs_dir, exist_ok=True)
        print("All data cleared and directories re-created.")
        return jsonify({"message": "All chat history, uploaded files, and vector databases have been cleared."}), 200
    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500

if __name__ == '__main__':
    # Ensure the base directories for chroma dbs and uploaded pdfs exist
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./uploaded_pdfs", exist_ok=True)
    print("Starting Flask application...")
    app.run(debug=True) # For development. Set debug=False for production.