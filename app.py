from flask import Flask, request, jsonify, render_template, redirect, url_for
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from vector import get_retriever, get_vector_store
import os
import uuid
import shutil
import re
import json # For JSON operations
import jsonlines # For JSONL operations

app = Flask(__name__)

# Base directory for storing chat history files
CHAT_HISTORY_DIR = "./chat_histories"

# In-memory stores (still used for active session, but now backed by disk)
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

def is_valid_chat_name(name):
    """
    Checks if a given name is valid for use as a chat_id (and thus a folder/file name).
    Avoids characters that are problematic in file paths.
    """
    if not name or not isinstance(name, str):
        return False, "Name cannot be empty."
    if not re.fullmatch(r"^[a-zA-Z0-9 _-]{1,100}$", name.strip()):
        return False, "Name can only contain letters, numbers, spaces, hyphens, and underscores (max 100 characters)."
    if name.strip().startswith('.') or name.strip().endswith('.'):
        return False, "Name cannot start or end with a dot."
    if '..' in name.strip():
        return False, "Name cannot contain '..'."
    if os.sep in name.strip() or '/' in name.strip() or '\\' in name.strip():
        return False, "Name cannot contain path separators."
    return True, ""

def get_chat_history_filepath(chat_id):
    """Returns the file path for a chat's history JSONL file."""
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f"{chat_id}.jsonl")

def load_chat_history_from_file(chat_id):
    """Loads chat history from a JSONL file."""
    filepath = get_chat_history_filepath(chat_id)
    history = []
    if os.path.exists(filepath):
        try:
            with jsonlines.open(filepath, 'r') as reader:
                for obj in reader:
                    if obj['type'] == 'human':
                        history.append(HumanMessage(content=obj['content']))
                    elif obj['type'] == 'ai':
                        history.append(AIMessage(content=obj['content']))
            print(f"[{chat_id}] Loaded chat history from {filepath}")
        except Exception as e:
            print(f"[{chat_id}] Error loading chat history from {filepath}: {e}")
            # Consider returning empty history if file is corrupt
            history = []
    return history

def save_chat_history_to_file(chat_id, history):
    """Saves chat history to a JSONL file."""
    filepath = get_chat_history_filepath(chat_id)
    try:
        with jsonlines.open(filepath, 'w') as writer:
            for message in history:
                if isinstance(message, HumanMessage):
                    writer.write({'type': 'human', 'content': message.content})
                elif isinstance(message, AIMessage):
                    writer.write({'type': 'ai', 'content': message.content})
        print(f"[{chat_id}] Saved chat history to {filepath}")
    except Exception as e:
        print(f"[{chat_id}] Error saving chat history to {filepath}: {e}")

# --- HTML Template Routes (no changes here for now, they are for direct Flask rendering) ---
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
    Accepts an optional 'chat_name'. If not provided, generates a UUID.
    Initializes an empty chat history and vector store directory.
    """
    data = request.get_json()
    desired_chat_name = data.get("chat_name")

    if desired_chat_name:
        is_valid, msg = is_valid_chat_name(desired_chat_name)
        if not is_valid:
            return jsonify({"error": f"Invalid chat name: {msg}"}), 400
        
        chat_id = desired_chat_name.strip() # Use the provided name as the ID

        # Check if this name is already in use (folder or history file)
        if os.path.exists(f"./chroma_db/{chat_id}") or \
           os.path.exists(f"./uploaded_pdfs/{chat_id}") or \
           os.path.exists(get_chat_history_filepath(chat_id)):
            return jsonify({"error": f"Chat name '{chat_id}' already exists. Please choose a different name."}), 409 # 409 Conflict
    else:
        chat_id = str(uuid.uuid4()) # Generate a unique ID if no name provided

    chat_histories[chat_id] = [] # Initialize in-memory
    save_chat_history_to_file(chat_id, []) # Create empty history file
    
    try:
        get_vector_store(chat_id) # Call with no pdf_files to just create the directory
        print(f"[{chat_id}] New chat created.")
        return jsonify({"chat_id": chat_id}), 201
    except Exception as e:
        print(f"Error creating chat '{chat_id}': {e}")
        return jsonify({"error": f"Failed to create chat: {str(e)}"}), 500


@app.route('/upload_pdfs/<chat_id>', methods=['POST'])
def upload_pdfs(chat_id):
    """
    API endpoint to upload PDF files for a given chat_id.
    Files are saved and then indexed into the chat's specific Chroma DB.
    """
    # Load chat history if not in memory (e.g., after server restart)
    if chat_id not in chat_histories:
        db_location_for_chat = f"./chroma_db/{chat_id}"
        if os.path.exists(db_location_for_chat) or os.path.exists(get_chat_history_filepath(chat_id)):
            chat_histories[chat_id] = load_chat_history_from_file(chat_id)
            print(f"[{chat_id}] Chat history re-initialized from file before PDF upload.")
        else:
            return jsonify({"error": "Chat ID not found. Cannot upload PDFs without an existing chat."}), 404

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
        vector_store_instance = get_vector_store(chat_id, pdf_files=pdf_paths)
        retrievers[chat_id] = vector_store_instance.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 8} # Changed from 6 to 8 for more retrieved context
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
    # Load chat history if not in memory (e.g., after server restart)
    if chat_id not in chat_histories:
        db_location_for_chat = f"./chroma_db/{chat_id}"
        # Check if the DB directory or history file exists for this chat_id
        if os.path.exists(db_location_for_chat) or os.path.exists(get_chat_history_filepath(chat_id)):
            chat_histories[chat_id] = load_chat_history_from_file(chat_id)
            print(f"[{chat_id}] Chat history re-initialized from file.")
        else:
            return jsonify({"error": "Chat ID not found. No previous session data found for this ID."}), 404

    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    current_retriever = retrievers.get(chat_id)
    if not current_retriever:
        try:
            current_retriever = get_retriever(chat_id)
            retrievers[chat_id] = current_retriever
            print(f"[{chat_id}] Retriever loaded from disk.")
        except Exception as e:
            print(f"[{chat_id}] Error loading retriever: {e}")
            return jsonify({"error": f"Failed to load documents for this chat. Error: {str(e)}"}), 500

    print(f"[{chat_id}] Question received: {question}")

    # Retrieve relevant documents
    docs = current_retriever.invoke(question)
    context = format_docs_with_sources(docs)
    print(f"[{chat_id}] Retrieved {len(docs)} documents.")

    current_chat_history = chat_histories[chat_id]
    formatted_history = format_chat_history(current_chat_history)
    
    chain = prompt | model
    result = chain.invoke({"context": context, "chat_history": formatted_history, "question": question})

    # Update in-memory history
    chat_histories[chat_id].append(HumanMessage(content=question))
    chat_histories[chat_id].append(AIMessage(content=result))
    
    # Save updated history to file
    save_chat_history_to_file(chat_id, chat_histories[chat_id])
    
    print(f"[{chat_id}] Answer generated and history updated.")

    return jsonify({"answer": result, "sources": [doc.metadata for doc in docs]}), 200

@app.route('/rename_chat', methods=['POST'])
def rename_chat():
    """
    API endpoint to rename an existing chat session.
    Updates chat ID in memory and renames associated file system directories/files.
    """
    data = request.get_json()
    old_chat_id = data.get("old_chat_id")
    new_chat_name = data.get("new_chat_name")

    if not old_chat_id or not new_chat_name:
        return jsonify({"error": "Missing old_chat_id or new_chat_name."}), 400

    is_valid, msg = is_valid_chat_name(new_chat_name)
    if not is_valid:
        return jsonify({"error": f"Invalid new chat name: {msg}"}), 400

    new_chat_name = new_chat_name.strip() # Clean input

    # Check if old chat exists in memory or on disk
    old_chroma_path = f"./chroma_db/{old_chat_id}"
    old_uploaded_path = f"./uploaded_pdfs/{old_chat_id}"
    old_history_path = get_chat_history_filepath(old_chat_id)

    if not (os.path.exists(old_chroma_path) or os.path.exists(old_history_path) or old_chat_id in chat_histories):
        return jsonify({"error": f"Original chat ID '{old_chat_id}' not found. Cannot rename."}), 404

    # Check if new name already exists as a directory or in memory
    new_chroma_path = f"./chroma_db/{new_chat_name}"
    new_uploaded_path = f"./uploaded_pdfs/{new_chat_name}"
    new_history_path = get_chat_history_filepath(new_chat_name)

    if os.path.exists(new_chroma_path) or \
       os.path.exists(new_uploaded_path) or \
       os.path.exists(new_history_path) or \
       new_chat_name in chat_histories:
        return jsonify({"error": f"New chat name '{new_chat_name}' already exists. Please choose a different name."}), 409 # 409 Conflict

    try:
        # 1. Rename directories/files on file system
        if os.path.exists(old_chroma_path):
            os.rename(old_chroma_path, new_chroma_path)
            print(f"Renamed Chroma DB from '{old_chat_id}' to '{new_chat_name}'")
        else:
            # If Chroma path doesn't exist, create an empty one for the new name
            os.makedirs(new_chroma_path)
            print(f"Chroma DB for '{old_chat_id}' not found, created empty one for '{new_chat_name}'")

        if os.path.exists(old_uploaded_path):
            os.rename(old_uploaded_path, new_uploaded_path)
            print(f"Renamed Uploaded PDFs from '{old_chat_id}' to '{new_chat_name}'")
        else:
            # If uploaded path doesn't exist, create an empty one for the new name
            os.makedirs(new_uploaded_path)
            print(f"Uploaded PDFs for '{old_chat_id}' not found, created empty one for '{new_chat_name}'")
        
        if os.path.exists(old_history_path):
            os.rename(old_history_path, new_history_path)
            print(f"Renamed chat history file from '{old_chat_id}.jsonl' to '{new_chat_name}.jsonl'")
        else:
            # Create an empty history file for the new name if old didn't exist
            save_chat_history_to_file(new_chat_name, [])
            print(f"Chat history file for '{old_chat_id}' not found, created empty one for '{new_chat_name}'")


        # 2. Update in-memory dictionaries
        if old_chat_id in chat_histories:
            chat_histories[new_chat_name] = chat_histories.pop(old_chat_id)
            print(f"Updated chat_histories: '{old_chat_id}' -> '{new_chat_name}'")
        
        if old_chat_id in retrievers:
            retrievers[new_chat_name] = retrievers.pop(old_chat_id)
            print(f"Updated retrievers: '{old_chat_id}' -> '{new_chat_name}'")
        
        print(f"Chat '{old_chat_id}' successfully renamed to '{new_chat_name}'.")
        return jsonify({"message": "Chat renamed successfully.", "new_chat_id": new_chat_name}), 200

    except Exception as e:
        print(f"Error renaming chat '{old_chat_id}' to '{new_chat_name}': {e}")
        # Basic attempt to rollback in case of error (more robust rollback needed for production)
        if os.path.exists(new_chroma_path) and not os.path.exists(old_chroma_path): os.rename(new_chroma_path, old_chroma_path)
        if os.path.exists(new_uploaded_path) and not os.path.exists(old_uploaded_path): os.rename(new_uploaded_path, old_uploaded_path)
        if os.path.exists(new_history_path) and not os.path.exists(old_history_path): os.rename(new_history_path, old_history_path)
        return jsonify({"error": f"Failed to rename chat: {str(e)}"}), 500


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
    chat_history_base_dir = "./chat_histories"

    try:
        if os.path.exists(chroma_db_dir):
            shutil.rmtree(chroma_db_dir)
            print(f"Deleted directory: {chroma_db_dir}")
        if os.path.exists(uploaded_pdfs_dir):
            shutil.rmtree(uploaded_pdfs_dir)
            print(f"Deleted directory: {uploaded_pdfs_dir}")
        if os.path.exists(chat_history_base_dir):
            shutil.rmtree(chat_history_base_dir)
            print(f"Deleted directory: {chat_history_base_dir}")

        os.makedirs(chroma_db_dir, exist_ok=True)
        os.makedirs(uploaded_pdfs_dir, exist_ok=True)
        os.makedirs(chat_history_base_dir, exist_ok=True)
        print("All data cleared and directories re-created.")
        return jsonify({"message": "All chat history, uploaded files, and vector databases have been cleared."}), 200
    except Exception as e:
        print(f"Error clearing data: {e}")
        return jsonify({"error": f"Failed to clear data: {str(e)}"}), 500

@app.route('/list_chats', methods=['GET'])
def list_chats():
    """
    API endpoint to list all available chat IDs (which are now names).
    Scans the chroma_db directory for chat folders.
    """
    chat_ids = []
    if os.path.exists("./chroma_db"):
        for item in os.listdir("./chroma_db"):
            item_path = os.path.join("./chroma_db", item)
            if os.path.isdir(item_path):
                chat_ids.append(item)
    print(f"Listing available chats: {chat_ids}")
    return jsonify({"chat_ids": chat_ids}), 200

@app.route('/load_chat_history/<chat_id>', methods=['GET'])
def load_chat_history_api(chat_id):
    """
    API endpoint to load the conversation history for a specific chat_id.
    """
    history = load_chat_history_from_file(chat_id)
    # Convert HumanMessage/AIMessage objects to a serializable format for JSON
    serializable_history = []
    for msg in history:
        if isinstance(msg, HumanMessage):
            serializable_history.append({'type': 'human', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            serializable_history.append({'type': 'ai', 'content': msg.content})
    
    return jsonify({"chat_history": serializable_history}), 200


if __name__ == '__main__':
    # Ensure the base directories for chroma dbs, uploaded pdfs, and chat histories exist
    os.makedirs("./chroma_db", exist_ok=True)
    os.makedirs("./uploaded_pdfs", exist_ok=True)
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    print("Starting Flask application...")
    app.run(debug=True) # For development. Set debug=False for production.