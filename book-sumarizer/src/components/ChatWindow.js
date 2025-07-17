import React, { useState, useEffect, useRef } from 'react';
import './ChatWindow.css'; 

function ChatWindow({ API_BASE_URL, chatId, onChatRenamed }) {
  const [chatHistory, setChatHistory] = useState([]);
  const [questionInput, setQuestionInput] = useState('');
  const [uploadStatus, setUploadStatus] = useState('');
  const [questionStatus, setQuestionStatus] = useState('');
  const [newChatNameInput, setNewChatNameInput] = useState('');
  const [renameStatus, setRenameStatus] = useState('');
  const chatWindowRef = useRef(null); 

  useEffect(() => {
    if (chatWindowRef.current) {
      chatWindowRef.current.scrollTop = chatWindowRef.current.scrollHeight;
    }
  }, [chatHistory]);

  useEffect(() => {
    setChatHistory([]); 
    setQuestionStatus('Loading chat history...');
    const fetchChatHistory = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/load_chat_history/${chatId}`);
        if (response.ok) {
          const data = await response.json();
          setChatHistory(data.chat_history);
          setQuestionStatus('');
        } else {
          setQuestionStatus(`Error loading history: ${data.error || 'Unknown error'}`);
          console.error('Failed to load chat history:', await response.text());
        }
      } catch (error) {
        setQuestionStatus('Error connecting to server to load history.');
        console.error('Error loading chat history:', error);
      }
    };

    if (chatId) {
      fetchChatHistory();
    }
  }, [chatId, API_BASE_URL]);

  const handleFileUpload = async (event) => {
    event.preventDefault();
    const files = event.target.files;
    if (files.length === 0) {
      setUploadStatus('Please select PDF files.');
      return;
    }

    const formData = new FormData();
    for (let i = 0; i < files.length; i++) {
      formData.append('pdfs', files[i]);
    }

    setUploadStatus('Uploading and indexing... This may take a while.');
    try {
      const response = await fetch(`${API_BASE_URL}/upload_pdfs/${chatId}`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (response.ok) {
        setUploadStatus(`Success: ${data.message}`);
      } else {
        setUploadStatus(`Error: ${data.error || 'Unknown error'}`);
        console.error('File upload failed:', data);
      }
    } catch (error) {
      console.error('Error uploading files:', error);
      setUploadStatus('Error connecting to server for upload.');
    }
  };

  const handleQuestionSubmit = async (event) => {
    event.preventDefault();
    const question = questionInput.trim();
    if (!question) return;

    setChatHistory(prev => [...prev, { type: 'human', content: question }]);
    setQuestionInput('');
    setQuestionStatus('Thinking...');

    try {
      const response = await fetch(`${API_BASE_URL}/chat/${chatId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();
      if (response.ok) {
        setChatHistory(prev => [...prev, { type: 'ai', content: data.answer, sources: data.sources }]);
        setQuestionStatus('');
      } else {
        setQuestionStatus(`Error: ${data.error || 'Unknown error'}`);
        console.error('Chat failed:', data);
      }
    } catch (error) {
      console.error('Error sending message:', error);
      setQuestionStatus('Error connecting to server.');
    }
  };

  const handleRenameChat = async (event) => {
    event.preventDefault();
    const newName = newChatNameInput.trim();
    if (!newName) {
      setRenameStatus("Please enter a new chat name.");
      return;
    }

    setRenameStatus('Renaming...');
    try {
      const response = await fetch(`${API_BASE_URL}/rename_chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ old_chat_id: chatId, new_chat_name: newName }),
      });

      const data = await response.json();
      if (response.ok) {
        setRenameStatus(`Success: ${data.message}`);
        onChatRenamed(data.new_chat_id); 
        setNewChatNameInput(''); 
      } else {
        setRenameStatus(`Error: ${data.error || 'Unknown error'}`);
        console.error('Rename failed:', data);
      }
    } catch (error) {
      console.error('Error renaming chat:', error);
      setRenameStatus('Error connecting to server for rename.');
    }
  };

  if (!chatId) {
    return <p className="select-chat-message">Select a chat to view its window.</p>;
  }

  return (
    <div className="chat-window-container">
      <h2>Chat ID: {chatId}</h2>

      <section className="rename-chat-section">
        <h3>Rename this Chat</h3>
        <form onSubmit={handleRenameChat}>
          <input
            type="text"
            value={newChatNameInput}
            onChange={(e) => setNewChatNameInput(e.target.value)}
            placeholder="Enter new chat name"
            required
          />
          <button type="submit">Rename Chat</button>
        </form>
        {renameStatus && <p className="status-message">{renameStatus}</p>}
      </section>

      <section className="upload-pdf-section">
        <h3>Upload PDFs for this Chat</h3>
        <form>
          <input type="file" multiple accept=".pdf" onChange={handleFileUpload} />
        </form>
        {uploadStatus && <p className="status-message">{uploadStatus}</p>}
      </section>

      <section className="chat-interaction-section">
        <h3>Conversation</h3>
        <div className="chat-window" ref={chatWindowRef}>
          {chatHistory.map((msg, index) => (
            <div key={index} className={`message-container ${msg.type}-message`}>
              {msg.type === 'human' ? 'You: ' : 'AI: '}
              {msg.content}
              {msg.type === 'ai' && msg.sources && msg.sources.length > 0 && (
                <div className="source-info">
                  Sources: {msg.sources.map(s => `${s.source_file} (Chunk ${s.chunk})`).join('; ')}
                </div>
              )}
            </div>
          ))}
        </div>
        <form onSubmit={handleQuestionSubmit}>
          <textarea
            value={questionInput}
            onChange={(e) => setQuestionInput(e.target.value)}
            placeholder="Type your question..."
            rows="3"
          ></textarea>
          <button type="submit">Send Message</button>
        </form>
        {questionStatus && <p className="status-message">{questionStatus}</p>}
      </section>
    </div>
  );
}

export default ChatWindow;