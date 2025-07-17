import React, { useState } from 'react';

function CreateChat({ API_BASE_URL, onCreateSuccess }) {
  const [chatNameInput, setChatNameInput] = useState('');
  const [statusMessage, setStatusMessage] = useState('');

  const handleCreateChat = async (event) => {
    event.preventDefault();
    setStatusMessage('Creating chat...');

    const requestBody = {};
    if (chatNameInput.trim()) {
      requestBody.chat_name = chatNameInput.trim();
    }

    try {
      const response = await fetch(`${API_BASE_URL}/create_chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody),
      });

      const data = await response.json();
      if (response.ok) {
        setStatusMessage(`Chat created! ID: ${data.chat_id}`);
        setChatNameInput('');
        onCreateSuccess(data.chat_id); 
      } else {
        setStatusMessage(`Error: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.error('Error creating chat:', error);
      setStatusMessage('Error connecting to server.');
    }
  };

  return (
    <section className="create-chat-section">
      <h3>Create New Chat</h3>
      <form onSubmit={handleCreateChat}>
        <input
          type="text"
          value={chatNameInput}
          onChange={(e) => setChatNameInput(e.target.value)}
          placeholder="Enter chat name (optional)"
        />
        <button type="submit">Create Chat</button>
      </form>
      {statusMessage && <p className="status-message">{statusMessage}</p>}
    </section>
  );
}

export default CreateChat;