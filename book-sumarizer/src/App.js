import React, { useState, useEffect } from 'react';
import ChatList from './components/ChatList';
import CreateChat from './components/CreateChat';
import ChatWindow from './components/ChatWindow';
import './App.css'; // For basic styling

function App() {
  const [selectedChatId, setSelectedChatId] = useState(null);
  const [chats, setChats] = useState([]); 

  const API_BASE_URL = 'http://127.0.0.1:5000'; 

  const fetchChats = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/list_chats`);
      if (response.ok) {
        const data = await response.json();
        setChats(data.chat_ids);
      } else {
        console.error('Failed to fetch chats:', await response.text());
      }
    } catch (error) {
      console.error('Error fetching chats:', error);
    }
  };

  useEffect(() => {
    fetchChats();
  }, []);

  const handleChatCreated = (newChatId) => {
    fetchChats(); 
    setSelectedChatId(newChatId); 
  };

  return (
    <div className="App">
      <header className="App-header">
        <h1>RAG Chat Assistant</h1>
      </header>
      <div className="main-layout">
        <div className="left-panel">
          <CreateChat API_BASE_URL={API_BASE_URL} onCreateSuccess={handleChatCreated} />
          <ChatList API_BASE_URL={API_BASE_URL} chats={chats} onSelectChat={setSelectedChatId} selectedChatId={selectedChatId} />
        </div>
        <div className="right-panel">
          {selectedChatId ? (
            <ChatWindow API_BASE_URL={API_BASE_URL} chatId={selectedChatId} onChatRenamed={handleChatCreated} />
          ) : (
            <p className="select-chat-message">Select a chat from the left or create a new one.</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;