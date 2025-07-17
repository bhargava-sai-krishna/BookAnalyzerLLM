import React from 'react';
import './ChatList.css'; 

function ChatList({ chats, onSelectChat, selectedChatId }) {
  return (
    <section className="chat-list-section">
      <h3>Existing Chats</h3>
      {chats.length > 0 ? (
        <ul className="chat-list">
          {chats.map(chatId => (
            <li 
              key={chatId} 
              className={chatId === selectedChatId ? 'selected' : ''}
              onClick={() => onSelectChat(chatId)}
            >
              {chatId}
            </li>
          ))}
        </ul>
      ) : (
        <p>No existing chats found.</p>
      )}
    </section>
  );
}

export default ChatList;