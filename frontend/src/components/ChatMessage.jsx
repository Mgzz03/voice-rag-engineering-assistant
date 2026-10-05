import { useState } from 'react'
import './ChatMessage.css'

function ChatMessage({ message, hideAssistantText = false }) {
  const [showSources, setShowSources] = useState(false)

  if (message.role === 'user') {
    return (
      <div className="message user-message">
        <div className="message-avatar">👤</div>
        <div className="message-content">
          <div className="message-text">{message.content}</div>
        </div>
      </div>
    )
  }

  if (message.role === 'assistant') {
    return (
      <div className="message assistant-message">
        <div className="message-avatar">🤖</div>
        <div className="message-content">
          {hideAssistantText ? (
            <div className="message-text voice-mode-indicator">
              <span className="voice-icon">🔊</span>
              <span>Playing audio response...</span>
            </div>
          ) : (
            <div className="message-text">{message.content}</div>
          )}

          {message.sources && message.sources.length > 0 && (
            <div className="sources-section">
              <button
                className="sources-toggle"
                onClick={() => setShowSources(!showSources)}
              >
                📚 {message.num_chunks_used} source{message.num_chunks_used !== 1 ? 's' : ''} used
                <span className="toggle-icon">{showSources ? '▲' : '▼'}</span>
              </button>

              {showSources && (
                <div className="sources-list">
                  {message.sources.map((source, idx) => (
                    <div key={idx} className="source-item">
                      <div className="source-header">
                        <span className="source-type">{source.doc_type.toUpperCase()}</span>
                        <span className="source-page">Page {source.page}</span>
                        <span className="source-score">Score: {source.score.toFixed(2)}</span>
                      </div>
                      <div className="source-text">{source.text}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    )
  }

  if (message.role === 'error') {
    return (
      <div className="message error-message">
        <div className="message-avatar">⚠️</div>
        <div className="message-content">
          <div className="message-text">{message.content}</div>
        </div>
      </div>
    )
  }

  return null
}

export default ChatMessage
