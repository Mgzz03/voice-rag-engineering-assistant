import { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ChatMessage from './components/ChatMessage'
import VoiceControls from './components/VoiceControls'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [inputText, setInputText] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [systemStatus, setSystemStatus] = useState(null)

  // Voice mode states
  const [voiceInputEnabled, setVoiceInputEnabled] = useState(false)
  const [voiceOutputEnabled, setVoiceOutputEnabled] = useState(false)
  const [isRecording, setIsRecording] = useState(false)
  const [speakingMessageIndex, setSpeakingMessageIndex] = useState(null)

  const messagesEndRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const audioChunksRef = useRef([])

  // Fetch system status on mount
  useEffect(() => {
    fetchSystemStatus()
  }, [])

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const fetchSystemStatus = async () => {
    try {
      const response = await axios.get('/api/status')
      setSystemStatus(response.data)
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    }
  }

  const sendQuestion = async (questionText) => {
    if (!questionText.trim()) return

    // Add user message to chat
    const userMessage = { role: 'user', content: questionText }
    setMessages(prev => [...prev, userMessage])
    setInputText('')
    setIsLoading(true)

    try {
      const response = await axios.post('/api/question', {
        question: questionText,
        temperature: 0.3,
        return_sources: true
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.answer,
        sources: response.data.sources,
        num_chunks_used: response.data.num_chunks_used
      }

      // Add message to chat
      setMessages(prev => [...prev, assistantMessage])

      // If voice output is enabled, speak the answer (outside of state setter)
      if (voiceOutputEnabled && systemStatus?.tts_available) {
        // Get the message index (current messages length + user message + assistant message - 1)
        const messageIndex = messages.length + 1 // +1 for user message, this will be assistant message index
        setSpeakingMessageIndex(messageIndex)
        speakText(response.data.answer).finally(() => {
          setSpeakingMessageIndex(null)
        })
      }
    } catch (error) {
      console.error('Error:', error)
      const errorMessage = {
        role: 'error',
        content: error.response?.data?.detail || 'Failed to get answer. Please try again.'
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendQuestion(inputText)
  }

  const startVoiceInput = async () => {
    if (!systemStatus?.stt_available) {
      alert('Voice input not available. Check server configuration.')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      audioChunksRef.current = []

      const mediaRecorder = new MediaRecorder(stream)
      mediaRecorderRef.current = mediaRecorder

      mediaRecorder.ondataavailable = (event) => {
        audioChunksRef.current.push(event.data)
      }

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/wav' })
        await transcribeAudio(audioBlob)
        stream.getTracks().forEach(track => track.stop())
      }

      mediaRecorder.start()
      setIsRecording(true)
    } catch (error) {
      console.error('Error starting recording:', error)
      alert('Could not access microphone. Please check permissions.')
    }
  }

  const stopVoiceInput = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
    }
  }

  const transcribeAudio = async (audioBlob) => {
    setIsLoading(true)
    try {
      const formData = new FormData()
      formData.append('audio', audioBlob, 'recording.wav')

      const response = await axios.post('/api/stt', formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      })

      const transcribedText = response.data.text
      if (transcribedText) {
        sendQuestion(transcribedText)
      } else {
        alert('No speech detected. Please try again.')
        setIsLoading(false)
      }
    } catch (error) {
      console.error('Transcription error:', error)
      alert('Failed to transcribe audio. Please try again.')
      setIsLoading(false)
    }
  }

  const speakText = async (text) => {
    if (!systemStatus?.tts_available) {
      console.log('TTS not available')
      return
    }

    console.log('Generating speech for:', text)

    try {
      const response = await axios.post('/api/tts', { text }, {
        responseType: 'blob'
      })

      console.log('TTS response received:', response.data.type, response.data.size, 'bytes')

      // Check if we actually got audio data
      if (response.data.size === 0) {
        console.error('Received empty audio blob')
        return
      }

      const audioUrl = URL.createObjectURL(response.data)
      const audio = new Audio(audioUrl)

      // Return a promise that resolves when audio finishes
      return new Promise((resolve, reject) => {
        audio.onerror = (e) => {
          console.error('Audio playback error:', e)
          URL.revokeObjectURL(audioUrl)
          reject(e)
        }

        audio.onended = () => {
          console.log('Audio playback finished')
          URL.revokeObjectURL(audioUrl)
          resolve()
        }

        console.log('Starting audio playback...')
        audio.play().then(() => {
          console.log('Audio playing successfully')
        }).catch((playError) => {
          console.error('Play error:', playError)
          URL.revokeObjectURL(audioUrl)
          reject(playError)
        })
      })
    } catch (error) {
      console.error('TTS error:', error)
      if (error.response) {
        console.error('Response status:', error.response.status)
        console.error('Response data:', error.response.data)
      }
      throw error
    }
  }

  const clearHistory = async () => {
    try {
      await axios.post('/api/clear-history')
      setMessages([])
    } catch (error) {
      console.error('Error clearing history:', error)
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Chatbot System</h1>
        <p>Built for zewailcity</p>

        <div className="status-badges">
          {systemStatus && (
            <>
              <span className={`badge ${systemStatus.rag_available ? 'online' : 'offline'}`}>
                RAG: {systemStatus.rag_available ? 'Online' : 'Offline'}
              </span>
              <span className={`badge ${systemStatus.stt_available ? 'online' : 'offline'}`}>
                Voice Input: {systemStatus.stt_available ? 'Available' : 'Unavailable'}
              </span>
              <span className={`badge ${systemStatus.tts_available ? 'online' : 'offline'}`}>
                Voice Output: {systemStatus.tts_available ? 'Available' : 'Unavailable'}
              </span>
            </>
          )}
        </div>
      </header>

      <VoiceControls
        voiceInputEnabled={voiceInputEnabled}
        voiceOutputEnabled={voiceOutputEnabled}
        onToggleVoiceInput={() => setVoiceInputEnabled(!voiceInputEnabled)}
        onToggleVoiceOutput={() => setVoiceOutputEnabled(!voiceOutputEnabled)}
        sttAvailable={systemStatus?.stt_available}
        ttsAvailable={systemStatus?.tts_available}
        onClearHistory={clearHistory}
      />

      <main className="chat-container">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Welcome!</h2>
              <p>Ask questions about:</p>
              <ul>
                <li>Course prerequisites and credits</li>
                <li>Academic policies</li>
                <li>Program curricula</li>
              </ul>
              {systemStatus?.stt_available && (
                <p className="voice-hint">
                  💡 Enable voice input to ask questions by speaking
                </p>
              )}
            </div>
          )}

          {messages.map((msg, idx) => (
            <ChatMessage
              key={idx}
              message={msg}
              hideAssistantText={voiceOutputEnabled && msg.role === 'assistant' && speakingMessageIndex === idx}
            />
          ))}

          {isLoading && (
            <div className="loading-indicator">
              <div className="spinner"></div>
              <span>Thinking...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <form className="input-form" onSubmit={handleSubmit}>
          {voiceInputEnabled ? (
            <div className="voice-input-container">
              <button
                type="button"
                className={`voice-button ${isRecording ? 'recording' : ''}`}
                onClick={isRecording ? stopVoiceInput : startVoiceInput}
                disabled={isLoading}
              >
                {isRecording ? (
                  <>
                    <span className="mic-icon recording">🎙️</span>
                    <span>Stop Recording</span>
                  </>
                ) : (
                  <>
                    <span className="mic-icon">🎤</span>
                    <span>Press to Speak</span>
                  </>
                )}
              </button>
            </div>
          ) : (
            <>
              <input
                type="text"
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                placeholder="Ask a question..."
                disabled={isLoading}
                className="text-input"
              />
              <button
                type="submit"
                disabled={isLoading || !inputText.trim()}
                className="send-button"
              >
                Send
              </button>
            </>
          )}
        </form>
      </main>
    </div>
  )
}

export default App
