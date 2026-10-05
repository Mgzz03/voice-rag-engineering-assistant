import './VoiceControls.css'

function VoiceControls({
  voiceInputEnabled,
  voiceOutputEnabled,
  onToggleVoiceInput,
  onToggleVoiceOutput,
  sttAvailable,
  ttsAvailable,
  onClearHistory
}) {
  return (
    <div className="voice-controls">
      <div className="controls-group">
        <button
          className={`control-button ${voiceInputEnabled ? 'active' : ''}`}
          onClick={onToggleVoiceInput}
          disabled={!sttAvailable}
          title={sttAvailable ? 'Toggle voice input' : 'Voice input not available'}
        >
          <span className="control-icon">🎤</span>
          <span className="control-label">
            Voice Input: {voiceInputEnabled ? 'ON' : 'OFF'}
          </span>
        </button>

        <button
          className={`control-button ${voiceOutputEnabled ? 'active' : ''}`}
          onClick={onToggleVoiceOutput}
          disabled={!ttsAvailable}
          title={ttsAvailable ? 'Toggle voice output' : 'Voice output not available'}
        >
          <span className="control-icon">🔊</span>
          <span className="control-label">
            Voice Output: {voiceOutputEnabled ? 'ON' : 'OFF'}
          </span>
        </button>

        <button
          className="control-button clear-button"
          onClick={onClearHistory}
          title="Clear conversation history"
        >
          <span className="control-icon">🗑️</span>
          <span className="control-label">Clear History</span>
        </button>
      </div>

      {voiceInputEnabled && sttAvailable && (
        <div className="mode-hint">
          🎙️ Voice input mode active - Click the microphone button to speak
        </div>
      )}

      {voiceOutputEnabled && ttsAvailable && (
        <div className="mode-hint">
          🔊 Voice output mode active - Answers will be spoken aloud
        </div>
      )}
    </div>
  )
}

export default VoiceControls
