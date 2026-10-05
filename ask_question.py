"""
Interactive Q&A interface for the RAG system.

Run this to ask questions about engineering programs, courses, and policies.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.rag_pipeline import RAGPipeline

# Optional voice support
try:
    from stt import FasterWhisperSTT
    from tts import PiperTTS
    from audio_recorder import AudioRecorder
    VOICE_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    VOICE_IMPORT_ERROR = str(e)


def play_audio(audio_path: str):
    """Play audio file using system default player."""
    import platform
    import subprocess

    system = platform.system()
    try:
        if system == "Linux":
            # Try common Linux audio players
            for player in ["aplay", "paplay", "ffplay"]:
                try:
                    subprocess.run([player, audio_path], check=True, capture_output=True)
                    return
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            print("⚠️  No audio player found (tried aplay, paplay, ffplay)")
        elif system == "Darwin":  # macOS
            subprocess.run(["afplay", audio_path], check=True)
        elif system == "Windows":
            os.startfile(audio_path)
    except Exception as e:
        print(f"⚠️  Could not play audio: {e}")


def main():
    """Run interactive Q&A session."""
    print("="*80)
    print("ENGINEERING PROGRAM Q&A SYSTEM")
    print("Powered by RAG (Retrieval-Augmented Generation)")
    print("="*80)

    # Initialize pipeline with adaptive chunk selection
    print("\nInitializing system...")
    try:
        rag = RAGPipeline(
            chunks_path="data/processed/engineering/chunks.jsonl",
            max_chunks=10,            # Maximum chunks to send
            min_chunks=1,             # Minimum chunks to send if any found
            score_threshold=0.5,      # Only use chunks with score >= 0.5
            score_gap_threshold=3.0,  # Stop if score drops by more than 3.0
            min_score_for_llm=2.0     # Don't call LLM if best score < 2.0
        )
        print("✓ System ready!")
        print("  Adaptive selection: 1-10 chunks based on relevance")
        print("  Context-aware: LLM handles conversation history")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("\nMake sure:")
        print("  1. The chunks file exists: data/processed/engineering/chunks.jsonl")
        print("  2. phi-3-mini-4k-instruct is running on http://192.168.1.149:1234")
        return

    # Initialize voice components (optional)
    stt = None
    tts = None
    recorder = None
    voice_input_enabled = False
    voice_output_enabled = False

    if VOICE_AVAILABLE:
        try:
            print("\nInitializing voice components...")
            stt = FasterWhisperSTT(model_name_or_path="small", device="cpu", compute_type="int8")
            tts = PiperTTS(model_path="voices/en_US-lessac-medium.onnx")
            recorder = AudioRecorder(sample_rate=16000, channels=1)
            print("✓ Voice support enabled!")
            print("  STT: Whisper (small)")
            print("  TTS: Piper (en_US-lessac-medium)")
        except Exception as e:
            print(f"⚠️  Voice support unavailable: {e}")
            print("  Text-only mode will be used")
            VOICE_AVAILABLE = False
    else:
        print(f"\n⚠️  Voice support unavailable: {VOICE_IMPORT_ERROR}")
        print("  Install dependencies: pip install faster-whisper piper-tts sounddevice soundfile")
        print("  Text-only mode will be used")

    print("="*80)
    print("Ask questions about:")
    print("  - Course prerequisites and credits")
    print("  - Academic policies")
    print("  - Program curricula")
    print("\nCommands:")
    print("  'quit' or 'exit' - End session")
    print("  'sources' - Show sources for last answer")
    print("  'clear' - Clear conversation history")
    print("  'history' - Show conversation summary")
    if VOICE_AVAILABLE:
        print("\nVoice Commands:")
        print("  'voice input' or 'vi' - Toggle voice input (speak questions)")
        print("  'voice output' or 'vo' - Toggle voice output (hear answers)")
        print("  'voice status' or 'vs' - Show current voice mode status")
    print("\n💬 Chat mode enabled - follow-up questions are supported!")
    print("="*80)

    last_result = None

    while True:
        # Get user input
        print("\n" + "-"*80)
        question = input("\nYour question: ").strip()

        if not question:
            continue

        if question.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break

        # Handle commands
        if question.lower() == 'sources':
            if last_result:
                print("\nSources for last answer:")
                for i, source in enumerate(last_result.get('sources', []), 1):
                    print(f"\n{i}. {source['doc_type'].upper()} (page {source['page']}, score={source['score']:.2f})")
                    print(f"   {source['text']}")
            else:
                print("\nNo previous answer to show sources for.")
            continue

        if question.lower() == 'clear':
            rag.clear_history()
            print("\n✓ Conversation history cleared!")
            continue

        if question.lower() == 'history':
            if rag.conversation_history:
                print(f"\n💬 Conversation History ({len(rag.conversation_history)//2} exchanges):")
                for i in range(0, len(rag.conversation_history), 2):
                    if i+1 < len(rag.conversation_history):
                        user_msg = rag.conversation_history[i]['content'][:100]
                        assistant_msg = rag.conversation_history[i+1]['content'][:100]
                        print(f"  {i//2 + 1}. Q: {user_msg}...")
                        print(f"     A: {assistant_msg}...")
            else:
                print("\nNo conversation history yet.")
            continue

        # Get answer
        print("\nSearching and generating answer...")
        try:
            result = rag.query(question, return_sources=True, temperature=0.3)
            last_result = result

            print("\n" + "="*80)
            print("ANSWER:")
            print("="*80)
            print(result['answer'])

            # Show brief source info with adaptive selection indicator
            num_chunks = result.get('num_chunks_used', len(result.get('sources', [])))
            print(f"\n📚 Used {num_chunks} source{'s' if num_chunks != 1 else ''} (adaptive selection)")
            print("   (Type 'sources' to see details)")

        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("The LLM might be slow or unavailable. Please try again.")


if __name__ == "__main__":
    main()
