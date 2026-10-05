"""
FastAPI web server for the RAG Q&A system with voice support.

Provides REST API and WebSocket endpoints for:
- Text-based Q&A
- Speech-to-Text (STT) for voice input
- Text-to-Speech (TTS) for voice output
"""

import sys
import os
import tempfile
import base64
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('web_server.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from src.rag_pipeline import RAGPipeline
from stt import FasterWhisperSTT
from tts import PiperTTS


# Global instances
rag: Optional[RAGPipeline] = None
stt: Optional[FasterWhisperSTT] = None
tts: Optional[PiperTTS] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global rag, stt, tts

    print("Initializing RAG pipeline...")
    rag = RAGPipeline(
        chunks_path="data/processed/engineering/chunks.jsonl",
        max_chunks=10,
        min_chunks=1,
        score_threshold=0.5,
        score_gap_threshold=3.0,
        min_score_for_llm=2.0
    )
    print("✓ RAG pipeline ready")

    try:
        print("Initializing voice components...")
        stt = FasterWhisperSTT(model_name_or_path="small", device="cpu", compute_type="int8")
        tts = PiperTTS(model_path="voices/en_US-lessac-medium.onnx")
        print("✓ Voice components ready")
    except Exception as e:
        print(f"⚠️  Voice support unavailable: {e}")
        stt = None
        tts = None

    yield

    # Cleanup
    print("Shutting down...")


app = FastAPI(
    title="Chatbot System",
    description="Zewail City",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QuestionRequest(BaseModel):
    question: str
    temperature: float = 0.3
    return_sources: bool = True


class QuestionResponse(BaseModel):
    answer: str
    sources: list[Dict[str, Any]]
    num_chunks_used: int


class StatusResponse(BaseModel):
    status: str
    rag_available: bool
    stt_available: bool
    tts_available: bool
    stt_model: Optional[str] = None
    tts_model: Optional[str] = None


class ClearHistoryResponse(BaseModel):
    status: str
    message: str


# REST API Endpoints

@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Get system status and available features."""
    return StatusResponse(
        status="online",
        rag_available=rag is not None,
        stt_available=stt is not None,
        tts_available=tts is not None,
        stt_model=stt.model_name_or_path if stt else None,
        tts_model="en_US-lessac-medium" if tts else None,
    )


@app.post("/api/question", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question and get an answer."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    try:
        result = rag.query(
            request.question,
            return_sources=request.return_sources,
            temperature=request.temperature
        )

        return QuestionResponse(
            answer=result["answer"],
            sources=result.get("sources", []),
            num_chunks_used=result.get("num_chunks_used", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/api/stt")
async def speech_to_text(audio: UploadFile = File(...)):
    """Convert speech audio to text."""
    if not stt:
        raise HTTPException(status_code=503, detail="STT not available")

    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            content = await audio.read()
            temp_audio.write(content)
            temp_audio_path = temp_audio.name

        # Transcribe
        result = stt.transcribe(temp_audio_path, vad_filter=True)

        # Cleanup
        os.unlink(temp_audio_path)

        return {
            "text": result.text,
            "language": result.language,
            "language_probability": result.language_probability,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error transcribing audio: {str(e)}")


@app.post("/api/tts")
async def text_to_speech(request: dict):
    """Convert text to speech audio."""
    if not tts:
        logger.error("TTS not available")
        raise HTTPException(status_code=503, detail="TTS not available")

    text = request.get("text", "")
    if not text:
        logger.error("No text provided for TTS")
        raise HTTPException(status_code=400, detail="No text provided")

    logger.info(f"TTS request received for text: {text[:100]}...")

    try:
        # Generate speech
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio_path = temp_audio.name

        logger.info(f"Temp file created: {temp_audio_path}")
        tts.synthesize_to_wav(text, temp_audio_path)

        # Read audio file
        with open(temp_audio_path, "rb") as f:
            audio_data = f.read()

        logger.info(f"Audio file read: {len(audio_data)} bytes")

        # Cleanup
        os.unlink(temp_audio_path)
        logger.info("Temp file cleaned up")

        # Return audio file
        return StreamingResponse(
            iter([audio_data]),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=speech.wav"
            }
        )
    except Exception as e:
        logger.error(f"TTS error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")


@app.post("/api/clear-history", response_model=ClearHistoryResponse)
async def clear_history():
    """Clear conversation history."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    rag.clear_history()
    return ClearHistoryResponse(
        status="success",
        message="Conversation history cleared"
    )


@app.get("/api/history")
async def get_history():
    """Get conversation history."""
    if not rag:
        raise HTTPException(status_code=503, detail="RAG pipeline not available")

    return {
        "history": rag.conversation_history,
        "num_exchanges": len(rag.conversation_history) // 2
    }


# WebSocket endpoint for real-time chat
@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()

    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()

            message_type = data.get("type")

            if message_type == "question":
                question = data.get("question", "")
                temperature = data.get("temperature", 0.3)

                if not rag:
                    await websocket.send_json({
                        "type": "error",
                        "error": "RAG pipeline not available"
                    })
                    continue

                try:
                    # Send processing status
                    await websocket.send_json({
                        "type": "processing",
                        "message": "Searching and generating answer..."
                    })

                    # Get answer
                    result = rag.query(
                        question,
                        return_sources=True,
                        temperature=temperature
                    )

                    # Send answer
                    await websocket.send_json({
                        "type": "answer",
                        "answer": result["answer"],
                        "sources": result.get("sources", []),
                        "num_chunks_used": result.get("num_chunks_used", 0)
                    })
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "error": str(e)
                    })

            elif message_type == "clear_history":
                if rag:
                    rag.clear_history()
                    await websocket.send_json({
                        "type": "history_cleared",
                        "message": "Conversation history cleared"
                    })

            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        print("Client disconnected")


if __name__ == "__main__":
    import uvicorn

    print("="*80)
    print("ENGINEERING Q&A WEB SERVER")
    print("="*80)
    print("\nStarting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    print("\nPress Ctrl+C to stop")
    print("="*80)

    uvicorn.run(app, host="0.0.0.0", port=8000)
