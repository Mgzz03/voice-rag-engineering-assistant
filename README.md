# Voice RAG Engineering Assistant

A voice-enabled retrieval augmented generation assistant for answering questions from engineering curriculum and policy documents.

## Capabilities

- Speech input and text-to-speech output.
- Document ingestion, chunking, and BM25 retrieval.
- LLM answer generation grounded in retrieved engineering documents.
- React web interface with chat and voice controls.
- Dataset previews, gold questions, and report metadata for evaluation.

## Main Components

| Path | Role |
| --- | --- |
| `src/rag_pipeline.py` | Retrieval and answer pipeline |
| `src/retriever_bm25.py` | Lexical document retrieval |
| `stt.py` | Speech-to-text integration |
| `tts.py` | Text-to-speech integration |
| `web_server.py` | Backend service |
| `frontend/` | React interface |

## Setup

Install the Python dependencies listed in `requirements-web.txt`, configure the model and service settings required by `src/llm_client.py`, then install the frontend dependencies:

```bash
cd frontend
npm install
npm run dev
```

The large voice model and source PDFs are excluded from the public repository. Place local models and document collections under the paths expected by the application before running the full workflow.

Included documentation: `Speech_Recognition_Chatbot_presentation.pdf` and `Voice_RAG_Chatbot_Paper.pdf`.
