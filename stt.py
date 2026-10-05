# stt.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Dict, Any


@dataclass
class STTSegment:
    start: float
    end: float
    text: str


@dataclass
class STTResult:
    text: str
    language: Optional[str]
    language_probability: Optional[float]
    segments: List[STTSegment]


class FasterWhisperSTT:
    """
    Local STT using faster-whisper (Whisper via CTranslate2).
    - device: usually "cpu" in WSL on Snapdragon
    - compute_type: "int8" is a good CPU default
    """

    def __init__(self, model_name_or_path: str = "small", device: str = "cpu", compute_type: str = "int8"):
        try:
            from faster_whisper import WhisperModel
        except ImportError as e:
            raise ImportError("Missing dependency. Install with: pip install faster-whisper") from e

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.compute_type = compute_type
        self._model = WhisperModel(model_name_or_path, device=device, compute_type=compute_type)

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        vad_parameters: Optional[Dict[str, Any]] = None,
        word_timestamps: bool = False,
    ) -> STTResult:
        kwargs: Dict[str, Any] = {
            "beam_size": beam_size,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
        }
        if language is not None:
            kwargs["language"] = language
        if vad_parameters is not None:
            kwargs["vad_parameters"] = vad_parameters

        segments, info = self._model.transcribe(audio_path, **kwargs)

        seg_list: List[STTSegment] = []
        texts: List[str] = []

        for s in segments:  # generator: iteration triggers transcription
            t = (s.text or "").strip()
            if t:
                texts.append(t)
            seg_list.append(STTSegment(start=float(s.start), end=float(s.end), text=t))

        full_text = " ".join(texts).strip()
        return STTResult(
            text=full_text,
            language=getattr(info, "language", None),
            language_probability=getattr(info, "language_probability", None),
            segments=seg_list,
        )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python stt.py <audio_file>")
        sys.exit(2)

    stt = FasterWhisperSTT(model_name_or_path="small", device="cpu", compute_type="int8")
    result = stt.transcribe(sys.argv[1], vad_filter=True)
    print(result.text)
