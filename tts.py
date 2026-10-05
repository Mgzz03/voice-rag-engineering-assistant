# tts.py
from __future__ import annotations

import wave
from functools import lru_cache
from pathlib import Path
from typing import Optional


class PiperTTS:
    """
    Local TTS using Piper.

    model_path should point to the local .onnx file.
    The matching .onnx.json must be in the same directory with the same base name.
    """

    def __init__(self, model_path: str):
        self.model_path = str(model_path)

        try:
            from piper.voice import PiperVoice
        except ImportError as e:
            raise ImportError("Missing dependency. Install with: pip install piper-tts") from e

        self._PiperVoice = PiperVoice

    @lru_cache(maxsize=2)
    def _load_voice(self, model_path: str):
        return self._PiperVoice.load(model_path)

    @property
    def sample_rate(self) -> int:
        voice = self._load_voice(self.model_path)
        return int(voice.config.sample_rate)

    def synthesize_to_wav(self, text: str, out_wav_path: str) -> str:
        import numpy as np
        import logging

        logger = logging.getLogger(__name__)
        voice = self._load_voice(self.model_path)

        out_path = Path(out_wav_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect audio chunks from Piper
        audio_chunks = []
        chunk_count = 0

        logger.info(f"Starting TTS synthesis for text: {text[:50]}...")

        for audio_chunk in voice.synthesize(text):
            chunk_count += 1

            # Piper AudioChunk has audio_int16_array property
            if hasattr(audio_chunk, 'audio_int16_array'):
                audio_data_chunk = audio_chunk.audio_int16_array
                audio_chunks.append(audio_data_chunk)
                logger.info(f"Chunk {chunk_count}: {len(audio_data_chunk)} samples")
            else:
                logger.error(f"AudioChunk missing audio_int16_array attribute")
                raise ValueError(f"Unknown audio chunk format: {type(audio_chunk)}")

        logger.info(f"Total chunks collected: {len(audio_chunks)}")

        # Combine all audio chunks
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks)
            logger.info(f"Combined audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
        else:
            # Empty audio for empty text
            audio_data = np.array([], dtype=np.int16)
            logger.warning("No audio chunks generated")

        # Write to WAV file
        with wave.open(str(out_path), "wb") as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(voice.config.sample_rate)
            wav_file.writeframes(audio_data.tobytes())

        file_size = out_path.stat().st_size
        logger.info(f"WAV file written: {out_path}, size: {file_size} bytes")

        return str(out_path)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python tts.py <text> <out.wav> [model.onnx]")
        sys.exit(2)

    text = sys.argv[1]
    out_wav = sys.argv[2]
    model = sys.argv[3] if len(sys.argv) >= 4 else "voices/en_US-lessac-medium.onnx"

    tts = PiperTTS(model)
    path = tts.synthesize_to_wav(text, out_wav)
    print(f"Wrote: {path} (sample_rate={tts.sample_rate})")
