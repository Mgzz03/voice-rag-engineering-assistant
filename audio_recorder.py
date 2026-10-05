# audio_recorder.py
"""Simple audio recorder for voice input."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional


class AudioRecorder:
    """Records audio from microphone for STT processing."""

    def __init__(self, sample_rate: int = 16000, channels: int = 1):
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError as e:
            raise ImportError(
                "Missing dependencies. Install with: pip install sounddevice soundfile"
            ) from e

        self.sample_rate = sample_rate
        self.channels = channels
        self._sd = sd
        self._sf = sf

    def record_until_silence(
        self,
        silence_duration: float = 1.5,
        silence_threshold: float = 0.01,
        max_duration: float = 30.0,
    ) -> str:
        """
        Record audio until silence is detected.

        Args:
            silence_duration: Seconds of silence to stop recording
            silence_threshold: Amplitude threshold for silence
            max_duration: Maximum recording duration in seconds

        Returns:
            Path to temporary WAV file
        """
        import numpy as np

        print("🎤 Recording... (speak now, will auto-stop after silence)")

        recording = []
        silence_samples = int(silence_duration * self.sample_rate)
        max_samples = int(max_duration * self.sample_rate)
        consecutive_silence = 0

        def callback(indata, frames, time, status):
            nonlocal consecutive_silence

            if status:
                print(f"Status: {status}")

            # Add to recording
            recording.append(indata.copy())

            # Check for silence
            amplitude = np.abs(indata).mean()
            if amplitude < silence_threshold:
                consecutive_silence += frames
            else:
                consecutive_silence = 0

            # Stop if enough silence or max duration reached
            if consecutive_silence >= silence_samples:
                raise self._sd.CallbackStop()
            if len(recording) * frames >= max_samples:
                raise self._sd.CallbackStop()

        try:
            with self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                callback=callback,
            ):
                while True:
                    self._sd.sleep(100)
        except self._sd.CallbackStop:
            pass

        if not recording:
            raise ValueError("No audio recorded")

        # Concatenate and save
        import numpy as np
        audio_data = np.concatenate(recording, axis=0)

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()

        self._sf.write(temp_path, audio_data, self.sample_rate)
        print(f"✓ Recording saved ({len(audio_data)/self.sample_rate:.1f}s)")

        return temp_path

    def record_fixed_duration(self, duration: float = 5.0) -> str:
        """
        Record audio for a fixed duration.

        Args:
            duration: Recording duration in seconds

        Returns:
            Path to temporary WAV file
        """
        print(f"🎤 Recording for {duration} seconds...")

        audio_data = self._sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        self._sd.wait()

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_path = temp_file.name
        temp_file.close()

        self._sf.write(temp_path, audio_data, self.sample_rate)
        print("✓ Recording complete")

        return temp_path


if __name__ == "__main__":
    import sys

    recorder = AudioRecorder()

    if len(sys.argv) > 1 and sys.argv[1] == "--fixed":
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
        path = recorder.record_fixed_duration(duration)
    else:
        path = recorder.record_until_silence()

    print(f"Saved to: {path}")
