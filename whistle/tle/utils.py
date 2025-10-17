import torch
import numpy as np
from typing import List
from transformers import WhisperProcessor, WhisperModel


def get_teacher_states(
    audio_list: List[np.ndarray],
    whisper: WhisperModel,
    processor: WhisperProcessor,
    sampling_rate: int = 16000,
) -> torch.Tensor:
    """
    Extract Whisper encoder states from audio inputs for teacher supervision.

    Args:
        audio_list: List of audio arrays, each as np.ndarray (samples,) or (channels, samples)
        whisper: Pre-loaded Whisper model (frozen)
        processor: Whisper processor for feature extraction
        sampling_rate: Expected sampling rate (default: 16000)

    Returns:
        torch.Tensor: Encoder hidden states (B, T_enc, H) where H=1280 for large-v3

    Raises:
        ValueError: If audio_list is empty or contains invalid audio
        RuntimeError: If feature extraction fails
    """
    if not audio_list:
        raise ValueError("audio_list cannot be empty")

    if not isinstance(audio_list, list):
        raise ValueError("audio_list must be a list of numpy arrays")

    device = next(whisper.parameters()).device

    try:
        # Feature extraction: log-Mel per Whisper spec
        inputs = processor(audio_list, sampling_rate=sampling_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(device)  # (B, n_mels, T_mel)

        with torch.no_grad():
            enc_out = whisper.encoder(input_features=input_features)
            E_teacher = enc_out.last_hidden_state  # (B, T_enc, H=1280)

        return E_teacher  # supervision target

    except Exception as e:
        raise RuntimeError(f"Failed to extract teacher states: {str(e)}") from e


def validate_audio_batch(
    audio_list: List[np.ndarray], expected_sr: int = 16000
) -> None:
    """
    Validate audio batch for consistency and basic quality checks.

    Args:
        audio_list: List of audio arrays to validate
        expected_sr: Expected sampling rate

    Raises:
        ValueError: If validation fails
    """
    if not audio_list:
        raise ValueError("Empty audio list")

    for i, audio in enumerate(audio_list):
        if not isinstance(audio, np.ndarray):
            raise ValueError(f"Audio {i} is not a numpy array")

        if audio.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            raise ValueError(f"Audio {i} has unsupported dtype: {audio.dtype}")

        if audio.size == 0:
            raise ValueError(f"Audio {i} is empty")

        # Basic duration check (too short/long might indicate issues)
        duration = len(audio) / expected_sr
        if duration < 0.1:  # Less than 100ms
            raise ValueError(f"Audio {i} too short: {duration:.2f}s")
        if duration > 30.0:  # More than 30s
            raise ValueError(f"Audio {i} too long: {duration:.2f}s")
