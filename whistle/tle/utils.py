import torch
import torch.nn.functional as F
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


def augment_teacher_states(
    E_teacher: torch.Tensor,
    noise_std: float = 0.01,
    time_jitter_max: float = 0.1,
) -> torch.Tensor:
    """
    Apply noise and time-jitter augmentations to teacher encoder states.

    Args:
        E_teacher: Teacher encoder states (B, T, H)
        noise_std: Standard deviation for Gaussian noise
        time_jitter_max: Maximum time shift in frames (int(time_jitter_max * T))

    Returns:
        Augmented teacher states (B, T, H)
    """
    B, T, H = E_teacher.shape
    device = E_teacher.device

    # Apply Gaussian noise
    if noise_std > 0:
        noise = torch.randn_like(E_teacher) * noise_std
        E_teacher = E_teacher + noise

    # Apply random time shift (jitter)
    if time_jitter_max > 0:
        max_shift = int(time_jitter_max * T)
        if max_shift > 0:
            # Sample random shifts for each batch item
            shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=device)

            augmented_states = []
            for i in range(B):
                shift = shifts[i].item()
                if shift != 0:
                    # Circular shift along time dimension
                    E_i = torch.roll(E_teacher[i], shifts=int(shift), dims=0)
                    augmented_states.append(E_i)
                else:
                    augmented_states.append(E_teacher[i])

            E_teacher = torch.stack(augmented_states, dim=0)

    return E_teacher
