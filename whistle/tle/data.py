import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperProcessor
from datasets import Dataset as HFDataset


class CommonVoiceDataset(Dataset):
    """
    Dataset wrapper for Mozilla Common Voice datasets from HuggingFace.

    Handles the specific format of Common Voice data with audio arrays and text.
    """

    def __init__(
        self,
        hf_dataset: HFDataset,
        split: str = "train",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
        text_column: str = "sentence",
        audio_column: str = "audio",
    ):
        """
        Args:
            hf_dataset: Loaded HuggingFace dataset
            split: Dataset split to use ("train", "validation", "test")
            sampling_rate: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            min_audio_length: Minimum audio length in seconds
            text_column: Name of text column in dataset
            audio_column: Name of audio column in dataset
        """
        self.dataset = hf_dataset[split] if split in hf_dataset else hf_dataset
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.text_column = text_column
        self.audio_column = audio_column

        # Filter valid samples
        self.valid_indices = []
        for i in range(len(self.dataset)):
            try:
                item = self.dataset[i]
                audio_info = item[self.audio_column]

                # Check if audio array is available
                if "array" not in audio_info:
                    continue

                audio_array = np.array(audio_info["array"])
                duration = len(audio_array) / audio_info.get(
                    "sampling_rate", sampling_rate
                )

                if self.min_audio_length <= duration <= self.max_audio_length:
                    self.valid_indices.append(i)

            except (KeyError, ValueError, TypeError):
                continue

        print(
            f"Loaded {len(self.valid_indices)} valid samples from {len(self.dataset)} total"
        )

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample."""
        data_idx = self.valid_indices[idx]
        item = self.dataset[data_idx]

        # Extract audio
        audio_info = item[self.audio_column]
        audio_array = np.array(audio_info["array"], dtype=np.float32)

        # Convert to mono if multi-channel
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)

        # Resample if needed
        source_sr = audio_info.get("sampling_rate", self.sampling_rate)
        if source_sr != self.sampling_rate:
            audio_array = librosa.resample(
                audio_array, orig_sr=source_sr, target_sr=self.sampling_rate
            )

        # Truncate if too long (shouldn't happen due to filtering, but safety check)
        max_samples = int(self.max_audio_length * self.sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        return {
            "audio_array": audio_array,
            "text": item[self.text_column],
            "audio_path": item.get("path", f"sample_{data_idx}"),
            "speaker_id": item.get("client_id", "unknown"),
            "language": item.get("locale", "unknown"),
        }


class MultilingualCommonVoiceDataset(Dataset):
    """
    Dataset that combines multiple Common Voice language variants for multilingual training.
    """

    def __init__(
        self,
        datasets: Dict[str, HFDataset],
        split: str = "train",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
        language_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            datasets: Dict mapping language codes to loaded HF datasets
            split: Dataset split to use
            sampling_rate: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            min_audio_length: Minimum audio length in seconds
            language_weights: Optional weights for language sampling
        """
        self.datasets = {}
        self.cumulative_sizes = []
        self.language_codes = []
        self.language_weights = language_weights or {}

        total_samples = 0
        for lang_code, hf_dataset in datasets.items():
            cv_dataset = CommonVoiceDataset(
                hf_dataset,
                split=split,
                sampling_rate=sampling_rate,
                max_audio_length=max_audio_length,
                min_audio_length=min_audio_length,
            )

            if len(cv_dataset) > 0:
                self.datasets[lang_code] = cv_dataset
                self.language_codes.append(lang_code)
                total_samples += len(cv_dataset)
                self.cumulative_sizes.append(total_samples)

        if not self.datasets:
            raise ValueError("No valid datasets found")

        print(
            f"Combined dataset with {total_samples} samples across {len(self.datasets)} languages"
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample from appropriate language dataset."""
        # Find which language dataset this index belongs to
        lang_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                lang_idx = i
                break

        lang_code = self.language_codes[lang_idx]
        dataset = self.datasets[lang_code]

        # Adjust index for this dataset
        dataset_start = self.cumulative_sizes[lang_idx - 1] if lang_idx > 0 else 0
        dataset_idx = idx - dataset_start

        sample = dataset[dataset_idx]
        sample["language"] = lang_code
        return sample


class TLECollator:
    """
    Collate function for batching audio-text pairs in TLE training.

    Handles:
    - Audio padding/truncation
    - Text tokenization and padding
    - Batch tensor creation
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        tokenizer: Any,
        max_text_length: int = 256,
        pad_to_multiple_of: Optional[int] = None,
    ):
        """
        Args:
            processor: Whisper processor for audio features
            tokenizer: Text tokenizer (e.g., AutoTokenizer)
            max_text_length: Maximum text sequence length
            pad_to_multiple_of: Pad sequences to multiple of this value
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of audio-text samples.

        Returns:
            Dict with:
            - audio_arrays: List of audio arrays (for teacher extraction)
            - texts: List of text strings
            - input_ids: Padded text token ids (B, L)
            - attention_mask: Attention mask for text (B, L)
        """
        audio_arrays = []
        texts = []

        for item in batch:
            audio_arrays.append(item["audio_array"])
            texts.append(item["text"])

        # Tokenize texts
        text_batch = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        return {
            "audio_arrays": audio_arrays,
            "texts": texts,
            "input_ids": text_batch["input_ids"],
            "attention_mask": text_batch["attention_mask"],
        }


def create_data_loader(
    dataset: CommonVoiceDataset | MultilingualCommonVoiceDataset,
    collator: TLECollator,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create a DataLoader for TLE training.

    Args:
        dataset: Dataset instance (CommonVoiceDataset or MultilingualCommonVoiceDataset)
        collator: TLECollator instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer

    Returns:
        Configured DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collator,
    )


def create_common_voice_data_loader(
    datasets: Dict[str, HFDataset],
    processor: WhisperProcessor,
    tokenizer: Any,
    split: str = "train",
    batch_size: int = 8,
    sampling_rate: int = 16000,
    max_audio_length: float = 30.0,
    min_audio_length: float = 0.5,
    max_text_length: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for Common Voice multilingual training.

    Args:
        datasets: Dict mapping language codes to loaded HF datasets
        processor: Whisper processor for audio
        tokenizer: Text tokenizer
        split: Dataset split ("train", "validation", "test")
        batch_size: Training batch size
        sampling_rate: Audio sampling rate
        max_audio_length: Maximum audio duration in seconds
        min_audio_length: Minimum audio duration in seconds
        max_text_length: Maximum text sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of DataLoader workers

    Returns:
        Configured DataLoader for multilingual TLE training
    """
    # Create multilingual dataset
    multilingual_dataset = MultilingualCommonVoiceDataset(
        datasets=datasets,
        split=split,
        sampling_rate=sampling_rate,
        max_audio_length=max_audio_length,
        min_audio_length=min_audio_length,
    )

    # Create collator
    collator = TLECollator(
        processor=processor, tokenizer=tokenizer, max_text_length=max_text_length
    )

    # Create data loader
    return create_data_loader(
        dataset=multilingual_dataset,
        collator=collator,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
