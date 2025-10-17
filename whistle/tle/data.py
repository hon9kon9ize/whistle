import torch
import numpy as np
from typing import List, Dict, Any, Optional
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperProcessor
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict


class PreprocessedAudioTextDataset(Dataset):
    """
    Dataset for preprocessed audio-text data.

    Assumes data is already filtered, resampled, and formatted correctly.
    Expects columns: audio_array (numpy array), text (string)
    """

    def __init__(
        self,
        hf_dataset: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
        split: str = "train",
        augment: bool = False,
    ):
        """
        Args:
            hf_dataset: Loaded HuggingFace dataset with preprocessed data
            split: Dataset split to use ("train", "test")
            augment: Whether to apply telephony augmentation (8kHz resampling + μ-law)
        """
        self.dataset = hf_dataset[split] if split in hf_dataset else hf_dataset
        self.augment = augment
        self.original_length = len(self.dataset)
        print(f"Loaded {self.original_length} preprocessed samples")

    def __len__(self) -> int:
        if self.augment:
            return 2 * self.original_length
        return self.original_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample."""
        if self.augment and idx >= self.original_length:
            # Augmented sample
            actual_idx = idx - self.original_length
            augment_sample = True
        else:
            # Original sample
            actual_idx = idx
            augment_sample = False

        item = self.dataset[actual_idx]

        # Get preprocessed audio array - handle both formats
        if "audio_array" in item:
            # Preprocessed format: audio_array is already extracted
            audio_array = np.array(item["audio_array"], dtype=np.float32)
        elif (
            "audio" in item
            and isinstance(item["audio"], dict)
            and "array" in item["audio"]
        ):
            # Original format: audio is nested under "audio" key
            audio_array = np.array(item["audio"]["array"], dtype=np.float32)
        else:
            raise KeyError(
                f"Could not find audio data in item. Expected 'audio_array' or 'audio.array'. Available keys: {list(item.keys())}"
            )

        # Apply telephony augmentation if requested for this sample
        if augment_sample:
            # Resample to 8kHz (assuming input is 16kHz)
            audio_array = librosa.resample(audio_array, orig_sr=16000, target_sr=8000)
            # Normalize to [-1, 1] for mu-law
            audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
            # Apply μ-law compression
            audio_array = librosa.mu_compress(audio_array, mu=255)

        return {
            "audio_array": audio_array,
            "text": item.get(
                "text", item.get("sentence", "")
            ),  # Handle both preprocessed and Common Voice formats
            "audio_path": item.get("audio_path", f"sample_{actual_idx}"),
            "speaker_id": item.get("speaker_id", item.get("client_id", "unknown")),
            "language": item.get("language", item.get("locale", "unknown")),
        }


class AudioTextDataset(Dataset):
    """
    General dataset wrapper for any audio-text datasets.

    Supports flexible column names for text and audio data.
    """

    def __init__(
        self,
        hf_dataset: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
        split: str = "train",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
        text_column: str = "text",
        audio_column: str = "audio",
        audio_array_key: str = "array",
        audio_sampling_rate_key: str = "sampling_rate",
        augment: bool = False,
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
            audio_array_key: Key for audio array within audio column (e.g., "array")
            audio_sampling_rate_key: Key for sampling rate within audio column
            augment: Whether to apply telephony augmentation (8kHz resampling + μ-law)
        """
        self.dataset = hf_dataset[split] if split in hf_dataset else hf_dataset
        self.sampling_rate = sampling_rate
        self.max_audio_length = max_audio_length
        self.min_audio_length = min_audio_length
        self.text_column = text_column
        self.audio_column = audio_column
        self.audio_array_key = audio_array_key
        self.audio_sampling_rate_key = audio_sampling_rate_key
        self.augment = augment

        # Filter valid samples using dataset.map for parallel processing
        def check_valid(example):
            try:
                audio_info = example[self.audio_column]

                # Handle different audio formats
                if isinstance(audio_info, dict) and self.audio_array_key in audio_info:
                    # Standard format: {"array": [...], "sampling_rate": 16000}
                    audio_array = np.array(audio_info[self.audio_array_key])
                    source_sr = audio_info.get(
                        self.audio_sampling_rate_key, sampling_rate
                    )
                elif isinstance(audio_info, (list, np.ndarray)):
                    # Direct array format
                    audio_array = np.array(audio_info)
                    source_sr = sampling_rate  # Assume already at target rate
                else:
                    return {"valid": False}

                duration = len(audio_array) / source_sr

                return {
                    "valid": self.min_audio_length <= duration <= self.max_audio_length
                }

            except Exception:
                return {"valid": False}

        self.dataset = self.dataset.map(check_valid, num_proc=1)
        self.dataset = self.dataset.filter(lambda x: x["valid"], num_proc=1)
        self.dataset = self.dataset.remove_columns(["valid"])

        self.original_length = len(self.dataset)
        print(f"Loaded {self.original_length} valid samples")

    def __len__(self) -> int:
        if self.augment:
            return 2 * self.original_length
        return self.original_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample."""
        if self.augment and idx >= self.original_length:
            # Augmented sample
            actual_idx = idx - self.original_length
            augment_sample = True
        else:
            # Original sample
            actual_idx = idx
            augment_sample = False

        item = self.dataset[actual_idx]

        try:
            # Extract audio
            audio_info = item[self.audio_column]

            if isinstance(audio_info, dict) and self.audio_array_key in audio_info:
                audio_array = np.array(
                    audio_info[self.audio_array_key], dtype=np.float32
                )
                source_sr = audio_info.get(
                    self.audio_sampling_rate_key, self.sampling_rate
                )
            elif isinstance(audio_info, (list, np.ndarray)):
                audio_array = np.array(audio_info, dtype=np.float32)
                source_sr = self.sampling_rate
            else:
                raise ValueError(f"Unsupported audio format in {self.audio_column}")

            # Convert to mono if multi-channel
            if len(audio_array.shape) > 1:
                audio_array = np.mean(audio_array, axis=0)

            # Resample if needed
            if source_sr != self.sampling_rate:
                audio_array = librosa.resample(
                    audio_array, orig_sr=source_sr, target_sr=self.sampling_rate
                )

            # Apply telephony augmentation if requested for this sample
            if augment_sample:
                # Resample to 8kHz
                audio_array = librosa.resample(
                    audio_array, orig_sr=self.sampling_rate, target_sr=8000
                )
                # Normalize to [-1, 1] for mu-law
                audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
                # Apply μ-law compression
                audio_array = librosa.mu_compress(audio_array, mu=255)

            # Truncate if too long (shouldn't happen due to filtering, but safety check)
            max_samples = int(self.max_audio_length * self.sampling_rate)
            if len(audio_array) > max_samples:
                audio_array = audio_array[:max_samples]

            return {
                "audio_array": audio_array,
                "text": item[self.text_column],
                "audio_path": item.get(
                    "path", item.get("audio_path", f"sample_{actual_idx}")
                ),
                "speaker_id": item.get("speaker_id", item.get("client_id", "unknown")),
                "language": item.get("language", item.get("locale", "unknown")),
            }
        except Exception as e:
            # If audio decoding fails, return a dummy sample or skip
            # For now, raise to avoid silent failures
            raise RuntimeError(f"Failed to load audio for sample {actual_idx}: {e}")


class CommonVoiceDataset(Dataset):
    """
    Dataset wrapper for Mozilla Common Voice datasets from HuggingFace.

    Handles the specific format of Common Voice data with audio arrays and text.
    """

    def __init__(
        self,
        hf_dataset: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
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


class MultilingualAudioTextDataset(Dataset):
    """
    Dataset that combines multiple audio-text datasets for multilingual training.
    Supports flexible column names for different dataset formats.
    """

    def __init__(
        self,
        datasets: Dict[
            str,
            HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
        ],
        split: str = "train",
        sampling_rate: int = 16000,
        max_audio_length: float = 30.0,
        min_audio_length: float = 0.5,
        text_column: str = "sentence",
        audio_column: str = "audio",
        audio_array_key: str = "array",
        audio_sampling_rate_key: str = "sampling_rate",
        language_weights: Optional[Dict[str, float]] = None,
        augment: bool = False,
    ):
        """
        Args:
            datasets: Dict mapping dataset keys to loaded HF datasets
            split: Dataset split to use
            sampling_rate: Target sampling rate
            max_audio_length: Maximum audio length in seconds
            min_audio_length: Minimum audio length in seconds
            text_column: Name of text column in dataset
            audio_column: Name of audio column in dataset
            audio_array_key: Key for audio array within audio column
            audio_sampling_rate_key: Key for sampling rate within audio column
            language_weights: Optional weights for dataset sampling
            augment: Whether to apply telephony augmentation
        """
        self.datasets = {}
        self.cumulative_sizes = []
        self.dataset_keys = []
        self.language_weights = language_weights or {}

        total_samples = 0
        for dataset_key, hf_dataset in datasets.items():
            dataset = AudioTextDataset(
                hf_dataset,
                split=split,
                sampling_rate=sampling_rate,
                max_audio_length=max_audio_length,
                min_audio_length=min_audio_length,
                text_column=text_column,
                audio_column=audio_column,
                audio_array_key=audio_array_key,
                audio_sampling_rate_key=audio_sampling_rate_key,
                augment=augment,
            )

            if len(dataset) > 0:
                self.datasets[dataset_key] = dataset
                self.dataset_keys.append(dataset_key)
                total_samples += len(dataset)
                self.cumulative_sizes.append(total_samples)

        if not self.datasets:
            raise ValueError("No valid datasets found")

        print(
            f"Combined dataset with {total_samples} samples across {len(self.datasets)} datasets"
        )

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get sample from appropriate dataset."""
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes):
            if idx < cum_size:
                dataset_idx = i
                break

        dataset_key = self.dataset_keys[dataset_idx]
        dataset = self.datasets[dataset_key]

        # Adjust index for this dataset
        dataset_start = self.cumulative_sizes[dataset_idx - 1] if dataset_idx > 0 else 0
        sample_idx = idx - dataset_start

        sample = dataset[sample_idx]
        sample["dataset"] = dataset_key
        return sample


class MultilingualCommonVoiceDataset(Dataset):
    """
    Dataset that combines multiple Common Voice language variants for multilingual training.
    """

    def __init__(
        self,
        datasets: Dict[
            str,
            HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
        ],
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


def create_preprocessed_data_loader(
    dataset: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
    processor: WhisperProcessor,
    tokenizer: Any,
    split: str = "train",
    batch_size: int = 8,
    max_text_length: int = 256,
    augment: bool = False,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for preprocessed audio-text training.

    Args:
        dataset: Preprocessed HF dataset with train/test splits
        processor: Whisper processor for audio features
        tokenizer: Text tokenizer
        split: Dataset split ("train", "test")
        batch_size: Training batch size
        max_text_length: Maximum text sequence length
        augment: Whether to apply telephony augmentation

    Returns:
        Configured DataLoader for TLE training
    """
    # Create preprocessed dataset
    preprocessed_dataset = PreprocessedAudioTextDataset(
        hf_dataset=dataset,
        split=split,
        augment=augment,
    )

    # Create collator
    collator = TLECollator(
        processor=processor, tokenizer=tokenizer, max_text_length=max_text_length
    )

    # Create data loader
    return create_data_loader(
        dataset=preprocessed_dataset,
        collator=collator,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle training data, not test
        num_workers=0,
        pin_memory=True,
    )


def create_data_loader(
    dataset: (
        AudioTextDataset
        | MultilingualAudioTextDataset
        | CommonVoiceDataset
        | MultilingualCommonVoiceDataset
        | PreprocessedAudioTextDataset
    ),
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


def create_tle_data_loader(
    datasets: Dict[
        str, HFDataset | DatasetDict | IterableDataset | IterableDatasetDict
    ],
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
    text_column: str = "sentence",
    audio_column: str = "audio",
    audio_array_key: str = "array",
    audio_sampling_rate_key: str = "sampling_rate",
    augment: bool = False,
) -> DataLoader:
    """
    Convenience function to create a DataLoader for multilingual audio-text training.

    Supports flexible column names for different dataset formats.

    Args:
        datasets: Dict mapping language/dataset keys to loaded HF datasets
        processor: Whisper processor for audio features
        tokenizer: Text tokenizer
        split: Dataset split ("train", "validation", "test")
        batch_size: Training batch size
        sampling_rate: Audio sampling rate
        max_audio_length: Maximum audio duration in seconds
        min_audio_length: Minimum audio duration in seconds
        max_text_length: Maximum text sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of DataLoader workers
        text_column: Name of text column in dataset (default: "sentence" for Common Voice)
        audio_column: Name of audio column in dataset (default: "audio")
        audio_array_key: Key for audio array within audio column (default: "array")
        audio_sampling_rate_key: Key for sampling rate within audio column (default: "sampling_rate")
        augment: Whether to apply telephony augmentation

    Returns:
        Configured DataLoader for multilingual TLE training
    """
    # Create multilingual dataset
    multilingual_dataset = MultilingualAudioTextDataset(
        datasets=datasets,
        split=split,
        sampling_rate=sampling_rate,
        max_audio_length=max_audio_length,
        min_audio_length=min_audio_length,
        text_column=text_column,
        audio_column=audio_column,
        audio_array_key=audio_array_key,
        audio_sampling_rate_key=audio_sampling_rate_key,
        augment=augment,
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
