import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from torch.utils.data import Dataset, DataLoader
import librosa
from transformers import WhisperProcessor
from datasets import Dataset as HFDataset
from datasets import DatasetDict, IterableDataset, IterableDatasetDict
import logging
import random

# Set up logging
logger = logging.getLogger(__name__)


def safe_get_audio_info(
    item: Dict[str, Any], audio_column: str = "audio"
) -> Optional[Dict[str, Any]]:
    """
    Safely extract audio information from a dataset item with error handling.

    Args:
        item: Dataset item dictionary
        audio_column: Name of the audio column

    Returns:
        Audio info dict or None if extraction fails
    """
    try:
        audio_info = item.get(audio_column)
        if audio_info is None:
            logger.warning(f"Missing audio column '{audio_column}' in item")
            return None

        # Handle different audio formats
        if isinstance(audio_info, dict):
            if "array" not in audio_info:
                logger.warning(f"Missing 'array' key in audio info")
                return None
            return audio_info
        elif isinstance(audio_info, (list, np.ndarray)):
            # Handle raw audio arrays
            return {"array": audio_info, "sampling_rate": 16000}  # Assume 16kHz
        else:
            logger.warning(f"Unsupported audio format: {type(audio_info)}")
            return None
    except Exception as e:
        logger.error(f"Error extracting audio info: {e}")
        return None


def apply_audio_augmentation(
    audio_array: np.ndarray, augmentation_type: str = "telephony", **kwargs
) -> np.ndarray:
    """
    Apply various audio augmentations to the input array.

    Args:
        audio_array: Input audio array
        augmentation_type: Type of augmentation ("telephony", "noise", "pitch_shift", "time_stretch")
        **kwargs: Additional parameters for specific augmentations

    Returns:
        Augmented audio array
    """
    if augmentation_type == "telephony":
        # 8kHz resampling + μ-law compression
        target_sr = kwargs.get("target_sr", 8000)
        audio_array = librosa.resample(
            audio_array, orig_sr=kwargs.get("orig_sr", 16000), target_sr=target_sr
        )
        # Normalize to [-1, 1] for mu-law
        audio_array = audio_array / (np.max(np.abs(audio_array)) + 1e-8)
        # Apply μ-law compression
        audio_array = librosa.mu_compress(audio_array, mu=255)

    elif augmentation_type == "noise":
        # Add Gaussian noise
        noise_std = kwargs.get("noise_std", 0.01)
        noise = np.random.normal(0, noise_std, size=audio_array.shape)
        audio_array = audio_array + noise

    elif augmentation_type == "pitch_shift":
        # Pitch shifting
        n_steps = kwargs.get("n_steps", 2)
        audio_array = librosa.effects.pitch_shift(
            audio_array, sr=kwargs.get("sr", 16000), n_steps=n_steps
        )

    elif augmentation_type == "time_stretch":
        # Time stretching
        rate = kwargs.get("rate", 1.1)
        audio_array = librosa.effects.time_stretch(audio_array, rate=rate)

    return audio_array


class PreprocessedAudioTextDataset(Dataset):
    """
    Dataset for preprocessed audio-text data.

    Expects columns: text (string), audio (dict with array/sampling_rate), language (string)
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
            augment: Whether to apply random audio augmentation (telephony, noise, pitch_shift, time_stretch)
        """
        self.dataset = hf_dataset[split] if split in hf_dataset else hf_dataset
        self.augment = augment
        self.original_length = len(self.dataset)
        logger.info(f"Loaded {self.original_length} preprocessed samples")

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

        # Get audio array from standardized format
        audio_info = item["audio"]
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        source_sr = audio_info["sampling_rate"]

        # Apply random augmentation if requested for this sample
        if augment_sample:
            # Randomly choose augmentation type
            augmentation_types = ["telephony", "noise", "pitch_shift", "time_stretch"]
            chosen_augmentation = random.choice(augmentation_types)

            # Apply the chosen augmentation
            audio_array = apply_audio_augmentation(
                audio_array,
                augmentation_type=chosen_augmentation,
                orig_sr=source_sr,
                sr=source_sr,
            )

        return {
            "audio_array": audio_array,
            "text": item["text"],
            "language": item["language"],
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
            augment: Whether to apply random audio augmentation (telephony, noise, pitch_shift, time_stretch)
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

        # Skip filtering for faster data loading - validation happens in __getitem__
        try:
            self.original_length = len(self.dataset)
            logger.info(f"Loaded {self.original_length} samples (unfiltered)")
        except (TypeError, AttributeError):
            # IterableDataset doesn't have __len__
            self.original_length = None
            logger.info("Loaded streaming dataset (length unknown)")

    def __len__(self) -> int:
        if self.original_length is None:
            # For streaming datasets, return a large number
            # The actual length will be determined by data availability
            return 1000000  # Large number for streaming
        if self.augment:
            return 2 * self.original_length
        return self.original_length

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single data sample."""
        # For streaming datasets, disable augmentation since we don't know total length
        if self.original_length is None:
            augment_sample = False
            actual_idx = idx
        else:
            # Original augmentation logic for regular datasets
            if self.augment and idx >= self.original_length:
                actual_idx = idx - self.original_length
                augment_sample = True
            else:
                actual_idx = idx
                augment_sample = False

        try:
            item = self.dataset[actual_idx]
        except (IndexError, StopIteration):
            # End of streaming dataset or invalid index
            raise IndexError(f"Index {idx} out of range")

        # Extract audio from standardized format
        audio_info = safe_get_audio_info(item, self.audio_column)
        if audio_info is None:
            raise ValueError(f"Missing or invalid audio data in sample {idx}")

        # Load and process audio array
        audio_array = np.array(audio_info["array"], dtype=np.float32)
        source_sr = audio_info.get("sampling_rate", self.sampling_rate)

        # Convert to mono if multi-channel
        if len(audio_array.shape) > 1:
            audio_array = np.mean(audio_array, axis=0)

        # Resample if needed
        if source_sr != self.sampling_rate:
            audio_array = librosa.resample(
                audio_array, orig_sr=source_sr, target_sr=self.sampling_rate
            )

        # Runtime length validation
        duration = len(audio_array) / self.sampling_rate
        if not (self.min_audio_length <= duration <= self.max_audio_length):
            raise ValueError(
                f"Audio duration {duration:.2f}s out of range [{self.min_audio_length}, {self.max_audio_length}] for sample {idx}"
            )

        # Apply random augmentation if requested for this sample
        if augment_sample:
            # Randomly choose augmentation type
            augmentation_types = ["telephony", "noise", "pitch_shift", "time_stretch"]
            chosen_augmentation = random.choice(augmentation_types)

            # Apply the chosen augmentation
            audio_array = apply_audio_augmentation(
                audio_array,
                augmentation_type=chosen_augmentation,
                orig_sr=self.sampling_rate,
                sr=self.sampling_rate,
            )

        # Truncate if too long (shouldn't happen due to filtering, but safety check)
        max_samples = int(self.max_audio_length * self.sampling_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]

        return {
            "audio_array": audio_array,
            "text": item[self.text_column],
            "language": item.get("language", "en"),
        }


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

        logger.info(
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
            augment: Whether to apply random audio augmentation
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

        logger.info(
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

        logger.info(
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

    Expects standardized dataset format:
    - text: string
    - audio: dict with "array" and "sampling_rate" keys
    - language: language code string

    Handles:
    - Audio padding/truncation
    - Text tokenization and padding
    - Language ID mapping
    - Batch tensor creation
    """

    def __init__(
        self,
        processor: WhisperProcessor,
        tokenizer: Any,
        max_text_length: int = 256,
        pad_to_multiple_of: Optional[int] = None,
        language_mapping: Optional[Dict[str, int]] = None,
        enable_memory_efficient: bool = True,
    ):
        """
        Args:
            processor: Whisper processor for audio features
            tokenizer: Text tokenizer (e.g., AutoTokenizer)
            max_text_length: Maximum text sequence length
            pad_to_multiple_of: Pad sequences to multiple of this value
            language_mapping: Dict mapping language codes to IDs (e.g., {"en": 0, "zh": 1, "yue": 2})
            enable_memory_efficient: Whether to use memory-efficient batching
        """
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.enable_memory_efficient = enable_memory_efficient
        # Default language mapping for en/zh/yue
        self.language_mapping = language_mapping or {
            "en": 0,  # English
            "zh": 1,  # Mandarin
            "yue": 2,  # Cantonese
        }

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of audio-text samples.

        Expected input format for each sample:
        - audio_array: numpy array of audio samples
        - text: string text
        - language: language code (e.g., "en", "zh", "yue")

        Returns:
            Dict with:
            - audio_arrays: List of audio arrays (for teacher extraction)
            - texts: List of text strings
            - input_ids: Padded text token ids (B, L)
            - attention_mask: Attention mask for text (B, L)
            - lang_ids: Language IDs tensor (B,)
            - lengths: Target sequence lengths tensor (B,) - number of Whisper encoder frames
        """
        audio_arrays = []
        texts = []
        languages = []
        lengths = []

        for item in batch:
            audio_arrays.append(item["audio_array"])
            texts.append(item["text"])
            # Extract language and map to ID
            lang_code = item.get("language", "en")  # Default to English
            lang_id = self.language_mapping.get(lang_code, 0)  # Default to 0 if unknown
            languages.append(lang_id)

            # Compute target length: approximate Whisper encoder frames
            # Whisper v3 processes at ~50Hz, so length ≈ duration * 50
            # We use the actual audio length to compute this
            audio_len = len(item["audio_array"])
            # Assuming 16kHz sampling rate, compute duration and then frames
            # Whisper encoder produces ~50 frames per second
            duration_seconds = audio_len / 16000.0
            target_length = max(1, int(duration_seconds * 50))  # At least 1 frame
            lengths.append(target_length)

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
            "lang_ids": torch.tensor(languages, dtype=torch.long),
            "lengths": torch.tensor(lengths, dtype=torch.long),
        }


def create_preprocessed_data_loader(
    dataset: HFDataset | DatasetDict | IterableDataset | IterableDatasetDict,
    processor: WhisperProcessor,
    tokenizer: Any,
    split: str = "train",
    batch_size: int = 8,
    max_text_length: int = 256,
    augment: bool = False,
    num_workers: int = 4,  # Enable multiprocessing by default
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
        augment: Whether to apply random audio augmentation
        num_workers: Number of worker processes for data loading

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

    # Create data loader with optimized settings
    return DataLoader(
        preprocessed_dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),  # Shuffle training data, not test
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collator,
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=2 if num_workers > 0 else None,  # Prefetch 2 batches per worker
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
        augment: Whether to apply random audio augmentation

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
