from transformers import AutoTokenizer, WhisperProcessor, WhisperModel
from whistle.tle.tle import TLEVAE, TLEVAEConfig, vae_loss
from whistle.tle.utils import get_teacher_states
from whistle.tle.data import create_tle_data_loader
from datasets import load_dataset
import torch
from typing import List, Optional, Tuple
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
import argparse


def load_whisper_models(
    device: str = "cuda",
) -> Tuple[WhisperModel, WhisperProcessor, AutoTokenizer]:
    """Load Whisper model, processor, and tokenizer."""
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
    whisper_model = whisper_model.to(device)
    whisper_model.eval()
    for param in whisper_model.parameters():
        param.requires_grad = False

    processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    tokenizer = AutoTokenizer.from_pretrained("openai/whisper-large-v3")

    return whisper_model, processor, tokenizer


def create_tle_config(
    tokenizer: AutoTokenizer, whisper_model: WhisperModel
) -> TLEVAEConfig:
    """Create TLEVAE configuration."""
    return TLEVAEConfig(
        vocab_size=tokenizer.vocab_size, whisper_hidden=whisper_model.config.d_model
    )


class TLELightningModule(pl.LightningModule):
    def __init__(
        self, cfg: TLEVAEConfig, whisper_model, processor, tokenizer, beta: float = 0.1
    ):
        super().__init__()
        self.cfg = cfg
        self.tle = TLEVAE(cfg)
        self.whisper = whisper_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.beta = beta

    def forward(self, input_ids, attention_mask=None, target_T=None):
        return self.tle(input_ids, attention_mask, target_T)

    def training_step(self, batch, batch_idx):
        # Get teacher states
        E_teacher = get_teacher_states(
            batch["audio_arrays"], self.whisper, self.processor
        )
        _, T, _ = E_teacher.shape

        # Text tokens
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # TLE forward
        E_tilde, mu, logvar = self.tle(input_ids, attention_mask, target_T=T)

        # Loss
        loss = vae_loss(E_tilde, E_teacher, mu, logvar, self.beta)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get teacher states
        E_teacher = get_teacher_states(
            batch["audio_arrays"], self.whisper, self.processor
        )
        _, T, _ = E_teacher.shape

        # Text tokens
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        # TLE forward
        E_tilde, mu, logvar = self.tle(input_ids, attention_mask, target_T=T)

        # Loss
        loss = vae_loss(E_tilde, E_teacher, mu, logvar, self.beta)

        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.tle.parameters(), lr=2e-4, weight_decay=0.01)
        return optimizer


class TLEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_datasets,
        val_datasets=None,
        processor=None,
        tokenizer=None,
        batch_size: int = 8,
        text_column: str = "sentence",
        audio_column: str = "audio",
        audio_array_key: str = "array",
        audio_sampling_rate_key: str = "sampling_rate",
    ):
        super().__init__()
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.text_column = text_column
        self.audio_column = audio_column
        self.audio_array_key = audio_array_key
        self.audio_sampling_rate_key = audio_sampling_rate_key

    def train_dataloader(self):
        return create_tle_data_loader(
            datasets=self.train_datasets,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split="train",
            batch_size=self.batch_size,
            max_audio_length=30.0,
            min_audio_length=0.5,
            max_text_length=256,
            text_column=self.text_column,
            audio_column=self.audio_column,
            audio_array_key=self.audio_array_key,
            audio_sampling_rate_key=self.audio_sampling_rate_key,
        )

    def val_dataloader(self):
        if self.val_datasets is None:
            return None
        return create_tle_data_loader(
            datasets=self.val_datasets,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split="test",
            batch_size=self.batch_size,
            max_audio_length=30.0,
            min_audio_length=0.5,
            max_text_length=256,
            text_column=self.text_column,
            audio_column=self.audio_column,
            audio_array_key=self.audio_array_key,
            audio_sampling_rate_key=self.audio_sampling_rate_key,
        )


def train_with_dataset(
    train_datasets,
    val_datasets=None,
    batch_size: int = 8,
    max_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_every: int = 1000,
    device: str = "cuda",
    text_column: str = "sentence",
    audio_column: str = "audio",
    audio_array_key: str = "array",
    audio_sampling_rate_key: str = "sampling_rate",
    use_wandb: bool = False,
):
    """Train TLE with provided datasets dictionary."""
    # Load models
    whisper_model, processor, tokenizer = load_whisper_models(device)
    cfg = create_tle_config(tokenizer, whisper_model)

    # Data module
    data_module = TLEDataModule(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        text_column=text_column,
        audio_column=audio_column,
        audio_array_key=audio_array_key,
        audio_sampling_rate_key=audio_sampling_rate_key,
    )

    # Model
    model = TLELightningModule(cfg, whisper_model, processor, tokenizer)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="tle-{epoch:02d}-{step:06d}",
        save_top_k=-1,
        every_n_train_steps=save_every,
    )

    # Trainer
    trainer_kwargs = {
        "max_epochs": max_epochs,
        "callbacks": [checkpoint_callback],
        "accelerator": device if device != "cpu" else "cpu",
        "devices": 1,
        "log_every_n_steps": 10,
    }
    if use_wandb:
        trainer_kwargs["logger"] = WandbLogger(project="tle-training")
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps

    # Use bfloat16 if available
    if device == "cuda" and torch.cuda.is_bf16_supported():
        trainer_kwargs["precision"] = "bf16-mixed"

    trainer = Trainer(**trainer_kwargs)

    # Train
    trainer.fit(model, data_module)

    return model


def train_with_commonvoice(
    language_codes: Optional[List[str]] = None,
    batch_size: int = 8,
    max_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_every: int = 1000,
    device: str = "cuda",
    use_wandb: bool = False,
):
    if language_codes is None:
        language_codes = ["en", "zh-CN", "zh-HK", "yue"]

    # Load Common Voice datasets
    train_datasets = {}
    val_datasets = {}
    for lang_code in language_codes:
        try:
            print(f"Loading {lang_code}...")
            train_datasets[lang_code] = load_dataset(
                "mozilla-foundation/common_voice_16_1", lang_code, split="train"
            )
            val_datasets[lang_code] = load_dataset(
                "mozilla-foundation/common_voice_16_1", lang_code, split="test"
            )
        except (ValueError, ConnectionError, RuntimeError) as e:
            print(f"Failed to load {lang_code}: {e}")
            continue

    if not train_datasets:
        raise ValueError("No datasets could be loaded")

    print("Train dataset sizes:")
    for key in train_datasets.keys():
        print(key, ":", len(train_datasets[key]))

    print("Val dataset sizes:")
    for key in val_datasets.keys():
        print(key, ":", len(val_datasets[key]))

    # Train with the loaded datasets
    return train_with_dataset(
        train_datasets=train_datasets,
        val_datasets=val_datasets,
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        save_every=save_every,
        device=device,
        text_column="sentence",  # Common Voice specific
        audio_column="audio",
        audio_array_key="array",
        audio_sampling_rate_key="sampling_rate",
        use_wandb=use_wandb,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TLE model with Common Voice datasets"
    )
    parser.add_argument(
        "--language-codes",
        type=str,
        default="en,zh-CN,zh-HK",
        help="Comma-separated list of language codes (default: en,zh-CN,zh-HK)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for training (default: 4)"
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=1,
        help="Maximum number of epochs (default: 1)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum number of training steps (default: None)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training (default: auto-detect cuda/cpu)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )

    args = parser.parse_args()

    # Parse language codes
    language_codes = args.language_codes.split(",") if args.language_codes else None

    # Auto-detect device if not specified
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    trained_model = train_with_commonvoice(
        language_codes=language_codes,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        save_every=args.save_every,
        device=device,
        use_wandb=args.use_wandb,
    )
