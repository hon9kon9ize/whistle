from transformers import AutoTokenizer, WhisperProcessor, WhisperModel
from whistle.tle.tle import TLEVAE, TLEVAEConfig, vae_loss
from whistle.tle.utils import get_teacher_states
from whistle.tle.data import create_tle_data_loader, create_preprocessed_data_loader
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
    # Whisper tokenizer has incorrect vocab_size reporting, use actual size
    # From testing: max token ID is 51865, so vocab_size should be 51866
    actual_vocab_size = 51866
    return TLEVAEConfig(
        vocab_size=actual_vocab_size, whisper_hidden=whisper_model.config.d_model
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
        dataset,
        processor=None,
        tokenizer=None,
        batch_size: int = 8,
        train_split: str = "train",
        test_split: str = "test",
        augment: bool = False,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_split = train_split
        self.test_split = test_split
        self.augment = augment

    def train_dataloader(self):
        return create_preprocessed_data_loader(
            dataset=self.dataset,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split=self.train_split,
            batch_size=self.batch_size,
            max_text_length=256,
            augment=self.augment,
        )

    def val_dataloader(self):
        return create_preprocessed_data_loader(
            dataset=self.dataset,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split=self.test_split,
            batch_size=self.batch_size,
            max_text_length=256,
            augment=self.augment,
        )


def train_with_dataset(
    dataset,
    batch_size: int = 8,
    max_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_every: int = 1000,
    device: str = "cuda",
    train_split: str = "train",
    test_split: str = "test",
    use_wandb: bool = False,
    augment: bool = False,
):
    """Train TLE with provided dataset that has train/test splits."""
    # Load models
    whisper_model, processor, tokenizer = load_whisper_models(device)
    cfg = create_tle_config(tokenizer, whisper_model)

    # Data module
    data_module = TLEDataModule(
        dataset=dataset,
        processor=processor,
        tokenizer=tokenizer,
        batch_size=batch_size,
        train_split=train_split,
        test_split=test_split,
        augment=augment,
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


def train_with_preprocessed_dataset(
    dataset_path: str,
    batch_size: int = 8,
    max_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_every: int = 1000,
    device: str = "cuda",
    train_split: str = "train",
    test_split: str = "test",
    subset: Optional[str] = None,
    use_wandb: bool = False,
    augment: bool = False,
):
    """
    Train TLE with a preprocessed dataset that has train/test splits.

    Args:
        dataset_path: Path to the preprocessed dataset (HuggingFace dataset path or local path)
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        max_steps: Maximum number of training steps
        save_every: Save checkpoint every N steps
        device: Device to use for training
        train_split: Name of the training split
        test_split: Name of the test/validation split
        subset: Dataset subset/configuration name
        use_wandb: Enable Weights & Biases logging
        augment: Apply telephony augmentation
    """
    print(f"Loading preprocessed dataset from: {dataset_path}")

    # Load the preprocessed dataset
    try:
        if subset is not None:
            dataset = load_dataset(dataset_path, subset)
        else:
            dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset from {dataset_path}: {e}")
        raise

    # Print dataset info
    try:
        print(f"Dataset loaded successfully")
        if subset is not None:
            print(f"Using subset: {subset}")
        print(f"Using train_split='{train_split}', test_split='{test_split}'")
    except:
        print("Could not determine dataset info")

    # Train with the loaded dataset
    return train_with_dataset(
        dataset=dataset,
        batch_size=batch_size,
        max_epochs=max_epochs,
        max_steps=max_steps,
        save_every=save_every,
        device=device,
        train_split=train_split,
        test_split=test_split,
        use_wandb=use_wandb,
        augment=augment,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TLE model with preprocessed audio-text datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the preprocessed dataset (HuggingFace dataset name or local path)",
    )
    parser.add_argument(
        "--train-split",
        type=str,
        default="train",
        help="Name of the training split (default: train)",
    )
    parser.add_argument(
        "--test-split",
        type=str,
        default="test",
        help="Name of the test/validation split (default: test)",
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
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Apply telephony augmentation (8kHz resampling + μ-law) to training and validation datasets",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/configuration name (for datasets with multiple subsets)",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Train
    trained_model = train_with_preprocessed_dataset(
        dataset_path=args.dataset,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        max_steps=args.max_steps,
        save_every=args.save_every,
        device=device,
        train_split=args.train_split,
        test_split=args.test_split,
        subset=args.subset,
        use_wandb=args.use_wandb,
        augment=args.augment,
    )
