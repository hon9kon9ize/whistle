from transformers import AutoTokenizer, WhisperProcessor, WhisperModel
from whistle.tle.tle import TLEVAE, TLEVAEConfig, vae_loss
from whistle.tle.utils import get_teacher_states, augment_teacher_states
from whistle.tle.data import create_preprocessed_data_loader
from datasets import load_dataset, load_from_disk
import torch
from typing import Optional, Tuple
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
import argparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_whisper_models(
    device: str = "cuda",
) -> Tuple[WhisperModel, WhisperProcessor, AutoTokenizer]:
    """Load Whisper model, processor, and tokenizer."""
    if not torch.cuda.is_available() and device == "cuda":
        device = "cpu"

    # Use Whisper Large V3 for best Cantonese support
    whisper_model_name = "openai/whisper-large-v3"  # Best Cantonese support
    print(f"Loading Whisper model: {whisper_model_name}")

    whisper_model = WhisperModel.from_pretrained(whisper_model_name)
    whisper_model = whisper_model.to(device)
    whisper_model.eval()
    for param in whisper_model.parameters():
        param.requires_grad = False

    processor = WhisperProcessor.from_pretrained(whisper_model_name)
    tokenizer = AutoTokenizer.from_pretrained(whisper_model_name)

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
        self,
        cfg: TLEVAEConfig,
        whisper_model,
        processor,
        tokenizer,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.cfg = cfg
        self.tle = TLEVAE(cfg)
        self.whisper = whisper_model
        self.processor = processor
        self.tokenizer = tokenizer
        self.learning_rate = learning_rate

    def on_after_backward(self):
        """Compute and log gradient norm after backward pass."""
        if self.training:
            # Compute gradient norm across all parameters
            total_norm = 0.0
            param_count = 0
            for p in self.tle.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
            if param_count > 0:
                total_norm = total_norm ** (1.0 / 2)
                self.log("grad_norm", total_norm)

    def get_current_beta(self) -> float:
        """Compute current beta value based on training step for annealing."""
        if not self.training:
            return self.cfg.beta_end  # use final beta for validation

        current_step = self.global_step
        if current_step < self.cfg.beta_warmup_steps:
            # Linear annealing from beta_start to beta_end
            progress = current_step / self.cfg.beta_warmup_steps
            beta = self.cfg.beta_start + progress * (
                self.cfg.beta_end - self.cfg.beta_start
            )
        else:
            beta = self.cfg.beta_end

        return beta

    def forward(self, input_ids, attention_mask=None, target_T=None):
        return self.tle(input_ids, attention_mask, target_T)

    def training_step(self, batch, batch_idx):
        # Get teacher states
        E_teacher = get_teacher_states(
            batch["audio_arrays"], self.whisper, self.processor
        )
        _, T, _ = E_teacher.shape

        # Apply noise and time-jitter augmentation to teacher states
        E_teacher = augment_teacher_states(
            E_teacher,
            noise_std=self.cfg.teacher_noise_std,
            time_jitter_max=self.cfg.teacher_time_jitter_max,
        )

        # Text tokens and language IDs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lang_ids = batch["lang_ids"]

        # TLE forward
        E_tilde, mu, logvar, length_pred = self.tle(
            input_ids, attention_mask, target_T=T, lang_ids=lang_ids
        )

        # Get current beta for annealing
        beta = self.get_current_beta()

        # Loss with free-bits and auxiliary length loss
        loss, mse_loss, kl_loss, length_loss = vae_loss(
            E_tilde,
            E_teacher,
            mu,
            logvar,
            beta,
            self.cfg.free_bits_threshold,
            length_pred=length_pred,
            length_target=batch["lengths"],
            length_loss_weight=self.cfg.length_loss_weight,
        )

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_beta", beta, prog_bar=True)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_length_loss", length_loss)
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr)
        return loss

    def validation_step(self, batch, batch_idx):
        # Get teacher states
        E_teacher = get_teacher_states(
            batch["audio_arrays"], self.whisper, self.processor
        )
        _, T, _ = E_teacher.shape

        # Text tokens and language IDs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lang_ids = batch["lang_ids"]

        # TLE forward
        E_tilde, mu, logvar, length_pred = self.tle(
            input_ids, attention_mask, target_T=T, lang_ids=lang_ids
        )

        # Use final beta for validation
        beta = self.cfg.beta_end

        # Loss with free-bits and auxiliary length loss
        loss, mse_loss, kl_loss, length_loss = vae_loss(
            E_tilde,
            E_teacher,
            mu,
            logvar,
            beta,
            self.cfg.free_bits_threshold,
            length_pred=length_pred,
            length_target=batch["lengths"],
            length_loss_weight=self.cfg.length_loss_weight,
        )

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse_loss", mse_loss)
        self.log("val_kl_loss", kl_loss)
        self.log("val_length_loss", length_loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.tle.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5000, T_mult=2, eta_min=1e-5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


class TLEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset,
        processor=None,
        tokenizer=None,
        batch_size: int = 8,
        train_split: str = "train",
        test_split: str = "valid",
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
            num_workers=8,
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
            num_workers=8,
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
    precision: str = "auto",
    learning_rate: float = 1e-3,
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

    # Print training info
    train_dataloader = data_module.train_dataloader()
    steps_per_epoch = len(train_dataloader)
    total_expected_steps = (
        steps_per_epoch * max_epochs
        if max_steps is None
        else min(max_steps, steps_per_epoch * max_epochs)
    )

    print(f"Training setup:")
    print(f"  Dataset: {len(data_module.dataset)} samples")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Max epochs: {max_epochs}")
    if max_steps is not None:
        print(f"  Max steps: {max_steps} (limited training)")
    print(f"  Total expected steps: {total_expected_steps}")
    print(f"  Device: {device}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Whisper model: whisper-large-v3 (best Cantonese support)")
    print()

    # Model
    model = TLELightningModule(cfg, whisper_model, processor, tokenizer, learning_rate)

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

    # Use mixed precision if available
    if precision == "auto":
        if device == "cuda":
            if torch.cuda.is_bf16_supported():
                trainer_kwargs["precision"] = "bf16-mixed"
            elif torch.cuda.is_available():
                trainer_kwargs["precision"] = "16-mixed"
        # else use default (32-bit)
    elif precision == "bf16":
        trainer_kwargs["precision"] = "bf16-mixed"
    elif precision == "16":
        trainer_kwargs["precision"] = "16-mixed"
    # precision == "32" uses default full precision

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
    test_split: str = "valid",
    subset: Optional[str] = None,
    use_wandb: bool = False,
    augment: bool = False,
    precision: str = "auto",
    learning_rate: float = 1e-3,
):
    """
    Train TLE with a preprocessed dataset that has train/test splits.

    Args:
        dataset_path: Path to the preprocessed dataset (HuggingFace dataset name/URL or local directory path)
        batch_size: Batch size for training
        max_epochs: Maximum number of epochs
        max_steps: Maximum number of training steps
        save_every: Save checkpoint every N steps
        device: Device to use for training
        train_split: Name of the training split
        test_split: Name of the test/validation split
        subset: Dataset subset/configuration name (only used for HuggingFace datasets)
        use_wandb: Enable Weights & Biases logging
        augment: Apply random audio augmentation
        precision: Mixed precision mode (auto, 32, 16, bf16)
        learning_rate: Learning rate for training
    """
    print(f"Loading preprocessed dataset from: {dataset_path}")

    # Load the preprocessed dataset
    try:
        import os

        if os.path.isdir(dataset_path):
            # Load from local directory
            dataset = load_from_disk(dataset_path)
        else:
            # Load from HuggingFace Hub or URL
            if subset is not None:
                dataset = load_dataset(dataset_path, subset)
            else:
                dataset = load_dataset(dataset_path)
    except Exception as e:
        logger.error(f"Failed to load dataset from {dataset_path}: {e}")
        raise

    # Print dataset info
    try:
        logger.info(f"Dataset loaded successfully")
        if subset is not None:
            logger.info(f"Using subset: {subset}")
        logger.info(f"Using train_split='{train_split}', test_split='{test_split}'")
    except:
        logger.warning("Could not determine dataset info")

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
        precision=precision,
        learning_rate=learning_rate,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TLE model with preprocessed audio-text datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the preprocessed dataset (HuggingFace dataset name/URL or local directory path)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset/configuration name (only used for HuggingFace datasets)",
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
        default="valid",
        help="Name of the test/validation split (default: valid)",
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
        help="Apply random audio augmentation (telephony, noise, pitch_shift, time_stretch) to training and validation datasets",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "32", "16", "bf16"],
        help="Precision for training (auto=detect best available, 32=full precision, 16=float16, bf16=bfloat16)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
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
        precision=args.precision,
        learning_rate=args.learning_rate,
    )
