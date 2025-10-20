from transformers import AutoTokenizer, WhisperProcessor, WhisperModel
from whistle.tle.tle import TLEVAE, TLEVAEConfig, vae_loss
from whistle.tle.utils import get_teacher_states, augment_teacher_states
from whistle.tle.data import create_preprocessed_data_loader
from datasets import load_dataset, load_from_disk
import torch
import torch.nn.functional as F
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

        # Cache for validation teacher states to avoid repeated Whisper inference
        self.validation_cache = {}
        self.cache_enabled = True

    def get_cached_teacher_states(self, audio_arrays, batch_cache_keys):
        """Get teacher states with caching for validation performance."""
        if not self.cache_enabled:
            # Fallback to direct computation
            return get_teacher_states(audio_arrays, self.whisper, self.processor)

        # Check cache first
        cached_states = []
        uncached_indices = []
        uncached_audio = []

        for i, cache_key in enumerate(batch_cache_keys):
            if cache_key in self.validation_cache:
                cached_states.append(self.validation_cache[cache_key])
            else:
                cached_states.append(None)  # Placeholder
                uncached_indices.append(i)
                uncached_audio.append(audio_arrays[i])

        # Compute missing teacher states
        if uncached_audio:
            new_states = get_teacher_states(
                uncached_audio, self.whisper, self.processor
            )
            # Cache them
            for idx, state in zip(uncached_indices, new_states):
                cache_key = batch_cache_keys[idx]
                self.validation_cache[cache_key] = state

            # Log cache misses (only occasionally to avoid spam)
            if len(self.validation_cache) % 100 == 0:
                print(
                    f"Validation cache: {len(self.validation_cache)} total entries, "
                    f"computed {len(uncached_audio)} new states this batch"
                )

        # Reconstruct batch in correct order
        final_states = []
        for i, state in enumerate(cached_states):
            if state is not None:
                final_states.append(state)
            else:
                # Find the computed state for this index
                uncached_pos = uncached_indices.index(i)
                final_states.append(new_states[uncached_pos])

        return torch.stack(final_states)

    def clear_validation_cache(self):
        """Clear the validation cache to free memory."""
        cache_size = len(self.validation_cache)
        self.validation_cache.clear()
        print(f"Cleared validation cache ({cache_size} entries)")

    def enable_cache(self, enabled: bool = True):
        """Enable or disable validation caching."""
        self.cache_enabled = enabled
        if enabled:
            print("Validation teacher state caching enabled")
        else:
            print("Validation teacher state caching disabled")

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

                # GRADIENT CLIPPING: Prevent exploding gradients
                max_grad_norm = 1.0
                torch.nn.utils.clip_grad_norm_(self.tle.parameters(), max_grad_norm)
                self.log("grad_norm_clipped", min(total_norm, max_grad_norm))

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

    def get_learning_rate_multiplier(self) -> float:
        """Compute learning rate multiplier based on beta annealing progress."""
        if not self.training:
            return 1.0  # no multiplier for validation

        current_step = self.global_step
        if current_step < self.cfg.beta_warmup_steps:
            return 1.0  # full learning rate during warmup
        else:
            # Reduce learning rate by 50% after beta warmup to prevent instability
            return 0.5

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
        E_tilde, mu, logvar = self.tle(
            input_ids, attention_mask, target_T=T, lang_ids=lang_ids
        )

        # Get current beta for annealing
        beta = self.get_current_beta()

        # Get learning rate multiplier for stability after beta warmup
        lr_multiplier = self.get_learning_rate_multiplier()

        # Compute losses on all timesteps (no masking needed since all 1500 are informative)
        E_teacher_flat = E_teacher.flatten(0, 1)  # (B*T, D)
        E_tilde_flat = E_tilde.flatten(0, 1)  # (B*T, D)

        # Compute MSE loss on all embeddings
        mse_loss = F.mse_loss(E_tilde_flat, E_teacher_flat)

        # Compute KL loss (per sequence, no masking needed)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z_dim)
        if self.cfg.free_bits_threshold > 0:
            kl_per_dim = torch.clamp(kl_per_dim - self.cfg.free_bits_threshold, min=0)
        kl_loss = kl_per_dim.mean()

        # Total loss
        loss = mse_loss + beta * kl_loss

        # Apply learning rate multiplier to loss for effective LR decay
        loss = loss * lr_multiplier

        # Compute cosine similarity metric (use all embeddings)
        cos_sim = torch.nn.functional.cosine_similarity(
            E_tilde_flat, E_teacher_flat, dim=-1
        ).mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_beta", beta, prog_bar=True)
        self.log("train_mse_loss", mse_loss)
        self.log("train_kl_loss", kl_loss)
        self.log("train_cos_sim", cos_sim)
        self.log("train_lr_multiplier", lr_multiplier)

        # KL explosion detection - log warning if KL loss gets too high
        if kl_loss.item() > 10.0:  # KL loss > 10 nats per dimension is concerning
            self.log("kl_explosion_warning", 1.0)
            print(f"WARNING: High KL loss detected: {kl_loss.item():.2f}")
        else:
            self.log("kl_explosion_warning", 0.0)
        # Log learning rate
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("learning_rate", lr)
        return loss

    def validation_step(self, batch, batch_idx):
        # Create cache keys for this batch (use audio content hash or index-based)
        # For simplicity, use batch_idx and sample indices within batch
        batch_cache_keys = [
            f"val_{batch_idx}_{i}" for i in range(len(batch["audio_arrays"]))
        ]

        # Get teacher states (with caching)
        E_teacher = self.get_cached_teacher_states(
            batch["audio_arrays"], batch_cache_keys
        )
        _, T, _ = E_teacher.shape

        # Text tokens and language IDs
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lang_ids = batch["lang_ids"]

        # TLE forward
        E_tilde, mu, logvar = self.tle(
            input_ids, attention_mask, target_T=T, lang_ids=lang_ids
        )

        # Use final beta for validation
        beta = self.cfg.beta_end

        # Compute losses on all timesteps (no masking needed since all 1500 are informative)
        E_teacher_flat = E_teacher.flatten(0, 1)  # (B*T, D)
        E_tilde_flat = E_tilde.flatten(0, 1)  # (B*T, D)

        # Compute MSE loss on all embeddings
        mse_loss = F.mse_loss(E_tilde_flat, E_teacher_flat)

        # Compute KL loss (per sequence, no masking needed)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z_dim)
        if self.cfg.free_bits_threshold > 0:
            kl_per_dim = torch.clamp(kl_per_dim - self.cfg.free_bits_threshold, min=0)
        kl_loss = kl_per_dim.mean()

        # Total loss
        loss = mse_loss + beta * kl_loss

        # Compute cosine similarity metric (use all embeddings)
        cos_sim = torch.nn.functional.cosine_similarity(
            E_tilde_flat, E_teacher_flat, dim=-1
        ).mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_mse_loss", mse_loss)
        self.log("val_kl_loss", kl_loss)
        self.log("val_cos_sim", cos_sim)
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
        val_samples: Optional[int] = None,
    ):
        super().__init__()
        self.dataset = dataset
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.train_split = train_split
        self.test_split = test_split
        self.val_samples = val_samples

    def train_dataloader(self):
        return create_preprocessed_data_loader(
            dataset=self.dataset,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split=self.train_split,
            batch_size=self.batch_size,
            max_text_length=256,
            num_workers=8,
        )

    def val_dataloader(self):
        dataset = self.dataset

        # Limit validation set to val_samples if specified
        if self.val_samples is not None:
            # Create a subset of the validation split
            import random

            val_split = dataset[self.test_split]
            total_samples = len(val_split)

            if total_samples > self.val_samples:
                # Randomly select val_samples indices
                selected_indices = random.sample(range(total_samples), self.val_samples)
                selected_indices.sort()  # Sort for better cache locality
                val_split = val_split.select(selected_indices)

                # Create a temporary dataset dict with the subset
                dataset = {self.test_split: val_split}
                logger.info(
                    f"Validation set limited to {self.val_samples} samples (from {total_samples})"
                )

        return create_preprocessed_data_loader(
            dataset=dataset,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split=self.test_split,
            batch_size=self.batch_size,
            max_text_length=256,
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
    val_samples: Optional[int] = None,
    use_wandb: bool = False,
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
        val_samples=val_samples,
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
    # Custom checkpoint callback that saves TLE model + optimizer state, but excludes Whisper weights
    # This keeps optimizer state for training resumption, but removes Whisper to save ~2GB
    class TLEOnlyCheckpoint(ModelCheckpoint):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            logger.info(
                "Using TLEOnlyCheckpoint: saving TLE model + optimizer state (excluding Whisper weights)"
            )

        def _save_checkpoint(self, trainer, filepath: str) -> None:
            # First save the full checkpoint normally using parent class
            super()._save_checkpoint(trainer, filepath)

            # Then load and filter it to remove Whisper weights
            ckpt = torch.load(filepath, map_location="cpu")

            if "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]

                # Filter to remove Whisper, processor, tokenizer
                filtered_state_dict = {}
                whisper_params_removed = 0

                for key, value in state_dict.items():
                    # Explicitly exclude Whisper, processor, tokenizer
                    if (
                        key.startswith("whisper.")
                        or key.startswith("processor.")
                        or key.startswith("tokenizer.")
                    ):
                        whisper_params_removed += 1
                        continue

                    # Keep everything else (TLE model + optimizer states)
                    filtered_state_dict[key] = value

                if whisper_params_removed > 0:
                    logger.info(
                        f"Removed {whisper_params_removed} Whisper/processor/tokenizer parameters from checkpoint"
                    )

                # Replace state_dict with filtered version
                ckpt["state_dict"] = filtered_state_dict

            # Save the modified checkpoint with optimizer states intact
            torch.save(ckpt, filepath)

            import os

            ckpt_size_gb = os.path.getsize(filepath) / 1e9
            logger.info(
                f"Saved checkpoint to {filepath} ({ckpt_size_gb:.2f}GB, "
                f"with optimizer state for training resumption, Whisper weights excluded)"
            )

    checkpoint_callback = TLEOnlyCheckpoint(
        dirpath="checkpoints",
        filename="tle-{epoch:02d}-{step:06d}.pt",
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
    elif precision == "bf16-mixed":
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
    val_samples: Optional[int] = None,
    use_wandb: bool = False,
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
        val_samples: Limit validation to N random samples (e.g., 1000 for faster validation)
        use_wandb: Enable Weights & Biases logging
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
        val_samples=val_samples,
        use_wandb=use_wandb,
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
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "32", "16", "bf16", "bf16-mixed"],
        help="Precision for training (auto=detect best available, 32=full precision, 16=float16, bf16=bfloat16, bf16-mixed=bf16 mixed precision)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for training (default: 1e-3)",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=None,
        help="Limit validation set to N random samples for faster validation (e.g., 1000). Default: None (use full validation set)",
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
        val_samples=args.val_samples,
        use_wandb=args.use_wandb,
        precision=args.precision,
        learning_rate=args.learning_rate,
    )
