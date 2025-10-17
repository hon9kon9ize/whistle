from transformers import AutoTokenizer, WhisperProcessor, WhisperModel
from whistle.tle.tle import TLEVAE, TLEVAEConfig, vae_loss, clip_gradients
from whistle.tle.utils import get_teacher_states
from whistle.tle.data import create_common_voice_data_loader
from datasets import load_dataset
import torch
from typing import List, Optional
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# Setup device and models
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Whisper teacher model
whisper_model = WhisperModel.from_pretrained("openai/whisper-large-v3")
whisper_model = whisper_model.to(device)
whisper_model.eval()
for param in whisper_model.parameters():
    param.requires_grad = False

processor = WhisperProcessor.from_pretrained("openai/whisper-large-v3")

# Use a tokenizer for your text (you can use Whisper's tokenizer or any BPE/WordPiece)
tok = AutoTokenizer.from_pretrained("openai/whisper-large-v3")

cfg = TLEVAEConfig(
    vocab_size=tok.vocab_size, whisper_hidden=whisper_model.config.d_model
)  # d_model=1280 for large-v3


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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.tle.parameters(), lr=2e-4, weight_decay=0.01)
        return optimizer


class CommonVoiceDataModule(pl.LightningDataModule):
    def __init__(
        self, language_codes: List[str], processor, tokenizer, batch_size: int = 8
    ):
        super().__init__()
        self.language_codes = language_codes
        self.processor = processor
        self.tokenizer = tokenizer
        self.batch_size = batch_size

    def setup(self, stage: str):
        # Load datasets
        datasets = {}
        for lang_code in self.language_codes:
            try:
                print(f"Loading {lang_code}...")
                datasets[lang_code] = load_dataset(
                    "mozilla-foundation/common_voice_17_0", lang_code, split="train"
                )
            except (ValueError, ConnectionError, RuntimeError) as e:
                print(f"Failed to load {lang_code}: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets could be loaded")

        self.train_dataset = datasets  # Store for dataloader

    def train_dataloader(self):
        return create_common_voice_data_loader(
            datasets=self.train_dataset,
            processor=self.processor,
            tokenizer=self.tokenizer,
            split="train",
            batch_size=self.batch_size,
            max_audio_length=30.0,
            min_audio_length=0.5,
            max_text_length=256,
        )


def train_with_lightning(
    language_codes: Optional[List[str]] = None,
    batch_size: int = 8,
    max_epochs: int = 1,
    max_steps: Optional[int] = None,
    save_every: int = 1000,
):
    if language_codes is None:
        language_codes = ["en", "zh-CN", "zh-HK"]

    # Data module
    data_module = CommonVoiceDataModule(language_codes, processor, tok, batch_size)

    # Model
    model = TLELightningModule(cfg, whisper_model, processor, tok)

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
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "log_every_n_steps": 10,
    }
    if max_steps is not None:
        trainer_kwargs["max_steps"] = max_steps
    trainer = Trainer(**trainer_kwargs)

    # Train
    trainer.fit(model, data_module)

    return model


if __name__ == "__main__":
    # Example usage
    trained_model = train_with_lightning(
        language_codes=["en", "zh-CN", "zh-HK"],
        batch_size=4,
        max_epochs=1,
        max_steps=100,  # Quick test run
    )
