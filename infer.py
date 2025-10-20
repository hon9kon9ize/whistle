#!/usr/bin/env python3
"""
TLE Inference Script

Demonstrates the complete TLE (Text-to-Latent Encoder) pipeline:
Text Input → TLE → Whisper Encoder Latents → Whisper Decoder → Text Output

Example: "你好，香港！" → TLE → latent → Whisper Decoder → "Hello, Hong Kong!"

Usage:
    python infer.py --text "你好，香港！" --checkpoint checkpoints/tle-00-001000.pt
    python infer.py --text "Hello world" --language en --checkpoint checkpoints/tle-00-001000.pt
"""

import torch
import torch.nn as nn
import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    pipeline,
)
from whistle.tle.tle import TLEVAE, TLEVAEConfig
from whistle.utils.checkpoint_utils import load_tle_checkpoint_from_lightning


class TLEInferencePipeline:
    """
    Complete TLE inference pipeline: Text → TLE → Whisper Latents → Whisper Decoder → Text
    """

    def __init__(
        self,
        tle_checkpoint_path: str,
        whisper_model_name: str = "openai/whisper-large-v3",
        device: str = "auto",
    ):
        """
        Initialize the TLE inference pipeline.

        Args:
            tle_checkpoint_path: Path to trained TLE model checkpoint
            whisper_model_name: Whisper model to use for decoding
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.device = self._setup_device(device)

        print(f"Loading TLE model from {tle_checkpoint_path}")
        # Create TLE config (same as training)
        self.tle_config = TLEVAEConfig(vocab_size=51866, whisper_hidden=1280)
        self.tle_model = load_tle_checkpoint_from_lightning(
            tle_checkpoint_path, self.tle_config
        )
        self.tle_model.to(self.device)
        self.tle_model.eval()

        print(f"Loading Whisper model: {whisper_model_name}")
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(
            whisper_model_name
        )
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_model_name)
        self.whisper_tokenizer = AutoTokenizer.from_pretrained(whisper_model_name)

        # Move Whisper to device
        self.whisper_model.to(self.device)
        self.whisper_model.eval()

        # Language mapping (same as training)
        self.language_mapping = {
            "en": 0,  # English
            "zh": 1,  # Mandarin
            "yue": 2,  # Cantonese
        }

        print("✅ TLE Inference Pipeline ready!")

    def _setup_device(self, device: str) -> torch.device:
        """Setup compute device."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _detect_language(self, text: str) -> str:
        """
        Simple language detection based on character sets.
        In production, you'd use a proper language detection model.
        """
        # Chinese characters (Mandarin/Cantonese)
        chinese_chars = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")

        if chinese_chars > len(text) * 0.3:  # >30% Chinese characters
            # Simple heuristic: if contains traditional chars, assume Cantonese
            traditional_chars = sum(1 for char in text if char in "個們會說沒這們")
            return "yue" if traditional_chars > 0 else "zh"
        else:
            return "en"

    def encode_text_to_latents(
        self,
        text: str,
        language: Optional[str] = None,
        target_length: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Encode text input to Whisper encoder latents using TLE.

        Args:
            text: Input text
            language: Language code ("en", "zh", "yue") or auto-detect
            target_length: Target sequence length (Whisper encoder frames)

        Returns:
            Whisper encoder latents: (1, T, H)
        """
        if language is None:
            language = self._detect_language(text)
            print(f"Auto-detected language: {language}")

        # Tokenize text
        tokenizer_inputs = self.whisper_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=256
        )

        input_ids = tokenizer_inputs["input_ids"].to(self.device)
        attention_mask = tokenizer_inputs["attention_mask"].to(self.device)

        # Map language to ID
        lang_id = self.language_mapping.get(language, 0)
        lang_ids = torch.tensor([lang_id], dtype=torch.long).to(self.device)

        # Estimate target length if not provided
        if target_length is None:
            target_length = 1500  # Fixed T for consistent latent length

        print(f"Encoding '{text}' ({language}) → latents with T={target_length}")

        with torch.no_grad():
            # TLE forward pass
            latents, _, _ = self.tle_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                target_T=target_length,
                lang_ids=lang_ids,
            )

        return latents

    def decode_latents_to_text(
        self,
        latents: torch.Tensor,
        language: str = "en",
        max_length: int = 256,
        num_beams: int = 5,
        temperature: float = 1.0,
    ) -> str:
        """
        Decode Whisper latents to text using Whisper decoder.

        Args:
            latents: Whisper encoder latents (B, T, H)
            language: Target language for generation
            max_length: Maximum output length
            num_beams: Beam search width
            temperature: Sampling temperature

        Returns:
            Generated text
        """
        print(f"Decoding latents → text (language: {language})")

        # Latents should already be in Whisper's hidden dimension (1280) from TLE
        # No projection needed - TLE is trained specifically for whisper-large-v3

        # Scale latents to match expected magnitude (experimental)
        # TLE latents are ~935 magnitude, random latents that work are ~1386
        target_magnitude = 1386.0
        current_magnitude = latents.norm().item()
        scale_factor = target_magnitude / current_magnitude
        latents = latents * scale_factor
        print(
            f"Scaled latents from magnitude {current_magnitude:.1f} to {latents.norm().item():.1f}"
        )

        with torch.no_grad():
            # Create proper EncoderOutput object
            from transformers.modeling_outputs import BaseModelOutput

            encoder_outputs = BaseModelOutput(last_hidden_state=latents)

            # Get decoder prompt IDs with language and task tokens
            forced_decoder_ids = self.whisper_processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

            # Generate using Whisper decoder with forced_decoder_ids
            # This ensures the decoder starts with the correct language and task tokens
            generated_ids = self.whisper_model.generate(
                encoder_outputs=encoder_outputs,
                forced_decoder_ids=forced_decoder_ids,
                max_length=max_length,
                num_beams=1,  # Greedy decoding for stability with custom encoder outputs
                temperature=temperature,
                do_sample=temperature > 0,
            )

        # Decode generated tokens to text
        generated_text = self.whisper_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        return generated_text.strip()

    def __call__(
        self,
        text: str,
        language: Optional[str] = None,
        target_length: Optional[int] = None,
        **decode_kwargs,
    ) -> str:
        """
        Complete inference pipeline: Text → TLE → Latents → Whisper Decoder → Text

        Args:
            text: Input text
            language: Language code or auto-detect
            target_length: Target latent sequence length
            **decode_kwargs: Additional arguments for decode_latents_to_text

        Returns:
            Generated text output
        """
        print(f"\n🔄 TLE Inference Pipeline")
        print(f"Input: '{text}'")

        # Step 1: Encode text to latents
        latents = self.encode_text_to_latents(text, language, target_length)

        # Step 2: Decode latents to text
        if language is None:
            language = self._detect_language(text)

        output_text = self.decode_latents_to_text(latents, language, **decode_kwargs)

        print(f"Output: '{output_text}'")
        print("✅ Pipeline complete!\n")

        return output_text


def main():
    parser = argparse.ArgumentParser(description="TLE Inference Pipeline")
    parser.add_argument("--text", type=str, required=True, help="Input text to process")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to TLE model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--language",
        type=str,
        choices=["en", "zh", "yue"],
        default=None,
        help="Language code (auto-detect if not specified)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="openai/whisper-large-v3",
        help="Whisper model to use for decoding",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run inference on",
    )
    parser.add_argument(
        "--target-length",
        type=int,
        default=None,
        help="Target latent sequence length (auto if not specified)",
    )
    parser.add_argument(
        "--max-length", type=int, default=256, help="Maximum output text length"
    )
    parser.add_argument("--num-beams", type=int, default=5, help="Beam search width")
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature (0.0 = greedy)",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = TLEInferencePipeline(
        tle_checkpoint_path=args.checkpoint,
        whisper_model_name=args.whisper_model,
        device=args.device,
    )

    # Run inference
    result = pipeline(
        text=args.text,
        language=args.language,
        target_length=args.target_length,
        max_length=args.max_length,
        num_beams=args.num_beams,
        temperature=args.temperature,
    )

    print(f"🎯 Final Result: {result}")


if __name__ == "__main__":
    main()
