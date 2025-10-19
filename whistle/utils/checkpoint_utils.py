"""
Utilities for loading and managing TLE checkpoints.

The checkpoints saved by ModelOnlyCheckpoint contain only the TLEVAE model
state_dict (not full Lightning state including optimizer), which is ~500MB
instead of ~1.6GB.
"""

import torch
from whistle.tle.tle import TLEVAE, TLEVAEConfig


def load_tle_checkpoint(checkpoint_path: str, cfg: TLEVAEConfig) -> TLEVAE:
    """
    Load a TLE model from a checkpoint saved by ModelOnlyCheckpoint.

    Args:
        checkpoint_path: Path to the checkpoint file (.pt file)
        cfg: TLEVAEConfig for the model

    Returns:
        TLEVAE model with loaded weights

    Example:
        >>> from whistle.utils.checkpoint_utils import load_tle_checkpoint
        >>> cfg = TLEVAEConfig(vocab_size=51866, whisper_hidden=1280)
        >>> model = load_tle_checkpoint("checkpoints/tle-00-001000.pt", cfg)
        >>> model.eval()
    """
    model = TLEVAE(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")

    # Filter out Whisper parameters - TLEVAE should only contain its own parameters
    tle_state_dict = {}
    tle_prefixes = [
        "text_encoder.",
        "text_to_H.",
        "mu_head.",
        "logvar_head.",
        "pe.",
        "resblocks.",
        "proj_out.",
    ]

    for key, value in state_dict.items():
        # Keep only TLEVAE parameters, filter out Whisper and other components
        if any(key.startswith(prefix) for prefix in tle_prefixes):
            tle_state_dict[key] = value
        elif (
            not key.startswith("whisper.")
            and not key.startswith("processor.")
            and not key.startswith("tokenizer.")
        ):
            # Also keep parameters that don't have component prefixes but aren't Whisper
            tle_state_dict[key] = value

    print(
        f"Loaded {len(tle_state_dict)} TLE parameters from {len(state_dict)} total parameters"
    )
    if len(tle_state_dict) != len(state_dict):
        filtered_count = len(state_dict) - len(tle_state_dict)
        print(
            f"Filtered out {filtered_count} non-TLE parameters (Whisper, processor, tokenizer)"
        )

    model.load_state_dict(tle_state_dict)
    print(f"Loaded TLE model from {checkpoint_path}")
    return model


def load_tle_checkpoint_from_lightning(
    checkpoint_path: str, cfg: TLEVAEConfig
) -> TLEVAE:
    """
    Load a TLE model from a full Lightning checkpoint (includes optimizer state).

    Use this if you have old checkpoints from before ModelOnlyCheckpoint was implemented.

    Args:
        checkpoint_path: Path to the .ckpt Lightning checkpoint file
        cfg: TLEVAEConfig for the model

    Returns:
        TLEVAE model with loaded weights

    Example:
        >>> from whistle.utils.checkpoint_utils import load_tle_checkpoint_from_lightning
        >>> cfg = TLEVAEConfig(vocab_size=51866, whisper_hidden=1280)
        >>> model = load_tle_checkpoint_from_lightning("checkpoints/epoch=0-step=3.ckpt", cfg)
    """
    model = TLEVAE(cfg)
    lightning_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = lightning_ckpt["state_dict"]

    # Remove "tle." prefix if present and filter out Whisper parameters
    state_dict_cleaned = {}
    tle_prefixes = [
        "tle.text_encoder.",
        "tle.text_to_H.",
        "tle.mu_head.",
        "tle.logvar_head.",
        "tle.pe.",
        "tle.resblocks.",
        "tle.proj_out.",
    ]

    for key, value in state_dict.items():
        # Handle Lightning checkpoint format with "tle." prefix
        if key.startswith("tle."):
            clean_key = key[4:]  # Remove "tle." prefix
            # Keep only TLEVAE parameters, filter out Whisper and other components
            if any(clean_key.startswith(prefix[4:]) for prefix in tle_prefixes) or (
                not clean_key.startswith("whisper.")
                and not clean_key.startswith("processor.")
                and not clean_key.startswith("tokenizer.")
            ):
                state_dict_cleaned[clean_key] = value
        else:
            # Keep parameters that don't have component prefixes but aren't Whisper
            if (
                not key.startswith("whisper.")
                and not key.startswith("processor.")
                and not key.startswith("tokenizer.")
            ):
                state_dict_cleaned[key] = value

    print(
        f"Loaded {len(state_dict_cleaned)} TLE parameters from {len(state_dict)} total parameters"
    )
    if len(state_dict_cleaned) != len(state_dict):
        filtered_count = len(state_dict) - len(state_dict_cleaned)
        print(
            f"Filtered out {filtered_count} non-TLE parameters (Whisper, processor, tokenizer)"
        )

    model.load_state_dict(state_dict_cleaned)
    print(f"Loaded TLE model from Lightning checkpoint {checkpoint_path}")
    return model
