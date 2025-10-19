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
        >>> cfg = TLEVAEConfig(vocab_size=51866, whisper_hidden=1280)
        >>> model = load_tle_checkpoint("checkpoints/tle-00-001000.pt", cfg)
        >>> model.eval()
    """
    model = TLEVAE(cfg)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
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
        >>> cfg = TLEVAEConfig(vocab_size=51866, whisper_hidden=1280)
        >>> model = load_tle_checkpoint_from_lightning("checkpoints/epoch=0-step=3.ckpt", cfg)
    """
    model = TLEVAE(cfg)
    lightning_ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = lightning_ckpt["state_dict"]

    # Remove "tle." prefix if present
    state_dict_cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("tle."):
            state_dict_cleaned[key[4:]] = value
        else:
            state_dict_cleaned[key] = value

    model.load_state_dict(state_dict_cleaned)
    print(f"Loaded TLE model from Lightning checkpoint {checkpoint_path}")
    return model
