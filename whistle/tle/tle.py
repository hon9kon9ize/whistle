import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import logging

# Set up logging
logger = logging.getLogger(__name__)


# ---------------------------
# Small Utilities
# ---------------------------
def lengths_to_mask(
    lengths: torch.Tensor, max_len: Optional[int] = None
) -> torch.Tensor:
    max_len = max_len or int(lengths.max().item())
    rng = torch.arange(max_len, device=lengths.device)[None, :]
    return rng < lengths[:, None]


class ResidualConv1dFiLM(nn.Module):
    """
    Conv1d residual block with FiLM (scale/shift) conditioning from z.
    in/out channels = H. Kernel 3, padding 1. GELU activations.
    """

    def __init__(self, hidden_size: int, z_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, padding=1)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        # FiLM from z → per-channel scale/shift
        self.film1 = nn.Linear(z_dim, 2 * hidden_size)
        self.film2 = nn.Linear(z_dim, 2 * hidden_size)

    def forward(self, x_tbh: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """
        x_tbh: (B, T, H)  (we'll permute to (B, H, T) for Conv1d)
        z:     (B, Z)
        """
        x = x_tbh.transpose(1, 2)  # (B, H, T) for Conv1d

        # Block 1 with FiLM
        y = self.conv1(x)  # (B, H, T)
        y = y.transpose(1, 2)  # (B, T, H) for norm + FiLM
        y = self.norm1(y)
        gamma, beta = self.film1(z).chunk(2, dim=-1)  # (B, H), (B, H)
        # Clamp FiLM modulation to prevent explosion
        gamma = torch.tanh(gamma) * 0.3  # Scale tanh to [-0.3, 0.3]
        beta = torch.tanh(beta) * 0.3
        y = y * (1 + gamma[:, None, :]) + beta[:, None, :]
        y = F.gelu(y)
        y = y.transpose(1, 2)  # (B, H, T) for next conv

        # Block 2 with FiLM
        y = self.conv2(y)  # (B, H, T)
        y = y.transpose(1, 2)  # (B, T, H) for norm + FiLM
        y = self.norm2(y)
        gamma2, beta2 = self.film2(z).chunk(2, dim=-1)
        # Clamp FiLM modulation to prevent explosion
        gamma2 = torch.tanh(gamma2) * 0.3  # Scale tanh to [-0.3, 0.3]
        beta2 = torch.tanh(beta2) * 0.3
        y = y * (1 + gamma2[:, None, :]) + beta2[:, None, :]
        y = F.gelu(y)

        # Residual connection (both in (B, T, H))
        return x_tbh + y


class PositionalEncoding(nn.Module):
    """Standard sine-cosine PE for (B, T, H)."""

    def __init__(self, hidden_size: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, T, _ = x.shape
        return x + self.pe[:T, :][None, :, :]


# ---------------------------
# TLE VAE: text -> E_tilde
# ---------------------------
@dataclass
class TLEVAEConfig:
    vocab_size: int
    text_hidden: int = 768  # text-side hidden (small)
    n_text_layers: int = 4
    n_text_heads: int = 8
    whisper_hidden: int = 1280  # H for whisper-large-v3 encoder
    z_dim: int = 256  # global latent
    n_res_blocks: int = 6
    beta: float = 0.1  # KL weight
    # KL scheduling parameters
    beta_start: float = (
        1.0  # starting beta for annealing (increased for better KL learning)
    )
    beta_end: float = 10.0  # final beta value (increased to enforce latent learning)
    beta_warmup_steps: int = (
        5000  # steps to anneal beta (increased for smoother transition)
    )
    # Free-bits parameters
    free_bits_threshold: float = (
        0.01  # KL per dim threshold (in nats) - REDUCED from 1.0 to 0.01 to prevent posterior collapse
    )
    # Language conditioning parameters
    num_languages: int = 3  # en, zh, yue
    lang_embed_dim: int = 32  # small language embedding dimension
    # Teacher state augmentation parameters
    teacher_noise_std: float = 0.01  # Gaussian noise std for teacher states
    teacher_time_jitter_max: float = 0.1  # Max time stretch factor (±10%)
    # Performance optimization parameters
    gradient_checkpointing: bool = (
        True  # Enable gradient checkpointing for memory efficiency
    )


class TLETextEncoder(nn.Module):
    """Tiny Transformer over token embeddings -> (B, L, D_text)."""

    def __init__(self, cfg: TLEVAEConfig):
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.text_hidden)
        self.lang_embed = nn.Embedding(cfg.num_languages, cfg.lang_embed_dim)
        # Project language embedding to match text_hidden dimension
        self.lang_proj = nn.Linear(cfg.lang_embed_dim, cfg.text_hidden)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.text_hidden,
            nhead=cfg.n_text_heads,
            dim_feedforward=4 * cfg.text_hidden,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=cfg.n_text_layers
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        lang_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.embed(input_ids)  # (B, L, D_text)

        # Add language conditioning if provided
        if lang_ids is not None:
            # lang_ids: (B,) -> (B, lang_embed_dim) -> (B, D_text) -> (B, 1, D_text)
            lang_emb = self.lang_proj(self.lang_embed(lang_ids))  # (B, D_text)
            lang_emb = lang_emb.unsqueeze(1)  # (B, 1, D_text)
            x = x + lang_emb  # Broadcast to all tokens in sequence

        if attention_mask is not None:
            # Transformer expects True = keep, so invert mask
            key_padding_mask = ~attention_mask.bool()
        else:
            key_padding_mask = None
        x = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return x


class TLEVAE(nn.Module):
    """
    Text-to-Latent VAE that outputs pseudo Whisper encoder states.
    Training usage:
      E_tilde, mu, logvar = model(input_ids, attn_mask, target_T=E.size(1))
      loss = mse(E_tilde, E) + beta * KL
    Inference (text-only):
      E_tilde, _, _ = model(input_ids, attn_mask, target_T=pred_T or heuristic)
    """

    def __init__(self, cfg: TLEVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.text_encoder = TLETextEncoder(cfg)

        # Project text features to Whisper hidden for time-interpolation seed
        self.text_to_H = nn.Linear(cfg.text_hidden, cfg.whisper_hidden)

        # Global latent heads from pooled text features
        self.mu_head = nn.Linear(cfg.text_hidden, cfg.z_dim)
        self.logvar_head = nn.Linear(cfg.text_hidden, cfg.z_dim)

        # Residual Conv stack over time with FiLM(z)
        self.pe = PositionalEncoding(cfg.whisper_hidden)
        self.resblocks = nn.ModuleList(
            [
                ResidualConv1dFiLM(cfg.whisper_hidden, cfg.z_dim)
                for _ in range(cfg.n_res_blocks)
            ]
        )
        self.proj_out = nn.Linear(
            cfg.whisper_hidden, cfg.whisper_hidden
        )  # final projection

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Clamp logvar for numerical stability AND variance preservation
        # Changed from [-5, 5] to [-3, 2] to:
        # - Prevent variance collapse (exp(logvar) was ~0.27, target > 0.5)
        # - Maintain numerical stability (still prevents NaN/explosion)
        # - Allow posterior to remain informative
        logvar = torch.clamp(logvar, min=-3, max=2)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def _interp_time(x_blh: torch.Tensor, target_T: int) -> torch.Tensor:
        """
        x_blh: (B, L, H) -> interpolate along L dimension to length T, return (B, T, H)
        Uses linear interpolation for temporal upsampling.
        """
        _, L, _ = x_blh.shape
        if L == target_T:
            return x_blh
        if L == 1:
            # Broadcast single token to target length
            return x_blh.expand(-1, target_T, -1)

        x_bhl = x_blh.transpose(1, 2)  # (B, H, L)
        x_bht = F.interpolate(
            x_bhl, size=target_T, mode="linear", align_corners=False
        )  # (B, H, T)
        return x_bht.transpose(1, 2)  # (B, T, H)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        target_T: Optional[int] = None,
        lang_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: (E_tilde, mu, logvar)
          E_tilde: predicted encoder states (B, T, H)
          mu, logvar: latent parameters (B, z_dim)
        """
        B, L = input_ids.shape

        # Input validation
        if attention_mask is not None:
            assert (
                attention_mask.shape == input_ids.shape
            ), f"attention_mask shape {attention_mask.shape} != input_ids shape {input_ids.shape}"
        if target_T is not None:
            assert target_T > 0, f"target_T must be positive, got {target_T}"
        if lang_ids is not None:
            assert lang_ids.shape == (
                B,
            ), f"lang_ids shape {lang_ids.shape} != (B,) = {(B,)}"

        # 1) Text encoder
        text_feats = self.text_encoder(
            input_ids, attention_mask, lang_ids
        )  # (B, L, D_text)

        # 2) Global latent z from pooled text features
        pooled = (
            text_feats
            * (attention_mask[:, :, None] if attention_mask is not None else 1)
        ).sum(1)
        denom = (
            attention_mask.sum(1).clamp(min=1)
            if attention_mask is not None
            else torch.full((B,), L, device=text_feats.device)
        )
        pooled = pooled / denom[:, None]  # (B, D_text)
        mu = self.mu_head(pooled)  # (B, z_dim)
        logvar = self.logvar_head(pooled)  # (B, z_dim)
        z = self.reparameterize(mu, logvar)  # (B, z_dim)

        # 3) Seed time sequence by interpolating text→H to target_T
        if target_T is None:
            # Heuristic if not given: ~ 2 frames per token (tune to your data)
            target_T = max(2, 2 * L)
        seed = self.text_to_H(text_feats)  # (B, L, H)
        x = self._interp_time(seed, target_T)  # (B, T, H)
        x = self.pe(x)  # add positional encoding

        # 4) Residual Conv stack with FiLM(z) - use gradient checkpointing for memory efficiency
        for block in self.resblocks:
            # Use gradient checkpointing to reduce memory usage during training
            if self.training and self.cfg.gradient_checkpointing:
                x = checkpoint(block, x, z, use_reentrant=False)
            else:
                x = block(x, z)  # (B, T, H)

        # 5) Final projection
        E_tilde = self.proj_out(x)  # (B, T, H)
        return E_tilde, mu, logvar


# ---------------------------
# Loss helpers
# ---------------------------
def vae_loss(
    E_tilde: torch.Tensor,
    E_teacher: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float,
    free_bits_threshold: float = 0.0,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MSE recon over (B,T,H) + beta * KL( q(z|y) || N(0,I) ) with optional free-bits and masking

    Args:
        E_tilde: predicted encoder states (B, T, H)
        E_teacher: teacher encoder states (B, T, H)
        mu: latent mean (B, z_dim)
        logvar: latent log variance (B, z_dim)
        beta: KL weight coefficient
        free_bits_threshold: minimum KL per dimension to penalize (in nats)
        mask: optional boolean mask (B, T) for valid positions. If provided, MSE is computed only over valid positions.

    Returns:
        Tuple of (total_loss, mse_loss, kl_loss)
    """
    if mask is not None:
        # Compute masked MSE: only consider valid positions
        diff = (E_tilde - E_teacher) ** 2  # (B, T, H)
        masked_diff = diff * mask.unsqueeze(-1)  # Zero out invalid positions
        mse = masked_diff.sum() / (
            mask.sum() * E_tilde.size(-1)
        )  # Average over valid elements
    else:
        mse = F.mse_loss(E_tilde, E_teacher)

    # Compute KL per dimension
    # KL(q(z) || N(0,I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, z_dim)

    # Apply free-bits: only penalize KL above threshold
    if free_bits_threshold > 0:
        kl_per_dim = torch.clamp(kl_per_dim - free_bits_threshold, min=0)

    # Average over batch and dimensions
    kl = kl_per_dim.mean()

    loss = mse + beta * kl

    return loss, mse, kl
