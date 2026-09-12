"""
VLM Backbone for CATE-π (§4.5).

PaliGemma-initialized backbone: SigLIP vision + Gemma language decoder.
This module:
1. Encodes the fused prefix S_t = [z_vis, z_lang, z_state, z̃_tac]
   with non-causal / fully bidirectional attention.
2. Provides KV cache from the prefix for the action expert suffix.
3. Supports LoRA fine-tuning of the Gemma weights (TacVLA's approach).
4. Handles language tokenization + embedding via PaliGemma's SentencePiece.
5. Projects proprioceptive state to a single continuous token (π0 convention).

Two forward modes share the same backbone weights (§4.7):
- Action generation: bidirectional prefix, block-causal suffix (via action expert)
- CoT reasoning: bidirectional prefix, autoregressive text suffix (via cot_head)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class AdaRMSNorm(nn.Module):
    """
    Adaptive RMS normalization conditioned on a signal (e.g. flow-matching τ).
    Used in the action expert to inject the denoising timestep.
    """

    def __init__(self, hidden_dim: int, cond_dim: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scale_proj = nn.Linear(cond_dim, hidden_dim)
        self.shift_proj = nn.Linear(cond_dim, hidden_dim)
        self.eps = 1e-6

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) hidden states.
            cond: (B, 1) or (B,) conditioning signal.
        """
        if cond.dim() == 1:
            cond = cond.unsqueeze(-1)

        # RMS norm
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x_normed = x / rms

        # Adaptive scale and shift
        scale = 1.0 + self.scale_proj(cond).unsqueeze(1)  # (B, 1, D)
        shift = self.shift_proj(cond).unsqueeze(1)  # (B, 1, D)

        return x_normed * scale + shift


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional attention mask."""

    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_dim = hidden_dim

        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, D) query input.
            attention_mask: (B, T, S) or (B, 1, T, S) float mask.
            kv: (B, S, D) optional separate key-value input for cross-attention.
        """
        B, T, _ = x.shape
        kv_input = kv if kv is not None else x
        S = kv_input.shape[1]

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(kv_input).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(kv_input).reshape(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, S)

        if attention_mask is not None:
            if attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)  # (B, 1, T, S)
            
            # Apply proportional soft-masking via log-additive bias (§4.4)
            # For gate=1.0, log(1)=0 (no penalty)
            # For gate in (0, 1), log(gate) provides a proportional penalty
            # For gate=0.0, we apply a hard -1e9 penalty
            zero_mask = (attention_mask == 0.0)
            log_bias = torch.log(attention_mask.clamp(min=1e-9))
            log_bias = log_bias.masked_fill(zero_mask, -1e9)
            
            attn_weights = attn_weights + log_bias

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v)  # (B, H, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        out = self.o_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm architecture."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(hidden_dim, num_heads, head_dim, dropout)
        self.norm1 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.norm2 = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, intermediate_dim),
            nn.GELU(),
            nn.Linear(intermediate_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-norm self/cross-attention
        residual = x
        x = self.norm1(x)
        x = self.attention(x, attention_mask=attention_mask, kv=kv)
        x = residual + x

        # Pre-norm FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x

        return x


class StateProjection(nn.Module):
    """
    Projects proprioceptive state vector to a single continuous token.
    Follows π0's convention: linear projection, not discretized/tokenized.
    """

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: (B, state_dim) proprioceptive state vector.
        Returns:
            token: (B, 1, hidden_dim) state token.
        """
        return self.projection(state).unsqueeze(1)


class VLMBackbone(nn.Module):
    """
    PaliGemma-based VLM backbone for CATE-π.

    Encodes the fused multimodal prefix with bidirectional attention,
    producing hidden states / KV cache for the action expert or CoT head.
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        num_layers: int = 18,
        num_heads: int = 8,
        head_dim: int = 256,
        intermediate_dim: int = 16384,
        vocab_size: int = 256000,
        max_seq_len: int = 8192,
        state_dim: int = 32,
        dropout: float = 0.0,
    ):
        """
        Args:
            hidden_dim: Gemma hidden dimension.
            num_layers: Number of transformer layers.
            num_heads: Number of attention heads.
            head_dim: Per-head dimension.
            intermediate_dim: FFN intermediate dimension.
            vocab_size: Vocabulary size (PaliGemma SentencePiece).
            max_seq_len: Maximum sequence length.
            state_dim: Proprioceptive state vector dimension.
            dropout: Dropout rate.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Language token embedding (shared with PaliGemma's embedding table)
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)

        # Proprioceptive state projection
        self.state_projection = StateProjection(state_dim, hidden_dim)

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                head_dim=head_dim,
                intermediate_dim=intermediate_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.final_norm = nn.RMSNorm(hidden_dim, eps=1e-6)

    def embed_language(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        Embed language tokens using PaliGemma's shared embedding table.

        Args:
            token_ids: (B, L) integer token IDs.
        Returns:
            embeddings: (B, L, hidden_dim) language token embeddings.
        """
        return self.token_embedding(token_ids)

    def embed_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Project proprioceptive state to a single token.

        Args:
            state: (B, state_dim) proprioceptive state.
        Returns:
            token: (B, 1, hidden_dim) state token.
        """
        return self.state_projection(state)

    def build_prefix(
        self,
        vision_tokens: torch.Tensor,
        language_tokens: torch.Tensor,
        state_token: torch.Tensor,
        tactile_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Concatenate all modality tokens into the ordered prefix.

        Token order (§5): [vision | language | state | tactile(gated)]

        Args:
            vision_tokens: (B, N_vis, D) from VisionEncoder.
            language_tokens: (B, N_lang, D) embedded language.
            state_token: (B, 1, D) projected state.
            tactile_tokens: (B, N_tac, D) gated tactile tokens.

        Returns:
            prefix: (B, N_total, D) fused prefix sequence.
        """
        return torch.cat([vision_tokens, language_tokens, state_token, tactile_tokens], dim=1)

    def encode_prefix(
        self,
        prefix: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Encode the fused prefix with bidirectional (non-causal) attention.

        This is the shared computation used by both the action expert
        and the CoT head.

        Args:
            prefix: (B, N_prefix, hidden_dim) fused prefix tokens.
            attention_mask: (B, N_prefix, N_prefix) attention mask from
                           attention_masks.build_prefix_mask().

        Returns:
            prefix_hidden: (B, N_prefix, hidden_dim) encoded prefix states.
        """
        x = prefix
        for layer in self.layers:
            x = layer(x, attention_mask=attention_mask)
        x = self.final_norm(x)
        return x

    def forward(
        self,
        vision_tokens: torch.Tensor,
        language_ids: torch.Tensor,
        state: torch.Tensor,
        tactile_tokens: torch.Tensor,
        prefix_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full prefix encoding pipeline.

        Args:
            vision_tokens: (B, N_vis, D) from VisionEncoder.
            language_ids: (B, L) language token IDs.
            state: (B, state_dim) proprioceptive state.
            tactile_tokens: (B, N_tac, D) gated tactile tokens.
            prefix_attention_mask: (B, N, N) prefix attention mask.

        Returns:
            prefix_hidden: (B, N_total, hidden_dim) encoded prefix.
        """
        # Embed modalities
        lang_tokens = self.embed_language(language_ids)  # (B, L, D)
        state_token = self.embed_state(state)  # (B, 1, D)

        # Build ordered prefix
        prefix = self.build_prefix(
            vision_tokens=vision_tokens,
            language_tokens=lang_tokens,
            state_token=state_token,
            tactile_tokens=tactile_tokens,
        )

        # Encode with bidirectional attention
        prefix_hidden = self.encode_prefix(prefix, attention_mask=prefix_attention_mask)

        return prefix_hidden

    @classmethod
    def from_config(cls, config: dict) -> "VLMBackbone":
        """Create from the backbone config dict."""
        gemma_cfg = config["backbone"]["gemma"]
        return cls(
            hidden_dim=gemma_cfg["hidden_dim"],
            num_layers=gemma_cfg["num_layers"],
            num_heads=gemma_cfg["num_heads"],
            head_dim=gemma_cfg["head_dim"],
            intermediate_dim=gemma_cfg["intermediate_dim"],
            vocab_size=gemma_cfg["vocab_size"],
            max_seq_len=gemma_cfg["max_position_embeddings"],
        )
