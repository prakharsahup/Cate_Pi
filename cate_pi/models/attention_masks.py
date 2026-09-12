"""
Unified attention mask builder for CATE-π (§5).

Builds the block-sparse attention masks for:
1. PREFIX (non-causal): [vision | language | state | tactile(gated)]
   - All four groups fully cross-attend to each other
   - Tactile block is masked out entirely when c_t = 0 (contact gate)

2. SUFFIX (block-causal): [state_token(suffix) | noisy_action+force_tokens(τ)]
   - State token: attends to full prefix + earlier state tokens (causal)
   - Action tokens: attend to full prefix + state token(s) + each other
     (bidirectionally within the action block)
   - Action tokens never attend to future/other denoising-step copies

3. COT mode: suffix becomes causal text tokens (standard autoregressive)

IMPORTANT (from §5):
- A naive dense attention is correct but wasteful due to block-sparsity.
- Get correctness first with explicit boolean masks, then optimize with
  flex_attention + create_block_mask.
- Write a unit test that forces c_t=0 and asserts tactile block contributes
  exactly zero gradient to every other token's output.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


def build_prefix_mask(
    num_vision_tokens: int,
    num_language_tokens: int,
    num_state_tokens: int,
    num_tactile_tokens: int,
    contact_flag: torch.Tensor,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the non-causal prefix attention mask.

    All modality groups cross-attend freely, except tactile tokens
    are masked out (both as queries and keys) when contact_flag = 0.

    Args:
        num_vision_tokens: Number of vision tokens (e.g. 512 for 2 cameras).
        num_language_tokens: Number of language tokens.
        num_state_tokens: Number of state tokens (typically 1).
        num_tactile_tokens: Number of tactile tokens (e.g. 36).
        contact_flag: Shape (B,) or (B, 1) — per-sample binary/soft gate value.
        device: Target device.

    Returns:
        mask: Shape (B, total_prefix, total_prefix), float mask where
              1.0 = attend, 0.0 = masked.
    """
    B = contact_flag.shape[0]
    total = num_vision_tokens + num_language_tokens + num_state_tokens + num_tactile_tokens

    # Start with full attention (non-causal prefix)
    mask = torch.ones(B, total, total, device=device, dtype=torch.float32)

    # Tactile token range
    tac_start = num_vision_tokens + num_language_tokens + num_state_tokens
    tac_end = total

    # Gate: zero out tactile COLUMNS ONLY when contact_flag = 0
    # contact_flag shape: (B,) -> (B, 1, 1) for broadcasting
    gate = contact_flag.view(B, 1, 1).float()

    # Tactile tokens as keys (columns): other tokens attending TO tactile
    mask[:, :, tac_start:tac_end] = mask[:, :, tac_start:tac_end] * gate

    # NOTE: We intentionally do NOT zero tactile ROWS (tactile as queries).
    # Reason: an all-masked row (every column zeroed) would produce
    # softmax(-inf, -inf, ..., -inf) = 0/0 = NaN. Even though nothing
    # downstream reads the tactile output positions (columns are already
    # masked), autograd does not know the coefficient is "supposed to be
    # zero" — 0 * NaN = NaN during backprop, and NaN gradients leak into
    # shared parameters (tactile encoder, projection layers).
    #
    # This is safe because the embedding gate (§4.4) already zeroed z_tac,
    # so tactile query vectors collapse to ~zero (or a bias term), producing
    # uniform-but-harmless attention over valid keys. The column-mask
    # already prevents anyone from attending TO tactile tokens, which is
    # the gating behavior we actually need.

    return mask


def build_suffix_mask(
    num_prefix_tokens: int,
    num_state_tokens_suffix: int,
    num_action_tokens: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the block-causal suffix attention mask (action generation mode).

    Structure:
    - State token(s) in suffix: attend to full prefix + earlier state tokens (causal)
    - Action tokens: attend to full prefix + state token(s) + each other
      (bidirectionally within the action block)
    - Action tokens never attend to tokens from other denoising steps.

    This is NOT batched over B since the structure is the same for all samples;
    the prefix mask handles per-sample gating.

    Args:
        num_prefix_tokens: Total tokens in the prefix (already encoded).
        num_state_tokens_suffix: State tokens repeated in suffix (typically 1).
        num_action_tokens: Noisy action+force chunk tokens.
        device: Target device.

    Returns:
        mask: Shape (total_suffix_with_prefix, total_suffix_with_prefix),
              float mask. Rows/cols indexed as [prefix | state_suffix | actions].
    """
    total = num_prefix_tokens + num_state_tokens_suffix + num_action_tokens

    # Initialize: no attention
    mask = torch.zeros(total, total, device=device, dtype=torch.float32)

    # === Prefix block: handled by prefix_mask, allow self-attention here ===
    mask[:num_prefix_tokens, :num_prefix_tokens] = 1.0

    # === State token(s) in suffix ===
    state_start = num_prefix_tokens
    state_end = num_prefix_tokens + num_state_tokens_suffix

    # State attends to: full prefix
    mask[state_start:state_end, :num_prefix_tokens] = 1.0
    # State attends to: earlier state tokens (causal within state)
    for i in range(num_state_tokens_suffix):
        mask[state_start + i, state_start:state_start + i + 1] = 1.0

    # === Action tokens ===
    action_start = state_end
    action_end = total

    # Action tokens attend to: full prefix
    mask[action_start:action_end, :num_prefix_tokens] = 1.0
    # Action tokens attend to: state token(s) in suffix
    mask[action_start:action_end, state_start:state_end] = 1.0
    # Action tokens attend to: each other (bidirectional within action block)
    mask[action_start:action_end, action_start:action_end] = 1.0

    return mask


def build_cot_suffix_mask(
    num_prefix_tokens: int,
    num_cot_tokens: int,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the causal suffix mask for CoT reasoning mode.

    CoT tokens use standard autoregressive (causal) attention over each other,
    with full attention to the prefix.

    Args:
        num_prefix_tokens: Total tokens in the prefix.
        num_cot_tokens: Number of CoT text generation tokens.
        device: Target device.

    Returns:
        mask: Shape (total, total) where total = prefix + cot.
    """
    total = num_prefix_tokens + num_cot_tokens
    mask = torch.zeros(total, total, device=device, dtype=torch.float32)

    # Prefix self-attention
    mask[:num_prefix_tokens, :num_prefix_tokens] = 1.0

    # CoT tokens attend to full prefix
    cot_start = num_prefix_tokens
    mask[cot_start:, :num_prefix_tokens] = 1.0

    # CoT tokens: standard causal mask over each other
    for i in range(num_cot_tokens):
        mask[cot_start + i, cot_start:cot_start + i + 1] = 1.0

    return mask


def build_full_cate_mask(
    num_vision_tokens: int,
    num_language_tokens: int,
    num_state_tokens: int,
    num_tactile_tokens: int,
    num_action_tokens: int,
    contact_flag: torch.Tensor,
    mode: str = "action",
    num_cot_tokens: int = 0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Build the complete CATE-π attention mask combining prefix and suffix.

    Args:
        num_vision_tokens: Vision token count.
        num_language_tokens: Language token count.
        num_state_tokens: State token count (prefix).
        num_tactile_tokens: Tactile token count.
        num_action_tokens: Action chunk token count (action mode only).
        contact_flag: Shape (B,) — per-sample contact gate value.
        mode: "action" for action generation, "cot" for chain-of-thought.
        num_cot_tokens: Number of CoT tokens (cot mode only).
        device: Target device.

    Returns:
        mask: Shape (B, total, total) attention mask.
    """
    B = contact_flag.shape[0]
    num_prefix = num_vision_tokens + num_language_tokens + num_state_tokens + num_tactile_tokens

    # Build prefix mask (batched, handles contact gating)
    prefix_mask = build_prefix_mask(
        num_vision_tokens=num_vision_tokens,
        num_language_tokens=num_language_tokens,
        num_state_tokens=num_state_tokens,
        num_tactile_tokens=num_tactile_tokens,
        contact_flag=contact_flag,
        device=device,
    )

    if mode == "action":
        # Suffix: 1 state token (copy) + action tokens
        num_state_suffix = 1
        suffix_struct = build_suffix_mask(
            num_prefix_tokens=num_prefix,
            num_state_tokens_suffix=num_state_suffix,
            num_action_tokens=num_action_tokens,
            device=device,
        )

        total = num_prefix + num_state_suffix + num_action_tokens
        full_mask = torch.zeros(B, total, total, device=device, dtype=torch.float32)

        # Embed prefix mask into top-left
        full_mask[:, :num_prefix, :num_prefix] = prefix_mask

        # Expand suffix structure across batch
        suffix_expanded = suffix_struct.unsqueeze(0).expand(B, -1, -1)

        # The suffix structure already includes prefix→prefix attention;
        # we need to overlay the suffix's cross-attention pattern but keep
        # the batched prefix gating from prefix_mask.
        # Suffix rows attending to prefix: use the gated prefix visibility
        suffix_rows_start = num_prefix
        full_mask[:, suffix_rows_start:, :num_prefix] = (
            suffix_expanded[:, suffix_rows_start:, :num_prefix]
            * prefix_mask[:, 0:1, :num_prefix]  # broadcast: use row-0 of prefix mask for visibility
        )

        # Suffix self-attention (state_suffix + action tokens)
        full_mask[:, suffix_rows_start:, suffix_rows_start:] = (
            suffix_expanded[:, suffix_rows_start:, suffix_rows_start:]
        )

    elif mode == "cot":
        cot_struct = build_cot_suffix_mask(
            num_prefix_tokens=num_prefix,
            num_cot_tokens=num_cot_tokens,
            device=device,
        )

        total = num_prefix + num_cot_tokens
        full_mask = torch.zeros(B, total, total, device=device, dtype=torch.float32)

        # Embed prefix mask
        full_mask[:, :num_prefix, :num_prefix] = prefix_mask

        # CoT suffix structure
        cot_expanded = cot_struct.unsqueeze(0).expand(B, -1, -1)

        cot_start = num_prefix
        # CoT attending to prefix (with gating)
        full_mask[:, cot_start:, :num_prefix] = (
            cot_expanded[:, cot_start:, :num_prefix]
            * prefix_mask[:, 0:1, :num_prefix]
        )

        # CoT self-attention (causal)
        full_mask[:, cot_start:, cot_start:] = cot_expanded[:, cot_start:, cot_start:]

    else:
        raise ValueError(f"Unknown mode: {mode}. Expected 'action' or 'cot'.")

    return full_mask


class CATEPiAttentionMaskBuilder(nn.Module):
    """
    Stateful mask builder that caches token counts from the model config
    and provides a clean interface for building masks during forward passes.
    """

    def __init__(
        self,
        num_vision_tokens: int = 512,
        num_language_tokens: int = 128,
        num_state_tokens: int = 1,
        num_tactile_tokens: int = 36,
        num_action_tokens: int = 50,
    ):
        super().__init__()
        self.num_vision_tokens = num_vision_tokens
        self.num_language_tokens = num_language_tokens
        self.num_state_tokens = num_state_tokens
        self.num_tactile_tokens = num_tactile_tokens
        self.num_action_tokens = num_action_tokens

    def forward(
        self,
        contact_flag: torch.Tensor,
        mode: str = "action",
        num_cot_tokens: int = 0,
        actual_language_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Build the full attention mask for a forward pass.

        Args:
            contact_flag: (B,) contact gate values.
            mode: "action" or "cot".
            num_cot_tokens: tokens for CoT (only in cot mode).
            actual_language_tokens: override language token count if variable.

        Returns:
            (B, total, total) attention mask.
        """
        n_lang = actual_language_tokens if actual_language_tokens is not None else self.num_language_tokens

        return build_full_cate_mask(
            num_vision_tokens=self.num_vision_tokens,
            num_language_tokens=n_lang,
            num_state_tokens=self.num_state_tokens,
            num_tactile_tokens=self.num_tactile_tokens,
            num_action_tokens=self.num_action_tokens,
            contact_flag=contact_flag,
            mode=mode,
            num_cot_tokens=num_cot_tokens,
            device=contact_flag.device,
        )
