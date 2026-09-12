"""
Chain-of-Thought Reasoning Head for CATE-π (§4.7).

This head reuses the SAME Gemma decoder weights as the VLM backbone,
but in autoregressive text-generation mode instead of the bidirectional
prefix + flow-matching suffix mode.

Two forward modes sharing one set of backbone weights:
| Mode              | Suffix content             | Attention over suffix     | Trigger          |
|-------------------|---------------------------|---------------------------|------------------|
| Action generation | noisy action + time embed  | block-causal, bidir action| every step       |
| CoT reasoning     | free-form text tokens      | standard causal/AR        | periodic/event   |

The CoT prompt structure follows Tactile-VLA exactly:
1. "Has the task been completed successfully?" → yes/no
2. If no: "What should be adjusted?" → analysis referencing force components
3. Emit corrective instruction → replaces L_t for subsequent steps

CRITICAL DESIGN ADDITION (§4.7):
The raw, ungated tactile history buffer feeds the CoT head ONLY.
This preserves the action pathway's gating efficiency while giving the
reasoning pathway access to contact/non-contact transitions needed for
slip detection and failure diagnosis.

Buffer: [(c_i, mean_force_i, peak_shear_i) for i in t-W..t]
→ Pooled into ~10 summary tokens via 1D pooling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from collections import deque


class RawTactileBuffer:
    """
    Always-on, ungated rolling buffer of tactile summary statistics.

    This buffer is NEVER gated — it stores the full force history including
    non-contact periods, which is essential for detecting transitions
    (e.g., slip = contact→non-contact) during CoT reasoning.

    Stores per-step: (contact_flag, mean_force, peak_shear)
    """

    def __init__(self, window_size: int = 50, features_per_step: int = 3):
        """
        Args:
            window_size: W — number of past steps to retain (~3-4 seconds).
            features_per_step: Features logged per step (contact, mean_force, peak_shear).
        """
        self.window_size = window_size
        self.features_per_step = features_per_step
        # Per-sample buffers (managed externally for batched operation)
        self._buffer: Optional[torch.Tensor] = None

    def reset(self, batch_size: int, device: torch.device):
        """Initialize/reset the buffer for a new episode."""
        self._buffer = torch.zeros(
            batch_size, self.window_size, self.features_per_step,
            device=device, dtype=torch.float32,
        )

    def update(
        self,
        contact_flag: torch.Tensor,
        raw_tactile: torch.Tensor,
    ):
        """
        Append a new timestep to the buffer.

        Args:
            contact_flag: (B,) contact gate values.
            raw_tactile: (B, rows, cols) raw tactile readings.
        """
        B = contact_flag.shape[0]
        device = contact_flag.device

        if self._buffer is None or self._buffer.shape[0] != B:
            self.reset(B, device)

        # Compute summary statistics
        if raw_tactile.dim() == 3:
            flat = raw_tactile.flatten(1)  # (B, N_taxels)
        else:
            flat = raw_tactile

        mean_force = flat.mean(dim=-1)  # (B,)
        peak_shear = flat.max(dim=-1).values  # (B,) (proxy for peak shear)

        # New entry: (contact_flag, mean_force, peak_shear)
        new_entry = torch.stack([contact_flag, mean_force, peak_shear], dim=-1)  # (B, 3)

        # Shift buffer left and append
        self._buffer = torch.cat([
            self._buffer[:, 1:, :],
            new_entry.unsqueeze(1),
        ], dim=1)

    def get_buffer(self) -> torch.Tensor:
        """
        Get the current buffer contents.

        Returns:
            buffer: (B, W, features_per_step) raw tactile history.
        """
        if self._buffer is None:
            raise RuntimeError("Buffer not initialized. Call reset() first.")
        return self._buffer


class TactileBufferTokenizer(nn.Module):
    """
    Converts the raw tactile buffer into a small set of summary tokens
    for the CoT reasoning head.

    Uses 1D average pooling to reduce the temporal window into a fixed
    number of summary tokens, then projects to backbone dimension.
    """

    def __init__(
        self,
        window_size: int = 50,
        features_per_step: int = 3,
        num_summary_tokens: int = 10,
        output_dim: int = 2048,
    ):
        """
        Args:
            window_size: Buffer temporal window size.
            features_per_step: Raw features per step.
            num_summary_tokens: Output token count (~10 per §4.7).
            output_dim: Output dimension (backbone hidden_dim).
        """
        super().__init__()
        self.num_summary_tokens = num_summary_tokens
        self.window_size = window_size

        # Each summary token covers window_size / num_summary_tokens steps
        self.pool_kernel = window_size // num_summary_tokens

        # Project pooled features to backbone dim
        self.projection = nn.Sequential(
            nn.Linear(features_per_step, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, buffer: torch.Tensor) -> torch.Tensor:
        """
        Tokenize the raw tactile buffer.

        Args:
            buffer: (B, W, features) from RawTactileBuffer.

        Returns:
            tokens: (B, num_summary_tokens, output_dim) summary tokens.
        """
        B, W, n_feat = buffer.shape

        # 1D average pooling along temporal axis
        # Reshape for avg_pool1d: (B, F, W) — pool over W
        x = buffer.transpose(1, 2)  # (B, F, W)
        x = F.avg_pool1d(x, kernel_size=self.pool_kernel, stride=self.pool_kernel)
        # x: (B, F, num_summary_tokens)
        x = x.transpose(1, 2)  # (B, num_summary_tokens, F)

        # Trim to exact token count (in case of rounding)
        x = x[:, :self.num_summary_tokens, :]

        # Project to backbone dimension
        tokens = self.projection(x)  # (B, num_summary_tokens, output_dim)

        return tokens


class CoTTriggerPolicy:
    """
    Determines when to trigger Chain-of-Thought reasoning.

    Two modes (§4.7):
    1. Fixed-interval: every N action chunks (paper baseline).
    2. Event-triggered: fires early on unexpected contact loss (extension).
       Uses the already-computed contact signal — reuses, doesn't rebuild.
    """

    def __init__(
        self,
        fixed_interval: int = 5,
        event_trigger_enabled: bool = True,
        contact_loss_window: int = 3,
    ):
        """
        Args:
            fixed_interval: N — trigger every N action chunks.
            event_trigger_enabled: Whether to enable event-based triggers.
            contact_loss_window: Frames of unexpected non-contact before triggering.
        """
        self.fixed_interval = fixed_interval
        self.event_trigger_enabled = event_trigger_enabled
        self.contact_loss_window = contact_loss_window

        self._step_counter = 0
        self._contact_history: List[float] = []
        self._expecting_contact = False

    def reset(self):
        """Reset trigger state for a new episode."""
        self._step_counter = 0
        self._contact_history = []
        self._expecting_contact = False

    def should_trigger(
        self,
        contact_flag: float,
        in_contact_phase: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check whether CoT should be triggered this step.

        Args:
            contact_flag: Current contact gate value (scalar, per-sample).
            in_contact_phase: Whether the task is currently in a phase where
                            sustained contact is expected (e.g. mid-grasp).

        Returns:
            trigger: Whether to fire CoT.
            reason: "fixed_interval", "contact_loss", or "none".
        """
        self._step_counter += 1
        self._contact_history.append(contact_flag)
        self._expecting_contact = in_contact_phase

        # Fixed-interval check
        if self._step_counter >= self.fixed_interval:
            self._step_counter = 0
            return True, "fixed_interval"

        # Event trigger: unexpected contact loss
        if self.event_trigger_enabled and self._expecting_contact:
            recent = self._contact_history[-self.contact_loss_window:]
            if len(recent) >= self.contact_loss_window and all(c < 0.5 for c in recent):
                self._step_counter = 0  # reset counter after event trigger
                return True, "contact_loss"

        return False, "none"


class ChainOfThoughtHead(nn.Module):
    """
    CoT reasoning head for CATE-π.

    Reuses the VLM backbone's Gemma decoder in autoregressive text-generation
    mode. Receives the encoded prefix (same as action generation) plus
    additional summary tokens from the raw tactile buffer.

    Generates diagnostic text following Tactile-VLA's prompt structure:
    1. Task completion check
    2. Failure analysis referencing force components
    3. Corrective instruction emission

    Implemented as a SEPARATE forward function from action generation
    (per §5's recommendation — easier to test and debug than a branching
    mode flag in every mask computation).
    """

    def __init__(
        self,
        hidden_dim: int = 2048,
        vocab_size: int = 256000,
        max_cot_tokens: int = 128,
        # Buffer tokenizer
        buffer_window: int = 50,
        buffer_features: int = 3,
        num_buffer_tokens: int = 10,
    ):
        """
        Args:
            hidden_dim: Backbone hidden dimension.
            vocab_size: Vocabulary size for text generation.
            max_cot_tokens: Maximum generated CoT tokens.
            buffer_window: Raw tactile buffer temporal window.
            buffer_features: Features per buffer step.
            num_buffer_tokens: Summary tokens from buffer.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.max_cot_tokens = max_cot_tokens

        # Raw tactile buffer (not a nn.Module — manages its own state)
        self.raw_buffer = RawTactileBuffer(
            window_size=buffer_window,
            features_per_step=buffer_features,
        )

        # Buffer tokenizer
        self.buffer_tokenizer = TactileBufferTokenizer(
            window_size=buffer_window,
            features_per_step=buffer_features,
            num_summary_tokens=num_buffer_tokens,
            output_dim=hidden_dim,
        )

        # CoT trigger policy
        self.trigger_policy = CoTTriggerPolicy()

        # Language model head for autoregressive generation
        # Note: in the full implementation, this shares weights with
        # VLMBackbone.token_embedding (tied embeddings). Here we define
        # it separately for modularity; weight tying happens at init time.
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        # Prompt template tokens (fixed, pre-encoded)
        # These would be tokenized at init time from the prompt strings
        self._cot_prompt_template = [
            "Based on the tactile feedback history, answer the following:",
            "1. Has the task been completed successfully?",
            "2. If not, what should be adjusted? Analyze the force components.",
            "3. Provide a corrective instruction:",
        ]

    def update_buffer(
        self,
        contact_flag: torch.Tensor,
        raw_tactile: torch.Tensor,
    ):
        """
        Update the raw tactile buffer with current step data.
        Called every control step, regardless of whether CoT is triggered.

        Args:
            contact_flag: (B,) contact gate values.
            raw_tactile: (B, rows, cols) raw tactile readings.
        """
        self.raw_buffer.update(contact_flag, raw_tactile)

    def check_trigger(
        self,
        contact_flag: float,
        in_contact_phase: bool = False,
    ) -> Tuple[bool, str]:
        """
        Check whether CoT reasoning should be triggered.

        Args:
            contact_flag: Scalar contact value for this sample.
            in_contact_phase: Whether sustained contact is expected.

        Returns:
            should_trigger: Boolean.
            reason: Trigger reason string.
        """
        return self.trigger_policy.should_trigger(contact_flag, in_contact_phase)

    def generate_reasoning(
        self,
        prefix_hidden: torch.Tensor,
        backbone_layers: nn.ModuleList,
        backbone_norm: nn.Module,
        token_embedding: nn.Embedding,
        prefix_mask: Optional[torch.Tensor] = None,
        prompt_ids: Optional[torch.Tensor] = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[torch.Tensor, str]:
        """
        Generate CoT reasoning text autoregressively.

        This method reuses the backbone's transformer layers in causal mode.
        The prefix is encoded (same as action generation), then text tokens
        are generated autoregressively.

        Args:
            prefix_hidden: (B, N_prefix_orig, D) encoded prefix from backbone.
            backbone_layers: The backbone's nn.ModuleList of TransformerBlocks.
            backbone_norm: The backbone's final RMSNorm.
            token_embedding: The backbone's token embedding table.
            prefix_mask: (B, N_prefix_orig, N_prefix_orig) original gating mask.
            prompt_ids: (1, T_prompt) or (B, T_prompt) tokenized prompt template.
            temperature: Sampling temperature.
            top_p: Nucleus sampling threshold.

        Returns:
            generated_ids: (B, T_gen) generated token IDs.
            corrective_instruction: Decoded text (first sample for logging).
        """
        B = prefix_hidden.shape[0]
        device = prefix_hidden.device

        # Get buffer summary tokens (ungated tactile history)
        buffer_data = self.raw_buffer.get_buffer()
        buffer_tokens = self.buffer_tokenizer(buffer_data)  # (B, N_buf, D)

        # Extend prefix with buffer tokens
        # These go ONLY into the CoT pathway, never into action generation
        extended_prefix = torch.cat([prefix_hidden, buffer_tokens], dim=1)
        N_prefix = extended_prefix.shape[1]

        # Start with a BOS/prompt token
        if prompt_ids is not None:
            if prompt_ids.shape[0] == 1:
                prompt_ids = prompt_ids.expand(B, -1)
            current_ids = prompt_ids.to(device)
        else:
            current_ids = torch.full((B, 1), fill_value=2, device=device, dtype=torch.long)  # BOS token

        generated_ids = []
        finished_mask = torch.zeros((B, 1), dtype=torch.bool, device=device)

        for step in range(self.max_cot_tokens):
            # Embed all current tokens (naive generation, no KV cache)
            token_emb = token_embedding(current_ids)  # (B, T_text, D)
            sequence = torch.cat([extended_prefix, token_emb], dim=1)

            # Forward through backbone layers (causal over suffix)
            x = sequence
            for layer in backbone_layers:
                T = x.shape[1]
                causal_mask = torch.ones(B, T, T, device=device)
                
                # Apply contact-gating key mask to all queries (both prefix and suffix)
                if prefix_mask is not None:
                    # Extract the key mask from the original prefix (B, 1, N_prefix_orig)
                    orig_key_mask = prefix_mask[:, 0:1, :]
                    # Buffer tokens are ungated, so pad their key mask with 1s
                    N_buffer = buffer_tokens.shape[1]
                    buffer_key_mask = torch.ones(B, 1, N_buffer, device=device)
                    full_prefix_key_mask = torch.cat([orig_key_mask, buffer_key_mask], dim=2)
                    
                    causal_mask[:, :, :N_prefix] = full_prefix_key_mask

                # Prefix is bidirectional, Suffix (from N_prefix onward) is causal
                suffix_len = T - N_prefix
                if suffix_len > 0:
                    suffix_causal = torch.tril(
                        torch.ones(suffix_len, suffix_len, device=device)
                    )
                    causal_mask[:, N_prefix:, N_prefix:] = suffix_causal
                x = layer(x, attention_mask=causal_mask)

            x = backbone_norm(x)

            # Get logits for the last position
            last_hidden = x[:, -1, :]  # (B, D)
            logits = self.lm_head(last_hidden)  # (B, vocab)

            # Temperature-scaled sampling with nucleus (top-p)
            logits = logits / temperature
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
            sorted_indices_to_remove[:, 0] = False

            for b in range(B):
                logits[b, sorted_indices[b, sorted_indices_to_remove[b]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # Check for EOS (token ID 1 by convention)
            is_eos = (next_token == 1)
            finished_mask = finished_mask | is_eos
            
            # Mask out tokens generated after EOS (pad with 0)
            next_token = next_token.masked_fill(finished_mask, 0)

            generated_ids.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=-1)

            if finished_mask.all():
                break

        if generated_ids:
            all_ids = torch.cat(generated_ids, dim=-1)  # (B, T_gen)
        else:
            all_ids = torch.zeros(B, 0, device=device, dtype=torch.long)

        return all_ids, ""

    def compute_cot_loss(
        self,
        prefix_hidden: torch.Tensor,
        target_ids: torch.Tensor,
        backbone_layers: nn.ModuleList,
        backbone_norm: nn.Module,
        token_embedding: nn.Embedding,
        prefix_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for CoT fine-tuning (Phase 3).

        Args:
            prefix_hidden: (B, N_prefix_orig, D) encoded prefix.
            target_ids: (B, T_target) ground-truth CoT token IDs.
            backbone_layers: Backbone transformer layers.
            backbone_norm: Backbone final norm.
            token_embedding: Backbone token embeddings.
            prefix_mask: (B, N_prefix_orig, N_prefix_orig) original gating mask.

        Returns:
            loss: Scalar cross-entropy loss.
        """
        B, T = target_ids.shape
        device = target_ids.device

        # Get buffer tokens
        buffer_data = self.raw_buffer.get_buffer()
        buffer_tokens = self.buffer_tokenizer(buffer_data)
        extended_prefix = torch.cat([prefix_hidden, buffer_tokens], dim=1)
        N_prefix = extended_prefix.shape[1]

        # Embed target tokens (teacher forcing)
        target_embs = token_embedding(target_ids)  # (B, T, D)

        # Full sequence: prefix + target tokens
        sequence = torch.cat([extended_prefix, target_embs], dim=1)

        # Forward
        x = sequence
        for layer in backbone_layers:
            T_seq = x.shape[1]
            causal_mask = torch.ones(B, T_seq, T_seq, device=device)
            
            # Apply contact-gating key mask to all queries (both prefix and suffix)
            if prefix_mask is not None:
                orig_key_mask = prefix_mask[:, 0:1, :]
                N_buffer = buffer_tokens.shape[1]
                buffer_key_mask = torch.ones(B, 1, N_buffer, device=device)
                full_prefix_key_mask = torch.cat([orig_key_mask, buffer_key_mask], dim=2)
                causal_mask[:, :, :N_prefix] = full_prefix_key_mask

            suffix_len = T_seq - N_prefix
            if suffix_len > 0:
                suffix_causal = torch.tril(
                    torch.ones(suffix_len, suffix_len, device=device)
                )
                causal_mask[:, N_prefix:, N_prefix:] = suffix_causal
            x = layer(x, attention_mask=causal_mask)
        x = backbone_norm(x)

        # LM head on suffix positions (predict next token)
        suffix_hidden = x[:, N_prefix:-1, :]  # shift: predict token t+1 from position t
        logits = self.lm_head(suffix_hidden)  # (B, T-1, vocab)

        # Cross-entropy loss
        target_shifted = target_ids[:, 1:]  # (B, T-1)
        loss = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            target_shifted.reshape(-1),
            ignore_index=-100,  # padding
        )

        return loss

    def reset_episode(self, batch_size: int, device: torch.device):
        """Reset buffer and trigger state for a new episode."""
        self.raw_buffer.reset(batch_size, device)
        self.trigger_policy.reset()

    @classmethod
    def from_config(cls, config: dict) -> "ChainOfThoughtHead":
        """Create from config dicts."""
        backbone_cfg = config["backbone"]["gemma"]
        buffer_cfg = config.get("cot_buffer", {})
        return cls(
            hidden_dim=backbone_cfg["hidden_dim"],
            vocab_size=backbone_cfg["vocab_size"],
            max_cot_tokens=config.get("language", {}).get("max_token_length", 128),
            buffer_window=buffer_cfg.get("window_size", 50),
            buffer_features=buffer_cfg.get("features_per_step", 3),
            num_buffer_tokens=buffer_cfg.get("num_summary_tokens", 10),
        )
