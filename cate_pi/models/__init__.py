"""
CATE-π: Contact-Aware Tactile-Expert VLA

A merged architecture combining π0's PaliGemma backbone + flow-matching,
Tactile-VLA's Tactile-Aware Action Expert, and TacVLA's contact-aware gating.
"""

from cate_pi.models.vision_encoder import VisionEncoder
from cate_pi.models.tactile_encoder import SpatioTemporalTactileEncoder
from cate_pi.models.contact_gate import ContactGate, HardContactGate, SoftContactGate
from cate_pi.models.vlm_backbone import VLMBackbone
from cate_pi.models.action_expert import TactileAwareActionExpert
from cate_pi.models.cot_head import ChainOfThoughtHead
from cate_pi.models.attention_masks import build_prefix_mask, build_suffix_mask, build_full_cate_mask

__all__ = [
    "VisionEncoder",
    "SpatioTemporalTactileEncoder",
    "ContactGate",
    "HardContactGate",
    "SoftContactGate",
    "VLMBackbone",
    "TactileAwareActionExpert",
    "ChainOfThoughtHead",
    "build_prefix_mask",
    "build_suffix_mask",
    "build_full_cate_mask",
]
