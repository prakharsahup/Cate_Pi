# CATE-π Implementation Walkthrough

## What Was Built

The complete CATE-π (Contact-Aware Tactile-Expert π-VLA) codebase — **31 files** across **7 packages** — implementing the merged architecture from π0, Tactile-VLA, and TacVLA.

---

## Architecture Diagram

```mermaid
graph TB
    subgraph Inputs["Inputs (per control step)"]
        IMG["📷 RGB Images<br/>(front + wrist cameras)"]
        LANG["📝 Language Instruction"]
        STATE["🦾 Proprioceptive State<br/>(joint angles + gripper)"]
        TAC["🖐️ Tactile Array<br/>(15×8 per contact site)"]
    end

    subgraph Encoders["Modality Encoders"]
        VIS_ENC["VisionEncoder<br/>SigLIP (patch-14, 224²)<br/>→ 512 vision tokens"]
        LANG_ENC["Language Encoder<br/>PaliGemma SentencePiece<br/>→ ≤128 language tokens"]
        STATE_ENC["State Projection<br/>Linear → 1 token"]
        TAC_ENC["SpatioTemporal<br/>TactileEncoder<br/>MLP + 1D Conv/GRU<br/>+ 2D sin-cos pos<br/>→ 36 tactile tokens"]
    end

    subgraph Gating["Contact-Aware Gating (§4.4)"]
        CONTACT["Contact Detector<br/>Hard: threshold counting<br/>Soft: sigmoid MLP"]
        GATE["Double Gate<br/>① Embedding: z̃ = g·z<br/>② Attention mask: M = g·1"]
        BCE["Aux BCE Loss<br/>(soft gate training)"]
    end

    subgraph Backbone["Shared VLM Backbone (§4.5)"]
        PREFIX["Fused Prefix S_t<br/>[z_vis | z_lang | z_state | z̃_tac]"]
        MASK["Block-Sparse<br/>Attention Mask<br/>(non-causal prefix)"]
        GEMMA["Gemma Transformer<br/>18 layers, 2048-dim<br/>(LoRA fine-tuned)<br/>Bidirectional attention"]
        PREFIX_HIDDEN["Encoded Prefix<br/>Hidden States"]
    end

    subgraph ActionMode["Action Generation Mode (every step)"]
        EXPERT["Tactile-Aware Action Expert<br/>8 layers, 512-dim<br/>AdaRMSNorm(τ) conditioning"]
        CFM["Conditional Flow Matching<br/>a_τ = τ·a + (1-τ)·ε<br/>10 Euler steps at inference"]
        ACTION["Augmented Action Chunk<br/>[pos(3) | orient(6) | grip(1)<br/>| F_normal(1) | F_shear(2)]<br/>× 50 steps, padded to 32"]
    end

    subgraph CoTMode["CoT Reasoning Mode (periodic/event)"]
        BUFFER["Raw Tactile Buffer<br/>(UNGATED, always-on)<br/>[(c_i, μ_force, peak_shear)]<br/>→ 10 summary tokens"]
        TRIGGER["Trigger Policy<br/>① Fixed interval (N chunks)<br/>② Event: contact 1→0 loss"]
        COT_GEN["Autoregressive Generation<br/>(same Gemma weights, causal)<br/>\"Force was too low. Retry<br/>with stronger grasp.\""]
        INJECT["Corrective Instruction<br/>→ replaces L_t"]
    end

    subgraph Controller["Hybrid Position-Force Controller (§7)"]
        SPLIT["Split Action Vector<br/>P_target + F_target"]
        HYBRID["Hybrid P/F Control<br/>P_hybrid = P_target + K·ΔF<br/>if ‖ΔF‖ > threshold"]
        PID_EXT["External Force PID<br/>(end-effector → environment)"]
        PID_GRASP["Grasp Force PID<br/>(gripper → object)<br/>DECOUPLED"]
        JOINT_CMD["Joint Commands<br/>→ Simulator"]
    end

    subgraph Sim["Simulation Environments (§9)"]
        MUJOCO["Path B: MuJoCo<br/>Shadow Hand<br/>(92 touch sensors)<br/>Quick prototyping"]
        PYBULLET["Path A: PyBullet<br/>+ TACTO/Taxim<br/>(optical tactile)<br/>High fidelity"]
    end

    %% Input → Encoder connections
    IMG --> VIS_ENC
    LANG --> LANG_ENC
    STATE --> STATE_ENC
    TAC --> TAC_ENC
    TAC --> CONTACT

    %% Gating
    CONTACT --> GATE
    TAC_ENC --> GATE
    GATE --> BCE
    GATE -->|"z̃_tac (gated)"| PREFIX

    %% Backbone
    VIS_ENC --> PREFIX
    LANG_ENC --> PREFIX
    STATE_ENC --> PREFIX
    PREFIX --> MASK
    MASK --> GEMMA
    PREFIX --> GEMMA
    GEMMA --> PREFIX_HIDDEN

    %% Action generation
    PREFIX_HIDDEN --> EXPERT
    GATE -->|"gate value → mask"| EXPERT
    EXPERT --> CFM
    CFM --> ACTION

    %% CoT (shares backbone weights)
    TAC -->|"raw, ungated"| BUFFER
    CONTACT -->|"c_t"| BUFFER
    CONTACT -->|"c_t transitions"| TRIGGER
    TRIGGER -->|"fire?"| COT_GEN
    BUFFER -->|"summary tokens"| COT_GEN
    PREFIX_HIDDEN -->|"same prefix"| COT_GEN
    COT_GEN --> INJECT
    INJECT -.->|"replaces L_t"| LANG

    %% Controller
    ACTION --> SPLIT
    SPLIT --> HYBRID
    HYBRID --> PID_EXT
    HYBRID --> PID_GRASP
    PID_EXT --> JOINT_CMD
    PID_GRASP --> JOINT_CMD

    %% Sim
    JOINT_CMD --> MUJOCO
    JOINT_CMD --> PYBULLET
    MUJOCO -.->|"obs"| Inputs
    PYBULLET -.->|"obs"| Inputs

    %% Styling
    classDef input fill:#4a9eff,stroke:#2563eb,color:#fff
    classDef encoder fill:#8b5cf6,stroke:#6d28d9,color:#fff
    classDef gate fill:#f59e0b,stroke:#d97706,color:#fff
    classDef backbone fill:#10b981,stroke:#059669,color:#fff
    classDef action fill:#ef4444,stroke:#dc2626,color:#fff
    classDef cot fill:#ec4899,stroke:#db2777,color:#fff
    classDef controller fill:#6366f1,stroke:#4f46e5,color:#fff
    classDef sim fill:#64748b,stroke:#475569,color:#fff

    class IMG,LANG,STATE,TAC input
    class VIS_ENC,LANG_ENC,STATE_ENC,TAC_ENC encoder
    class CONTACT,GATE,BCE gate
    class PREFIX,MASK,GEMMA,PREFIX_HIDDEN backbone
    class EXPERT,CFM,ACTION action
    class BUFFER,TRIGGER,COT_GEN,INJECT cot
    class SPLIT,HYBRID,PID_EXT,PID_GRASP,JOINT_CMD controller
    class MUJOCO,PYBULLET sim
```

---

## Data Flow Summary

```mermaid
sequenceDiagram
    participant Sim as Simulator
    participant Enc as Encoders
    participant Gate as Contact Gate
    participant VLM as VLM Backbone
    participant AE as Action Expert
    participant CoT as CoT Head
    participant Ctrl as Controller

    loop Every Control Step
        Sim->>Enc: RGB + State + Tactile
        Enc->>Gate: tactile_tokens + raw_tactile
        Gate->>Gate: compute c_t, apply double gate
        Enc->>VLM: vision + language + state tokens
        Gate->>VLM: gated tactile tokens
        VLM->>VLM: bidirectional prefix encoding
        VLM->>AE: prefix_hidden + state_hidden
        AE->>AE: flow matching (10 Euler steps)
        AE->>Ctrl: [pos, orient, grip, F_normal, F_shear] × 50
        Ctrl->>Ctrl: hybrid P/F with PID
        Ctrl->>Sim: joint commands
        Gate-->>CoT: raw buffer update (ungated)
    end

    Note over CoT: Periodic or event-triggered
    CoT->>CoT: check trigger policy
    alt Triggered
        CoT->>VLM: reuse prefix (causal mode)
        VLM->>CoT: prefix_hidden
        CoT->>CoT: generate diagnostic text
        CoT-->>Enc: corrective instruction → replaces L_t
    end
```

---

## File Structure (31 files, ~200KB)

| Package | File | Description | Lines |
|---------|------|-------------|-------|
| `configs/` | [`backbone_pi0_base.yaml`](file:///p:/Pi-VLA/cate_pi/configs/backbone_pi0_base.yaml) | PaliGemma backbone + action expert + action space | ~85 |
| | [`tactile_sensor.yaml`](file:///p:/Pi-VLA/cate_pi/configs/tactile_sensor.yaml) | Sensor layout, encoder, gate, CoT buffer | ~60 |
| | [`training_phase0.yaml`](file:///p:/Pi-VLA/cate_pi/configs/training_phase0.yaml) | Phase 0: backbone init verification | ~20 |
| | [`training_phase1.yaml`](file:///p:/Pi-VLA/cate_pi/configs/training_phase1.yaml) | Phase 1: gate warm-up | ~50 |
| | [`training_phase2.yaml`](file:///p:/Pi-VLA/cate_pi/configs/training_phase2.yaml) | Phase 2: CFM fine-tuning | ~70 |
| | [`training_phase3.yaml`](file:///p:/Pi-VLA/cate_pi/configs/training_phase3.yaml) | Phase 3: CoT fine-tuning | ~70 |
| **`models/`** | [`attention_masks.py`](file:///p:/Pi-VLA/cate_pi/models/attention_masks.py) | Block-sparse mask: prefix (bidir) + suffix (causal) + contact gating | ~250 |
| | [`vision_encoder.py`](file:///p:/Pi-VLA/cate_pi/models/vision_encoder.py) | SigLIP wrapper, multi-camera, per-camera embeddings | ~160 |
| | [`tactile_encoder.py`](file:///p:/Pi-VLA/cate_pi/models/tactile_encoder.py) | Spatio-temporal: MLP + conv/GRU + 2D sin-cos pos, 36 tokens | ~270 |
| | [`contact_gate.py`](file:///p:/Pi-VLA/cate_pi/models/contact_gate.py) | Hard + soft gate with double gating + aux BCE loss | ~200 |
| | [`vlm_backbone.py`](file:///p:/Pi-VLA/cate_pi/models/vlm_backbone.py) | PaliGemma/Gemma backbone, prefix encoding, state projection | ~260 |
| | [`action_expert.py`](file:///p:/Pi-VLA/cate_pi/models/action_expert.py) | Tactile-aware CFM action expert, Euler ODE, per-channel metrics | ~350 |
| | [`cot_head.py`](file:///p:/Pi-VLA/cate_pi/models/cot_head.py) | CoT reasoning: raw buffer + trigger policy + autoregressive gen | ~400 |
| **`controllers/`** | [`hybrid_position_force.py`](file:///p:/Pi-VLA/cate_pi/controllers/hybrid_position_force.py) | Classical P/F controller, decoupled ext/grasp, PID, safety | ~280 |
| **`data/`** | [`dataset.py`](file:///p:/Pi-VLA/cate_pi/data/dataset.py) | Multi-modal dataset with synthetic fallback | ~260 |
| | [`action_normalization.py`](file:///p:/Pi-VLA/cate_pi/data/action_normalization.py) | Per-channel quantile normalization (prevents force channel loss) | ~200 |
| | [`force_adverb_labeler.py`](file:///p:/Pi-VLA/cate_pi/data/force_adverb_labeler.py) | Auto-labels force→adverb for Tactile-VLA Table 2 experiments | ~240 |
| **`sim/`** | [`pybullet_tacto_env.py`](file:///p:/Pi-VLA/cate_pi/sim/pybullet_tacto_env.py) | Path A: high-fidelity optical tactile + domain randomization | ~350 |
| | [`mujoco_shadowhand_env.py`](file:///p:/Pi-VLA/cate_pi/sim/mujoco_shadowhand_env.py) | Path B: 92 touch sensors, turn-key prototyping | ~270 |
| **`training/`** | [`train_phase1_gate_warmup.py`](file:///p:/Pi-VLA/cate_pi/training/train_phase1_gate_warmup.py) | Phase 1: isolated gate training with F1/gate distribution tracking | ~260 |
| | [`train_phase2_cfm_finetune.py`](file:///p:/Pi-VLA/cate_pi/training/train_phase2_cfm_finetune.py) | Phase 2: end-to-end CFM with per-channel loss monitoring | ~300 |
| | [`train_phase3_cot_finetune.py`](file:///p:/Pi-VLA/cate_pi/training/train_phase3_cot_finetune.py) | Phase 3: CoT on failure/diagnosis data with anti-forgetting mix | ~280 |
| **`eval/`** | [`eval_suite.py`](file:///p:/Pi-VLA/cate_pi/eval/eval_suite.py) | Success rate, force accuracy, latency benchmarking | ~280 |
| | [`ablations.py`](file:///p:/Pi-VLA/cate_pi/eval/ablations.py) | 10-variant ablation table (mirrors TacVLA structure) | ~250 |

---

## Key Design Decisions Implemented

| Decision | Rationale |
|----------|-----------|
| **Double gating** (mask + embedding zero) | TacVLA shows both are needed — masking alone leaks positional offsets, zeroing alone lets attention attend to zero tokens |
| **Soft learnable gate** (sigmoid MLP + BCE) | Addresses TacVLA's stated limitation: hard threshold doesn't handle ambiguous near-contact frames |
| **Raw ungated CoT buffer** | Gating blinds the reasoning pathway to contact→non-contact transitions needed for slip detection |
| **Per-channel quantile normalization** | Without it, position (meters) dominates force (Newtons) in CFM loss — force channels get ignored |
| **Decoupled external/grasp force** | Physically distinct quantities (pressing on environment vs. squeezing object) — conflating them is a common bug |
| **Event-triggered CoT** | Catches slips faster than fixed-interval by detecting unexpected contact loss mid-grasp |
| **Separate action/CoT forward paths** | §5 recommendation: easier to test and debug than one function with mode branching through mask logic |

---

## Training Pipeline

```mermaid
graph LR
    P0["Phase 0<br/>Load π0 checkpoint<br/>Freeze all<br/>Verify forward pass"]
    P1["Phase 1<br/>Train tactile encoder<br/>+ contact gate<br/>Contact classification"]
    P2["Phase 2<br/>End-to-end CFM<br/>LoRA backbone<br/>[pos + force] actions"]
    P3["Phase 3<br/>CoT fine-tuning<br/>Freeze action expert<br/>Failure diagnosis data"]

    P0 --> P1 --> P2 --> P3

    style P0 fill:#94a3b8,stroke:#64748b,color:#fff
    style P1 fill:#f59e0b,stroke:#d97706,color:#fff
    style P2 fill:#ef4444,stroke:#dc2626,color:#fff
    style P3 fill:#ec4899,stroke:#db2777,color:#fff
```
