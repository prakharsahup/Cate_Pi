# CATE-π: Contact-Aware Tactile-Expert VLA
### A merged architecture combining LeRobot's π0, Tactile-VLA, and TacVLA — full implementation plan

---

## 0. TL;DR

You are building a VLA that:

1. Uses **π0's** PaliGemma-backbone + flow-matching action-expert scaffold as the structural chassis.
2. Replaces π0's plain action expert with **Tactile-VLA's Tactile-Aware Action Expert**, which predicts a joint `[position, force]` action vector, plus **Tactile-VLA-CoT's** periodic chain-of-thought reasoning loop for failure recovery.
3. Wraps the tactile token stream in **TacVLA's contact-aware gating**, so tactile tokens only participate in cross-attention when physical contact is actually detected — this is what fixes the "always-on tactile" inefficiency that TacVLA's own ablation shows hurts Tactile-VLA-style unconditional fusion.

Working name: **CATE-π** (Contact-Aware Tactile-Expert π-VLA). Rename freely — nothing below depends on the name.

Target deployment per your stated goal: a robotic hand/gripper in simulation. Section 9 gives concrete simulator choices that match your existing PyBullet experience from the CAIR/ANVESH work.

---

## 1. Provenance and design rationale

| Component | Source | What you take from it |
|---|---|---|
| Backbone + flow-matching action generation | π0 (Black et al., 2024), LeRobot `PI0Policy` | PaliGemma (SigLIP + Gemma) VLM trunk, separate smaller-width "expert" transformer for actions, conditional flow matching (CFM) objective, block-wise attention structure, action chunking |
| Tactile-Aware Action Expert + Tactile-VLA-CoT | Tactile-VLA (Huang et al., 2025, arXiv:2507.09160) | Augmenting the action vector with an explicit force target alongside position, a hybrid position-force controller, and a chain-of-thought variant that uses the VLM's own decoder to diagnose tactile failures and re-plan |
| Contact-aware Gating | TacVLA (Zhang et al., 2026, arXiv:2603.12665) | Binary contact-flag gating that zeroes/masks tactile tokens out of cross-attention when no contact is detected, keeping a fixed token topology |

### Why merge these three specifically

**Tactile-VLA's weak point is exactly what TacVLA fixes.** Tactile-VLA fuses a single tactile token into the prefix unconditionally at every timestep, whether or not the gripper is touching anything. TacVLA's own ablation (`Finetuned Pi0.5 + Tactile (w/o Gating)`) shows this unconditional fusion measurably *hurts* performance relative to a vision-only baseline on at least one task (Task 3: 60% vs. 65% for vision-only) — the tactile tokens interfere with visual localization before contact is established. Gating the tactile pathway by a contact flag removes that interference while preserving the force-grounding benefits Tactile-VLA demonstrates.

**TacVLA's weak point is exactly what Tactile-VLA-CoT adds.** TacVLA has no notion of *why* an interaction failed — it's a reactive policy. Tactile-VLA-CoT adds an explicit reasoning loop: the VLM's own language decoder periodically inspects the tactile/force history, diagnoses failure causes ("shear force should be larger"), and emits a corrective sub-instruction. TacVLA's discussion section explicitly flags this as future work ("extending state-dependent modality routing to longer-horizon... task settings"); Tactile-VLA-CoT is a working solution to exactly that gap.

**π0 is the substrate that makes both integrations tractable.** Tactile-VLA is itself initialized from π0's pretrained checkpoint, and TacVLA is initialized from π0.5's checkpoint and fine-tuned with LoRA. Both downstream papers already assume the π0 scaffold — so building on LeRobot's `PI0Policy` implementation directly is not an approximation, it's the natural common ancestor of both.

### Two things this plan adds that are *not* in any of the three papers (clearly flagged as your own design choices, not replicated results)

1. **Soft, learnable gate as an option alongside the hard threshold gate.** TacVLA's own discussion section names its binary threshold heuristic as a limitation ("does not allow gradual or learnable adjustment of modality importance"). Section 4.4 gives you both: the paper-faithful hard gate (for a clean reproduction/ablation baseline) and a learnable sigmoid gate trained with an auxiliary contact-classification loss (for the "perform really well" version).
2. **A raw, ungated tactile history buffer feeding the CoT head.** If you gate tactile tokens out during non-contact phases for the *action* pathway, you must not also blind the *reasoning* pathway — detecting a slip or a failed grasp often requires seeing the *transition* from contact to non-contact, which a purely gated stream would hide. Section 4.7 specifies a small always-on summary buffer dedicated to this.

---

## 2. Notation

| Symbol | Meaning |
|---|---|
| `t` | control timestep |
| `H` | action chunk horizon (π0 default: 50) |
| `I_t` | RGB frame(s) at time t (multi-view) |
| `L_t` | language instruction tokens |
| `q_t` | proprioceptive/robot state |
| `X_t` | raw tactile sensor reading (spatial array) at time t |
| `c_t` | binary contact flag, c_t ∈ {0,1} |
| `g_t` | soft gate value, g_t ∈ [0,1] (your extension) |
| `z_vis, z_lang, z_state, z_tac` | token embeddings per modality |
| `S_t` | fused prefix token sequence |
| `a_t` | augmented action vector: position/orientation/gripper + force channels |
| `τ` | flow-matching integration timestep, τ ∈ [0,1] |
| `v_θ` | learned vector field (flow matching) |
| `P_target, F_target` | action-expert outputs: target pose, target contact force |

---

## 3. System I/O contract

**Inputs per control step:**
- 1–2 RGB images (front-facing + wrist camera, matching TacVLA's setup)
- Language instruction string (persists across steps until CoT overwrites it)
- Proprioceptive state vector (joint angles + gripper/finger state)
- Tactile array reading, shape `[rows × cols]` per contact site (e.g. 15×8 per finger, matching TacVLA's sensor)

**Outputs:**
- An action chunk `a_{t:t+H}` of augmented actions (position + orientation + gripper + force targets)
- A binary/soft contact signal `c_t` (also logged — useful for eval and for triggering CoT)
- Periodically (not every step): a natural-language diagnostic + corrective instruction string

---

## 4. Module-by-module specification

### 4.1 Vision encoder

- Use SigLIP (the vision half of PaliGemma) exactly as π0 does — do not build a new vision tower.
- Patch size 14, resolution 224×224 per camera → 256 tokens/image. Two cameras (front + wrist) → 512 vision tokens before any pooling.
- If you need to cut token count for compute reasons, reduce resolution before you reduce cameras — wrist-camera visibility is what most contact-rich tasks rely on when the gripper occludes the front view.
- Optional (stretch): stack the last `H_vis` frames per Tactile-VLA's convention if you want the vision stream itself to carry short-horizon dynamics (e.g. approach velocity). Start with single-frame-per-camera; only add temporal stacking if your ablations show it's needed — it directly multiplies token count and compute.

### 4.2 Language + proprioceptive state encoder

- Reuse PaliGemma's SentencePiece tokenizer and embedding table for language tokens — this is what lets you initialize from a pretrained checkpoint at all.
- Max token length: 48 is π0's default; raise this to ~96–128 if you're using Tactile-VLA-CoT, since corrective instructions injected back into the prompt add length.
- Proprioceptive state: follow π0's continuous-state convention — a single linear-projected token, not a discretized/tokenized state (π0.5 supports a discrete variant; stick with continuous for the higher control frequency it enables).

### 4.3 Tactile encoder (spatio-temporal)

Neither source paper's tactile encoder alone is enough: Tactile-VLA's MLP collapses an entire temporal window into *one* token (good for dynamics, loses spatial structure); TacVLA's MLP keeps spatial structure via 36 tokens with 2D sin-cos positions but only looks at the current frame (good for geometry, loses dynamics). Combine both:

```
for each timestep i in [t-K+1, ..., t]:
    e_i = shared_MLP(taxel_grid_i)              # per-taxel or patch-level embed
proj = 1D temporal conv (or small GRU) over {e_i}_{i=t-K+1..t}
tactile_tokens = flatten(proj) + sincos_2d_pos_embed   # TacVLA-style spatial tokens
tactile_summary_token = mean_pool(tactile_tokens)      # optional single-token digest
```

- `K` (temporal window) 4–8 steps is a reasonable start, matching Tactile-VLA's "last H tactile measurements" framing at 100 Hz-ish sensor rates downsampled to control rate.
- Keep the total tactile token count modest (TacVLA's 36 tokens is a good reference point) — this is the multimodal-fusion tax on every forward pass, and it's the exact quantity the gate exists to zero out.

### 4.4 Contact detector and gate

**Contact detection (paper-faithful, hard):**
```
c_t = 1  if  count(taxels where pressure_i > p_thresh) > n_thresh
     else 0
```
Threshold both `p_thresh` and `n_thresh` from a small calibration set of known-contact vs. known-no-contact frames before you touch the policy — don't hand-pick these mid-training.

**Gating, applied at two levels (TacVLA's design, keep both):**
```
M_tac = c_t · 1                 # attention-mask gate: excludes tactile tokens from attention when c_t = 0
z̃_tac = c_t · z_tac             # embedding gate: zeroes the token values themselves
```
This double application matters: masking alone still lets zero-but-nonzero positional/embedding offsets leak in; zeroing alone still lets a naive implementation attend to an all-zero-but-present token. Do both, as TacVLA does.

**Your extension — learnable soft gate (recommended for the "perform well" build, keep the hard gate available as an ablation arm):**
```
g_t = sigmoid(MLP(mean_pool(z_tac)))         # small 2-layer MLP, scalar output
z̃_tac = g_t · z_tac
L_gate = BCE(g_t, c_t)                        # heuristic c_t used as a weak label, not ground truth
```
Train `g_t` with the auxiliary BCE loss against the heuristic contact flag, but let it flow gradients from the main CFM loss too, so the network can learn to keep tactile influence partially active in genuinely ambiguous near-contact frames rather than the hard on/off of the paper version. This directly answers TacVLA's own stated limitation.

### 4.5 Shared VLM backbone

- PaliGemma init: SigLIP + Gemma decoder, loaded from a pretrained π0/π0.5 checkpoint (`lerobot/pi0_base` or `lerobot/pi05_libero_base` on the Hub are reasonable starting points).
- Attention over the fused prefix `S_t = [z_vis, z_lang, z_state, z̃_tac]` is **non-causal / fully bidirectional** — every source paper agrees on this for the prefix. This is what lets vision, language, state, and (when active) tactile tokens cross-attend freely.
- Model size: the papers report ~2–2.6B for the Gemma component. If your GPU budget doesn't comfortably fit that plus an action expert plus gradients, two escape hatches, both legitimate: (a) LoRA-finetune the backbone and only fully finetune the small modules (TacVLA's own approach — LoRA on top of a `pi05_base` checkpoint), or (b) drop to a smaller backbone variant (e.g. a SmolVLA-scale ~500M-parameter VLA) and accept a capability/generalization trade-off. Given you're prototyping on a single robotic hand in simulation rather than chasing broad generalist coverage, (a) is the better default — full LoRA fine-tuning is exactly what lets a student-scale compute budget still touch a multi-billion-parameter pretrained prior.

### 4.6 Tactile-Aware Action Expert (flow-matching head)

This is architecturally π0's "Expert Gemma" — a second transformer sharing the same per-layer block structure as the main backbone but with a narrower hidden width (Tactile-VLA reports ~300M parameters for this module; π0's action expert is sized similarly) — retargeted to output the augmented `[position, force]` vector instead of pure kinematics.

**Forward pass (single denoising step):**
```
a_τ = τ · a + (1-τ) · ε                     # linear interpolation, ε ~ N(0, I)
h = ExpertTransformer(
        prefix_kv = KV(S_t)  from backbone,       # block-causal: state token + action tokens attend to full prefix
        suffix    = [state_token, a_τ tokens],
        time_cond = AdaRMSNorm(τ)                 # inject flow-matching timestep
    )
v_θ = h                                      # predicted vector field
```

**Attention structure inside the suffix** (this is the subtle part — see Section 5 for the full mask):
- The state token attends to the full prefix and to earlier state tokens causally.
- Action tokens (including the force channels) have full bidirectional visibility over the entire non-padded prefix, the state token(s), and each other — they do *not* attend causally to one another. This matters because force and position at a given chunk-step are jointly denoised, and the two should be free to condition on each other within a step.

**Training objective (Conditional Flow Matching):**
```
L_CFM = E_{τ~Beta, ε~N(0,I), (o,a)~D} [ || v_θ(a_τ, τ, o) - (a - ε) ||² ]
```
- `τ` sampled from a Beta distribution skewed toward low (noisier) values, matching π0's convention.
- **Critical detail Tactile-VLA is explicit about:** the loss "penalizes deviations in both the kinematic and force dimensions." Position (meters) and force (Newtons) live on wildly different numeric scales — if you don't normalize per-channel before computing the loss, the position channels will dominate the gradient and the model will effectively ignore force. Quantile-normalize each action dimension independently to `[-1, 1]` (same idea as π0-FAST's normalization step) before flow matching, and store the per-dimension normalization stats alongside the checkpoint.

**Inference:** integrate the learned ODE from `τ=0` to `τ=1` with ~10 Euler steps (π0's convention), execute the resulting chunk (or a receding-horizon prefix of it) at your control rate, then re-query.

### 4.7 Chain-of-thought reasoning head (Tactile-VLA-CoT, gate-aware)

This reuses the *same* Gemma decoder that forms the VLM backbone, just in its native autoregressive text-generation mode instead of the bidirectional-prefix-plus-flow-matching-suffix mode. Your model therefore has **two forward modes** sharing one set of backbone weights:

| Mode | Suffix content | Attention over suffix | Trigger |
|---|---|---|---|
| Action generation | noisy action tokens + time embedding | block-causal, bidirectional within the action block | every control step |
| CoT reasoning | free-form text tokens | standard causal/autoregressive | periodic (see below) |

**Trigger policy** (extend Tactile-VLA-CoT's "fixed interval" with an event-triggered option, since you already have the contact signal computed for the gate — reuse it, don't build a second detector):
- Fixed-interval trigger: every `N` action chunks (paper's approach — simple, reliable baseline).
- Event trigger (your addition, cheap to add since `c_t` already exists): fire early if `c_t` transitions 1→0 unexpectedly during a phase where sustained contact was expected (e.g. mid-wipe, mid-grasp) — this catches slips faster than waiting for the next fixed interval.

**Prompt structure for the CoT turn** (follow Tactile-VLA's structure exactly — it's simple and it works):
1. "Has the task been completed successfully?" → yes/no
2. If no: "What should be adjusted?" → analysis referencing specific force components, grounded in the tactile/force history
3. Emit a corrective instruction string, which **replaces** `L_t` for subsequent control steps until the next CoT turn.

**The raw-buffer detail (your addition, and the reason it matters):** the main action-generation pathway only ever sees *gated* tactile tokens — by design, they're zeroed during non-contact. But diagnosing "the wipe failed because normal force was too low" or "the object slipped" often requires seeing the force *trend*, including the moment contact was lost. So maintain a small, ungated rolling buffer:
```
buffer_t = [(c_i, mean_force_i, peak_shear_i) for i in t-W..t]   # W ~ a few seconds worth of steps
```
Pool this into a handful of extra summary tokens (e.g. 10, via simple binning/1D pooling) and feed them into the language decoder *only* during CoT turns — never into the per-step action-generation prefix. This keeps the action pathway's efficiency gains from gating intact while giving the reasoning pathway the history it actually needs.

**Fine-tuning data for this head:** a small, targeted dataset pairing failure trajectories with textual failure-cause annotations (Tactile-VLA used demonstrations like wiping with insufficient force, paired with "the force was too light, a stronger force is needed"). Mix in a small fraction of the model's original VQA/instruction-following data during this phase — Tactile-VLA's paper states this dual purpose explicitly: it both preserves general reasoning and extends it to tactile signals, mitigating catastrophic forgetting of language ability.

---

## 5. The unified attention mask

This is the single most error-prone part of the implementation, because you're merging three different mask conventions into one block-sparse structure. Token order and mask logic, end to end:

```
[ vision tokens | language tokens | state token | tactile tokens (gated) ]  <- PREFIX, non-causal
                                                                              (all four groups fully cross-attend to
                                                                               each other; tactile block is masked
                                                                               out entirely when c_t = 0, per §4.4)
                        ↓ (prefix hidden states / KV cache)
[ state token (suffix copy) | noisy action+force tokens (τ) ]               <- SUFFIX, block-causal
   - state token: attends to full prefix + earlier state tokens (causal across state history)
   - action tokens: attend to full prefix + state token(s) + each other, bidirectionally within the action block
   - action tokens never attend to future/other denoising-step copies
```

Implementation note straight from the π0/LeRobot team's own experience report: a naive dense attention implementation over this mask is correct but wasteful, because the mask is heavily block-sparse. If you're implementing from scratch, don't hand-roll a `score_mod` addition (it defangs kernel fusion); precompute the block mask and pass it to PyTorch's `flex_attention` with `create_block_mask`, or start simpler with an explicit boolean mask and eager attention until correctness is verified, then optimize. Get correctness first — a subtly wrong mask (e.g. letting non-contact tactile tokens leak into attention through the state-token path) will silently degrade the whole point of the gating mechanism and is very hard to catch from loss curves alone. Write a unit test that forces `c_t=0` and asserts the tactile block contributes exactly zero gradient to every other token's output.

When you switch into **CoT mode**, the suffix changes shape entirely (free-form causal text tokens replace the action+time suffix) — implement this as a genuinely separate forward function that reuses the same prefix-encoding code path and backbone weights, not as a single mega-function with branching mask logic. It will be much easier to test and debug as two paths that share a component than one path with a mode flag threaded through every mask computation.

---

## 6. Augmented action space definition

Define your action vector explicitly before writing any code — this is the piece most tutorials hand-wave and most implementation bugs hide in.

**For a parallel-jaw gripper (simplest case):**
```
a_t = [Δx, Δy, Δz,                # Cartesian position delta (3)
       roll, pitch, yaw or 6D rot,# orientation (3 or 6)
       gripper_width,             # (1)
       F_normal_target,           # target grasp/contact normal force (1)
       F_shear_x, F_shear_y]      # target shear force components (2)
```

**For a multi-finger dexterous hand (e.g. Allegro/Shadow-scale, N finger joints):**
```
a_t = [Δpose(6 or 7 for wrist),
       q_finger_1 ... q_finger_N,      # per-joint targets
       F_normal_1 ... F_normal_M,      # per-contact-site (per fingertip) target normal force
       F_shear_1  ... F_shear_M]       # per-contact-site target shear force
```

- Pad the vector to a fixed maximum dimension (π0 pads to 32 across embodiments) and zero-mask unused channels — this is what keeps the checkpoint reusable if you later add a second embodiment/hand.
- Log both the *raw* and the *quantile-normalized* action stats at data-collection time; you'll need the raw units (Newtons, meters) for the low-level controller and the normalized units for the flow-matching loss (§4.6).

---

## 7. Hybrid position-force controller

This is the low-level layer that turns the action expert's `(P_target, F_target)` output into actual joint commands — it is *not* learned, it's a classical controller, and it's what makes the force predictions actually matter physically rather than being a training signal with no execution consequence.

```
ΔF = F_target - F_measured

Phybrid = P_target + K·ΔF   if ||ΔF|| > τ_threshold
        = P_target          otherwise
```
- `K` is a gain matrix (tune per-axis; start conservative to avoid oscillation, especially in sim where contact dynamics can be stiffer than real gripper compliance).
- `τ_threshold` exists purely to avoid chattering — without it, small measurement noise causes constant tiny corrective moves. Tune this against your simulator's tactile noise floor.
- A PID loop then actually drives the joints to `Phybrid`.
- **Decouple two force channels onto two independent actuation axes**, exactly as Tactile-VLA does: Cartesian end-effector position regulates *net external* force (how hard you're pressing on the environment), while gripper width / finger closure regulates *internal grasp* force (how hard you're squeezing the object). Conflating these into one control channel is a common bug — they are physically distinct quantities.

---

## 8. Training curriculum

Four phases, going from cheap-to-verify to expensive-and-fragile. Don't skip straight to phase 3 — each earlier phase exists to catch a class of bug before it's buried under a much more expensive training run.

**Phase 0 — Backbone initialization.**
Load a pretrained π0/π0.5 checkpoint (LeRobot Hub). Freeze everything. Verify you can run a forward pass and reproduce the checkpoint's reported behavior on a known eval task before touching anything tactile. This is your regression baseline.

**Phase 1 — Tactile/gate warm-up (new modules only).**
Freeze the VLM backbone and action expert entirely. Train *only* the tactile encoder + contact detector/gate (§4.3–4.4) on an auxiliary contact-classification task: predict `c_t` (or a continuous force-magnitude proxy) directly from `X_t`, supervised by your calibrated heuristic labels. This is cheap, fast to iterate on, and lets you validate your tactile pipeline and simulator's tactile signal quality before it's entangled with the much harder joint optimization. Not in any source paper — it's a practical scaffolding step that pays for itself in debugging time.

**Phase 2 — End-to-end CFM fine-tuning.**
Unfreeze the tactile encoder, gate, and Tactile-Aware Action Expert; LoRA-adapt (don't fully unfreeze, for compute reasons — see §4.5) the shared VLM backbone. Train on the CFM loss (§4.6) over demonstration data with the augmented `[position, force]` action space. This is the phase that gives you Tactile-VLA's core capability: force-grounded, tactile-conditioned action generation.

**Phase 3 — CoT fine-tuning.**
Freeze the action expert (protect the control skill you just trained). LoRA-adapt the language decoder head on your small targeted failure/diagnosis dataset (§4.7), mixed with a small ratio of general instruction-following data to guard against catastrophic forgetting of language ability — this mix is explicit in Tactile-VLA's own recipe, not optional.

**Phase 4 (stretch, optional) — Closed-loop fine-tuning.**
If you have simulation budget left: DAgger-style or on-policy fine-tuning to address compounding error from open-loop chunk execution, especially around the CoT-triggered replanning boundary (the moment right after a corrective instruction is issued is exactly where distribution shift from your demonstration data is largest).

---

## 9. Data collection and simulation setup

You stated you want this on a robotic hand/gripper **in simulation** — two concrete, compatible-with-your-background paths:

**Path A — PyBullet + TACTO/Taxim (recommended given your existing PyBullet fluency from the CAIR/ANVESH work).**
- [TACTO](https://github.com/facebookresearch/tacto) simulates DIGIT-style optical tactile sensors by bridging PyBullet contact physics to a `pyrender` optical model — it was built specifically to plug into PyBullet scenes.
- [Taxim](https://github.com/CMURoboTouch/Taxim) is a data-driven alternative (calibrated from <100 real contact examples) also integrated with PyBullet, targeting GelSight-style sensors, and reportedly captures intrinsic sensor noise better than TACTO's purely physics-based rendering.
- Either gives you spatially-resolved tactile images/taxel grids that map naturally onto the TacVLA-style tokenization in §4.3. Mount the sensor on a simple parallel gripper (e.g. a Robotiq 2F-85 or WSG-50 model) to start — add a second/third finger only once the single-contact-site pipeline works end to end.

**Path B — MuJoCo + Shadow Dexterous Hand (Gymnasium-Robotics), for faster turn-key prototyping.**
- `HandManipulateBlockTouchSensors` ships with 92 built-in MuJoCo touch sensors distributed across the palm and finger phalanges — lower spatial fidelity than an optical tactile sim, but zero sensor-integration work, and a real dexterous hand model out of the box.
- Good for validating the *architecture and training pipeline* fast, before investing in a higher-fidelity tactile sensor sim.

**Recommended sequencing:** stand up Path B first to get the full pipeline (encoders → gate → backbone → action expert → controller → CoT) running end-to-end with minimal simulator friction, then port to Path A once the architecture is verified, since Path A's higher-fidelity tactile signal is what will actually let you demonstrate Tactile-VLA-style force-language grounding (e.g. "grasp gently" vs. "grasp hard") convincingly.

**Demonstration collection in sim:** you don't need real teleoperation. Two options:
- Scripted/heuristic policies with domain randomization (friction, mass, object geometry) — fast to generate large datasets, but demonstrations won't naturally vary force the way a human would.
- A lightweight RL-trained expert per task, distilled into demonstrations via behavioral cloning — closer to your CAIR internship's behavioral-cloning experience, and lets you *programmatically* vary target force by shaping the RL reward, which solves the "how do I get force-varied demonstrations without a human" problem directly.

**Auto-labeling force-adverb language (this is what makes Tactile-VLA's core experiment reproducible without manual annotation):** since the simulator gives you ground-truth applied force, bin the measured peak contact force during each demonstration into an adverb vocabulary programmatically, e.g. `<1N → "gently"`, `1–3N → "normally"`, `>3N → "firmly"/"hard"`, and synthesize the instruction string from a small template set. This reproduces Tactile-VLA's Table 2 experiment (learned-word generalization to unseen adverbs and unseen tasks) without any manual force-language labeling.

---

## 10. Repository layout

```
cate_pi/
  configs/
    backbone_pi0_base.yaml
    tactile_sensor.yaml
    training_phase{0,1,2,3}.yaml
  models/
    vision_encoder.py          # SigLIP wrapper
    tactile_encoder.py         # spatio-temporal tactile encoder, §4.3
    contact_gate.py            # hard + soft gate, §4.4
    vlm_backbone.py            # PaliGemma/Gemma wrapper, prefix encoding
    action_expert.py           # Tactile-Aware Action Expert, flow matching, §4.6
    cot_head.py                # dual-mode decoding, raw tactile buffer, §4.7
    attention_masks.py         # unified block-sparse mask builder, §5
  controllers/
    hybrid_position_force.py   # §7, NOT a learned module
  data/
    dataset.py
    action_normalization.py    # per-channel quantile stats
    force_adverb_labeler.py    # auto-labeling, §9
  sim/
    pybullet_tacto_env.py      # Path A
    mujoco_shadowhand_env.py   # Path B
  training/
    train_phase1_gate_warmup.py
    train_phase2_cfm_finetune.py
    train_phase3_cot_finetune.py
  eval/
    eval_suite.py              # §11
    ablations.py
```

---

## 11. Evaluation protocol and ablations

**Core metrics, per task:**
- Success rate (%)
- Applied-force accuracy: correlation between commanded force adverb and measured peak force (reproduces Tactile-VLA's Table 2 methodology)
- Generalization: held-out objects/materials (randomize friction, mass in sim to create OOD test conditions, mirroring Tactile-VLA's ID/OOD object split)
- Robustness: performance under simulated camera occlusion (mirrors TacVLA's block-camera test) and under injected tactile-sensor dropout/noise

**Ablation table to run (mirrors TacVLA's own ablation structure — this is also your strongest evidence for a paper if you pursue publication):**

| Variant | What it isolates |
|---|---|
| Full CATE-π | — |
| − contact gating (tactile always on) | value of §4.4 |
| hard gate vs. soft learnable gate | value of your extension |
| − CoT head | value of §4.7's reasoning loop |
| fixed-interval CoT vs. event-triggered CoT | value of your event-trigger extension |
| − force channels (position-only action space) | value of the tactile-aware action expert itself |
| gated CoT buffer vs. raw always-on buffer (§4.7) | value of the raw-buffer design decision |

**Latency/compute check:** flow matching's ~10-step iterative denoising is not free — benchmark end-to-end control-loop latency (encoders → gate → backbone → 10 Euler steps → controller) against your target control frequency *before* committing to a chunk size or backbone scale. This is the kind of thing that's easy to discover too late.

---

## 12. Known failure modes and mitigations

- **Force channels get ignored by the loss.** Symptom: model predicts near-constant force regardless of language. Cause: unnormalized loss scale (§4.6). Fix: verify per-channel normalization stats before any long training run — plot the normalized-loss contribution per channel early.
- **Gate collapses to always-on or always-off.** Symptom: soft gate `g_t` saturates near 0 or 1 for all inputs. Cause: BCE auxiliary loss weight too high/low relative to the main CFM loss, or contact-label imbalance in your data. Fix: log `g_t` distribution during training; balance contact/no-contact frames in your demonstration data explicitly.
- **CoT forgets general reasoning.** Symptom: reasoning outputs become degenerate/templated after Phase 3. Cause: skipped the general-data mixing step. Fix: don't skip it — it's explicit in Tactile-VLA's recipe for exactly this reason.
- **Silent mask bugs.** Symptom: gating "works" per the loss curve but ablation shows no difference vs. no-gating. Cause: a leak in the block-sparse mask (§5) letting gated-out tactile tokens still influence outputs through an indirect path. Fix: the unit test described in §5 — make it a CI check, not a one-time verification.
- **Compounding error after CoT replanning.** Symptom: the model recovers from failure in training-distribution replays but not in longer rollouts. Cause: the state right after a corrective instruction is injected is out-of-distribution relative to normal demonstrations. Fix: Phase 4 closed-loop fine-tuning, or at minimum, include some replanning-boundary states in your Phase 2/3 training data if you can generate them in sim.

---

## 13. Implementation roadmap

1. **Week 1–2:** Path B sim environment (MuJoCo/Shadow Hand) + scripted demonstration collection + action normalization pipeline. No tactile fusion yet — get a plain π0-style position-only policy training and executing in sim.
2. **Week 2–3:** Tactile encoder + hard contact gate (§4.3–4.4, hard version only). Phase 1 warm-up training. Validate contact classification accuracy in isolation.
3. **Week 3–5:** Tactile-Aware Action Expert with augmented `[position, force]` action space (§4.6) + hybrid controller (§7). Phase 2 training. This is the point where you should be able to reproduce a Tactile-VLA-style force-language experiment (§9's auto-labeling).
4. **Week 5–6:** Soft learnable gate variant + ablation harness (§11). This is your first publishable comparison table.
5. **Week 6–8:** CoT head (§4.7), including the raw tactile buffer and event-triggered variant. Phase 3 training on a small synthetic failure dataset generated in sim (scripted failure injection, e.g. deliberately under-applying force, paired with auto-generated diagnostic text).
6. **Week 8+:** Port to Path A (PyBullet + TACTO/Taxim) for higher-fidelity tactile signal; re-run the full ablation suite; robustness evaluation (occlusion, sensor dropout).

---

## 14. Positioning for publication

Given you're building toward a research profile, the honest framing of this work's contribution is: **not a new component, but a demonstrated synergy between three existing components that individually have complementary, named weaknesses** — Tactile-VLA's own paper doesn't gate, TacVLA's own paper doesn't reason, and neither paper's ablations test the *combination*. Your strongest result, if the ablations in §11 come out as hypothesized, is showing that contact-gating recovers the efficiency TacVLA argues for *without* sacrificing the CoT-based recovery capability Tactile-VLA argues for, plus the soft-gate and event-triggered-CoT extensions as incremental but concrete improvements over both papers' explicitly-stated limitations. Frame the paper's related-work section around exactly the "what's missing from each" argument in Section 1 of this document — it's already written in the right shape for that.

---

## 15. Key references

- Black, K. et al. "π0: A Vision-Language-Action Flow Model for General Robot Control." arXiv:2410.24164, 2024.
- Huang, J., Wang, S., Lin, F., Hu, Y., Wen, C., Gao, Y. "Tactile-VLA: Unlocking Vision-Language-Action Model's Physical Knowledge for Tactile Generalization." arXiv:2507.09160, 2025.
- Zhang, K., Zhang, H., Xu, Z., Zhang, Z., Prince, M.R.I., Li, X., Han, X., Zhou, Y., Ajoudani, A., She, Y. "TacVLA: Contact-Aware Tactile Fusion for Robust Vision-Language-Action Manipulation." arXiv:2603.12665, 2026.
- LeRobot `PI0Policy` implementation and π0/π0-FAST integration blog post, Hugging Face.
- Wang, S., Lambeta, M., Chou, P.-W., Calandra, R. "TACTO: A Fast, Flexible, and Open-source Simulator for High-Resolution Vision-based Tactile Sensors." arXiv:2012.08456.
- Si, Z., Yuan, W. "Taxim: An Example-based Simulation Model for GelSight Tactile Sensors." arXiv:2109.04027.
