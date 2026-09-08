# %% [markdown]
# # Experiment 4c — measured steering with variance + placebo control
#
# **What exp4 / exp4b left open.** exp4 tried the **top-2 purest
# slots per domain** at strengths `[0, 0.5*theta, 1*theta, 2*theta,
# 4*theta]` and produced zero measurable domain steering on Pythia
# generation. exp4b widened the target to **top-8 slots**, added a
# GAIN multiplier (sweep `[1, 4, 8]`) on the residual-add patch, and
# ran both a **BOOST** and a **SUPPRESS** direction. Verdict from
# exp4b at BOOST: with the loudest setting (gain=8, strength=4*theta)
# the nudged delta was `~12.29%` of the residual norm — large enough
# that the null result was no longer explainable by an under-powered
# nudge. The SUPPRESS direction, however, was structurally
# saturated: even at the loudest setting `delta_ratio` never exceeded
# `0.362%` of the residual norm, so the SUPPRESS sweep produced no
# useful comparison. Both directions were also single-sample — no
# variance, no placebo, so we cannot tell whether a single bar that
# moved by X hit% reflects a real slot-specific effect or noise /
# any-direction-push.
#
# **This experiment (exp4c) addresses exactly that.** It keeps
# everything in exp4b's training / purity / sampling / hook / metrics
# stack **bit-identical** and changes only the steering-eval block:
#
# 1. **BOOST only.** exp4b measured SUPPRESS as structurally saturated
#    (max `delta_ratio = 0.362%` vs BOOST's `12.29%` at identical
#    settings); the SUPPRESS sweep, its table, and its plots are
#    removed. BOOST is the one direction that actually moves the
#    delta-norm needle, so it is the only direction worth measuring.
#
# 2. **GAIN fixed at 8.0.** exp4b already showed that BOOST only
#    matters at GAIN=8; the `[1, 4]` rows were structural zero. So
#    we fix `gain = 8.0` (single value) and put the resolution into
#    the strength grid instead. `strength=0` at gain=8 is still the
#    bit-exact no-op baseline (both decode paths identical, delta=0).
#
# 3. **Finer strength grid.** `[0.0, 1.5, 2.0, 2.5, 3.0, 4.0] * theta`
#    (6 levels). 0.0 = literal no-op; the rest span the loudness range
#    where exp4b's GAIN=8 BOOST curve went from quiet to collapsing.
#
# 4. **Targeted vs control (placebo).** For every (domain, strength)
#    cell we run TWO conditions:
#
#    - **`"targeted"`** — boost the domain's top-8 purest slots
#      (exp4b behaviour).
#    - **`"control"`**  — boost 8 RANDOM slots that are NOT that
#      domain's slots. Pick from live slots whose argmax-domain is
#      different from this domain, sampled with a **fixed seed** so
#      the choice is reproducible and logged. Same count (8), same
#      strength, same gain. This is the placebo: if "control" moves
#      hit%/native% as much as "targeted", the effect is not
#      slot-specific (any 8-slot push will do).
#
# 5. **REPEATS = 3 per (domain, condition, strength) cell.** Each
#    repeat uses a different generation seed (`torch.manual_seed(1000
#    + rep)`). Total = 4 domains × 2 conditions × 6 strengths × 3
#    repeats = **144 runs**. Per cell we report `mean ± std` for
#    hit%, native%, rep%, entropy, delta_ratio, and `n_collapsed /
#    REPEATS`. Every individual repeat is kept in the JSON.
#
# 6. **Verdict block.** Computed automatically per domain from the
#    repeat means / stds. For each domain we look for the **largest
#    strength** satisfying three conditions simultaneously:
#
#    (a) targeted hit% mean exceeds its own strength=0 baseline mean
#        by **more than 2× pooled std**,
#    (b) targeted hit% mean exceeds the control hit% mean at the
#        **same strength** by **more than 2× pooled std**, AND
#    (c) the cell is not collapsed in the majority of repeats.
#
#    Call that the domain's **"clean steering window"**. If no
#    strength qualifies, print "no clean window found". The same is
#    computed for native%. The verdict answers the real question:
#    *is there a (domain, condition)-specific effect, large enough to
#    be visible above sampling noise and above a placebo of equal
#    magnitude, that does not require fluency collapse?*
#
# **Outputs**
# - `exp4c_model.pt` — registry state_dict (NOT Pythia).
# - `exp4c_summary.json` — purity verdict + training summary (same
#   schema as exp4).
# - `exp4c_steering.json` — score-scale stats, derived strengths,
#   fixed gain, the full 144-row per-(domain, condition, strength,
#   repeat) table (hit%, native%, rep%, entropy, delta_ratio,
#   collapsed), per-cell aggregates (`mean ± std`), targeted and
#   control slot ids + purities, cfg.
# - `exp4c_results.png` — per-domain: targeted vs control curves
#   with error bars, x = delta_ratio.
# - `exp4c_delta_ratio.png` — delta_ratio vs strength, targeted vs
#   control.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings -> Accelerator -> GPU T4 x2** (or P100)
# 2. **Settings -> Internet -> On**  <- required to download Pythia + data
#
# Expected runtime: ~35-40 minutes total. Training is identical to
# exp3b / exp4b (~19 min). The steering block is 144 runs (4 domains
# x 2 conditions x 6 strengths x 3 repeats) — about 1.5x exp4b's
# 96-run sweep, so ~12-14 minutes on a T4.

# %%
# Kaggle's default image may not have a fresh-enough `transformers` for Pythia.
# Make it self-bootstrapping: only pip-install if not already present.
import subprocess, sys
try:
    import transformers
    print(f"transformers already present, version: {transformers.__version__}")
except ImportError:
    print("installing transformers...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "transformers"])
try:
    import datasets
    print(f"datasets already present, version: {datasets.__version__}")
except ImportError:
    print("installing datasets...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])


# %%
import os, math, time, json, random, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("!! no GPU detected -- go to Settings > Accelerator > GPU")

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
os.makedirs(OUT, exist_ok=True)


# %%
class CFG:
    # --- pretrained model (frozen, never trained) ---
    pythia_name   = "EleutherAI/pythia-410m"
    extract_layer = 12          # hidden_states[12] = output of layer 12 (middle of 24)
    hidden_size   = 1024        # pythia-410m hidden_size

    # --- registry (exp0d winning config, byte-identical to exp0d) ---
    n_slots      = 4096         # 16x expansion (matches exp0d)
    k            = 16           # top-k per token (matches exp0d)
    k_aux        = 32           # AuxK revival capacity (matches exp0d)
    dead_after   = 800          # slots unused for this many steps = dead (matches exp0d)

    # --- data ---
    tokens_per_domain = 3_000_000
    ctx                = 256
    batch_size         = 24     # 6 rows/domain in mixed batches (matches exp0d structure)

    # --- training ---
    steps       = 3000          # 3000 steps * 24 * 256 = ~18.4M total tokens
                               # -> ~4.6M tokens/domain (above the soft 3M target,
                               # well under exp0d's 15M/domain, and stays inside
                               # the 3000-4000 step window from the spec)
    lr_registry = 3e-3          # registry needs a higher LR (matches exp0d)
    warmup      = 200
    grad_clip   = 1.0

    # --- losses (recon + AuxK only, NO LM loss, NO alpha) ---
    lambda_recon = 1.0
    lambda_auxk  = 0.03

    # --- eval ---
    eval_every        = 500
    eval_batches      = 40
    eval_loss_batches = 4      # smaller per-domain batch count for Pythia CE sanity check

cfg = CFG()
print(json.dumps({k: v for k, v in vars(CFG).items() if not k.startswith("_")},
                 indent=2, ensure_ascii=False))


# %% [markdown]
# ## 1. Data -- four distant domains (same sources as exp1)
#
# Same four HuggingFace sources as exp1/exp1h, in the same load order. The
# only difference: tokenization is done with **Pythia's own** GPTNeoX BPE
# tokenizer instead of a custom from-scratch BPE -- activations are
# tokenizer/model-specific, so we cannot reuse exp0/exp1's trained tokenizer.

# %%
from datasets import load_dataset

DOMAINS = {
    "medicine":    dict(path="qiaojin/PubMedQA",            name="pqa_artificial",
                        split="train",
                        extract=lambda ex: "\n\n".join(ex["context"]["contexts"])),
    "law":         dict(path="FiscalNote/billsum",          name=None,
                        split="train",
                        extract=lambda ex: ex["text"]),
    "code":        dict(path="codeparrot/codeparrot-clean-valid", name=None,
                        split="train",
                        extract=lambda ex: ex["content"]),
    "literature":  dict(path="manu/project_gutenberg",      name="default",
                        split="en",
                        extract=lambda ex: ex["text"]),
}
DOMAIN_NAMES = list(DOMAINS.keys())
N_DOMAINS    = len(DOMAIN_NAMES)
assert N_DOMAINS == 4, "exp3b is hard-coded to 4 domains"

def stream_texts(spec, limit=None):
    """Reads texts from HuggingFace via streaming (no full dataset download).
    Uses each domain's `extract` callable to flatten the example into a string."""
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=True)
    n = 0
    for ex in ds:
        t = spec["extract"](ex)
        if t and len(t.strip()) > 50:
            yield t
            n += 1
            if limit and n >= limit:
                return


# %% [markdown]
# ## 2. Frozen Pythia -- pretrained feature extractor
#
# `EleutherAI/pythia-410m` (24 layers, hidden_size=1024, ~405M params) is loaded
# once, put into `.eval()`, and **every parameter is frozen** (`requires_grad=False`).
# All Pythia forward calls are wrapped in `torch.no_grad()`. The model is a
# pure feature extractor in this experiment: the only trainable component is
# the registry.
#
# `hidden_states[12]` (the output of the 13th of 24 layers -- roughly the middle
# of the stack) is taken as the residual-stream activation tensor `h` of
# shape `[B, T, 1024]`. `hidden_states` is a 25-tuple when
# `output_hidden_states=True` (embeddings + outputs after each of the 24
# layers), so `index 12` picks the post-layer-12 residual stream.

# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"loading {cfg.pythia_name}...")
t0 = time.time()
pythia = AutoModelForCausalLM.from_pretrained(cfg.pythia_name).to(DEVICE).eval()
tokenizer = AutoTokenizer.from_pretrained(cfg.pythia_name)
print(f"  loaded in {time.time()-t0:.0f}s")
print(f"  pythia params : {sum(p.numel() for p in pythia.parameters())/1e6:.1f}M")
print(f"  tokenizer     : vocab={tokenizer.vocab_size}, "
      f"len(tokenizer)={len(tokenizer)}, "
      f"eos_token_id={tokenizer.eos_token_id}")

# Freeze ALL pythia parameters
for p in pythia.parameters():
    p.requires_grad_(False)

frozen_total = sum(p.numel() for p in pythia.parameters())
frozen_train = sum(p.numel() for p in pythia.parameters() if p.requires_grad)
print(f"  pythia params frozen: {frozen_total:,} total, "
      f"{frozen_train:,} trainable (must be 0)")


# %%
def build_corpus(domain, spec, target_tokens):
    """Tokenize domain texts with Pythia's tokenizer into a single token array."""
    model_tag = cfg.pythia_name.split("/")[-1]
    model_tag = "".join(c if c.isalnum() or c in ".-_" else "_" for c in model_tag)
    path = f"{OUT}/{domain}.{model_tag}.npy"
    if os.path.exists(path):
        arr = np.load(path)
        print(f"{domain}: loaded {len(arr):,} tokens from {path}")
        return arr

    t0 = time.time()
    buf, total = [], 0
    eos_id = tokenizer.eos_token_id
    for text in stream_texts(spec):
        ids = tokenizer.encode(text)
        if eos_id is not None:
            ids = ids + [eos_id]
        buf.append(np.array(ids, dtype=np.uint16))
        total += len(ids)
        if total >= target_tokens:
            break
    arr = np.concatenate(buf)[:target_tokens]
    np.save(path, arr)
    if len(arr) < target_tokens * 0.95:
        print(f"WARNING: {domain} produced only {len(arr):,} of {target_tokens:,} "
              f"requested tokens -- the source was exhausted.")
    print(f"{domain}: {len(arr):,} tokens in {time.time()-t0:.0f}s")
    return arr

corpora = {d: build_corpus(d, s, cfg.tokens_per_domain) for d, s in DOMAINS.items()}

# train/eval split (98/2 like exp1h)
data = {}
for d, arr in corpora.items():
    n = int(0.98 * len(arr))
    data[d] = {"train": arr[:n], "eval": arr[n:]}
    print(f"{d}: train={n:,}  eval={len(arr)-n:,}")


# %%
def get_batch(domain, split, bs):
    arr = data[domain][split]
    ix = np.random.randint(0, len(arr) - cfg.ctx - 1, size=bs)
    x = np.stack([arr[i:i+cfg.ctx] for i in ix]).astype(np.int64)
    y = np.stack([arr[i+1:i+1+cfg.ctx] for i in ix]).astype(np.int64)
    return (torch.from_numpy(x).to(DEVICE, non_blocking=True),
            torch.from_numpy(y).to(DEVICE, non_blocking=True))

def get_mixed_batch(bs):
    """4-way balanced: bs//4 rows per domain in DOMAINS dict order.
    Returns (x, y, domain_labels). Training only uses x; y is kept for the
    optional Pythia CE sanity check."""
    assert bs % N_DOMAINS == 0, f"batch_size={bs} must be divisible by N_DOMAINS={N_DOMAINS}"
    per = bs // N_DOMAINS
    xs, ys = [], []
    for d in DOMAIN_NAMES:
        x, y = get_batch(d, "train", per)
        xs.append(x); ys.append(y)
    x = torch.cat(xs, dim=0)
    y = torch.cat(ys, dim=0)
    domain_labels = torch.cat(
        [torch.full((per,), i, dtype=torch.long, device=x.device)
         for i in range(N_DOMAINS)], dim=0)
    return x, y, domain_labels


# %% [markdown]
# ## 3. The registry -- sparse dictionary (exp0d design, byte-identical)
#
# Same `ConceptRegistry` as exp0/exp0d/exp1: TopK sparse coding, encoder
# + decoder with decoder columns unit-normalized and renormalized
# periodically (every 200 training steps), AuxK dead-slot revival,
# fp32 loss casting. The only differences from exp0d:
#
# - `d_model` is now `1024` (Pythia's hidden size) instead of `256`
# - The forward `target` is `h.detach()` from the *frozen* Pythia forward,
#   not a detached residual from a trainable trunk. Functionally identical
#   -- the registry learns to reconstruct the input -- but the source is
#   never updated.
# - There is no alpha-blending curriculum. The registry's recon output is
#   never fed back into Pythia; Pythia is frozen and untouched.

# %%
class ConceptRegistry(nn.Module):
    """TopK sparse dictionary. Spreads a dense representation across numbered slots."""

    def __init__(self, d_model, n_slots, k, k_aux):
        super().__init__()
        self.n_slots, self.k, self.k_aux = n_slots, k, k_aux
        self.encoder = nn.Linear(d_model, n_slots, bias=True)
        self.decoder = nn.Linear(n_slots, d_model, bias=False)
        self.b_dec   = nn.Parameter(torch.zeros(d_model))

        # init decoder as transpose of encoder (standard practice for SAEs),
        # then unit-normalize the decoder columns
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

        # counter: last step each slot fired (used to detect dead slots)
        self.register_buffer("last_fired", torch.zeros(n_slots, dtype=torch.long))

        # per-slot additive steering bias, added to the pre-topk scores
        # (`pre`) BEFORE top-k selection. Non-persistent so it never
        # enters state_dict and never gets a gradient. Default all zeros
        # -> training-time forward is bit-equivalent to exp3b (the
        # addition is mathematically a no-op).
        self.register_buffer(
            "steer_bias", torch.zeros(n_slots), persistent=False)

    @staticmethod
    def _topk(x, k):
        v, i = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        return out.scatter_(-1, i, v)

    def set_steer_bias(self, bias):
        """In-place setter. `bias` may be any [n_slots] tensor on any
        device; it is copied into the registry's `steer_bias` buffer
        (which lives on the same device as the rest of the module).
        Returns the previous bias for symmetry / manual restore."""
        prev = self.steer_bias.detach().clone()
        target = self.steer_bias
        if not isinstance(bias, torch.Tensor):
            bias = torch.as_tensor(bias, dtype=target.dtype, device=target.device)
        else:
            bias = bias.to(dtype=target.dtype, device=target.device)
        with torch.no_grad():
            target.copy_(bias)
        return prev

    def clear_steer_bias(self):
        """Force `steer_bias` back to all zeros. Cheap idempotent guard
        used between every steering run so a stray bias from one run
        can't leak into the next."""
        with torch.no_grad():
            self.steer_bias.zero_()

    @contextlib.contextmanager
    def steering(self, bias):
        """Context manager: install `bias` for the duration of the
        `with` block, restore the previous value on exit (always zero
        for us at run time). Use
        `with registry.steering(bias):` around the generation calls
        you want to steer. We additionally force a final
        `clear_steer_bias()` so a stray bias can never leak into the
        next run even if an exception is raised mid-block."""
        prev = self.set_steer_bias(bias)
        try:
            yield
        finally:
            self.set_steer_bias(prev)
            self.clear_steer_bias()

    def forward(self, h, step=None, dead_after=None):
        # compute pre, then add the per-slot steering bias BEFORE top-k
        # selection. With steer_bias == 0 (default at training time and
        # during the exp3b purity readout) this is a bit-exact no-op, so
        # the trained registry behavior is unchanged.
        pre         = F.relu(self.encoder(h - self.b_dec))
        pre_steered = pre + self.steer_bias
        acts        = self._topk(pre_steered, self.k)
        recon = self.decoder(acts) + self.b_dec

        aux_loss = h.new_zeros(())
        if self.training and step is not None:
            with torch.no_grad():
                if step % 200 == 0:
                    self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)
                fired = (acts > 0).flatten(0, -2).any(0)
                self.last_fired[fired] = step

            # AuxK: let dead slots try to rebuild the residual error
            dead = (step - self.last_fired) > dead_after
            if dead.any() and step > dead_after:
                resid = (h - recon).detach()
                pre_dead = pre.masked_fill(~dead, 0.0)
                kk = min(self.k_aux, int(dead.sum()))
                aux_acts = self._topk(pre_dead, kk)
                aux_recon = self.decoder(aux_acts)
                aux_loss = F.mse_loss(aux_recon.float(), resid.float())

        return recon, acts, aux_loss

    def dead_fraction(self, step, dead_after):
        return ((step - self.last_fired) > dead_after).float().mean().item()


# %%
registry = ConceptRegistry(cfg.hidden_size, cfg.n_slots, cfg.k, cfg.k_aux).to(DEVICE)
n_reg = sum(p.numel() for p in registry.parameters())
print(f"registry params: {n_reg/1e6:.2f}M  "
      f"(n_slots={cfg.n_slots}, k={cfg.k}, k_aux={cfg.k_aux}, d_in={cfg.hidden_size})")
print(f"sanity:  n_slots={cfg.n_slots}, k={cfg.k}, k_aux={cfg.k_aux}, "
      f"dead_after={cfg.dead_after} -- matches exp0d exactly")


# %% [markdown]
# ## 4. Training -- activations streamed on-the-fly, never persisted
#
# Every training step does:
#
# 1. Sample a 4-way balanced mixed batch of token ids (`get_mixed_batch`).
# 2. **Frozen Pythia forward** under `torch.no_grad()` with
#    `output_hidden_states=True` -- pull `hidden_states[cfg.extract_layer]`
#    as the activation tensor `h` of shape `[B, T, 1024]`. `h` lives only in
#    memory; it is **never** saved to disk.
# 3. **Registry forward** on `h` (the only learnable component of this
#    experiment). Compute recon MSE against `h.detach()` + AuxK auxiliary
#    loss. Backprop + step the registry's optimizer.
# 4. Discard `h`; the next step computes fresh activations from the next batch.
#
# Pythia's own next-token CE is printed every `eval_every` steps as a
# sanity check that the frozen model is behaving normally (small, stable
# number). It is **not** part of the loss surface.

# %%
opt = torch.optim.AdamW(
    registry.parameters(), lr=cfg.lr_registry,
    betas=(0.9, 0.95), weight_decay=0.1,
)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

def lr_scale(step):
    if step < cfg.warmup:
        return step / max(1, cfg.warmup)
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0)))

@torch.no_grad()
def pythia_ce_loss(n_batches=4):
    """Pythia's own next-token cross-entropy on held-out data. Sanity check
    that the frozen model is behaving normally -- nothing to optimize here."""
    pythia.eval()
    losses = {d: 0.0 for d in DOMAINS}
    counts = {d: 0 for d in DOMAINS}
    for d in DOMAINS:
        for _ in range(n_batches):
            x, y = get_batch(d, "eval", cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                out = pythia(x)
            logits = out.logits.float()
            n_tok = y.numel()
            loss_sum = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1), reduction="sum",
            )
            losses[d] += loss_sum.item()
            counts[d] += n_tok
    return {d: losses[d] / max(1, counts[d]) for d in DOMAINS}


# %%
hist = []
t0 = time.time()
pythia.eval()  # frozen model stays in eval() the entire time

for step in range(1, cfg.steps + 1):
    for g in opt.param_groups:
        g["lr"] = cfg.lr_registry * lr_scale(step)

    # 1. token batch
    x, _y, _dl = get_mixed_batch(cfg.batch_size)

    # 2. frozen pythia forward -> streamed activation h (never persisted)
    with torch.no_grad():
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            pythia_out = pythia(x, output_hidden_states=True)
    h = pythia_out.hidden_states[cfg.extract_layer]  # [B, T, 1024]
    # `h` is only ever held in this scope; never assigned to a module
    # attribute, never saved, never returned from this loop iteration.

    # 3. registry forward + backward (only learnable part of this experiment)
    registry.train()
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        recon, acts, aux_loss = registry(h, step=step, dead_after=cfg.dead_after)
        # target detached: registry learns to reconstruct the input
        # without pushing the (already-frozen) trunk anywhere
        recon_loss = F.mse_loss(recon.float(), h.detach().float())
        loss = cfg.lambda_recon * recon_loss + cfg.lambda_auxk * aux_loss

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(registry.parameters(), cfg.grad_clip)
    scaler.step(opt)
    scaler.update()

    # 4. `h` goes out of scope at the next iteration; never persisted

    if step % 100 == 0 or step == 1:
        dead = registry.dead_fraction(step, cfg.dead_after)
        el = time.time() - t0
        print(f"step {step:5d} | recon {recon_loss.item():.4f} "
              f"| aux {aux_loss.item():.4f} | dead {dead*100:4.1f}% | {el:.0f}s")
        hist.append(dict(step=step, recon=recon_loss.item(),
                         aux=aux_loss.item(), dead=dead))

    if step % cfg.eval_every == 0:
        ce = pythia_ce_loss(n_batches=cfg.eval_loss_batches)
        ce_str = "  ".join(f"{d}={v:.3f}" for d, v in ce.items())
        print(f"   >> pythia CE (sanity): {ce_str}")

print(f"\ntraining done in {(time.time()-t0)/60:.1f} minutes")
# Save registry state_dict only -- NOT pythia
torch.save(registry.state_dict(), f"{OUT}/exp4c_model.pt")
print(f"saved registry state_dict to {OUT}/exp4c_model.pt "
      f"({sum(p.numel() for p in registry.parameters()):,} params)")


# %% [markdown]
# ## 5. The decisive measurement -- slot purity vs raw-dim purity
#
# For each domain, stream fresh activations through Pythia + the trained
# registry and count:
#
# - **slot firings**: for every token, the registry fires exactly `k=16`
#   slots (TopK). Count how often each slot fires on each domain.
# - **raw-dim top-k**: at matched sparsity `k=16`, count how often each of
#   the 1024 Pythia hidden dimensions lands in the top-16 by absolute
#   value for each token, per domain.
#
# `purity_i = max_d counts[d][i] / sum_d counts[d][i]`. The verdict is the
# mean purity gap between registry slots and raw dims.

# %%
@torch.no_grad()
def collect_activity(n_batches):
    """Streamed activation extraction -- activations live only in this loop.
    Counts (a) slot firings per slot per domain, (b) raw-dim top-k hits
    per dim per domain (matched sparsity k=16), (c) which input tokens
    most activate each slot (for the eyeball-proof readout later)."""
    slot_cnt  = {d: torch.zeros(cfg.n_slots,       device=DEVICE) for d in DOMAINS}
    dim_cnt   = {d: torch.zeros(cfg.hidden_size,  device=DEVICE) for d in DOMAINS}
    tok_score = torch.zeros(cfg.n_slots, len(tokenizer), device=DEVICE)

    max_tok_id_corpus = max(int(arr.max()) for arr in corpora.values())
    print(f"token id range check: max_token_id={max_tok_id_corpus}  "
          f"tokenizer.vocab_size={tokenizer.vocab_size}  "
          f"len(tokenizer)={len(tokenizer)}")

    pythia.eval()
    registry.eval()

    for d in DOMAINS:
        for _ in range(n_batches):
            x, _y = get_batch(d, "eval", cfg.batch_size)

            # Streamed: frozen pythia forward -> grab layer-12 residual
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                pythia_out = pythia(x, output_hidden_states=True)
            h = pythia_out.hidden_states[cfg.extract_layer].float()  # [B, T, 1024]

            # Registry on the streamed activation
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                recon, acts, _ = registry(h, step=None, dead_after=cfg.dead_after)

            acts_f = acts.float().flatten(0, 1)                 # [B*T, n_slots]
            slot_cnt[d] += (acts_f > 0).float().sum(0)

            # raw-dim top-k at matched sparsity (same k as the registry)
            h_flat = h.flatten(0, 1)                            # [B*T, 1024]
            _topv, topi = torch.topk(h_flat.abs(), cfg.k, dim=-1)
            dim_cnt[d].scatter_add_(0, topi.flatten(),
                                    torch.ones_like(topi.flatten(), dtype=torch.float))

            flat_tok = x.flatten()
            assert int(flat_tok.max()) < tok_score.shape[1], f"token id {int(flat_tok.max())} exceeds tok_score width {tok_score.shape[1]}"
            tok_score.index_add_(1, flat_tok, acts_f.t())

    return slot_cnt, dim_cnt, tok_score

slot_cnt, dim_cnt, tok_score = collect_activity(cfg.eval_batches)


# %%
def purity_stats(counts, min_activity=50):
    """4-way purity over a per-domain count dict.
    purity_i = max_d counts[d][i] / sum_d counts[d][i]  for unit i (live units only)."""
    stack = torch.stack([counts[d] for d in DOMAIN_NAMES])     # [n_domains, N]
    total = stack.sum(0)
    live  = total > min_activity
    pur   = stack.max(0).values[live] / total[live]
    return dict(
        n_live=int(live.sum()),
        n_total=len(total),
        mean=pur.mean().item(),
        median=pur.median().item(),
        frac_above_80=(pur > 0.80).float().mean().item(),
        frac_above_95=(pur > 0.95).float().mean().item(),
        values=pur.cpu().numpy(),
        total=total.cpu().numpy(),
    )

P_slots = purity_stats(slot_cnt)
P_dims  = purity_stats(dim_cnt)

print("=" * 64)
print(f"{'':22} {'registry slots':>18} {'raw dims':>18}")
print("-" * 64)
print(f"{'live units':22} {P_slots['n_live']:>10}/{P_slots['n_total']:<7} "
      f"{P_dims['n_live']:>10}/{P_dims['n_total']:<7}")
print(f"{'mean purity':22} {P_slots['mean']:>18.3f} {P_dims['mean']:>18.3f}")
print(f"{'median purity':22} {P_slots['median']:>18.3f} {P_dims['median']:>18.3f}")
print(f"{'purity > 0.80':22} {P_slots['frac_above_80']:>17.1%} {P_dims['frac_above_80']:>17.1%}")
print(f"{'purity > 0.95':22} {P_slots['frac_above_95']:>17.1%} {P_dims['frac_above_95']:>17.1%}")
print("=" * 64)


# %%
# Pythia own next-token CE -- final sanity check that the frozen model
# itself is behaving normally (small, stable number; nothing to optimize).
final_ce = pythia_ce_loss(n_batches=cfg.eval_batches)
mean_ce = sum(final_ce.values()) / N_DOMAINS
print(f"\npythia own next-token CE on held-out data (final sanity check):")
for d in DOMAINS:
    print(f"   {d:12s} = {final_ce[d]:.3f}")
print(f"   {'mean':12s} = {mean_ce:.3f}")


# %% [markdown]
# ## 6. Verdict
#
# Same bar as exp0's original threshold: **gap > 0.10** between registry-slot
# mean purity and raw-dim mean purity at matched sparsity k=16. Tiered
# verdict so a weak-positive signal is not collapsed into a hard pass/fail.

# %%
gap = P_slots["mean"] - P_dims["mean"]
print(f"\nmean-purity gap (registry slots vs raw dims): {gap:+.3f}\n")

if gap > 0.10:
    verdict = ("\u2705 pass -- exp0's specialization finding replicates on real "
               "Pythia activations; the registry is clearly purer than the raw "
               "Pythia hidden dimensions at matched sparsity.")
    nxt = ("next: try a slightly larger pretrained model (pythia-1.4b) to see "
           "if the gap widens with richer activations, or move to a downstream "
           "intervention experiment (e.g. slot-conditioned generation steering).")
elif gap > 0.05:
    verdict = ("\U0001F7E1 weak-positive signal -- there is some specialization, "
               "but the gap is smaller than exp0's original bar.")
    nxt = ("consider: more training tokens, larger pretrained model, or a "
           "registry with k_aux / n_slots tuned to real-activation statistics.")
else:
    verdict = ("\u274C fail -- the registry did not specialize meaningfully "
               "beyond the raw-dim baseline on real Pythia activations.")
    nxt = ("walk through this checklist: (1) is dead_frac high? "
           "(2) is recon_loss plateauing? (3) is the chosen layer too early / "
           "too late in the stack? if all three are fine, exp0's effect may "
           "have been an artifact of the toy setup and the rest of the project "
           "needs to be rethought.")

print(verdict)
print("next step:", nxt)


# %% [markdown]
# ## 7. Plots

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

steps = [h["step"] for h in hist]
ax[0].plot(steps, [h["recon"] for h in hist], label="recon MSE", color="teal")
ax[0].plot(steps, [h["aux"] for h in hist], label="AuxK loss", color="darkorange", alpha=0.7)
ax[0].set_title("registry losses"); ax[0].set_xlabel("step"); ax[0].legend(fontsize=8)

ax[1].plot(steps, [h["dead"] * 100 for h in hist], color="darkorange")
ax[1].set_title("dead-slot fraction %"); ax[1].set_xlabel("step")

bins = np.linspace(0.25, 1.0, 31)
ax[2].hist(P_dims["values"],  bins=bins, alpha=.5, label="raw dims",      color="gray")
ax[2].hist(P_slots["values"], bins=bins, alpha=.7, label="registry slots", color="teal")
ax[2].axvline(0.25, color="k", ls=":", lw=1, label="uniform floor (0.25)")
ax[2].set_title("purity distribution -- registry vs raw (Pythia hidden dims)")
ax[2].set_xlabel("purity"); ax[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/exp4c_results.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp4c_results.png")


# %% [markdown]
# ## 8. Reading the slots -- eyeball proof
#
# For each domain, the top-purity slots and the input tokens that most
# activate them. Tokens are decoded with Pythia's own tokenizer (subword
# BPE, so multi-token words may appear as fragments; the leading-space
# marker is preserved).

# %%
c = torch.stack([slot_cnt[d] for d in DOMAIN_NAMES])           # [4, n_slots]
total_all = c.sum(0)
pur_all = torch.where(total_all > 0,
                      c.max(0).values / total_all.clamp(min=1),
                      torch.zeros_like(total_all))
dom_all = c.argmax(0)

for d_idx, d in enumerate(DOMAIN_NAMES):
    mask = (dom_all == d_idx) & (total_all > 200) & (pur_all > 0.70)
    idx = torch.nonzero(mask).flatten()
    if len(idx) == 0:
        print(f"\n### {d}: not enough pure slots (need purity > 0.70 and fires > 200)")
        continue
    idx = idx[total_all[idx].argsort(descending=True)][:6]
    print(f"\n{'=' * 60}\n### {d} slots (top {len(idx)})\n{'=' * 60}")
    for s in idx.tolist():
        top_tok = tok_score[s].topk(12).indices.tolist()
        words = [repr(tokenizer.decode([t]))[1:-1] for t in top_tok]
        print(f"slot {s:4d} | purity {pur_all[s]:.2f} | fires {int(total_all[s]):7,} "
              f"| {' . '.join(words[:10])}")


# %%
summary = dict(
    verdict=verdict,
    slots=dict(
        mean=P_slots["mean"],
        median=P_slots["median"],
        live=P_slots["n_live"],
        above_80=P_slots["frac_above_80"],
        above_95=P_slots["frac_above_95"],
    ),
    raw_dims=dict(
        mean=P_dims["mean"],
        median=P_dims["median"],
        live=P_dims["n_live"],
        above_80=P_dims["frac_above_80"],
        above_95=P_dims["frac_above_95"],
    ),
    gap=gap,
    final_dead_frac=hist[-1]["dead"],
    final_recon_loss=hist[-1]["recon"],
    final_aux_loss=hist[-1]["aux"],
    pythia_next_token_ce=dict(mean=mean_ce, **final_ce),
    config={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
)
with open(f"{OUT}/exp4c_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(f"\nsaved {OUT}/exp4c_summary.json")


# %% [markdown]
# ## 9. Steering eval — calibrate strength to Pythia's real score scale
#
# exp2b's lesson: hand-picked strength constants (e.g. 5, 10, 20) are
# either too small (no measurable steering) or too large (fluency
# collapses at the very first non-zero step). Before we steer anything
# we have to measure the **actual scale** of the registry's pre-topk
# slot scores on Pythia-410m layer-12 activations:
#
# - `mean(pre)` — overall scale of the encoded slot scores.
# - `std(pre)`  — spread of the score distribution (the natural `sigma`).
# - `topk_threshold = avg_k-th_largest(pre)` — the average per-token
#   value at rank `k=16` in the descending sort of `pre`. This is the
#   **actual top-k selection threshold**: a slot whose `pre` value
#   exceeds `topk_threshold` will be selected, a slot below it will
#   not. Anchoring BOOST strengths to this threshold (rather than to
#   `sigma`) answers the real question: how much do we have to push a
#   slot to *guarantee* it makes the cut on every token?
#
# `theta = pre_stats["topk_threshold"]` is what the BOOST
# strength list below is scaled by.

# %%
PRE_STATS_N_BATCHES = 10   # ~10 eval batches is enough to get a stable std

@torch.no_grad()
def measure_pre_stats(n_batches=PRE_STATS_N_BATCHES, k=cfg.k):
    """Stream n_batches of mixed-domain eval through frozen Pythia,
    capture the registry's pre-topk slot scores (`pre`) at every token,
    compute mean / std / the average per-token k-th largest value of
    `pre`. `pre` is reconstructed in this function as
    `F.relu(encoder(h - b_dec)) + steer_bias` with the registry's own
    parameters, so the numbers are bit-equivalent to what
    `registry.forward()` would compute on the same input. With
    `steer_bias == 0` (its state right now) `pre` is identical to what
    the steering block will see."""
    pythia.eval()
    registry.eval()
    chunks = []
    encoder = registry.encoder
    b_dec   = registry.b_dec
    bias    = registry.steer_bias  # zeros at this point
    for _ in range(n_batches):
        # Mixed-domain batches mirror training; the stats cover all
        # four domains at once for a single summary number.
        x, _y, _dl = get_mixed_batch(cfg.batch_size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            pythia_out = pythia(x, output_hidden_states=True)
        h = pythia_out.hidden_states[cfg.extract_layer].float()  # [B, T, 1024]
        pre = F.relu(encoder(h - b_dec)) + bias                  # [B, T, n_slots]
        chunks.append(pre.detach().float().cpu().reshape(-1, cfg.n_slots))
    all_pre = torch.cat(chunks, dim=0)             # [N_tokens, n_slots]
    mean = all_pre.mean().item()
    std  = all_pre.std().item()
    # per-token k-th largest: sort each row descending, take column k-1
    sorted_desc, _ = torch.sort(all_pre, dim=-1, descending=True)
    kth = sorted_desc[:, k - 1]                    # [N_tokens]
    avg_kth = kth.mean().item()
    return dict(mean=mean, std=std, topk_threshold=avg_kth, k=k,
                n_tokens=all_pre.shape[0], n_batches=n_batches)

pre_stats = measure_pre_stats(PRE_STATS_N_BATCHES, k=cfg.k)
sigma = pre_stats["std"]
theta = pre_stats["topk_threshold"]
print("=" * 78)
print(f"pre stats: mean={pre_stats['mean']:.4f}  std={sigma:.4f}  "
      f"topk_threshold(k={pre_stats['k']})={theta:.4f}  "
      f"(over {pre_stats['n_tokens']:,} tokens from "
      f"{pre_stats['n_batches']} mixed batches)")
print(f"  -> BOOST strengths will be scaled by theta = {theta:.4f}")
print("=" * 78)


# %% [markdown]
# ## 10. Target slots — top-8 purest slots per domain (TARGETED condition)
#
# For each of the 4 domains we pick the **top-8 highest-purity slots**
# (drawn from the registry's `total_all / pur_all / dom_all` arrays
# computed in section 8). Same gates as the "top slots per domain"
# printout: slot must have `dom_all == d_idx` and `total_all > 200`
# fires. **No purity cutoff** — the gate is relaxed; we rank by purity
# descending and take the top 8 (or fewer if fewer qualify). If the
# count is below 8 we log it and take whatever is available. These
# slots are the ones we'll boost in the `"targeted"` condition of the
# sweep (positive `steer_bias` entries, same bias value per slot).
#
# Why top-8 instead of top-2: exp4 used top-2 because we expected a
# small handful of sharply-pure slots to dominate steering; instead the
# delta was too quiet to detect. Top-8 distributes the bias over more
# slots (same per-slot bias value), which the registry's top-k then
# accumulates into more decoder columns and a larger residual delta.

# %%
TARGET_SLOTS_PER_DOMAIN = 8   # top-8 purest slots per domain (exp4 used top-2)

target_slots = {}
target_slots_meta = {}
print("=" * 78)
print(f"TARGET SLOTS  (top {TARGET_SLOTS_PER_DOMAIN} by purity, "
      f"gates: dom_all == d_idx, total_all > 200 fires)")
print("=" * 78)
for d_idx, d in enumerate(DOMAIN_NAMES):
    mask = (dom_all == d_idx) & (total_all > 200)
    idx = torch.nonzero(mask).flatten()
    n_qualifying = int(mask.sum().item())
    idx = idx[pur_all[idx].argsort(descending=True)][:TARGET_SLOTS_PER_DOMAIN]
    n_picked = len(idx)
    if n_picked < TARGET_SLOTS_PER_DOMAIN:
        print(f"  {d:>10s}  WARNING: only {n_picked}/{TARGET_SLOTS_PER_DOMAIN} "
              f"target slots qualified (n_qualifying={n_qualifying}); "
              f"taking what is available.")
    target_slots[d] = [int(i) for i in idx.cpu().tolist()]
    target_slots_meta[d] = dict(
        slot_ids=target_slots[d],
        purities=[float(pur_all[i].item()) for i in target_slots[d]],
        fires=[int(total_all[i].item()) for i in target_slots[d]],
        n_qualifying=n_qualifying,
        n_picked=n_picked,
    )
    purity_str = ", ".join(f"{pur_all[i].item():.3f}" for i in target_slots[d])
    activity_str = ", ".join(f"{int(total_all[i].item()):,}" for i in target_slots[d])
    note = "" if n_picked >= TARGET_SLOTS_PER_DOMAIN else \
           f"  [only {n_picked}/{TARGET_SLOTS_PER_DOMAIN} qualified]"
    print(f"  {d:>10s}  target_slots = {target_slots[d]}  "
          f"purities = [{purity_str}]  fires = [{activity_str}]{note}")
print("=" * 78)


# %% [markdown]
# ## 10b. Control slots — 8 PURITY-MATCHED non-domain slots (CONTROL placebo)
#
# For each domain we also pick a **control** set of 8 slots that are
# explicitly NOT in `target_slots[d]`. This is the placebo: at
# identical (count, strength, gain) the only thing different from
# `"targeted"` is *which* slots get boosted. If `"targeted"` moves
# hit%/native% more than `"control"` does, the effect is slot-specific
# (the registry's per-domain direction actually matters); if the two
# move together, the effect is just "any 8 slots" — i.e. we are
# pushing a generic directional bias onto the residual stream, not a
# domain-specific signal.
#
# The previous implementation was a uniform `randperm` over the
# candidate pool. That made the two conditions differ in TWO ways
# (domain AND purity / fire-count), which weakened the verdict: any
# `targeted > control` gap could be explained by the targeted set
# having higher purity (and so a larger decoder footprint per slot)
# rather than by slot-domain identity. To make the verdict test
# domain-slot-specificity — not "high-purity-slots-beat-random-slots" —
# the control set must match the targeted set's purity profile.
#
# Selection rules (logged so the choice is reproducible and auditable):
# - Candidate pool = every live slot whose `dom_all != d_idx`
#   (argmax-domain is NOT this domain). The `total_all > 200` gate
#   from the targeted selector is kept for symmetry: the control slots
#   are also "live, frequently-firing" — not dead slots, not
#   low-activity background slots, so the registry still has a real
#   decoder column attached to each.
# - For each `p_i` in the targeted set's purity list `P = [p_1..p_8]`
#   (descending), greedily pick the not-yet-picked candidate whose
#   purity is closest to `p_i` (smallest absolute purity gap).
#   Ties on purity are broken by closest fire-count; remaining ties
#   are broken by a deterministic per-domain RNG so the choice is
#   reproducible. The deterministic seed is consumed UP-FRONT as a
#   single `torch.rand(n_cand, generator=g)` — it only influences the
#   answer in the rare event of an exact purity + fire-count tie, so
#   the primary match is fully determined by the data.
# - Same count (8) as `target_slots[d]`. If fewer than 8 candidates
#   exist for a domain, we log it and take what is available — same
#   fall-back as the targeted selector.
# - For each domain we log the targeted purities, the matched control
#   purities, the per-slot absolute purity gap, and the same for fire
#   counts, plus the mean absolute purity gap and mean absolute fire-
#   count gap, so the match quality is auditable from the log alone.
#   Picked-slot ids, their (non-this-domain) argmax-domain, purity
#   and fire-count are all logged into `control_slots_meta[d]` and
#   written verbatim into the steering JSON.

# %%
CONTROL_SEED_BASE = 9000   # base offset; per-domain seed = CONTROL_SEED_BASE + d_idx

control_slots = {}
control_slots_meta = {}
print("=" * 78)
print(f"CONTROL SLOTS  ({TARGET_SLOTS_PER_DOMAIN} PURITY-MATCHED non-domain slots per domain, "
      f"gates: dom_all != d_idx AND total_all > 200)")
print("=" * 78)
for d_idx, d in enumerate(DOMAIN_NAMES):
    cand_mask = (dom_all != d_idx) & (total_all > 200)
    cand_idx_all = torch.nonzero(cand_mask).flatten()
    n_qualifying = int(cand_mask.sum().item())
    n_cand = int(len(cand_idx_all))
    # Pre-extract candidate purity / fire as plain python lists so the
    # greedy inner loop stays free of torch.item() overhead.
    cand_purs  = [float(pur_all[i].item())    for i in cand_idx_all]
    cand_fires = [int(total_all[i].item())   for i in cand_idx_all]
    # Deterministic seed consumed UP-FRONT as one rand tensor; used
    # only as the final tie-breaker when both the purity gap and the
    # fire-count gap are exactly equal between two candidates. The
    # primary match is fully determined by the data.
    g = torch.Generator(device="cpu").manual_seed(CONTROL_SEED_BASE + d_idx)
    rng_tiebreak = torch.rand(n_cand, generator=g).tolist() if n_cand > 0 else []

    # targeted set's purities + fires (descending purity, the order
    # section 10 picked them in). These define the profile the
    # control set must mimic.
    target_purs  = [float(p) for p in target_slots_meta[d]["purities"]]
    target_fires = [int(f)   for f in target_slots_meta[d]["fires"]]
    n_target = len(target_purs)
    # If we somehow have fewer candidates than targeted slots we fall
    # back to however many we can pick; the loop below naturally
    # terminates when no more candidates are available.
    n_pick = min(n_target, n_cand)

    used = [False] * n_cand
    picked_slots    = []
    matched_purs    = []
    matched_fires   = []
    gaps_pur        = []
    gaps_fire       = []
    for k in range(n_pick):
        p_target    = target_purs[k]
        fire_target = target_fires[k]
        best_j       = -1
        best_pur_gap = None
        best_fire_gap = None
        best_rng     = None
        for j in range(n_cand):
            if used[j]:
                continue
            pur_gap   = abs(cand_purs[j] - p_target)
            fire_gap  = abs(cand_fires[j] - fire_target)
            rng_val   = rng_tiebreak[j]
            if best_j == -1 or pur_gap < best_pur_gap:
                best_j, best_pur_gap = j, pur_gap
                best_fire_gap, best_rng = fire_gap, rng_val
            elif pur_gap == best_pur_gap:
                if fire_gap < best_fire_gap:
                    best_j, best_pur_gap = j, pur_gap
                    best_fire_gap, best_rng = fire_gap, rng_val
                elif fire_gap == best_fire_gap:
                    if rng_val < best_rng:
                        best_j, best_pur_gap = j, pur_gap
                        best_fire_gap, best_rng = fire_gap, rng_val
        if best_j == -1:
            break
        used[best_j] = True
        picked_slots.append(int(cand_idx_all[best_j].item()))
        matched_purs.append(cand_purs[best_j])
        matched_fires.append(cand_fires[best_j])
        gaps_pur.append(abs(cand_purs[best_j] - p_target))
        gaps_fire.append(abs(cand_fires[best_j] - fire_target))

    n_picked = len(picked_slots)
    if n_picked < TARGET_SLOTS_PER_DOMAIN:
        print(f"  {d:>10s}  WARNING: only {n_picked}/{TARGET_SLOTS_PER_DOMAIN} "
              f"control candidates qualified (n_qualifying={n_qualifying}); "
              f"taking what is available.")
    control_slots[d] = [int(i) for i in picked_slots]
    # also log the (non-this-domain) argmax-domain of each control slot
    # so the JSON makes the "definitely not this domain" property auditable.
    mean_pur_gap  = (sum(gaps_pur)  / len(gaps_pur))  if gaps_pur  else 0.0
    mean_fire_gap = (sum(gaps_fire) / len(gaps_fire)) if gaps_fire else 0.0
    control_slots_meta[d] = dict(
        slot_ids=control_slots[d],
        argmax_domains=[DOMAIN_NAMES[int(dom_all[i].item())] for i in control_slots[d]],
        targeted_purities=target_purs,
        matched_purities=matched_purs,
        targeted_fires=target_fires,
        matched_fires=matched_fires,
        purities=matched_purs,
        fires=matched_fires,
        per_slot_abs_purity_gap=gaps_pur,
        per_slot_abs_fire_gap=gaps_fire,
        mean_abs_purity_gap=float(mean_pur_gap),
        mean_abs_fire_gap=float(mean_fire_gap),
        n_qualifying=n_qualifying,
        n_picked=n_picked,
        seed=CONTROL_SEED_BASE + d_idx,
    )
    target_pur_str = ", ".join(f"{p:.3f}" for p in target_purs[:n_picked])
    matched_pur_str = ", ".join(f"{p:.3f}" for p in matched_purs)
    fire_str   = ", ".join(f"{f:,}" for f in matched_fires)
    dom_str    = ", ".join(DOMAIN_NAMES[int(dom_all[i].item())][:4] for i in control_slots[d])
    note = "" if n_picked >= TARGET_SLOTS_PER_DOMAIN else \
           f"  [only {n_picked}/{TARGET_SLOTS_PER_DOMAIN} qualified]"
    print(f"  {d:>10s}  control_slots = {control_slots[d]}  "
          f"argmax_dom = [{dom_str}]")
    print(f"           targeted_purities = [{target_pur_str}]")
    print(f"           matched_purities  = [{matched_pur_str}]  "
          f"mean_abs_purity_gap = {mean_pur_gap:.3f}")
    print(f"           fires             = [{fire_str}]  "
          f"mean_abs_fire_gap = {mean_fire_gap:,.0f}{note}")
print("=" * 78)


# %% [markdown]
# ## 11. Domain vocabularies — distinctive-vocab + slot-native-vocab
#
# Two per-domain vocab sets, both built once from the same corpus the
# registry was trained on. Used as the targets for the `hit%` and
# `native%` metrics below:
#
# - **distinctive vocab** (`domain_vocab[d]`) — top N tokens by
#   `freq_in_d / (freq_in_other_3_domains + eps)`, restricted to tokens
#   that appear at least `MIN_FREQ_IN_DOMAIN` times in d. Same recipe
#   as exp2. **hit%** = fraction of generated tokens found in this set.
# - **slot-native vocab** (`slot_native_vocab[d]`) — top N tokens by
#   raw firing score at exactly `target_slots[d]`, aggregated by
#   summing `tok_score` over the chosen slots. Same recipe as exp2d.
#   **native%** = fraction of generated tokens found in this set. This
#   is the metric-mismatch cross-check: if hit% stays flat while
#   native% rises, BOOST is steering generation toward the target
#   slots' own associated tokens but the distinctive-vocab metric
#   doesn't notice.

# %%
N_VOCAB_PER_DOMAIN     = 150
N_SLOT_NATIVE_VOCAB    = 30
MIN_FREQ_IN_DOMAIN     = 10

@torch.no_grad()
def build_domain_vocab(n_tokens=N_VOCAB_PER_DOMAIN, min_freq=MIN_FREQ_IN_DOMAIN):
    """Per-domain distinctive-vocabulary set. freq_in_d / sum_other +
    small eps to avoid divide-by-zero. Low-frequency tokens in d are
    filtered out by `min_freq`. token-id range is the corpus max,
    NOT `tokenizer.vocab_size` — Pythia's `len(tokenizer)` is what
    bounds `tok_score` (the T-040 fix)."""
    vocab_counts = {}
    n_corpus = max(int(arr.max()) for arr in corpora.values()) + 1
    for d in DOMAIN_NAMES:
        arr = data[d]["train"].astype(np.int64)
        counts = torch.bincount(
            torch.from_numpy(arr), minlength=n_corpus,
        ).to(DEVICE).float()
        vocab_counts[d] = counts
    eps = 1.0
    out = {}
    for d in DOMAIN_NAMES:
        others = sum(vocab_counts[o] for o in DOMAIN_NAMES if o != d)
        ratio  = vocab_counts[d] / (others + eps)
        ratio  = ratio * (vocab_counts[d] >= min_freq).float()
        top    = torch.topk(ratio, min(n_tokens, n_corpus)).indices.cpu().tolist()
        out[d] = top
    return out

domain_vocab      = build_domain_vocab()
domain_vocab_sets = {d: set(v) for d, v in domain_vocab.items()}

# exp2d slot-native vocab: top tokens by raw firing score summed
# across target_slots[d]. Aggregated from `tok_score` (the same
# `[n_slots, vocab]` tensor section 8 prints from) — no new forward
# passes, no recompute.
slot_native_vocab = {}
for d in DOMAIN_NAMES:
    slots = target_slots[d]
    agg = tok_score[slots].sum(dim=0)                              # [vocab]
    top_idx = torch.topk(agg, N_SLOT_NATIVE_VOCAB).indices.cpu().tolist()
    slot_native_vocab[d] = [int(t) for t in top_idx]
slot_native_vocab_sets = {d: set(v) for d, v in slot_native_vocab.items()}

print("Per-domain distinctive-vocab (hit% target):")
for d in DOMAIN_NAMES:
    sample = [tokenizer.decode([t]) for t in domain_vocab[d][:8]]
    print(f"  {d:>10s}  ({len(domain_vocab[d])} tokens)  sample: {sample}")

print("\nPer-domain slot-native vocab (native% target, top tokens fired by target_slots[d]):")
for d in DOMAIN_NAMES:
    sample = [tokenizer.decode([t]) for t in slot_native_vocab[d][:8]]
    print(f"  {d:>10s}  ({len(slot_native_vocab[d])} tokens)  sample: {sample}")


# %% [markdown]
# ## 12. Forward-hook patch + sampling generation harness
#
# Two pieces:
#
# 1. **Forward hook** registered on Pythia's layer-12 block
#    (`gpt_neox.layers[11]`, the block whose output is
#    `hidden_states[12]`). The hook replaces the residual `h` with
#
#        h_steered = h + GAIN * ( decode(topk(encode(h) + steer_bias))
#                                - decode(topk(encode(h))) )
#
#    Both decode paths are computed with the SAME `h`; only the bias
#    differs (the GAIN is a scalar multiplier on the resulting delta).
#    So a zero `steer_bias` is bit-exact no-op: the two decodes are
#    identical, the delta is zero, h_steered = h. The hook handles both
#    prefill `[B, T, 1024]` and decode-step `[B, 1, 1024]` shapes
#    (registry is per-position; we apply elementwise). All inside
#    `torch.no_grad()`, registry + Pythia in `.eval()`. On transformers
#    >= 5.0 (Kaggle's image ships 5.0.0) `GPTNeoXLayer.forward` returns
#    a BARE TENSOR, so the hook adds the registry delta to the output
#    directly and returns it; the tuple form `(hidden_states, ...)` is
#    kept as a compatibility fallback for older transformers where the
#    layer returned a tuple.
#
#    The hook is a callable class instance so it can carry mutable
#    state (current GAIN + a running delta_ratio accumulator) without
#    resorting to a global. The accumulator is reset between runs by
#    `reset_accumulator()` and the GAIN is set per-run via `set_gain()`.
#
# 2. **Sampling generation**. All generation uses nucleus sampling
#    (`temperature=0.8, top_p=0.9`), never greedy / argmax — greedy
#    fabrication collapses fluency onto a single high-purity slot's
#    top tokens (exp2b lesson; project rule). Identical to exp4.

# %%
class _SteeringHook:
    """Forward hook installed on Pythia's layer-12 block. Computes the
    registry's residual-add patch
        delta = decode(topk(pre + steer_bias)) - decode(topk(pre))
    multiplies by the current GAIN, and replaces the layer output with
    `h + GAIN * delta`. Also accumulates the `delta_ratio` diagnostic:
    whenever steer_bias is non-zero we record
        delta_ratio = (GAIN * delta).norm(dim=-1).mean() / h.norm(dim=-1).mean()
    (mean over tokens) per Pythia forward call. The accumulator is
    reset between runs by `reset_accumulator()` and read by the sweep
    loop after each generation. When steer_bias == 0 the patch is a
    bit-exact no-op (delta is exactly zero) and we skip the diagnostic
    so the literal baseline row records `delta_ratio = 0.0`."""

    def __init__(self, registry):
        self.registry = registry
        self.gain = 1.0
        self._dr_sum = 0.0
        self._dr_count = 0

    def reset_accumulator(self):
        self._dr_sum = 0.0
        self._dr_count = 0

    def set_gain(self, g):
        self.gain = float(g)

    @property
    def delta_ratio(self):
        return self._dr_sum / self._dr_count if self._dr_count > 0 else 0.0

    @property
    def delta_ratio_n(self):
        return self._dr_count

    def _patch_delta(self, h):
        reg = self.registry
        pre         = F.relu(reg.encoder(h - reg.b_dec))
        pre_steered = pre + reg.steer_bias
        acts_base   = reg._topk(pre,         reg.k)
        acts_steer  = reg._topk(pre_steered, reg.k)
        dec_base    = reg.decoder(acts_base)
        dec_steer   = reg.decoder(acts_steer)
        return dec_steer - dec_base

    def __call__(self, module, inputs, output):
        reg = self.registry
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        delta = self._patch_delta(h)
        steered = h + self.gain * delta
        # delta_ratio diagnostic: only accumulate when steer_bias is
        # non-zero (otherwise delta is bit-exact zero and there is no
        # nudge to measure). Cheap: reuses the delta and h already
        # computed in this hook -- no extra forward.
        if float(reg.steer_bias.abs().sum().item()) != 0.0:
            denom = h.norm(dim=-1).mean().item()
            if denom > 0.0:
                ratio = (self.gain * delta).norm(dim=-1).mean().item() / denom
                self._dr_sum += ratio
                self._dr_count += 1
        if is_tuple:
            return (steered,) + tuple(output[1:])
        return steered


_steering_hook = _SteeringHook(registry)
_steering_layer = pythia.gpt_neox.layers[cfg.extract_layer - 1]
_steering_handle = _steering_layer.register_forward_hook(_steering_hook)
print(f"registered steering hook on pythia.gpt_neox.layers[{cfg.extract_layer - 1}] "
      f"(output is hidden_states[{cfg.extract_layer}]); "
      f"handle={_steering_handle}")


# %%
GEN_LEN             = 200        # tokens generated per steering run
GEN_TEMPERATURE     = 0.8
GEN_TOP_P           = 0.9
COLLAPSE_REP        = 0.5        # rep% > this -> flagged
COLLAPSE_ENT        = 1.0        # entropy < this -> flagged (nats)
REPEATS             = 3          # repeats per (domain, condition, strength) cell
SEED_BASE           = 1000       # generation seed = SEED_BASE + rep
EOT = tokenizer.eos_token_id     # neutral "between documents" boundary token
if EOT is None:
    # Pythia's tokenizer always defines eos_token_id but be safe.
    EOT = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

# exp4c: BOOST strengths anchored to theta (the average per-token k-th
# largest value of `pre`) — i.e. "how much do we have to push a slot
# to make sure it makes the top-k cut". 6 levels: 0.0 = literal bit-
# exact no-op baseline (steer_bias all zeros, both decode paths in the
# hook identical, delta=0); the rest are a finer sweep of the loudness
# range where exp4b's GAIN=8 BOOST curve went from quiet to collapsing.
# We dropped the 1.0*theta step in favour of three intermediate
# points (1.5, 2.5, 3.0) so the per-strength delta_ratio curves
# below the collapse point are well-resolved.
BOOST_STRENGTHS = [0.0, 1.5 * theta, 2.0 * theta, 2.5 * theta, 3.0 * theta, 4.0 * theta]

# GAIN multiplier on the residual delta:
#     h_steered = h + GAIN * ( decode(topk(pre + steer_bias))
#                              - decode(topk(pre)) )
# exp4b measured that BOOST only mattered at GAIN=8; GAIN=1/4
# produced essentially zero delta. So we fix GAIN=8 (single value, no
# sweep) and put the resolution into the strength grid. gain=8 with
# strength=0 is still the bit-exact no-op baseline.
GAIN = 8.0

print(f"\nSteering config:")
print(f"  GEN_LEN             = {GEN_LEN}")
print(f"  GEN_TEMPERATURE     = {GEN_TEMPERATURE}")
print(f"  GEN_TOP_P           = {GEN_TOP_P}")
print(f"  COLLAPSE_REP        = {COLLAPSE_REP}")
print(f"  COLLAPSE_ENT        = {COLLAPSE_ENT}")
print(f"  REPEATS             = {REPEATS}")
print(f"  SEED_BASE           = {SEED_BASE}")
print(f"  EOT (neutral prompt token) = {EOT}")
print(f"  BOOST_STRENGTHS     = {[round(s, 4) for s in BOOST_STRENGTHS]}")
print(f"  GAIN (fixed)        = {GAIN}")
print(f"  theta (topk_threshold) = {theta:.4f}")
print(f"  total runs          = {len(DOMAIN_NAMES)} domains * "
      f"2 conditions * {len(BOOST_STRENGTHS)} strengths * {REPEATS} repeats "
      f"= {len(DOMAIN_NAMES) * 2 * len(BOOST_STRENGTHS) * REPEATS}")


# %%
@torch.no_grad()
def generate(prompt_ids, n_new=GEN_LEN):
    """Nucleus-sampling generation from `prompt_ids` (1-D list/array
    of token ids). Pads on the LEFT with `EOT` if the prompt is shorter
    than `cfg.ctx`; truncates on the LEFT if longer. Returns
    `(generated_tokens, entropies_per_step)`. Steering is the caller's
    responsibility -- wrap this in `with registry.steering(bias):` to
    apply a bias during generation. The steering forward hook patches
    layer-12 activations on every pythia() call.

    **Sampling, NEVER greedy / argmax** (project rule: temperature=0.8,
    top_p=0.9 nucleus). `entropy` returned per step is the full
    softmax distribution's entropy in nats, computed BEFORE
    temperature / top_p filtering so the metric reflects what the
    model thinks, not what we kept from it."""
    pythia.eval()
    registry.eval()
    pad_id = EOT if EOT is not None else 0
    ids = [int(t) for t in prompt_ids]
    if len(ids) < cfg.ctx:
        ids = [pad_id] * (cfg.ctx - len(ids)) + ids
    elif len(ids) > cfg.ctx:
        ids = ids[-cfg.ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated, entropies = [], []
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            o = pythia(x)
        logits = o.logits[0, -1].float()                       # [vocab]
        # entropy BEFORE sampling filtering (project rule: reflects model belief)
        p_full = F.softmax(logits, dim=-1)
        ent = -(p_full * (p_full + 1e-12).log()).sum().item()
        entropies.append(ent)
        # temperature scaling
        logits = logits / GEN_TEMPERATURE
        # top-p (nucleus) filter: zero out tokens above the cumulative
        # probability cutoff so we only sample from the smallest set
        # whose total mass is >= top_p.
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        # shift so the first token is always kept, then zero out any
        # token whose cumulative-prob-with-self exceeds top_p
        keep_mask = (cumprobs - sorted_probs) < GEN_TOP_P
        sorted_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
        # re-normalize over the kept set
        kept_probs = F.softmax(sorted_logits, dim=-1)
        # sample from the kept set
        nxt_sorted = torch.multinomial(kept_probs, num_samples=1).item()
        nxt = int(sorted_idx[nxt_sorted].item())
        generated.append(nxt)
        x = torch.cat([x[:, 1:],
                       torch.tensor([[nxt]], device=DEVICE)], dim=1)
    return generated, entropies


def domain_vocab_hit_rate(tokens, vocab_set):
    if not tokens:
        return 0.0
    in_set = sum(1 for t in tokens if int(t) in vocab_set)
    return in_set / len(tokens)


def repetition_rate(tokens):
    """Fraction of bigrams in `tokens` that are repeats of an EARLIER
    bigram in the same generation. A pure repetition of length 2
    repeated gives 0.5, of length 1 repeated gives ~1.0."""
    if len(tokens) < 2:
        return 0.0
    seen, reps = set(), 0
    for i in range(1, len(tokens)):
        b = (int(tokens[i - 1]), int(tokens[i]))
        if b in seen:
            reps += 1
        else:
            seen.add(b)
    return reps / (len(tokens) - 1)


def avg_entropy(ent_list):
    if not ent_list:
        return 0.0
    return sum(ent_list) / len(ent_list)


def is_collapsed(rep, ent):
    """Fluency-collapse flag. Evaluated for every run; the only
    exempt row is the literal baseline (`strength == 0`) where by
    construction no bias was applied (handled at the call site).
    exp2e bug: previously the first non-zero strength was exempted
    by absolute-magnitude threshold; sub-1 magnitudes were skipped
    even when they actually broke fluency. We do NOT exempt anything
    except the literal zero-bias baseline."""
    return (rep > COLLAPSE_REP) or (ent < COLLAPSE_ENT)


# Sanity-check: at this point steer_bias should still be all-zeros.
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must be all-zeros at the start of the steering block"


# %% [markdown]
# ## 13. Steering sweep — TARGETED vs CONTROL × REPEATS, BOOST only
#
# For each domain `d`, for each `condition` in `["targeted", "control"]`,
# and each `strength` in `BOOST_STRENGTHS`, with the hook's GAIN fixed
# at `GAIN = 8.0`, we run `REPEATS = 3` independent generations, each
# with a different generation seed (`torch.manual_seed(SEED_BASE + rep)`,
# so seed `SEED_BASE + 1`, `SEED_BASE + 2`, `SEED_BASE + 3`).
#
# - `"targeted"` condition: `steer_bias[target_slots[d]] = +strength`.
# - `"control"`  condition: `steer_bias[control_slots[d]] = +strength`.
#
# Biased slots count, strength, and gain are identical between the two
# conditions; only the **set of slot ids** changes. Same neutral
# `[EOT]` prompt. Same `GEN_LEN = 200` tokens.
#
# Per repeat we record: hit, native_hit, rep, ent, delta_ratio,
# collapsed, n_tokens. All `4 domains × 2 conditions × 6 strengths ×
# 3 repeats = 144` individual runs are saved.
#
# Total cells (after aggregation) = 4 × 2 × 6 = 48, each with
# `mean ± std` across the 3 repeats.

# %%
skipped_domains = []
# raw_runs[d][condition] = list of per-repeat dicts, one entry per (strength, rep)
raw_runs = {d: {"targeted": [], "control": []} for d in DOMAIN_NAMES}

print("=" * 78)
print(f"exp4c steering sweep  (GAIN={GAIN}, REPEATS={REPEATS}, "
      f"neutral prompt = [EOT], n_gen={GEN_LEN}, "
      f"strengths={[round(s, 4) for s in BOOST_STRENGTHS]}, "
      f"theta={theta:.4f})")
print("=" * 78)
print(f"{'dom':>10s}  {'cond':>9s}  {'rep':>3s}  {'strength':>10s}  "
      f"{'hit%':>6s}  {'nat%':>6s}  {'rep%':>5s}  {'ent':>7s}  "
      f"{'dR%':>8s}  {'n':>4s}  note")
print("-" * 78)

def _make_bias(slot_ids, strength):
    """Build a [n_slots] steer_bias tensor with `+strength` at every
    slot id in `slot_ids`. If `strength == 0` the bias stays all-zero
    (a literal no-op baseline: bit-exact Pythia + registry)."""
    bias = torch.zeros(cfg.n_slots, device=DEVICE)
    if strength != 0.0:
        for s in slot_ids:
            bias[s] = strength
    return bias

for d in DOMAIN_NAMES:
    registry.clear_steer_bias()              # leakage guard between domains
    _steering_hook.set_gain(GAIN)
    if (not target_slots[d]) or (not control_slots[d]):
        print(f"{d:>10s}  SKIPPED: target_slots={len(target_slots[d])}, "
              f"control_slots={len(control_slots[d])}")
        skipped_domains.append(d)
        continue
    for condition in ("targeted", "control"):
        slot_ids = target_slots[d] if condition == "targeted" else control_slots[d]
        for strength in BOOST_STRENGTHS:
            for rep_n in range(1, REPEATS + 1):
                # seed ONLY the generation path; do NOT seed before
                # the bias build / hook setup (those are deterministic
                # and don't consume RNG in a meaningful way).
                torch.manual_seed(SEED_BASE + rep_n)
                bias = _make_bias(slot_ids, strength)
                _steering_hook.reset_accumulator()
                _steering_hook.set_gain(GAIN)
                with registry.steering(bias):
                    toks, ents = generate([EOT], n_new=GEN_LEN)
                hit        = domain_vocab_hit_rate(toks, domain_vocab_sets[d])
                native_hit = domain_vocab_hit_rate(toks, slot_native_vocab_sets[d])
                rep_rate   = repetition_rate(toks)
                ent        = avg_entropy(ents)
                dr         = _steering_hook.delta_ratio
                # collapse check: applied to every non-zero strength row;
                # strength=0 gets a free pass (no bias -> can't collapse).
                collapsed  = (strength != 0.0) and is_collapsed(rep_rate, ent)
                raw_runs[d][condition].append(dict(
                    condition=condition, domain=d, rep_n=rep_n,
                    strength=float(strength), gain=float(GAIN),
                    hit=hit, native_hit=native_hit,
                    rep=rep_rate, ent=ent, delta_ratio=float(dr),
                    collapsed=bool(collapsed),
                    n_tokens=len(toks),
                    steered_slots=[int(s) for s in slot_ids],
                ))
                note = ""
                if collapsed:
                    note = "  **"
                elif strength == 0.0:
                    note = "  (baseline)"
                print(f"{d:>10s}  {condition:>9s}  {rep_n:>3d}  {strength:>10.4f}  "
                      f"{hit*100:5.1f}%  {native_hit*100:5.1f}%  "
                      f"{rep_rate*100:4.1f}%  {ent:7.3f}  "
                      f"{dr*100:7.3f}%  {len(toks):4d}{note}")
    registry.clear_steer_bias()              # leakage guard between domains
    _steering_hook.set_gain(GAIN)
    _steering_hook.reset_accumulator()
print("-" * 78)
print(f"(collapse threshold: rep% > {COLLAPSE_REP} OR entropy < "
      f"{COLLAPSE_ENT} nats; the literal baseline row is always exempt.)")


# %% [markdown]
# ## 14. Aggregating repeats — mean / std / collapsed-count per cell
#
# Each `(domain, condition, strength)` cell has `REPEATS = 3` runs.
# We compute per-cell:
#
# - `mean` and `std` (population, ddof=0 so the std is the spread you
#   see in the bar plot, not Bessel-corrected) for hit, native_hit,
#   rep, ent, delta_ratio.
# - `n_collapsed` (count of repeats where `collapsed == True`).
#
# Stored as `cell_results[d][condition][strength] = {mean, std, n_collapsed, repeats}`.

# %%
def _agg(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    s = (sum((v - m) ** 2 for v in values) / n) ** 0.5   # population std (ddof=0)
    return m, s

def _per_cell(rep_rows):
    """Reduce a list of REPEATS dicts (one cell) into a single aggregate
    dict. All means are population means (REPEATS small, no need for
    Bessel correction); `*_std` are the matching population stds."""
    hits  = [r["hit"]         for r in rep_rows]
    nats  = [r["native_hit"]  for r in rep_rows]
    reps  = [r["rep"]         for r in rep_rows]
    ents  = [r["ent"]         for r in rep_rows]
    drs   = [r["delta_ratio"] for r in rep_rows]
    hm, hs = _agg(hits)
    nm, ns = _agg(nats)
    rm, rs = _agg(reps)
    em, es = _agg(ents)
    dm, ds = _agg(drs)
    return dict(
        n=len(rep_rows),
        hit_mean=hm, hit_std=hs,
        native_mean=nm, native_std=ns,
        rep_mean=rm, rep_std=rs,
        ent_mean=em, ent_std=es,
        dr_mean=dm,  dr_std=ds,
        n_collapsed=sum(1 for r in rep_rows if r["collapsed"]),
    )

cell_results = {d: {c: {} for c in ("targeted", "control")} for d in DOMAIN_NAMES}
for d in DOMAIN_NAMES:
    for c in ("targeted", "control"):
        for s in BOOST_STRENGTHS:
            cell_rows = [r for r in raw_runs[d][c] if r["strength"] == s]
            cell_results[d][c][s] = _per_cell(cell_rows)


# %% [markdown]
# ## 15. Per-cell table — domain | condition | strength | hit/native/rep/ent (m±s) | collapsed
#
# ONE consolidated table, grouped by domain, with `targeted` rows
# immediately followed by `control` rows for each strength. Every
# metric is reported as `mean ± std` across the 3 repeats; the
# collapsed column is `n_collapsed / REPEATS`.

# %%
print("=" * 130)
print(f"EXP4C PER-CELL TABLE  (gain={GAIN}, REPEATS={REPEATS}, "
      f"every metric = mean ± std across repeats)")
print("=" * 130)
hdr = (f"{'domain':>10s} | {'cond':>9s} | {'strength':>10s} | {'dr_t%':>7s} | {'dr_c%':>7s} | "
       f"{'hit%':>13s} | {'native%':>13s} | {'rep%':>13s} | "
       f"{'ent':>13s} | {'collapsed':>10s}")
print(hdr)
print("-" * len(hdr))
for d in DOMAIN_NAMES:
    for c in ("targeted", "control"):
        for s in BOOST_STRENGTHS:
            cell = cell_results[d][c].get(s, None)
            if cell is None or cell["n"] == 0:
                print(f"{d:>10s} | {c:>9s} | {s:>10.4f} | (no repeats)")
                continue
            # always show BOTH conditions' dr_mean on every row so a
            # dr_mismatch is visible at a glance, even when the verdict
            # passes for this cell.
            dr_t_cell = cell_results[d]["targeted"].get(s, None)
            dr_c_cell = cell_results[d]["control"].get(s, None)
            dr_t_str = (f"{dr_t_cell['dr_mean']*100:6.3f}"
                        if (dr_t_cell is not None and dr_t_cell['n'] > 0)
                        else "   -- ")
            dr_c_str = (f"{dr_c_cell['dr_mean']*100:6.3f}"
                        if (dr_c_cell is not None and dr_c_cell['n'] > 0)
                        else "   -- ")
            hit_str = f"{cell['hit_mean']*100:5.1f}±{cell['hit_std']*100:4.1f}"
            nat_str = f"{cell['native_mean']*100:5.1f}±{cell['native_std']*100:4.1f}"
            rep_str = f"{cell['rep_mean']*100:4.1f}±{cell['rep_std']*100:4.1f}"
            ent_str = f"{cell['ent_mean']:5.2f}±{cell['ent_std']:4.2f}"
            coll    = f"{cell['n_collapsed']}/{REPEATS}"
            print(f"{d:>10s} | {c:>9s} | {s:>10.4f} | {dr_t_str:>7s} | {dr_c_str:>7s} | "
                  f"{hit_str:>13s} | {nat_str:>13s} | {rep_str:>13s} | "
                  f"{ent_str:>13s} | {coll:>10s}")
    print("-" * len(hdr))
print("=" * len(hdr))


# %% [markdown]
# ## 16. Verdict block — clean steering window per domain
#
# For each domain we look at the (condition, strength) grid and pick
# the **largest strength** that simultaneously satisfies FOUR
# conditions on the targeted-vs-control comparison:
#
# (a) **`targeted` hit% mean exceeds its own `strength=0` baseline
#     mean by more than `2 * pooled_std`** at that strength. Pooled std
#     = sqrt( std(targeted_at_s)^2 + std(targeted_at_0)^2 ). This is
#     the targeted-vs-its-own-baseline test: did this strength do
#     anything at all? **Zero-variance special case**: if `pooled_std
#     == 0` AND the two means differ, the verdict treats this as
#     "perfect separation (zero variance)" — the strongest possible
#     evidence — rather than skipping the cell as untestable.
# (b) **`targeted` hit% mean exceeds the `control` hit% mean at the
#     SAME strength by more than `2 * pooled_std`**. Pooled std =
#     sqrt( std(targeted_at_s)^2 + std(control_at_s)^2 ). This is the
#     placebo-vs-target test: is the targeted effect larger than the
#     effect of pushing an equally-sized but domain-random bias? Same
#     zero-variance special case as (a).
# (c) **The cell is not collapsed in the majority of repeats**
#     (`n_collapsed * 2 < REPEATS`, i.e. `<= 1` of `3`).
# (d) **The two conditions actually injected a comparable perturbation
#     magnitude**: the per-cell `delta_ratio` of targeted and control
#     agree within 35% of the larger of the two, i.e.
#         |dr_t - dr_c| / max(dr_t, dr_c, 1e-9) <= 0.35.
#     If `delta_ratio` for control is much smaller than for targeted
#     (or vice versa) at the same strength / gain / slot-count, the
#     "targeted > control" comparison is confounded by a magnitude
#     mismatch — control's smaller nudge is just harder to see — and
#     the cell is disqualified. The disqualification is logged with
#     the actual `dr_t`, `dr_c` numbers and the gap percentage.
#
# **Important: this is an EXPLORATORY screen, not a significance test.**
# With `REPEATS = 3` the population std (ddof=0) is itself noisy, and
# the `2 x pooled_std` rule is a heuristic for "this cell looks
# non-trivially different from zero / from the placebo". The verdict
# is a flag for human inspection, not a p-value. The "perfect
# separation" row is the one exception worth treating as strong
# evidence: zero variance in both samples with clearly different
# means is the cleanest possible signal we could ask for at n=3.
#
# The "largest strength" wins (we are looking for the strongest
# intervention that still keeps generation fluent and still produces
# a slot-specific signal). If no strength satisfies all four, we
# print "no clean window found" for that domain. Same verdict is
# computed for `native%` separately.

# %%
def _pooled_std(std_a, std_b):
    return (std_a ** 2 + std_b ** 2) ** 0.5

DR_MATCH_TOL = 0.35   # condition (d): |dr_t - dr_c| / max(dr_t, dr_c, 1e-9) <= 0.35

def _clean_window(d, metric):
    """Return the largest strength where all four (a), (b), (c), (d)
    conditions hold for `metric` in {'hit', 'native'}, or None.

    Conditions:
      (a) targeted above its own baseline by >2x pooled std at this
          strength (or PERFECT SEPARATION: pooled std == 0 AND means
          differ — zero-variance-but-different is the strongest signal,
          not "untestable").
      (b) targeted above control at the SAME strength by >2x pooled
          std (or PERFECT SEPARATION, same rule).
      (c) majority-not-collapsed: n_collapsed * 2 < REPEATS.
      (d) delta_ratio mismatch: |dr_t - dr_c| / max(dr_t, dr_c, 1e-9)
          <= DR_MATCH_TOL. The two conditions must have actually
          injected a comparable perturbation magnitude. If they do
          not, the comparison is confounded by a magnitude gap and the
          cell is disqualified; the disqualification is printed with
          the actual `dr_t`, `dr_c` numbers and the gap percentage.

    Note on the n=3, 2x-pooled-std rule: this is an EXPLORATORY screen,
    not a significance test. With three repeats the population std
    (ddof=0) is itself noisy, and the `2 x pooled_std` heuristic simply
    surfaces cells that look non-trivially different from zero / from
    the placebo. The verdict is a flag for human inspection. The
    "perfect separation" exception is worth treating as strong
    evidence: zero variance in both samples with clearly different
    means is the cleanest possible signal we could ask for at n=3.

    `metric='hit'` uses hit% / hit_std; `metric='native'` uses
    native% / native_std. The targeted row is `cell_results[d]['targeted']`
    and the placebo row is `cell_results[d]['control']`.

    Walk strengths from largest to smallest; first hit wins."""
    assert metric in ("hit", "native")
    key = "hit_mean" if metric == "hit" else "native_mean"
    sk  = "hit_std"  if metric == "hit" else "native_std"
    cells_t = cell_results[d]["targeted"]
    cells_c = cell_results[d]["control"]
    base = cells_t.get(0.0, None)
    if base is None or base["n"] == 0:
        return None
    base_mean = base[key]
    base_std  = base[sk]
    for s in sorted(BOOST_STRENGTHS, reverse=True):
        if s == 0.0:
            continue
        ct = cells_t.get(s, None)
        cc = cells_c.get(s, None)
        if ct is None or cc is None or ct["n"] == 0 or cc["n"] == 0:
            continue
        # (c) majority-not-collapsed
        if ct["n_collapsed"] * 2 >= REPEATS:
            continue
        # (a) targeted above its own baseline by >2 pooled std, OR
        # perfect separation (pooled std == 0 AND means differ)
        pooled_a = _pooled_std(ct[sk], base_std)
        perfect_separation_a = False
        if pooled_a <= 0:
            if ct[key] <= base_mean:
                continue  # equal OR worse -> not a win
            perfect_separation_a = True
        elif not (ct[key] - base_mean > 2.0 * pooled_a):
            continue
        # (b) targeted above control at same strength by >2 pooled std,
        # OR perfect separation (pooled std == 0 AND means differ)
        pooled_b = _pooled_std(ct[sk], cc[sk])
        perfect_separation_b = False
        if pooled_b <= 0:
            if ct[key] <= cc[key]:
                continue  # equal OR worse than the placebo -> not a win
            perfect_separation_b = True
        elif not (ct[key] - cc[key] > 2.0 * pooled_b):
            continue
        # (d) delta_ratio mismatch: the two conditions must have
        # actually injected a comparable perturbation magnitude.
        dr_t = float(ct["dr_mean"])
        dr_c = float(cc["dr_mean"])
        dr_gap = abs(dr_t - dr_c) / max(dr_t, dr_c, 1e-9)
        if dr_gap > DR_MATCH_TOL:
            print(f"  {d} strength={s:.4f}: dr mismatch targeted={dr_t*100:.2f}% "
                  f"vs control={dr_c*100:.2f}% (gap={dr_gap*100:.1f}%) "
                  f"-- not magnitude-matched")
            continue
        return dict(strength=s,
                    targeted_mean=ct[key], targeted_std=ct[sk],
                    control_mean=cc[key],  control_std=cc[sk],
                    baseline_mean=base_mean, baseline_std=base_std,
                    n_collapsed=ct["n_collapsed"],
                    pooled_std_a=pooled_a, pooled_std_b=pooled_b,
                    perfect_separation_a=perfect_separation_a,
                    perfect_separation_b=perfect_separation_b,
                    dr_targeted=dr_t, dr_control=dr_c, dr_gap=dr_gap)
    return None

def _fmt_sep(sep_flag, pooled):
    """Print `pooled_std=X.XX%` normally; say `perfect separation
    (zero variance)` when zero pooled std was treated as significant
    because the two means differ."""
    return "perfect separation (zero variance)" if sep_flag \
           else f"pooled_std={pooled*100:.2f}%"

print("=" * 78)
print("EXP4C VERDICT -- clean steering window per domain")
print("(largest strength where targeted > its own baseline AND > control,")
print(" each by >2x pooled std (or perfect separation),")
print(" majority-not-collapsed, AND delta_ratio match within 35%)")
print("=" * 78)
verdict = {}
for d in DOMAIN_NAMES:
    w_hit = _clean_window(d, "hit")
    w_nat = _clean_window(d, "native")
    verdict[d] = dict(hit_window=w_hit, native_window=w_nat)
    print(f"\n  {d}:")
    if w_hit is None:
        print(f"    hit%     : no clean window found")
    else:
        sep_a = _fmt_sep(w_hit['perfect_separation_a'], w_hit['pooled_std_a'])
        sep_b = _fmt_sep(w_hit['perfect_separation_b'], w_hit['pooled_std_b'])
        dr_t_pct = w_hit['dr_targeted'] * 100
        dr_c_pct = w_hit['dr_control'] * 100
        dr_gap_pct = w_hit['dr_gap'] * 100
        print(f"    hit%     : clean window at strength = {w_hit['strength']:.4f}  "
              f"(= {w_hit['strength']/theta:.2f}*theta)")
        print(f"               targeted hit% = "
              f"{w_hit['targeted_mean']*100:.1f}±{w_hit['targeted_std']*100:.1f}%   "
              f"vs baseline {w_hit['baseline_mean']*100:.1f}±{w_hit['baseline_std']*100:.1f}%, "
              f"{sep_a}, "
              f"dr_t={dr_t_pct:.2f}% dr_c={dr_c_pct:.2f}% (gap={dr_gap_pct:.1f}%)")
        print(f"               control  hit% = "
              f"{w_hit['control_mean']*100:.1f}±{w_hit['control_std']*100:.1f}%, "
              f"{sep_b}, "
              f"dr_t={dr_t_pct:.2f}% dr_c={dr_c_pct:.2f}% (gap={dr_gap_pct:.1f}%), "
              f"collapsed {w_hit['n_collapsed']}/{REPEATS}")
    if w_nat is None:
        print(f"    native%  : no clean window found")
    else:
        sep_a = _fmt_sep(w_nat['perfect_separation_a'], w_nat['pooled_std_a'])
        sep_b = _fmt_sep(w_nat['perfect_separation_b'], w_nat['pooled_std_b'])
        dr_t_pct = w_nat['dr_targeted'] * 100
        dr_c_pct = w_nat['dr_control'] * 100
        dr_gap_pct = w_nat['dr_gap'] * 100
        print(f"    native%  : clean window at strength = {w_nat['strength']:.4f}  "
              f"(= {w_nat['strength']/theta:.2f}*theta)")
        print(f"               targeted nat% = "
              f"{w_nat['targeted_mean']*100:.1f}±{w_nat['targeted_std']*100:.1f}%   "
              f"vs baseline {w_nat['baseline_mean']*100:.1f}±{w_nat['baseline_std']*100:.1f}%, "
              f"{sep_a}, "
              f"dr_t={dr_t_pct:.2f}% dr_c={dr_c_pct:.2f}% (gap={dr_gap_pct:.1f}%)")
        print(f"               control  nat% = "
              f"{w_nat['control_mean']*100:.1f}±{w_nat['control_std']*100:.1f}%, "
              f"{sep_b}, "
              f"dr_t={dr_t_pct:.2f}% dr_c={dr_c_pct:.2f}% (gap={dr_gap_pct:.1f}%), "
              f"collapsed {w_nat['n_collapsed']}/{REPEATS}")
print("=" * 78)


# %% [markdown]
# ## 17. Steering plots
#
# Two figures:
#
# 1. `exp4c_results.png` -- one row of 4 panels (one per domain).
#    Each panel shows hit% (left axis) and native% (right axis) as a
#    function of `delta_ratio` (the actual measured nudge magnitude,
#    not the requested strength -- so we are plotting "what happened"
#    not "what we asked for"). Two series per panel: `targeted` (solid
#    line + circle markers) and `control` (dashed line + square
#    markers). Error bars = ±1 std across the 3 repeats. The
#    vertical "collapsed" red dashed line at the highest
#    non-collapsed strength is dropped here (the table tells us
#    that); instead we just plot up to whatever the data covers.
#
# 2. `exp4c_delta_ratio.png` -- delta_ratio vs requested strength,
#    two lines per domain (targeted vs control). Sanity check that
#    the nudge magnitude is identical between targeted and control
#    (same slot count, same strength, same gain -- they should
#    overlap).

# %%
import matplotlib.pyplot as plt

colors_d = {"medicine": "tab:blue", "law": "tab:orange",
            "code": "tab:green", "literature": "tab:red"}

fig, ax = plt.subplots(2, 4, figsize=(28, 10), sharex=False)

# row 0: hit% (distinctive) vs delta_ratio; row 1: native% vs delta_ratio
for row_idx, metric in enumerate(("hit", "native")):
    key  = "hit_mean"   if metric == "hit" else "native_mean"
    sk   = "hit_std"    if metric == "hit" else "native_std"
    title_metric = "hit% (distinctive)" if metric == "hit" else "native% (slot-native)"
    for col_idx, d in enumerate(DOMAIN_NAMES):
        a = ax[row_idx, col_idx]
        c = colors_d[d]
        for cond, ls, mk in (("targeted", "-", "o"), ("control", "--", "s")):
            cells = cell_results[d][cond]
            xs, ys, es = [], [], []
            for s in BOOST_STRENGTHS:
                cell = cells.get(s, None)
                if cell is None or cell["n"] == 0:
                    continue
                # skip the strength=0 baseline in the curve so the
                # x-axis covers only non-zero strengths (delta_ratio
                # is exactly 0 there and would clobber the scale).
                if s == 0.0:
                    continue
                xs.append(cell["dr_mean"] * 100)         # % units
                ys.append(cell[key] * 100)
                es.append(cell[sk] * 100)
            if xs:
                a.errorbar(xs, ys, yerr=es, ls=ls, marker=mk, color=c,
                           label=f"{cond}", capsize=3, lw=1.4, ms=7)
        # mark the baseline (s=0) as a single horizontal marker so the
        # reader can see where the unsteered model sits in this panel.
        base = cell_results[d]["targeted"].get(0.0, None)
        if base is not None and base["n"] > 0:
            a.axhline(base[key] * 100, color=c, ls=":", lw=1, alpha=0.6,
                      label=f"baseline (s=0): {base[key]*100:.1f}%")
        a.set_title(f"{d} -- {title_metric}")
        a.set_xlabel("delta_ratio = nudge_norm / residual_norm  (%)")
        a.set_ylabel(f"{metric}%")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)

plt.suptitle(f"exp4c steering: targeted (solid) vs control (dashed) ±1 std, "
             f"x = measured delta_ratio; GAIN={GAIN}, REPEATS={REPEATS}",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(f"{OUT}/exp4c_results.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp4c_results.png")


# %% [markdown]
# ## 17b. delta_ratio diagnostic plot
#
# Sanity check that the nudged-delta magnitude is the SAME between
# targeted and control (same slot count, same strength, same gain) —
# the two curves should overlap, because the only difference between
# conditions is the slot ids, not the strength or the number of slots.
# If they DO differ, something about the registry's decoder-column
# norms is different across slots and the comparison would be
# contaminated. Two panels: left = delta_ratio vs requested strength
# (linear scale on strength); right = delta_ratio vs requested
# strength (symlog so the strength=0 baseline does not clobber the
# axis).

# %%
fig, ax = plt.subplots(1, 2, figsize=(20, 5), sharey=True)

for col_idx, xscale in enumerate(("linear", "symlog")):
    a = ax[col_idx]
    for cond, ls, mk in (("targeted", "-", "o"), ("control", "--", "s")):
        for d in DOMAIN_NAMES:
            cells = cell_results[d][cond]
            xs, ys = [], []
            for s in BOOST_STRENGTHS:
                cell = cells.get(s, None)
                if cell is None or cell["n"] == 0:
                    continue
                if s == 0.0:
                    continue   # delta_ratio identically 0 at baseline
                xs.append(s / theta)            # x in theta-units
                ys.append(cell["dr_mean"] * 100)
            if xs:
                a.plot(xs, ys, ls, marker=mk, color=colors_d[d], alpha=0.85,
                       label=f"{cond}  {d}", ms=6, lw=1.2)
    if xscale == "symlog":
        a.set_xscale("symlog", linthresh=0.5)
    a.set_title(f"delta_ratio vs strength ({xscale})")
    a.set_xlabel("strength (x theta)"); a.grid(alpha=0.3)
    a.legend(fontsize=7, ncol=4)
ax[0].set_ylabel("delta_ratio = nudge_norm / residual_norm  (%)")
plt.suptitle(f"exp4c delta_ratio diagnostic -- targeted vs control (lines should overlap)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/exp4c_delta_ratio.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp4c_delta_ratio.png")


# %% [markdown]
# ## 18. Steering JSON
#
# `exp4c_steering.json`: score-scale stats, derived strengths, the
# fixed GAIN, the full 144-row per-(domain, condition, strength, rep)
# table, the per-cell aggregates (`mean ± std`), target/control slot
# ids + purities, the verdict block, and cfg.
# `exp4c_summary.json` (saved earlier, section 8) carries the
# purity-readout summary separately.

# %%
def _slot_top_tokens(slot_ids, k=12):
    """Return a list of (slot_id, [(token_str, score), ...]) for the
    given slots. Reads from `tok_score` so the values match the
    section-8 readout."""
    out = []
    for s in slot_ids:
        row = tok_score[s]
        vals, idx = torch.topk(row, k)
        toks = [tokenizer.decode([int(t)]) for t in idx.cpu().tolist()]
        scores = [float(v) for v in vals.cpu().tolist()]
        out.append((int(s), list(zip(toks, scores))))
    return out


# Summary stats for the delta_ratio diagnostic across all non-baseline
# repeats (both conditions, every domain). Tells us at a glance how
# loud the nudge actually was.
def _delta_ratio_stats_all():
    rs = []
    for d in DOMAIN_NAMES:
        for c in ("targeted", "control"):
            for s in BOOST_STRENGTHS:
                cell = cell_results[d][c].get(s, None)
                if cell is None:
                    continue
                rs.append(cell["dr_mean"])
    if not rs:
        return dict(min=0.0, max=0.0, mean=0.0, median=0.0, n=0)
    return dict(
        min=float(min(rs)),
        max=float(max(rs)),
        mean=float(sum(rs) / len(rs)),
        median=float(sorted(rs)[len(rs) // 2]),
        n=len(rs),
    )


# Build the steering payload. `runs` carries every individual repeat
# (144 entries). `cells` carries the per-cell aggregate dicts (48
# entries = 4*2*6). `control_slots` and `target_slots` carry the slot
# ids + metadata + their top tokens. `verdict` carries the clean-window
# per domain.
def _verdict_to_json(v):
    if v is None:
        return None
    return dict(
        strength=float(v["strength"]),
        targeted_mean=float(v["targeted_mean"]),
        targeted_std=float(v["targeted_std"]),
        control_mean=float(v["control_mean"]),
        control_std=float(v["control_std"]),
        baseline_mean=float(v["baseline_mean"]),
        baseline_std=float(v["baseline_std"]),
        n_collapsed=int(v["n_collapsed"]),
        pooled_std_a=float(v["pooled_std_a"]),
        pooled_std_b=float(v["pooled_std_b"]),
        perfect_separation_a=bool(v["perfect_separation_a"]),
        perfect_separation_b=bool(v["perfect_separation_b"]),
        dr_targeted=float(v["dr_targeted"]),
        dr_control=float(v["dr_control"]),
        dr_gap=float(v["dr_gap"]),
    )

steering_payload = dict(
    cfg={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
    score_scale=dict(
        mean=pre_stats["mean"],
        std=pre_stats["std"],
        topk_threshold=theta,
        topk_k=pre_stats["k"],
        n_tokens_measured=pre_stats["n_tokens"],
        n_batches_measured=pre_stats["n_batches"],
    ),
    strengths=dict(
        theta=float(theta),
        boost=[float(s) for s in BOOST_STRENGTHS],
        gain=float(GAIN),
        repeats=int(REPEATS),
        seed_base=int(SEED_BASE),
        control_seed_base=int(CONTROL_SEED_BASE),
        generation=dict(
            n_new=GEN_LEN,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
        ),
        collapse=dict(rep=COLLAPSE_REP, ent=COLLAPSE_ENT),
        verdict=dict(
            dr_match_tol=float(DR_MATCH_TOL),
            rule="EXPLORATORY screen (n=3, 2x pooled std); not a significance test",
        ),
    ),
    target_slots=target_slots_meta,
    control_slots=control_slots_meta,
    target_slots_top_tokens={
        d: _slot_top_tokens(target_slots[d], k=12) for d in DOMAIN_NAMES
    },
    control_slots_top_tokens={
        d: _slot_top_tokens(control_slots[d], k=12) for d in DOMAIN_NAMES
    },
    skipped_domains=skipped_domains,
    vocabs=dict(
        distinctive={d: [int(t) for t in v] for d, v in domain_vocab.items()},
        slot_native={d: [int(t) for t in v] for d, v in slot_native_vocab.items()},
        n_vocab_per_domain=N_VOCAB_PER_DOMAIN,
        n_slot_native_vocab=N_SLOT_NATIVE_VOCAB,
        min_freq_in_domain=MIN_FREQ_IN_DOMAIN,
    ),
    delta_ratio_stats=_delta_ratio_stats_all(),
    cells={},
    verdict={d: dict(
        hit_window=_verdict_to_json(verdict[d]["hit_window"]),
        native_window=_verdict_to_json(verdict[d]["native_window"]),
    ) for d in DOMAIN_NAMES},
    runs=[],
)

# per-cell aggregates (mean ± std)
for d in DOMAIN_NAMES:
    for c in ("targeted", "control"):
        for s in BOOST_STRENGTHS:
            cell = cell_results[d][c].get(s, None)
            if cell is None:
                continue
            steering_payload["cells"].setdefault(d, {}).setdefault(c, {})[f"{s:.6f}"] = {
                "strength": float(s),
                "gain": float(GAIN),
                "n": int(cell["n"]),
                "hit_mean": float(cell["hit_mean"]),
                "hit_std":  float(cell["hit_std"]),
                "native_mean": float(cell["native_mean"]),
                "native_std":  float(cell["native_std"]),
                "rep_mean": float(cell["rep_mean"]),
                "rep_std":  float(cell["rep_std"]),
                "ent_mean": float(cell["ent_mean"]),
                "ent_std":  float(cell["ent_std"]),
                "dr_mean":  float(cell["dr_mean"]),
                "dr_std":   float(cell["dr_std"]),
                "n_collapsed": int(cell["n_collapsed"]),
            }

# all individual repeats (144 = 4*2*6*3)
for d in DOMAIN_NAMES:
    for c in ("targeted", "control"):
        for r in raw_runs[d][c]:
            steering_payload["runs"].append(dict(
                condition=r["condition"], domain=r["domain"], rep_n=r["rep_n"],
                strength=r["strength"], gain=r["gain"],
                hit=r["hit"], native_hit=r["native_hit"],
                rep=r["rep"], ent=r["ent"],
                delta_ratio=r["delta_ratio"],
                collapsed=r["collapsed"], n_tokens=r["n_tokens"],
                steered_slots=list(r["steered_slots"]),
            ))

with open(f"{OUT}/exp4c_steering.json", "w", encoding="utf-8") as f:
    json.dump(steering_payload, f, indent=2, ensure_ascii=False, default=str)

print(f"saved {OUT}/exp4c_steering.json "
      f"({len(steering_payload['runs'])} individual repeats + "
      f"{sum(len(steering_payload['cells'].get(d, {}).get(c, {})) for d in DOMAIN_NAMES for c in ('targeted','control'))} cells, "
      f"{len(DOMAIN_NAMES)} domains, "
      f"2 conditions, {len(BOOST_STRENGTHS)} strengths, "
      f"{REPEATS} repeats)")


# %%
# Remove the steering hook now that the steering block is done so the
# hook never accidentally fires if the cell is re-run.
_steering_handle.remove()
print(f"removed steering hook from pythia.gpt_neox.layers[{cfg.extract_layer - 1}]")
