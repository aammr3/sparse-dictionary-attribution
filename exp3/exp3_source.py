# %% [markdown]
# # Experiment 3 — Does slot specialization replicate on real pretrained activations?
#
# **The single question:** exp0/exp0d showed that *unsupervised* sparse-dictionary
# slots specialize cleanly into per-domain detectors on activations from a tiny
# from-scratch transformer. The natural next question is: does that still happen
# when the activations come from a **real pretrained model**?
#
# **This experiment:** Load `EleutherAI/pythia-160m`, freeze it entirely, and
# stream its layer-6 residual stream into a 4096-slot TopK dictionary trained
# *only* as an autoencoder on those activations. No language-model loss, no
# domain supervision, no alpha blending, no trunk modification. Pythia itself
# is never trained, never fine-tuned -- it is a frozen feature extractor.
#
# **Why this matters:** exp0/exp0d's mechanism was demonstrated on a tiny toy
# model that was itself shaped by the same optimization pressure as the
# registry. On a real pretrained model, the activations already carry rich
# semantic structure (the model "knows" about medicine, law, code, literature
# because it was trained on text from the open web). If the registry can
# still find domain-specialized slots there -- and find them clearly *more*
# specialized than the raw hidden dimensions at matched sparsity -- then
# exp0's specialization finding replicates on real pretrained
# representations, not just on a from-scratch toy.
#
# **Setup (mirrors exp0d, but on real activations):**
# - Pythia-160m, **fully frozen** (`requires_grad=False` on every parameter,
#   `torch.no_grad()` around every forward call)
# - Activation source: `output_hidden_states=True`, take `hidden_states[6]`
#   (middle of the 12-layer stack), shape `[B, T, 768]`
# - **Streamed on-the-fly** inside the training loop -- never persisted to
#   disk (4 domains x millions of tokens x 768 dims would be far too large)
# - 4 domains (medicine / law / code / literature) -- same HuggingFace
#   sources as exp1, tokenized with **Pythia's own** GPTNeoX BPE tokenizer
# - Registry config: `n_slots=4096, k=16, k_aux=32, dead_after=800` -- identical
#   to exp0d's winning configuration
# - Only loss: reconstruction (MSE) + AuxK auxiliary loss. No LM loss, no
#   alpha curriculum, no supervision
#
# **The decisive metric:** registry-slot purity vs. raw-dim purity at matched
# sparsity `k=16`, computed exactly the same way exp0/exp0d did
# (top-firing histograms across the 4 domain labels,
# `purity = max(domain_counts) / sum(domain_counts)`). If the registry's
# mean purity is meaningfully higher than the raw-dim baseline (>0.10 gap),
# then exp0's specialization finding replicates on real pretrained
# activations.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings -> Accelerator -> GPU T4 x2** (or P100)
# 2. **Settings -> Internet -> On**  <- required to download Pythia + data
#
# Expected runtime: ~12-18 minutes total. Pythia's forward is more expensive
# per token than exp0's tiny transformer, so the token budget is shrunk to
# ~3M tokens/domain (~5x less than exp0d's 15M) to keep wall-time well
# under the 20-minute Kaggle T4x2 GPU budget.
#
# ---
# **Why this is Phase G, first experiment.** Phases A-F built the unsupervised
# finding on a toy model. Phase G asks: does the same effect appear on real
# pretrained activations? If yes, the registry mechanism is doing real work on
# real representations; if no, exp0's result was an artifact of the toy
# setup and the rest of the project needs to be rethought. exp3 is the
# cheapest possible version of that question (smallest pretrained model,
# smallest dictionary, one frozen trunk, no language-model interaction).

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
import os, math, time, json, random
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
    pythia_name   = "EleutherAI/pythia-160m"
    extract_layer = 6           # hidden_states[6] = output of layer 6 (middle of 12)
    hidden_size   = 768         # pythia-160m hidden_size

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
assert N_DOMAINS == 4, "exp3 is hard-coded to 4 domains"

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
# `EleutherAI/pythia-160m` (12 layers, hidden_size=768, ~162M params) is loaded
# once, put into `.eval()`, and **every parameter is frozen** (`requires_grad=False`).
# All Pythia forward calls are wrapped in `torch.no_grad()`. The model is a
# pure feature extractor in this experiment: the only trainable component is
# the registry.
#
# `hidden_states[6]` (the output of the 7th of 12 layers -- roughly the middle
# of the stack) is taken as the residual-stream activation tensor `h` of
# shape `[B, T, 768]`. `hidden_states` is a 13-tuple when
# `output_hidden_states=True` (embeddings + outputs after each of the 12
# layers), so `index 6` picks the post-layer-6 residual stream.

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
# - `d_model` is now `768` (Pythia's hidden size) instead of `256`
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

    @staticmethod
    def _topk(x, k):
        v, i = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        return out.scatter_(-1, i, v)

    def forward(self, h, step=None, dead_after=None):
        pre  = F.relu(self.encoder(h - self.b_dec))
        acts = self._topk(pre, self.k)
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
#    as the activation tensor `h` of shape `[B, T, 768]`. `h` lives only in
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
    h = pythia_out.hidden_states[cfg.extract_layer]  # [B, T, 768]
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
torch.save(registry.state_dict(), f"{OUT}/exp3_model.pt")
print(f"saved registry state_dict to {OUT}/exp3_model.pt "
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
#   the 768 Pythia hidden dimensions lands in the top-16 by absolute
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

            # Streamed: frozen pythia forward -> grab layer-6 residual
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                pythia_out = pythia(x, output_hidden_states=True)
            h = pythia_out.hidden_states[cfg.extract_layer].float()  # [B, T, 768]

            # Registry on the streamed activation
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
                recon, acts, _ = registry(h, step=None, dead_after=cfg.dead_after)

            acts_f = acts.float().flatten(0, 1)                 # [B*T, n_slots]
            slot_cnt[d] += (acts_f > 0).float().sum(0)

            # raw-dim top-k at matched sparsity (same k as the registry)
            h_flat = h.flatten(0, 1)                            # [B*T, 768]
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
    nxt = ("next: try a slightly larger pretrained model (pythia-410m) to see "
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
plt.savefig(f"{OUT}/exp3_results.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp3_results.png")


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
with open(f"{OUT}/exp3_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(f"\nsaved {OUT}/exp3_summary.json")
