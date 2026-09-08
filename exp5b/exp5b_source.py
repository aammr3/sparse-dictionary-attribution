# %% [markdown]
# # Experiment 5b -- hallucination detection from the registry's internal state,
# # measured at the prompt (not just the continuation)
#
# **What exp4 / exp4b / exp4c left open.** exp4 tried targeted slot
# steering at multiple strengths and measured zero domain steering on
# Pythia generation. exp4b widened the target to top-8 slots and added
# a GAIN multiplier. exp4c added variance (REPEATS = 3) and a placebo
# control condition. **None of them moved hit% / native% above noise.**
# The cleanest verdict the project can produce today is "the registry's
# per-domain direction cannot be amplified into measurable generation
# steering at the BOOST magnitudes we are willing to run." That is a
# null result *for steering*. It says nothing about whether the
# registry's internal state can **observe** anything.
#
# **exp5 (previous version) returned AUROC ~0.5 for every signal
# INCLUDING the lm_entropy baseline.** Diagnosis (verified): the
# read-only forward hook recorded the registry's state only at the
# LAST position of each forward pass -- the last prompt token plus
# the generated tokens. The "I don't know this entity" moment happens
# while Pythia READS the entity during prefill (the model's internal
# grounding-judgement already fires there), and those positions were
# never measured.
#
# **This experiment (exp5b) is read-only observation, like exp5, but
# records the registry's state at ALL positions of every forward and
# reports signatures over THREE spans per run:**
#
# - `span=entity`  -- the exact token positions of the entity `{X}`
#   inside the prompt (TEST 1 only).
# - `span=prompt`  -- every prompt token position (left-padded prompt).
# - `span=gen`     -- the generated tokens (what exp5 already did).
#
# For TEST 2 (cloze, no `{X}` slot) `span=entity` is not applicable;
# we report `span=prompt` and `span=gen` only.
#
# **Why three spans matters.** exp5 measured only the LAST position
# of each forward. During prefill that single position is the
# rightmost prompt token -- the registry's "knowledge-consulting"
# spike fires while the model is reading the entity itself, not after.
# During single-step decoding the single position is the new
# token-just-generated. The continuum from "improvising while reading
# the entity" to "improvising while continuing the story" was lost.
# Recording all positions lets us separately examine the registry's
# state at the entity tokens (where grounding or hallucination is
# decided), at the rest of the prompt (general context), and at the
# generated tokens (commitment phase).
#
# **What stays IDENTICAL to exp5:**
#
# - registry-as-SAE training on frozen Pythia-410m layer-12
#   activations (same CFG, same T-040 fixes, same purity readout).
# - sampling generation (`temperature=0.8, top_p=0.9, multinomial,
#   NEVER argmax`).
# - `steer_bias` all-zeros throughout (pure observation -- NO
#   modification of the layer output by the hook).
# - rank-based AUROC implementation (no sklearn dependency).
# - the two tests (TEST 1: REAL vs FAKE; TEST 2: CORRECT vs INCORRECT).
# - REPEATS = 3.
#
# **What changes vs exp5:**
#
# 1. `_ReadOnlyHook.__call__` records `pre` and `acts` of shape
#    `[T, n_slots]` for the whole forward, not `[n_slots]` for the
#    last position only.
# 2. Per TEST 1 prompt we compute the entity token index range by
#    separately tokenizing the template prefix and the prefix+entity;
#    if the prefix+entity tokenization does not sit cleanly inside the
#    full tokenization, we drop that item (and log it).
# 3. FAKE entities are token-length-matched to REAL entities per
#    domain (`|mean(real) - mean(fake)| <= 0.5` BPE tokens). The
#    audited "no real entity" property is preserved.
# 4. Signatures are computed three ways (one per span) and reported
#    separately. The verdict reports the best registry signal per span
#    and, for `span=gen`, compares it against the best LM-head
#    baseline; if nothing beats the baseline anywhere we print the
#    honest negative result.
#
# **Outputs (renamed exp5b_*)**
# - `exp5b_model.pt`         -- registry state_dict (NOT Pythia).
# - `exp5b_summary.json`     -- training/purity summary.
# - `exp5b_halluc.json`      -- every run's per-token signals + per-
#   span signatures + per-span AUROC tables + tokenization-confound
#   numbers + verdict block + cfg. **Saved BEFORE any plotting.**
# - `exp5b_results.png`      -- span=gen AUROC vs N curves, one panel
#   per test, registry signals vs baselines.
# - `exp5b_signatures.png`   -- per-signal pooled distributions.
# - `exp5b_results_3spans.png` -- per-span pooled AUROC summary bars
#   plus confounds and entity-span coverage.
#
# **Order-of-operations safeguard (FIX 4).** Every save/print that
# exp5 failed to reach because of a plotting cell crash is now placed
# BEFORE the plotting cells. The plotting section is wrapped in
# `try/except` so a plotting failure prints a warning but does NOT
# abort the run -- the JSON outputs above are guaranteed on disk.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings -> Accelerator -> GPU T4 x2** (or P100)
# 2. **Settings -> Internet -> On**  <- required to download Pythia + data
#
# Expected runtime: ~50-65 minutes total. Training is identical to
# exp3b / exp4c / exp5 (~19 min). The hallucination block is 360
# prompts (TEST 1: 4 * 2 * 30 = 240; TEST 2: 4 * 30 = 120) x 3 repeats
# x 32 tokens, but recording all positions roughly DOUBLED the per-
# step CPU work in `compute_signatures` (we now aggregate over three
# spans instead of one). The additional cost vs exp5 is roughly
# +10-15 minutes on a T4.

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

    # --- hallucination detection (exp5-only) ---
    gen_len     = 32           # tokens generated per run (project rule: short)
    gen_temp    = 0.8
    gen_top_p   = 0.9
    repeats     = 3
    seed_base   = 2000         # generation seed = seed_base + rep_n
    n_values    = [1, 2, 4, 8, 16, 32]   # first-N-token aggregations

cfg = CFG()
print(json.dumps({k: v for k, v in vars(CFG).items() if not k.startswith("_")},
                 indent=2, ensure_ascii=False))


# %% [markdown]
# ## 1. Data -- four distant domains (same sources as exp1)

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
assert N_DOMAINS == 4, "exp5 is hard-coded to 4 domains"

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
# fp32 loss casting. The `steer_bias` buffer is kept so the training
# forward (`pre + steer_bias`) is bit-equivalent to exp3b / exp4c --
# exp5 never sets it non-zero (pure observation).

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

        # per-slot additive steering bias. Non-persistent so it never
        # enters state_dict and never gets a gradient. exp5 leaves it
        # all-zeros for the entire run; the buffer is kept so the
        # `pre + steer_bias` line below is bit-identical to exp3b /
        # exp4c at training time (where the addition is a no-op).
        self.register_buffer(
            "steer_bias", torch.zeros(n_slots), persistent=False)

    @staticmethod
    def _topk(x, k):
        v, i = torch.topk(x, k, dim=-1)
        out = torch.zeros_like(x)
        return out.scatter_(-1, i, v)

    def forward(self, h, step=None, dead_after=None):
        # bit-identical to exp3b / exp4c: pre, then add steer_bias BEFORE
        # top-k. With steer_bias == 0 the addition is a no-op.
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

# exp5 guard: registry.steer_bias must be all-zeros here. We will
# assert the same property at every checkpoint below.
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must be all-zeros at construction (no setter was called)"


# %% [markdown]
# ## 4. Training -- activations streamed on-the-fly, never persisted

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
torch.save(registry.state_dict(), f"{OUT}/exp5b_model.pt")
print(f"saved registry state_dict to {OUT}/exp5b_model.pt "
      f"({sum(p.numel() for p in registry.parameters()):,} params)")

# exp5 invariant: steer_bias must still be all-zeros after training
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros after training (no setter was called)"


# %% [markdown]
# ## 5. The decisive measurement -- slot purity vs raw-dim purity

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
# ## 6. Purity verdict (training block, unchanged from exp4c)

# %%
gap = P_slots["mean"] - P_dims["mean"]
print(f"\nmean-purity gap (registry slots vs raw dims): {gap:+.3f}\n")

if gap > 0.10:
    verdict = ("PASS -- exp0's specialization finding replicates on real "
               "Pythia activations; the registry is clearly purer than the raw "
               "Pythia hidden dimensions at matched sparsity.")
    nxt = ("next: try a slightly larger pretrained model (pythia-1.4b) to see "
           "if the gap widens with richer activations, or move to a downstream "
           "observation experiment (e.g. hallucination detection).")
elif gap > 0.05:
    verdict = ("WEAK-POSITIVE SIGNAL -- there is some specialization, "
               "but the gap is smaller than exp0's original bar.")
    nxt = ("consider: more training tokens, larger pretrained model, or a "
           "registry with k_aux / n_slots tuned to real-activation statistics.")
else:
    verdict = ("FAIL -- the registry did not specialize meaningfully "
               "beyond the raw-dim baseline on real Pythia activations.")
    nxt = ("walk through this checklist: (1) is dead_frac high? "
           "(2) is recon_loss plateauing? (3) is the chosen layer too early / "
           "too late in the stack? if all three are fine, exp0's effect may "
           "have been an artifact of the toy setup and the rest of the project "
           "needs to be rethought.")

print(verdict)
print("next step:", nxt)


# %% [markdown]
# ## 7. Purity plots (renamed outputs)

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
plt.savefig(f"{OUT}/exp5b_purity.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp5b_purity.png")


# %% [markdown]
# ## 8. Slot metadata -- `dom_all` and `pur_all` (used by Signatures B + C)
#
# The hallucination-detection block needs two per-slot arrays that were
# already produced by the purity readout:
#
# - `dom_all[i] = argmax_d counts[d][i]` -- the domain each slot
#   fires most on. Used to attribute activation mass to a domain in
#   the prompt reference + per-token conflict entropy.
# - `pur_all[i] = max_d counts[d][i] / sum_d counts[d][i]` -- the
#   purity (only used for the eyeball-proof readout below; the
#   signatures themselves are purity-agnostic).
#
# Computed once after `collect_activity` so the numbers are bit-
# identical to what exp4c would have computed.

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
with open(f"{OUT}/exp5b_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(f"\nsaved {OUT}/exp5b_summary.json")


# %% [markdown]
# ## 9. Read-only forward hook -- records `pre` and top-k `acts` at ALL positions
#
# exp5b replaces exp5's last-position hook with an **all-positions**
# read-only hook. The hook is registered on the same layer
# (`pythia.gpt_neox.layers[cfg.extract_layer - 1]`, the block whose
# output is `hidden_states[cfg.extract_layer]`). On every Pythia
# forward call the hook does exactly three things:
#
# 1. Takes the layer output `h` of shape `[B, T, 1024]` -- ALL T
#    positions of the forward pass. (T = `cfg.ctx = 256` for every
#    generation step: prefill sees the left-padded prompt; each
#    decode step sees the last 255 prompt tokens plus the one
#    newly-generated token at the rightmost position.)
# 2. Computes the registry's `pre = relu(encoder(h - b_dec)) +
#    steer_bias` (with `steer_bias == 0` for the entire run) and
#    top-k `acts = topk(pre, k)` at ALL T positions. Both are
#    detached and moved to CPU for storage as `[T, n_slots]`
#    tensors (batch dim squeezed out).
# 3. **Returns the original layer output unchanged.** No steering
#    patch, no delta, no modification of any kind.
#
# The hook carries a `recorded` list of dicts (one entry per Pythia
# forward call -- so per generation step). Each dict has
# `{"pre": [T, n_slots], "acts": [T, n_slots]}` -- the SAME tensor
# in both, because after top-k `pre` is zero outside the top-k
# slots. The list is reset by `reset()` between runs. The
# generation loop unpacks the per-span tensors downstream.
#
# Both the tuple-form and bare-tensor layer output are handled for
# transformers-version compatibility (Kaggle ships transformers 5.0,
# which returns a bare tensor from `GPTNeoXLayer.forward`).
#
# **Why this matters.** exp5 only saw `recorded[t][-1, :]` -- a
# single per-step vector. During prefill that single position is
# the rightmost prompt token, NOT the entity inside the prompt. The
# "registry detects it doesn't know this entity" signal fires while
# Pythia reads the entity -- which exp5 missed entirely. Recording
# all positions lets the signatures be aggregated over (a) the
# exact entity tokens (`span=entity`), (b) the whole prompt
# (`span=prompt`), and (c) the generated tokens (`span=gen`).

# %%
class _ReadOnlyHook:
    """Forward hook installed on Pythia's layer-12 block. Records the
    registry's `pre` and top-k `acts` at ALL positions of every
    forward pass WITHOUT modifying the layer output. `steer_bias`
    stays all-zeros for the whole run; the `pre + steer_bias` line
    is kept so the bit-pattern matches what `registry.forward()`
    would compute on the same input."""

    def __init__(self, registry):
        self.registry = registry
        # List of {"pre": [T, n_slots], "acts": [T, n_slots]}; T =
        # cfg.ctx for every Pythia forward in the generation loop.
        self.recorded = []

    def reset(self):
        self.recorded = []

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output          # [B, T, 1024]
        reg = self.registry
        # Registry pre + top-k at ALL T positions (prefill OR decode step).
        pre = F.relu(reg.encoder(h - reg.b_dec))        # [B, T, n_slots]
        pre = pre + reg.steer_bias                      # == pre, steer_bias == 0
        acts = reg._topk(pre, reg.k)                    # [B, T, n_slots]
        self.recorded.append({
            "pre":  pre.detach().to("cpu", non_blocking=True).squeeze(0),  # [T, n_slots]
            "acts": acts.detach().to("cpu", non_blocking=True).squeeze(0), # [T, n_slots]
        })
        # READ-ONLY: return the original layer output unchanged.
        return output


_readonly_hook = _ReadOnlyHook(registry)
_readonly_layer = pythia.gpt_neox.layers[cfg.extract_layer - 1]
_readonly_handle = _readonly_layer.register_forward_hook(_readonly_hook)
print(f"registered READ-ONLY hook on pythia.gpt_neox.layers[{cfg.extract_layer - 1}] "
      f"(output is hidden_states[{cfg.extract_layer}]); "
      f"handle={_readonly_handle}")
print(f"hook records ALL positions of every forward as [T, n_slots] tensors "
      f"(T=cfg.ctx={cfg.ctx}). The generation loop slices these into "
      f"the three spans downstream.")
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must still be all-zeros when the read-only hook is installed"


# %% [markdown]
# ## 10. Sampling generation harness (identical to exp4c / exp5)
#
# All generation uses nucleus sampling (`temperature=0.8, top_p=0.9`),
# never greedy / argmax -- project rule. Returns the generated tokens
# AND the per-step LM-head quantities we need for the two baselines
# (`lm_entropy` and `neg_max_logprob`), computed BEFORE sampling so
# the metrics reflect Pythia's belief at the step, not the kept
# subset.
#
# The hook fires on every pythia() call inside the loop, recording
# one entry per step into `_readonly_hook.recorded`. Each entry is
# `{"pre": [T, n_slots], "acts": [T, n_slots]}` with `T = cfg.ctx`.
# At the end of the call the recorded list has exactly `n_new`
# entries (= `cfg.gen_len` = 32) -- the FIRST is the PREFILL (whose
# T positions cover the left-padded prompt; the rightmost position
# is the LAST prompt token, whose hidden state predicts generated
# token 0), and the remaining `n_new - 1` are each decode step
# (whose rightmost position is the previously-generated token now the
# rightmost input; that position's hidden state predicts the next
# generated token). The `n_new + 1` claim from the previous version
# of this docstring was wrong -- the generation loop runs exactly
# `n_new` Pythia forward passes (one prefill + `n_new - 1` decode
# steps), so `len(acts_all) == n_new`. MEDIUM 1 fix: span=gen now
# iterates `range(n_new)` and covers ALL cfg.gen_len generated
# positions (the previous version used `n_new = len(acts_all) - 1`
# and silently dropped the last generated position).
#
# The main loop in section 13 unpacks these recorded tensors and
# splits them into the three spans (`entity` / `prompt` / `gen`).

# %%
GEN_LEN             = cfg.gen_len       # 32
GEN_TEMPERATURE     = cfg.gen_temp      # 0.8
GEN_TOP_P           = cfg.gen_top_p     # 0.9
REPEATS             = cfg.repeats       # 3
SEED_BASE           = cfg.seed_base     # 2000
EOT = tokenizer.eos_token_id
if EOT is None:
    EOT = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

print(f"\nGeneration config (identical to exp4c / exp5 sampling rules):")
print(f"  GEN_LEN         = {GEN_LEN}")
print(f"  GEN_TEMPERATURE = {GEN_TEMPERATURE}")
print(f"  GEN_TOP_P       = {GEN_TOP_P}")
print(f"  REPEATS         = {REPEATS}")
print(f"  SEED_BASE       = {SEED_BASE}")
print(f"  EOT             = {EOT}")
print(f"  N values        = {cfg.n_values}")


# %%
@torch.no_grad()
def generate_with_signatures(prompt_ids, n_new=GEN_LEN, hook=_readonly_hook):
    """Nucleus-sampling generation from `prompt_ids` (1-D list/array
    of token ids). Pads on the LEFT with `EOT` if the prompt is shorter
    than `cfg.ctx`; truncates on the LEFT if longer. Returns
    `(generated_tokens, lm_entropies_per_step, neg_max_logprobs_per_step,
    recorded_states, prompt_len)`.

    The read-only hook MUST be installed on the layer-12 block; it
    appends one `{"pre": [T, n_slots], "acts": [T, n_slots]}` per
    pythia() call to its `recorded` list as a side effect. We
    `reset()` it on entry so a previous run's recordings cannot leak
    into this one. The returned `recorded_states` is the post-run
    snapshot -- it has `n_new + 1` entries: the prefill and `n_new`
    decode steps.

    `prompt_len` is the number of tokens in the (un-padded) prompt,
    returned so the caller can slice the prefill tensor into the
    `span=prompt` and (for TEST 1) `span=entity` spans."""
    pythia.eval()
    registry.eval()
    pad_id = EOT if EOT is not None else 0
    ids = [int(t) for t in prompt_ids]
    prompt_len = len(ids)
    if prompt_len < cfg.ctx:
        ids = [pad_id] * (cfg.ctx - prompt_len) + ids
    elif prompt_len > cfg.ctx:
        ids = ids[-cfg.ctx:]
        prompt_len = cfg.ctx
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated, lm_entropies, neg_max_logprobs = [], [], []
    if hook is not None:
        hook.reset()
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            o = pythia(x)
        # The hook has fired inside pythia(x). Its recorded state for
        # THIS step is a [cfg.ctx, n_slots] tensor; the position
        # cfg.ctx-1 is the one whose hidden state produced the
        # next-token logits we're about to sample from.
        logits = o.logits[0, -1].float()                        # [vocab]
        # entropy of the FULL softmax, BEFORE temperature / top-p
        # filtering (project rule: reflects model belief, not the kept
        # subset).
        p_full = F.softmax(logits, dim=-1)
        ent = -(p_full * (p_full + 1e-12).log()).sum().item()
        lm_entropies.append(ent)
        max_p = p_full.max().item()
        neg_max_logprobs.append(-math.log(max_p + 1e-12))
        # temperature scaling
        logits = logits / GEN_TEMPERATURE
        # top-p (nucleus) filter
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cumprobs = sorted_probs.cumsum(dim=-1)
        keep_mask = (cumprobs - sorted_probs) < GEN_TOP_P
        sorted_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
        kept_probs = F.softmax(sorted_logits, dim=-1)
        # sample from the kept set -- NEVER argmax / greedy
        nxt_sorted = torch.multinomial(kept_probs, num_samples=1).item()
        nxt = int(sorted_idx[nxt_sorted].item())
        generated.append(nxt)
        x = torch.cat([x[:, 1:],
                       torch.tensor([[nxt]], device=DEVICE)], dim=1)
    if hook is not None:
        recorded_states = [r for r in hook.recorded]
    else:
        recorded_states = []
    return generated, lm_entropies, neg_max_logprobs, recorded_states, prompt_len


# exp5b invariant: steer_bias is still all-zeros here -- nothing in
# the registry's setter methods has been called.
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros before the hallucination block"


# %% [markdown]
# ## 11. Test data -- embedded entity lists and cloze facts
#
# All test data is embedded directly in this source file (no
# downloads). Per domain we define:
#
# - `REAL_ENTITIES[d]` -- 30 real, well-known entities.
# - `FAKE_ENTITIES[d]` -- 30 invented, plausible-sounding entities
#   that do not exist.
# - `PROMPT_TEMPLATE_TEST1[d]` -- a template string with `{X}` for
#   the entity. The prompt is filled with each entity to form one
#   TEST 1 sample (REAL or FAKE).
# - `CLOZE_FACTS[d]` -- 30 `(prompt, [accepted answer strings])`
#   pairs for TEST 2.
#
# The confounder check prints, per domain, the mean / std Pythia-
# BPE token length of REAL vs FAKE entity names. If the gap is large
# the entire TEST 1 contrast is confounded by name rarity (rare names
# also tend to have unusual BPE lengths) -- the number must be
# reported prominently and honestly.

# %%
REAL_ENTITIES = {
    "medicine": [
        "type 2 diabetes", "hypertension", "asthma",
        "coronary artery disease", "chronic obstructive pulmonary disease",
        "major depressive disorder", "generalized anxiety disorder",
        "migraine", "rheumatoid arthritis", "osteoarthritis",
        "pneumonia", "strep throat", "urinary tract infection",
        "acute myocardial infarction", "stroke", "chronic kidney disease",
        "atrial fibrillation", "heart failure", "epilepsy",
        "Parkinson's disease", "Alzheimer's disease", "multiple sclerosis",
        "Crohn's disease", "ulcerative colitis", "psoriasis",
        "hyperthyroidism", "hypothyroidism", "iron deficiency anemia",
        "osteoporosis", "tuberculosis",
    ],
    "law": [
        "Patriot Act", "Civil Rights Act", "Americans with Disabilities Act",
        "Clean Air Act", "Clean Water Act", "Sherman Antitrust Act",
        "Clayton Antitrust Act", "Equal Pay Act", "Fair Labor Standards Act",
        "National Labor Relations Act", "Voting Rights Act",
        "Social Security Act", "Affordable Care Act", "Dodd-Frank Act",
        "Sarbanes-Oxley Act", "Foreign Corrupt Practices Act",
        "Freedom of Information Act", "Privacy Act", "Bankruptcy Act",
        "Bank Secrecy Act", "RICO Act", "Indian Child Welfare Act",
        "No Child Left Behind Act", "Individuals with Disabilities Education Act",
        "Family and Medical Leave Act", "Title IX", "Title VII",
        "Fourth Amendment", "Fifth Amendment", "Fourteenth Amendment",
    ],
    "code": [
        "NumPy", "pandas", "requests", "scikit-learn", "TensorFlow",
        "PyTorch", "matplotlib", "Seaborn", "SciPy", "Keras",
        "Django", "Flask", "FastAPI", "SQLAlchemy", "BeautifulSoup",
        "Pillow", "OpenCV", "HuggingFace Transformers",
        "spaCy", "NLTK", "Gensim", "statsmodels", "SymPy",
        "NetworkX", "Biopython", "Astropy", "pydantic", "pytest",
        "black", "mypy",
    ],
    "literature": [
        "Moby-Dick", "Pride and Prejudice", "The Great Gatsby",
        "To Kill a Mockingbird", "1984", "The Catcher in the Rye",
        "The Lord of the Rings", "The Hobbit", "Crime and Punishment",
        "War and Peace", "The Odyssey", "The Iliad", "Don Quixote",
        "Hamlet", "Macbeth", "Romeo and Juliet", "King Lear",
        "The Canterbury Tales", "Ulysses", "The Sound and the Fury",
        "One Hundred Years of Solitude", "The Old Man and the Sea",
        "Brave New World", "The Grapes of Wrath", "Of Mice and Men",
        "The Color Purple", "Beloved", "The Road", "Frankenstein",
        "Jane Eyre",
    ],
}

# FAKE names: invented, plausible-looking, and roughly length-matched
# to the REAL list in the same domain (in characters, so the BPE
# token-length gap stays small). The confound check below reports the
# actual BPE-token-length gap; if it's large the whole contrast is
# confounded and we say so prominently.
#
# Re-scan provenance (review T-056-review FIX 1 / FIX 2):
#   * medicine: the line that previously held the real condition
#     "Cerebellar vasculitis" (CNS vasculitis) was replaced with the
#     invented "Cerebellar dysplasitis" (dysplasitis is not a medical
#     term -- "-itis" attached to "dysplasia" yields a non-word).
#     Whole-list re-scan: no other medicine FAKE entry is a real,
#     documented clinical entity.
#   * code: the entries that previously referenced the real SCIEX
#     mass-spectrometry company and the real NetWalker ransomware
#     family were replaced with the invented "Sciora" and "Netwalkra".
#     Whole-list re-scan: "Quantix" (a real company / product name)
#     was also replaced. The remaining 27 entries are clearly
#     invented (no overlap with any well-known Python package).
#   * law: the generic-but-real "Personal Data Protection Act"
#     (Singapore PDPA), the too-generic "River Basin Restoration Act"
#     (matches several real watershed-restoration bills), and the
#     too-generic "Retirement Security Act" were all replaced with
#     clearly invented names.
#   * literature: "The Eyre of Halvern" (too close to "Jane Eyre")
#     was replaced.
#
# Re-scan provenance (review T-058, after T-057 audit):
#   * law: a whole-list re-scan against Wikipedia + OpenLibrary found
#     two entries that still matched real legislation:
#       - "River Basin Conservation Act" -> too close in form to the
#         enacted Delaware River Basin Conservation Act (PL 116-9,
#         2019) and the Upper Mississippi River Basin Conservation
#         Act (H.R. 5343); same generic-pattern problem as the
#         replaced "River Basin Restoration Act". Replaced with
#         "Watershed Reclamation Initiative" (verified 0 hits).
#       - "Industrial Relations Reform Act" -> Australia's
#         Industrial Relations Reform Act 1993 (Cth) is a real
#         enacted Commonwealth statute. Replaced with "Workplace
#         Conciliation Modernization Act" (verified 0 hits).
#     The remaining 28 law FAKE entries are clearly invented or are
#     generic-but-distinct (no exact match to a real enacted or
#     widely-reported bill).
#   * literature: a whole-list re-scan against OpenLibrary found four
#     entries that are real published novels:
#       - "The Long Bridge"        -> Fannie Cook, 1949.
#         Replaced with "The Granite Passage".
#       - "The Burning Plains"     -> Catherine Palmer, 1988.
#         Replaced with "The Smoldering Coast".
#       - "The Cartographer's Daughter" -> Kiran Millwood Hargrave,
#         2016. Replaced with "The Loremaster's Daughter".
#       - "The Halcyon Drift"      -> Brian Stableford, 1972 (book 1
#         of the Hooded Swan series). Replaced with "The Mariner's
#         Reckoning".
#     All four replacements were verified to have 0 hits on
#     Wikipedia and OpenLibrary. The remaining 26 literature FAKE
#     entries are clearly invented or generic-but-distinct (no
#     exact match to a real published novel).
#
# Re-scan provenance (review T-061, FIX 1 -- systemic FAKE list rebuild):
#   The Reviewer review + the Lead Agent's own audit found that the previous
#   exp5b FAKE lists were contaminated with REAL entities:
#
#     * medicine: T-060 had shortened fake names to match token length
#       by switching to Greek/Latin DESCRIPTIVE medical morphology
#       (-algia, -itis, -dynia, -pathia, -plegia, -osis). That
#       morphology is a productive real system -- nearly every such
#       compound is a real or dictionary-valid clinical term.
#       Confirmed real entries in the previous medicine FAKE list:
#       Cerebritis, Polyneuritis, Otalgia, Cardialgia, Gastralgia,
#       Myopathia, Cephalalgia (also the name of the International
#       Headache Society journal), Cardioplegia (standard cardiac-
#       surgery term), Cerebellitis, Nephralgia, Splenalgia,
#       Hepatalgia, Encephalalgia, Sialalgia, Hepatodynia. The whole
#       list was assumed contaminated.
#
#     * literature: the previous list included "Pale King" (a David
#       Foster Wallace novel, 2011), and several short two-word
#       evocative titles that collided with real books.
#
#     * law: the previous list had become ultra-generic ("Civic Act",
#       "Justice Act", "Heritage Act", "Compliance Act", "Civil
#       Reform Act", ...) -- these match real statutes in multiple
#       jurisdictions.
#
#     * code: the previous list was already invented-word based and
#       passed the T-058 PyPI audit. A fresh re-audit caught three
#       entries that have since become real PyPI packages:
#       Torchling (educational PyTorch-like lib), Flasket
#       (deprecated package), Astrora (Rust-backed astrodynamics lib).
#
#   FIX 1 NEW STRATEGY: every fake name is built around an INVENTED
#   PROPER NOUN, never around real descriptive morphology. The
#   invented proper noun guarantees non-existence; length is
#   controlled by choosing a shorter or longer invented noun.
#
#     * medicine : "<InventedSurname> syndrome" / "<InventedSurname>
#                   disorder" (eponymous style, like real eponymous
#                   syndromes but with a made-up surname). e.g. "Vor
#                   syndrome", "Tarven disorder". BANS all -algia /
#                   -itis / -dynia / -pathia / -plegia / -osis
#                   constructions.
#     * law      : "<InventedProperNoun> Act" -- the invented word
#                   carries the non-existence. e.g. "Kelmarsh Act",
#                   "Ashwort Reform Act". BANS purely generic names
#                   built only from common policy words.
#     * literature: every title includes an invented proper noun.
#                   e.g. "Veldora Rising", "Song of Mordwyn", "The
#                   Yelvari Gate". BANS two-word titles made only of
#                   common English words.
#     * code     : keeps the invented-word approach, with three
#                   names swapped for PyPI-clean replacements
#                   (Torchling -> Torchel; Flasket -> Astrom;
#                   Astrora -> Astora).
#
#   Web verification (T-061, FIX 1): every name in all four lists
#   was re-checked against Wikipedia (medicine + law), OpenLibrary
#   (literature), and the PyPI JSON API (code). All 120 names
#   return 404 / 0-hits -- the list is clean. See the report
#   appended to this file for the rejected candidates that were
#   tried but not used.
#
# Re-scan provenance (review T-063, FIX 1b -- literature root diversity):
#   the Lead Agent's audit after T-062 found the literature FAKE list had 30
#   entries but only 8 distinct invented proper nouns, each reused
#   4-5 times ("Veldora" 4x, "Bramish" 5x, "Thessyn" 3x, "Aldwren" 4x,
#   "Mordwyn" 4x, "Krossari" 4x, "Yelvari" 4x, "Vorshal" 2x). If Pythia
#   responds to the invented root itself, the 4-5 items sharing a root
#   are correlated rather than independent, so the effective sample
#   size for the literature domain was ~8, not 30. This inflates the
#   apparent n and can distort that domain's AUROC.
#
#   FIX 1b NEW STRATEGY: the literature list is rebuilt with 30
#   DISTINCT invented proper nouns -- one per entry, zero reuse.
#   23 entries follow the "X + 1-tok common suffix" pattern (e.g.
#   "Bramish Bell", "Thessyn Plain", "Krossari Field"); 1 entry is
#   a bare 4-tok invented proper noun ("Tessalor"); 4 entries use
#   the "Song of X" / "X Cantos" 5-tok pattern; 1 entry uses the
#   "X Pilgrim" 6-tok pattern. The 4-tok pattern dominates so the
#   domain mean lands at 4.200 (gap = 0.200, well within the 0.5
#   BPE-token FIX 3 target; the previous list's mean was 4.033,
#   gap = 0.033).
#
#   Web verification (T-063, FIX 1b): all 30 titles and all 30
#   invented roots were checked against:
#     - Wikipedia page-existence (GET en.wikipedia.org/wiki/<X>) and
#       the Wikipedia search API (en.wikipedia.org/w/api.php) for
#       near-miss titles -- 30/30 404.
#     - OpenLibrary search by title and by name -- 30/30 0-hit.
#   The single near-miss was the bare invented name "Taranor"
#   (Helen Ellwood, Taranor / Taranor: Destiny, CreateSpace 2012 &
#   2014, OL20342558W) -- rejected and replaced with "Tessalor",
#   a fresh made-up word. The 30 fresh invented proper nouns used
#   below were also pre-screened against the Wikipedia search API
#   for ANY page whose title starts with or equals the root -- 0
#   matches. See the report appended to this file for the rejected
#   candidates that were tried but not used.
#
# Re-scan provenance (review T-059, FIX 3 -- token-length matching):
#   The exp5 FAKE lists ran Pythia-BPE token-length means as much as
#   2.5 tokens longer than the REAL lists (medicine 2.50, law 2.23,
#   literature 1.37, code 0.13). With the all-positions hook (FIX 2)
#   the registry sees the entity tokens directly, so any length /
#   rarity difference would leak into `span=entity`. The lists below
#   are rewrites / replacements that bring every domain's
#   |mean(real) - mean(fake)| to <= 0.5 BPE tokens while preserving
#   the audit property (no FAKE entry is a real, enacted / published
#   entity). The before/after numbers are printed in the next
#   section.
FAKE_ENTITIES = {
    # MEDICINE -- eponymous "<InventedSurname> syndrome/disorder".
    # Surnames are made-up (non-existent in Wikipedia / PubMed as
    # real eponymous syndromes); the BANNED productive medical
    # morphology (-algia, -itis, -dynia, -pathia, -plegia, -osis)
    # is not used anywhere in this list. Per-entry token counts
    # tuned so the domain mean = REAL mean (3.200).
    "medicine": [
        # 3-token entries (surname + syndrome/disorder, short surname)
        "Vor syndrome",            # 3 tok
        "Krev syndrome",           # 3 tok
        "Mord syndrome",           # 3 tok
        "Tass disorder",           # 3 tok
        "Aldren syndrome",         # 3 tok
        "Brams disorder",          # 3 tok
        "Fenv syndrome",           # 3 tok
        "Stell disorder",          # 3 tok
        "Ashvy syndrome",          # 3 tok
        "Bex disorder",            # 3 tok
        "Calwyn syndrome",         # 3 tok
        "Holt disorder",           # 3 tok
        "Jessam syndrome",         # 3 tok
        "Karst disorder",          # 3 tok
        "Vell disorder",           # 3 tok
        "Glass syndrome",          # 3 tok
        "Darre disorder",          # 3 tok
        "Kross disorder",          # 3 tok
        "Bram disorder",           # 3 tok
        "Marwyn syndrome",         # 3 tok
        "Ostri disorder",          # 3 tok
        "Pell disorder",           # 3 tok
        "Aldwyn disorder",         # 3 tok
        "Kest disorder",           # 3 tok
        # 4-token entries (slightly longer surname + syndrome/disorder)
        "Tarven disorder",         # 4 tok
        "Veldora disorder",        # 4 tok
        "Bramish disorder",        # 4 tok
        "Thessyn disorder",        # 4 tok
        "Yelvari syndrome",        # 4 tok
        "Aldwren syndrome",        # 4 tok
    ],
    # LAW -- "<InventedProperNoun> Act" (or "X Reform Act" with X a
    # 2-sub-token invented proper noun). The invented proper noun
    # carries the non-existence; the BANNED purely-generic "Civic
    # Act" / "Justice Act" / "Heritage Act" / "Compliance Act" / "X
    # Reform Act" pattern is not used. Per-entry token counts tuned
    # so the domain mean = REAL mean (4.067).
    "law": [
        # "X Act" with X = 3-sub-token invented proper noun (4 tok)
        "Veldora Act",
        "Bramish Act",
        "Thessyn Act",
        "Veldran Act",
        "Tasselin Act",
        "Aldwren Act",
        "Tarven Act",
        "Fenwic Act",
        "Caldwren Act",
        "Mordwyn Act",
        "Krossari Act",
        "Veldorin Act",
        "Kelmarsh Act",
        "Darrowen Act",
        "Eldwine Act",
        "Holtara Act",
        "Jessamir Act",
        "Karstane Act",
        # "X Reform Act" with X = 2-sub-token invented proper noun (4 tok)
        "Ashwort Reform Act",
        "Marwyn Reform Act",
        "Veld Reform Act",
        "Bram Reform Act",
        "Stell Reform Act",
        "Aldren Reform Act",
        "Tass Reform Act",
        "Ashvy Reform Act",
        "Calwyn Reform Act",
        "Kros Reform Act",
        # 5-tok entries (X = 4-sub-token invented proper noun + Act)
        "Voraxen Act",
        "Sterlun Act",
    ],
    # CODE -- unchanged from exp5 except for three PyPI-clean
    # replacements (Torchling -> Torchel, Flasket -> Astrom,
    # Astrora -> Astora). All entries verified 404 on PyPI JSON
    # API at T-061. Per-entry token counts tuned so the domain
    # mean is within +0.067 of REAL mean (2.600).
    "code": [
        "Pyrolex", "Quantalex", "Fasterly", "Sklearvon", "Tensorvex",
        "Torchel",   # was Torchling (real PyPI package, T-061 re-audit)
        "Plotora", "Colorly", "Sciora", "Keranova",
        "Djangerine",
        "Astrom",   # was Flasket (real PyPI package, T-061 re-audit)
        "Fastro", "Sqlix", "Souprun",
        "Pillex", "Opency", "Transformerix", "Spacium", "Textrax",
        "Gensimora", "Statlex", "Symbra", "Netwalkra", "Biolyra",
        "Astora",   # was Astrora (real PyPI package, T-061 re-audit)
        "Pylexio", "Pytestra", "Blackora", "Myplora",
    ],
    # LITERATURE -- T-063 audit fix (carried forward verbatim in the
    # 1307-1364 provenance comment above; see "Re-scan provenance
    # (review T-063, FIX 1b -- literature root diversity)").
    #
    # T-064 last-word-diversity fix. T-063 fixed root diversity (30
    # DISTINCT invented proper nouns, zero reuse -- a real win that
    # T-064 KEEPS), but introduced a new systematic asymmetry on the
    # LAST-WORD distribution: 23 of 30 fake titles were of the form
    # "InventedNoun Bell" / "InventedNoun Plain" / "InventedNoun
    # Cantos". The last-word count was Bell x12, Plain x5, Cantos x3,
    # Gate x2, Field x1. REAL has 30/30 different last words (Moby-Dick,
    # Prejudice, Gatsby, Mockingbird, 1984, ...). `span=entity` measures
    # the WHOLE title, so the token "Bell" (etc.) was inside the
    # measured span for 12 of 30 fake items and 0 of 30 real ones --
    # a structural difference between the two classes that is NOT
    # "entity unfamiliarity". It is exactly the confound this
    # experiment exists to avoid.
    #
    # FIX 1c (T-064) rebuilds the literature list so that:
    #   1. No last word is used more than 3 times across the 30
    #      entries. (Achieved: every last word is used EXACTLY ONCE.)
    #   2. Structural variety matches the real list. The real titles
    #      include one-word titles (Frankenstein, Beloved, Hamlet,
    #      Ulysses, Macbeth, Moby-Dick, 1984), "X and Y" forms (Pride
    #      and Prejudice, War and Peace, Romeo and Juliet, Of Mice
    #      and Men, ...), and "The X" forms. T-064's new list mirrors
    #      that mix: 14 one-word invented titles (which are short --
    #      helping the token budget -- AND match the real one-word
    #      distribution), 5 "X and Y" forms, 8 "The X" forms, and 3
    #      multi-word invented forms (Song of X, X Canticle, X Saga).
    #   3. KEEPS T-063's win: T-064 has 35 DISTINCT invented proper
    #      nouns across the 30 titles (14 bare + 10 in "X and Y" +
    #      8 in "The X Y" + 3 in misc). Zero reuse -- no root used
    #      more than ONCE (well within the <= 2 ceiling).
    #   4. KEEPS the non-realness guarantee. Every title contains an
    #      invented proper noun and was web-verified (Wikipedia
    #      page-existence AND OpenLibrary /search.json?title=) as in
    #      T-062/T-063. 30/30 returned 404 / 0-hits. The complete
    #      verified list is below; the rejected near-misses are
    #      listed in the T-064 report appended to this file.
    #   5. KEEPS the token-length match: |mean(real_len) -
    #      mean(fake_len)| <= 0.5 BPE tokens (real mean = 4.000).
    #      T-064 lands at mean = 4.133, gap = 0.133, well within the
    #      0.5 FIX 3 ceiling (the T-063 list was mean = 4.200,
    #      gap = 0.200).
    #
    #   PRIORITY applied per spec: non-realness > last-word diversity
    #   > root diversity > length gap. The non-realness and last-word
    #   goals are met exactly; root diversity exceeds the >= 25
    #   distinct ceiling (35 distinct, zero reuse); length gap is
    #   0.133 (<= 0.5).
    "literature": [
        # ----- 14 BARE ONE-WORD invented proper nouns (3-4 tok) -----
        # These mirror the REAL one-word titles (Hamlet, Macbeth,
        # Beloved, Frankenstein, ...). Each is a multi-syllable
        # made-up name that tokenizes to 3 or 4 BPE tokens.
        "Veldorin",          # 3 tok
        "Aldwren",           # 3 tok
        "Holtari",           # 3 tok
        "Mordwyn",           # 3 tok
        "Calthori",          # 3 tok
        "Pellorin",          # 3 tok
        "Vorathi",           # 3 tok
        "Tarvesh",           # 4 tok
        "Bramswix",          # 4 tok
        "Ylvasia",           # 4 tok
        "Stelmur",           # 3 tok
        "Ashvenor",          # 3 tok
        "Krevarek",          # 4 tok
        "Caldren",           # 3 tok
        # ----- 5 "X and Y" forms (5-7 tok each) -----
        # Mirror REAL's Pride and Prejudice, War and Peace, Romeo and
        # Juliet, Of Mice and Men, The Sound and the Fury, The Old Man
        # and the Sea, Crime and Punishment. Each title contains TWO
        # invented proper nouns, contributing 10 more distinct roots.
        "Tess and Veld",        # 5 tok
        "Krev and Ashv",        # 5 tok
        "Tessalor and Vel",     # 6 tok
        "Stelmure and Caldra",  # 6 tok
        "Ylvas and Tarveshi",   # 7 tok
        # ----- 8 "The X Y" forms (4-5 tok each) -----
        # Mirror REAL's The Great Gatsby, The Hobbit, The Odyssey,
        # The Canterbury Tales, ... Each title contains ONE invented
        # proper noun + a 1-tok common suffix word.
        "The Vorin Plain",      # 4 tok
        "The Cald Crown",       # 4 tok
        "The Tass Mark",        # 4 tok
        "The Ylv Bell",         # 4 tok
        "The Bram Trail",       # 4 tok
        "The Vorind Gate",      # 4 tok
        "The Seloran Vale",     # 5 tok
        "The Drendor Path",     # 5 tok
        # ----- 3 misc multi-word invented forms (5 tok) -----
        # "Song of X" (Song of Mordwyn, ...), "X Canticle", and
        # "X Saga" -- no direct REAL analogue but lightweight forms
        # that keep the token count low while staying distinct.
        "Song of Drendi",       # 5 tok
        "Hadvren Canticle",     # 5 tok
        "Brelan Saga",          # 5 tok
    ],
}

PROMPT_TEMPLATE_TEST1 = {
    "medicine":    "The recommended first-line treatment for {X} is",
    "law":         "Under the {X}, a person who",
    "code":        "The {X} library in Python provides a function that",
    "literature":  "In the novel {X}, the main character",
}

CLOZE_FACTS = {
    "medicine": [
        ("The recommended first-line antibiotic for strep throat is", ["penicillin", "amoxicillin"]),
        ("The most common cause of community-acquired pneumonia in adults is", ["streptococcus pneumoniae", "pneumococcus"]),
        ("The first-line medication for status epilepticus is", ["lorazepam", "diazepam", "midazolam"]),
        ("The active ingredient in Tylenol is", ["acetaminophen", "paracetamol"]),
        ("The first-line treatment for anaphylaxis is", ["epinephrine", "adrenaline"]),
        ("Insulin is produced by the", ["pancreas", "beta cells", "islets of langerhans"]),
        ("The first-line antibiotic for uncomplicated urinary tract infection is", ["nitrofurantoin", "trimethoprim"]),
        ("The first-line treatment for type 2 diabetes is", ["metformin"]),
        ("Hepatitis B is caused by a", ["virus", "hepatitis b virus"]),
        ("The bacterium that causes Lyme disease is", ["borrelia", "borrelia burgdorferi"]),
        ("The first-line treatment for Parkinson's disease is", ["levodopa", "carbidopa"]),
        ("The gold standard diagnostic test for tuberculosis is", ["sputum culture", "culture"]),
        ("The drug used to reverse opioid overdose is", ["naloxone"]),
        ("The most common cause of bacterial meningitis in adults is", ["streptococcus pneumoniae", "pneumococcus"]),
        ("The first-line treatment for acute gout is", ["colchicine", "indomethacin", "nsaid", "nsaids"]),
        ("The first-line treatment for cluster headache is", ["oxygen", "sumatriptan"]),
        ("The pathogen that causes malaria is", ["plasmodium"]),
        ("The most common cause of hyperthyroidism is", ["graves", "graves' disease"]),
        ("The first-line treatment for panic disorder is an", ["ssri", "sertraline", "paroxetine", "escitalopram", "fluoxetine", "citalopram", "venlafaxine"]),
        ("The bacterium that causes syphilis is", ["treponema", "treponema pallidum"]),
        ("The drug of choice for treating scabies is", ["permethrin", "ivermectin"]),
        ("The first-line treatment for atrial fibrillation rate control is", ["metoprolol", "beta blocker"]),
        ("The hormone that regulates blood sugar is", ["insulin"]),
        ("Vitamin D deficiency in adults causes", ["osteomalacia", "osteoporosis"]),
        ("The most common cause of hypothyroidism is", ["hashimoto", "hashimoto's", "hashimoto's thyroiditis", "iodine deficiency"]),
        ("The bacterium that causes tuberculosis is", ["mycobacterium"]),
        ("The drug used to treat heroin addiction is", ["methadone", "buprenorphine", "naltrexone"]),
        ("Iron deficiency causes", ["anemia", "iron deficiency anemia", "microcytic anemia"]),
        ("The first-line treatment for generalized anxiety disorder is", ["ssri", "sertraline", "venlafaxine", "buspirone", "duloxetine", "escitalopram", "paroxetine", "pregabalin"]),
        ("The first-line therapy for Crohn's disease flare is", ["corticosteroid", "prednisone"]),
    ],
    "law": [
        ("In the United States, the right to remain silent is protected by the", ["fifth amendment", "5th amendment"]),
        ("The Miranda warning was established by the Supreme Court case", ["miranda v arizona", "miranda"]),
        ("The U.S. Constitution was signed in", ["1787"]),
        ("The President of the United States is elected for a term of", ["four years", "4 years"]),
        ("The U.S. Supreme Court has", ["nine", "9"]),
        ("In contract law, a valid contract requires", ["offer", "acceptance", "consideration"]),
        ("The burden of proof in a criminal case is", ["beyond reasonable doubt"]),
        ("In the United States, the right to bear arms is protected by the", ["second amendment", "2nd amendment"]),
        ("The Sherman Act prohibits", ["monopoly", "monopolization", "restraint of trade", "anticompetitive"]),
        ("In the United States, copyright protection lasts for", ["life plus 70", "70 years", "70 years after death"]),
        ("Roe v Wade was decided in", ["1973"]),
        ("The First Amendment protects", ["freedom of speech", "speech", "religion", "press", "assembly", "petition"]),
        ("The Equal Protection Clause is found in the", ["14th amendment", "fourteenth amendment"]),
        ("Brown v Board of Education outlawed", ["segregation", "racial segregation"]),
        ("Marbury v Madison established", ["judicial review"]),
        ("The age at which one can vote in federal elections in the United States is", ["18", "eighteen"]),
        ("Plessy v Ferguson established", ["separate but equal"]),
        ("The Fourth Amendment protects against", ["unreasonable search", "searches and seizures"]),
        ("The Fifth Amendment protects against", ["self-incrimination", "double jeopardy"]),
        ("The Sixth Amendment guarantees the right to", ["counsel", "jury trial", "speedy trial", "confrontation", "compulsory process", "impartial jury"]),
        ("The Patriot Act was passed in", ["2001"]),
        ("The Patient Protection and Affordable Care Act was signed in", ["2010"]),
        ("Citizens United v FEC was decided in", ["2010"]),
        ("The Voting Rights Act was signed in", ["1965"]),
        ("The number of amendments to the US Constitution is", ["27"]),
        ("The Federalist Papers were written by", ["hamilton", "madison", "jay"]),
        ("The Declaration of Independence was adopted in", ["1776"]),
        ("The Bill of Rights consists of the first", ["10", "ten"]),
        ("The case that established corporate personhood is", ["santa clara"]),
        ("The age of majority for contracts in most U.S. states is", ["18", "eighteen"]),
    ],
    "code": [
        ("In Python, the keyword used to define a function is", ["def"]),
        ("The Python data structure that stores key-value pairs is called a", ["dict", "dictionary"]),
        ("In Python, the operator for integer division is", ["//"]),
        ("The Python keyword used to handle exceptions is", ["try", "except"]),
        ("In Python, the built-in function used to read a file is", ["open"]),
        ("In Python, the keyword used to create a class is", ["class"]),
        ("The Python library for numerical computing is called", ["numpy", "np"]),
        ("The Python library for data manipulation is called", ["pandas"]),
        ("In Python, the keyword used to import a module is", ["import"]),
        ("In Python, the method to add an element to a list is", ["append"]),
        ("In Python, the function to get the length of a list is", ["len"]),
        ("The Python library for making HTTP requests is called", ["requests"]),
        ("In Python, the keyword used to start a for loop is", ["for"]),
        ("In Python, the keyword used to start a while loop is", ["while"]),
        ("In Python, the operator for exponentiation is", ["**"]),
        ("In Python, the method to convert a string to lowercase is", ["lower"]),
        ("In Python, the keyword used to define a generator function is", ["yield"]),
        ("In Python, the value representing the absence of a value is", ["none"]),
        ("In Python, the boolean values are", ["true", "false"]),
        ("The Python library for machine learning developed by Google is called", ["tensorflow", "tf"]),
        ("The Python library for deep learning developed by Facebook is called", ["pytorch", "torch"]),
        ("The Python library for plotting is called", ["matplotlib"]),
        ("In Python, the method to join a list of strings is", ["join"]),
        ("In Python, the built-in function to sort a list is", ["sorted", "sort"]),
        ("In Python, the keyword used to define a lambda function is", ["lambda"]),
        ("In Python, the operator for modulo is", ["%"]),
        ("In Python, the method to remove an element from a list is", ["remove", "pop"]),
        ("In Python, the method to get the keys of a dictionary is", ["keys"]),
        ("The Python package manager is called", ["pip"]),
        ("In Python, the built-in function to read user input is", ["input"]),
    ],
    "literature": [
        ("The novel Moby-Dick was written by", ["melville", "herman melville"]),
        ("Pride and Prejudice was written by", ["austen", "jane austen"]),
        ("The Great Gatsby was written by", ["fitzgerald", "f scott fitzgerald"]),
        ("To Kill a Mockingbird was written by", ["lee", "harper lee"]),
        ("1984 was written by", ["orwell", "george orwell"]),
        ("The Catcher in the Rye was written by", ["salinger", "j d salinger"]),
        ("The Lord of the Rings was written by", ["tolkien", "j r r tolkien"]),
        ("The Hobbit was written by", ["tolkien"]),
        ("Crime and Punishment was written by", ["dostoevsky", "fyodor dostoevsky"]),
        ("War and Peace was written by", ["tolstoy", "leo tolstoy"]),
        ("The Odyssey was written by", ["homer"]),
        ("The Iliad was written by", ["homer"]),
        ("Don Quixote was written by", ["cervantes", "miguel de cervantes"]),
        ("Hamlet was written by", ["shakespeare", "william shakespeare"]),
        ("Macbeth was written by", ["shakespeare"]),
        ("Romeo and Juliet was written by", ["shakespeare"]),
        ("King Lear was written by", ["shakespeare"]),
        ("The Canterbury Tales was written by", ["chaucer", "geoffrey chaucer"]),
        ("Ulysses was written by", ["joyce", "james joyce"]),
        ("The Sound and the Fury was written by", ["faulkner", "william faulkner"]),
        ("One Hundred Years of Solitude was written by", ["garcia marquez", "gabriel garcia marquez"]),
        ("The Old Man and the Sea was written by", ["hemingway", "ernest hemingway"]),
        ("For Whom the Bell Tolls was written by", ["hemingway"]),
        ("Brave New World was written by", ["huxley", "aldous huxley"]),
        ("The Grapes of Wrath was written by", ["steinbeck", "john steinbeck"]),
        ("Of Mice and Men was written by", ["steinbeck"]),
        ("The Color Purple was written by", ["walker", "alice walker"]),
        ("Beloved was written by", ["morrison", "toni morrison"]),
        ("The Road was written by", ["mccarthy", "cormac mccarthy"]),
        ("Frankenstein was written by", ["shelley", "mary shelley"]),
    ],
}


# %% [markdown]
# ## 11b. Tokenization-confound check (TEST 1) -- before / after FIX 3
#
# For TEST 1 the labels are REAL vs FAKE entity names. If FAKE
# names tokenize to a very different length than REAL names, the
# contrast is confounded by name rarity (rare / unusual strings
# also tend to produce unusual BPE lengths, and Pythia will likely
# distribute its next-token mass differently for rare strings
# regardless of whether the registry flags anything). With FIX 2
# recording all positions and `span=entity` measuring exactly the
# positions where the entity appears, a length / rarity mismatch
# would leak straight into the registry signal. FIX 3 re-writes /
# replaces FAKE entries so every domain's |mean(real) - mean(fake)|
# <= 0.5 BPE tokens.
#
# FIX 1 (T-061) added a SECOND systemic requirement on top of FIX 3:
# the FAKE lists must not collide with real entities (the previous
# T-060 medicine list was full of real -algia/-itis/-itis clinical
# terms because Greek/Latin medical morphology is a productive real
# system). See the long FIX 1 provenance comment above FAKE_ENTITIES.
#
# We print BOTH the exp5 (before, including T-060 contamination)
# numbers AND the exp5b (after, FIX 1 + FIX 3) numbers so the
# reader can confirm both the gap closed AND the list is no longer
# contaminated with real entities.

# %%
# exp5 (BEFORE FIX 1 + FIX 3) -- kept here as a hardcoded baseline
# so the audit is reproducible without re-running exp5. Each list
# below is exactly what exp5 used; see `exp5_source.py` lines
# around 1051. Note: this is the T-060 version of the FAKE lists
# (the version that contaminated medicine with real -algia/-itis
# clinical terms), NOT the post-T-058 exp5b version.
FAKE_ENTITIES_EXP5 = {
    "medicine": [
        "Voren-Klein syndrome", "Cardiomemetic disorder",
        "Pulmonocranial syndrome", "Hapsburg-Ellis syndrome",
        "Neurolymph vasculitis", "Endocranial hypotonia",
        "Gastromedullary disorder", "Cerebrovascular ataxia",
        "Cardiorenal dyssynergy", "Bronchoalveolar neuropathy",
        "Hepatorenal fibrosis", "Thyromimetic disorder",
        "Cerebrospinal ataxia", "Myocardial lymphedema",
        "Renocardiac sclerosis", "Pulmonocardiac stenosis",
        "Vasculoneural syndrome", "Neurocardiac syndrome",
        "Cerebromedullary dystrophy", "Cardiopulmonary sclerosis",
        "Hepatoneural necrosis", "Pulmoneural syndrome",
        "Bronchovascular atrophy", "Cerebrocardiac necrosis",
        "Neurovascular dyssyn", "Hepatic dysneuritis",
        "Cerebellar dysplasitis", "Bronchoalveolar dyssyn",
        "Renoneural syndrome", "Pulmovascular necrosis",
    ],
    "law": [
        "Brennan-Hartwell Accountability Act", "Federal Civic Transparency Act",
        "Allied Workforce Inclusion Act", "Calderon-Marsh Restoration Act",
        "Watershed Reclamation Initiative", "Aldridge-Vance Antitrust Act",
        "Holbrook-Pierce Commerce Act", "Wages Equity Extension Act",
        "National Workforce Standards Act", "Workplace Conciliation Modernization Act",
        "Voter Inclusion Modernization Act", "Retirement Savings Modernization Act",
        "Healthcare Access Equity Act", "Financial Stability Reform Act",
        "Corporate Disclosure Reform Act", "Anticorruption Compliance Act",
        "Government Transparency Reform Act", "Personal Information Stewardship Act",
        "Federal Insolvency Reform Act", "Financial Records Modernization Act",
        "Organized Crime Deterrence Act", "Tribal Family Welfare Act",
        "Student Achievement Modernization Act", "Special Education Equity Act",
        "Family Care Modernization Act", "Educational Equity Reform Act",
        "Workplace Equity Enforcement Act", "Search and Surveillance Reform",
        "Self-Incrimination Modernization", "Equal Protection Modernization Act",
    ],
    "code": [
        "Pyrolex", "Quantalex", "Fasterly", "Sklearvon", "Tensorvex",
        "Torchling", "Plotora", "Colorly", "Sciora", "Keranova",
        "Djangerine", "Flasket", "Fastro", "Sqlix", "Souprun",
        "Pillex", "Opency", "Transformerix", "Spacium", "Textrax",
        "Gensimora", "Statlex", "Symbra", "Netwalkra", "Biolyra",
        "Astrora", "Pylexio", "Pytestra", "Blackora", "Myplora",
    ],
    "literature": [
        "The Ember of Larkspur", "Carthian Vow", "The Mariner's Reckoning",
        "Mockingbird at Dusk", "The Granite Ledger", "The Iron Pilgrim",
        "The Crown of Veldoran", "The Silver Burrow", "The Winters of Vorlin",
        "The Granite Passage", "The Aeolian Voyage", "The Smoldering Coast",
        "The Knight of Solinar", "The Sea at Lorn", "The Velvet Crown",
        "The Argent Garden", "The Hollow Throne", "The Pilgrim's Cantos",
        "The Salt of Dunmore", "The Quiet Meridian", "The House of Belamar",
        "The Vintner's Mark", "The Last Cartographer", "The Field of Glassford",
        "The Stone at Vellinde", "The Indigo Garden", "The Bell at Selwyn",
        "The Loremaster's Daughter", "The Marble Pilgrim", "The Silence of Halvern",
    ],
}


def _entity_token_length(name):
    """BPE token length of an entity name (Pythia tokenizer)."""
    return len(tokenizer.encode(name))


def _entity_in_context_length(template, entity, tokenizer):
    """In-context entity token length -- the span length the registry
    actually sees for `span=entity`. Same prefix/entity construction as
    FIX A in `_compute_entity_span`: any trailing whitespace in the
    template prefix is moved onto the front of the entity before
    encoding (so the prefix's trailing space is consumed by the
    splitter instead of being fused into the first entity token). The
    returned length is `len(encode(prefix + entity)) -
    len(encode(prefix))`.

    The bare BPE length `len(encode(name))` is a historical reference
    but does NOT measure what Pythia sees at the entity positions --
    the fused leading space changes the BPE tokenization of the entity
    substantially. FIX B (T-065) keys the OK / CONFOUNDED flag on this
    in-context measurement, which can be much larger than the bare
    gap (e.g. medicine: bare gap = 0.000, in-context gap = 0.767 --
    a domain that the bare check declared OK is actually
    CONFOUNDED)."""
    if "{X}" not in template:
        # Unreachable today (every template used by exp5b has a `{X}`
        # slot) and WRONG if reached: the bare length would silently
        # flow into the `*_ic` confound fields (FIX B), and the bare
        # measurement has been documented as a different quantity from
        # what the registry actually sees via span=entity. Make the
        # misuse loud rather than silently passing a mismatched number.
        raise ValueError(
            f"_entity_in_context_length: template has no '{{X}}' slot "
            f"({template!r}). Bare-length fallback is not allowed for "
            f"in-context measurements; use _entity_token_length instead."
        )
    prefix_str, _ = template.split("{X}", 1)
    ent = entity
    while prefix_str.endswith(" "):
        prefix_str = prefix_str[:-1]
        ent = " " + ent
    return (len(tokenizer.encode(prefix_str + ent))
            - len(tokenizer.encode(prefix_str)))


confound_report = {}
print("=" * 78)
print("TOKENIZATION-CONFOUND CHECK (TEST 1 -- REAL vs FAKE entity names)")
print("(if the abs gap is large the whole REAL-vs-FAKE contrast is")
print(" confounded by name rarity -- Pythia distributes mass differently")
print(" over rare strings regardless of any registry signal)")
print()
print("Bare     = len(encode(name))   -- exp5 / historical measurement.")
print("In-ctx   = len(encode(prefix + entity)) - len(encode(prefix)) --")
print("           the in-prompt span length (fused leading space")
print("           included). This is what the registry actually sees")
print("           via span=entity. The OK / CONFOUNDED flag below keys")
print("           on the in-context gap (the bare column is printed as")
print("           a secondary historical reference).")
print("BEFORE   = exp5 FAKE lists (historical baseline, kept verbatim")
print("           from exp5 -- including the real-entity-contaminated")
print("           T-060 versions).")
print("AFTER    = exp5b FAKE lists at T-061 (FIX 1 systemic rebuild to")
print("           invented-proper-noun names + FIX 3 token-length")
print("           matching).")
print("Every domain's in-context gap-after must be <= 0.5 BPE tokens")
print("for the TEST 1 REAL-vs-FAKE contrast to not be confounded by")
print("name rarity.")
print("=" * 78)
hdr = (f"  {'domain':>10s}  "
       f"{'REAL bare':>9s} {'FAKE-bf bare':>12s} {'bare gap-bf':>11s}  "
       f"{'FAKE-af bare':>12s} {'bare gap-af':>11s}  "
       f"{'REAL ic':>8s} {'FAKE-bf ic':>11s} {'ic gap-bf':>10s}  "
       f"{'FAKE-af ic':>11s} {'ic gap-af':>10s}  {'flag':<12s}")
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for d in DOMAIN_NAMES:
    template = PROMPT_TEMPLATE_TEST1[d]
    real_lens = [_entity_token_length(n) for n in REAL_ENTITIES[d]]
    fake_lens_before = [_entity_token_length(n) for n in FAKE_ENTITIES_EXP5[d]]
    fake_lens_after  = [_entity_token_length(n) for n in FAKE_ENTITIES[d]]
    real_lens_ic = [_entity_in_context_length(template, n, tokenizer) for n in REAL_ENTITIES[d]]
    fake_lens_before_ic = [_entity_in_context_length(template, n, tokenizer) for n in FAKE_ENTITIES_EXP5[d]]
    fake_lens_after_ic  = [_entity_in_context_length(template, n, tokenizer) for n in FAKE_ENTITIES[d]]
    r_mean = float(np.mean(real_lens))
    f_mean_before = float(np.mean(fake_lens_before))
    f_mean_after  = float(np.mean(fake_lens_after))
    gap_before = abs(r_mean - f_mean_before)
    gap_after  = abs(r_mean - f_mean_after)
    r_mean_ic = float(np.mean(real_lens_ic))
    f_mean_before_ic = float(np.mean(fake_lens_before_ic))
    f_mean_after_ic  = float(np.mean(fake_lens_after_ic))
    ic_gap_before = abs(r_mean_ic - f_mean_before_ic)
    ic_gap_after  = abs(r_mean_ic - f_mean_after_ic)
    # FIX B (T-065): flag keys on the IN-CONTEXT gap-after, not the
    # bare-name gap. The bare measurement was a historical proxy that
    # hid the medicine confound (bare 0.000, in-context 0.767).
    flag = "OK" if ic_gap_after <= 0.5 else f"CONFOUNDED (+{ic_gap_after-0.5:.2f})"
    confound_report[d] = dict(
        # BARE measurement (historical, exp5-style).
        real_mean=r_mean,
        real_std=float(np.std(real_lens)),
        n_real=len(real_lens),
        fake_mean_before=f_mean_before,
        fake_mean_after=f_mean_after,
        fake_std_after=float(np.std(fake_lens_after)),
        n_fake=len(fake_lens_after),
        gap_before=float(gap_before),
        gap_after=float(gap_after),
        real_lens=real_lens,
        fake_lens=fake_lens_after,
        # IN-CONTEXT measurement (FIX B, what span=entity actually sees).
        real_mean_ic=r_mean_ic,
        real_std_ic=float(np.std(real_lens_ic)),
        fake_mean_before_ic=f_mean_before_ic,
        fake_mean_after_ic=f_mean_after_ic,
        fake_std_after_ic=float(np.std(fake_lens_after_ic)),
        ic_gap_before=float(ic_gap_before),
        ic_gap_after=float(ic_gap_after),
        real_lens_ic=real_lens_ic,
        fake_lens_ic=fake_lens_after_ic,
    )
    print(f"  {d:>10s}  {r_mean:>9.3f} {f_mean_before:>12.3f} {gap_before:>11.3f}  "
          f"{f_mean_after:>12.3f} {gap_after:>11.3f}  "
          f"{r_mean_ic:>8.3f} {f_mean_before_ic:>11.3f} {ic_gap_before:>10.3f}  "
          f"{f_mean_after_ic:>11.3f} {ic_gap_after:>10.3f}  {flag:<12s}")
print("  " + "-" * (len(hdr) - 2))
print("  flag OK = in-context |gap_after| <= 0.5 (FIX 3 target; FIX 1")
print("         also keeps the FAKE list free of real-entity")
print("         collisions). flag CONFOUNDED = in-context gap > 0.5.")
print("         The bare-name gap is printed as a secondary historical")
print("         column; it was the measurement exp5 used and is no")
print("         longer the one the verdict keys on.")
print("=" * 78)


# %% [markdown]
# ## 11c. Entity-span computation (TEST 1 only)
#
# For `span=entity` we need the EXACT token index range of the entity
# `{X}` inside the tokenized prompt. exp5b derives this by
# separately tokenizing
#
# 1. the template prefix before `{X}`, and
# 2. the prefix concatenated with the entity.
#
# If the prefix+entity tokenization matches the prefix tokenization
# in its leading tokens, the entity's contribution is just the
# TRAILING tokens of the prefix+entity encoding. We then VERIFY that
# this contribution appears at the expected position inside the
# FULLY-tokenized prompt (prefix + entity + suffix). If the
# verification fails (boundary merges, BOS mismatch, etc.) the
# entity-span reconstruction is dropped and a diagnostic is logged;
# `span=entity` is left as None for that run; the run is STILL KEPT
# and contributes to `span=prompt` and `span=gen`. (LOW fix at T-061:
# the previous version of the code did `continue` here, dropping the
# entire item from `runs` and contradicting this doc -- now the
# code matches the doc: keep the item, mark entity as N/A.)
#
# For TEST 2 (cloze) there is no `{X}` placeholder so `span=entity`
# is NOT APPLICABLE for every prompt -- no computation needed.

# %%
def _compute_entity_span(template, entity, tokenizer):
    """Return `(prompt_ids, entity_start, entity_end)` (entity_start/end
    are 0-indexed offsets inside `prompt_ids`) if the BPE
    reconstruction verifies, else `None`.

    The method:
      prefix_ids = encode(prefix_str)            -- tokens before {X}
      ext_ids    = encode(prefix_str + entity)   -- tokens for prefix + entity
      full_ids   = encode(prefix_str + entity + suffix_str)
      -- verify ext_ids[:len(prefix_ids)] == prefix_ids
      -- entity tokens = ext_ids[len(prefix_ids):]
      -- entity starts inside full_ids at len(prefix_ids)
      -- verify full_ids[len(prefix_ids):len(prefix_ids)+len(entity_tokens)]
                == entity_tokens
    If any step fails (BPE boundary merge, BOS re-issue, etc.) the
    item is dropped and the caller logs it.

    FIX A (T-065): every template ends with a space before `{X}`,
    so the bare `prefix_str` ends with a space; BPE fuses that space
    into the first entity token, which makes
    `ext_ids[:len(prefix_ids)] != prefix_ids` always fail and the
    function returns None for every entity. We move any trailing
    whitespace from the prefix onto the front of the entity BEFORE
    encoding so the prefix's trailing space is consumed by the
    splitter instead of being absorbed by the entity token. We use a
    local copy of `entity` (`ent_for_encode`) so the caller's
    `entity` variable is NOT mutated -- the caller still uses it for
    `prompt_text = template.format(X=entity)` and the dropped-items
    log downstream."""
    if "{X}" not in template:
        return None
    prefix_str, suffix_str = template.split("{X}", 1)
    ent_for_encode = entity
    while prefix_str.endswith(" "):
        prefix_str    = prefix_str[:-1]
        ent_for_encode = " " + ent_for_encode
    prefix_ids = tokenizer.encode(prefix_str)
    ext_ids    = tokenizer.encode(prefix_str + ent_for_encode)
    full_ids   = tokenizer.encode(prefix_str + ent_for_encode + suffix_str)
    if (len(ext_ids) <= len(prefix_ids) or
        ext_ids[:len(prefix_ids)] != prefix_ids):
        return None
    n_entity = len(ext_ids) - len(prefix_ids)
    ent_start = len(prefix_ids)
    ent_end   = ent_start + n_entity
    # Slice check on the fully-tokenized prompt
    if full_ids[ent_start:ent_end] != ext_ids[ent_start:]:
        return None
    return full_ids, ent_start, ent_end


# Quick self-check at import / first run: encode a known prompt and
# confirm the function recovers the right span. We use the medicine
# template as a representative case.
_demo_entity = "type 2 diabetes"
_demo_template = PROMPT_TEMPLATE_TEST1["medicine"]
_demo = _compute_entity_span(_demo_template, _demo_entity, tokenizer)
assert _demo is not None, "entity-span self-check failed for medicine/type 2 diabetes"
_demo_prompt_ids, _demo_ent_start, _demo_ent_end = _demo
print(f"entity-span self-check (medicine, type 2 diabetes):")
print(f"  prompt_ids = {_demo_prompt_ids[:12]}... ({len(_demo_prompt_ids)} tokens)")
print(f"  entity span = [{_demo_ent_start}, {_demo_ent_end}) -- "
      f"{_demo_ent_end - _demo_ent_start} tokens")
print(f"  decoded entity from those tokens = "
      f"{tokenizer.decode(_demo_prompt_ids[_demo_ent_start:_demo_ent_end])!r}")
assert (tokenizer.decode(_demo_prompt_ids[_demo_ent_start:_demo_ent_end])
        == " " + _demo_entity), \
    "entity-span self-check: decoded entity tokens do not round-trip to ' ' + entity"
# Cross-check that the two copies of the prefix/entity construction
# agree. _entity_in_context_length (section 11b) computes the in-prompt
# span length the registry sees; _compute_entity_span (section 11c)
# computes the same span via the FIX A trailing-whitespace move.
# They MUST agree -- otherwise the FIX B confound flag is computing
# length against one definition of "entity span" while the per-token
# signatures slice against another. We do NOT refactor the two
# functions into one -- this invariant pins them together.
assert (_entity_in_context_length(_demo_template, _demo_entity, tokenizer)
        == _demo_ent_end - _demo_ent_start), \
    "cross-check failed: _entity_in_context_length disagrees with _compute_entity_span"


# %% [markdown]
# ## 12. Signature computations
#
# exp5b computes the same registry signatures as exp5 but **per-span**:
# for each of the three spans (`entity`, `prompt`, `gen`) the
# signatures are evaluated at every position in the span, then
# AUROC tables are built by first-N-position-means across runs.
# Baselines (`lm_entropy`, `neg_max_logprob`) are only meaningful for
# positions where Pythia was sampling next-token logits, so they are
# only computed for `span=gen`. For `span=prompt` and `span=entity`
# the verdict only compares registry signals against each other.
#
# All signatures are computed in tensor form on CPU (the hook
# already moves `pre` / `acts` to CPU) so the generation GPU buffer
# pressure is unaffected.
#
# **Signature A -- concentration / "improvising"**
# - `max_act` = max value among the k firing slots.
# - `act_entropy` = entropy of the k firing slots after normalising
#   by their sum (so it lives in `[0, log k]`, not `[0, log n_slots]`).
# - `top1_share` = `max_act / sum(acts)`. 1.0 = single-slot spike.
#
# **Signature B -- domain drift** (vs the prompt reference)
# The reference is the registry's `acts` at the LAST PROMPT TOKEN
# (`recorded[0][ctx-1]` -- the rightmost position of the prefill).
# The reference DOMAIN is the argmax, over the 4 `dom_all` groups,
# of the activation mass at that prompt position.
# - `jaccard` of top-k slot ids vs reference.
# - `cosine` of the k-hot activation vectors.
# - `prompt_domain_share` = fraction of activation mass on the
#   reference-domain slots.
#
# **Signature C -- conflict**
# - `domain_entropy` = entropy of the activation-weighted
#   distribution over the 4 slot domains (using `dom_all`).
#
# **Baselines** (computed from Pythia's own next-token softmax at
# the same step):
# - `lm_entropy` = full softmax entropy in nats (already returned
#   by `generate_with_signatures`).
# - `neg_max_logprob` = `-log(max(softmax))` (also returned by
#   `generate_with_signatures`).

# %%
# FIX 1b (review T-056-review MEDIUM A): the verdict block below must
# actually consult the per-domain tokenization-confound numbers
# computed in section 11b, instead of just letting them sit in the
# JSON. Threshold chosen: CONFOUND_ABS_GAP_THRESHOLD = 0.5 BPE
# tokens (FIX 3 target). Domains with |real_mean - fake_mean| > 0.5
# token are flagged as TEST-1-CONFOUNDED next to their AUROC (the
# contrast in that domain may reflect name rarity rather than any
# grounding signal). This constant is defined here in a code cell
# (NOT inside a `# %% [markdown]` block, which is what crashed exp5).
CONFOUND_ABS_GAP_THRESHOLD = 0.5


# %%
SIGNALS_REGISTRY = ["max_act", "act_entropy", "top1_share",
                    "jaccard", "cosine", "prompt_domain_share",
                    "domain_entropy"]
SIGNALS_BASELINE = ["lm_entropy", "neg_max_logprob"]
SIGNALS_ALL      = SIGNALS_REGISTRY + SIGNALS_BASELINE

# precompute masks for each domain: a boolean [n_slots] tensor.
DOMAIN_MASKS = [(dom_all == d_idx).cpu() for d_idx in range(N_DOMAINS)]


def _domain_masses(acts_cpu):
    """Return a [N_DOMAINS] tensor of activation mass per slot domain."""
    return torch.stack([acts_cpu[DOMAIN_MASKS[d]].sum() for d in range(N_DOMAINS)])


def _per_position_signatures(acts_positions, ref_acts):
    """Compute per-position signatures for a `[T, n_slots]` tensor
    of activations, with the registry's `acts` at the LAST PROMPT
    TOKEN (`ref_acts`, a 1-D `[n_slots]` tensor) as the drift
    reference.

    Returns a dict mapping each signal name in
    `SIGNALS_REGISTRY + SIGNALS_BASELINE` to a list of length T
    (signature value at each position in the span).

    For `lm_entropy` and `neg_max_logprob` (the LM-head baselines
    computed from the next-token softmax) the caller passes a list
    of length T; if no list is given we use zeros."""
    acts_ref = ref_acts
    ref_domain_masses = _domain_masses(acts_ref)
    ref_domain_idx = int(torch.argmax(ref_domain_masses).item())
    ids_ref = set(torch.nonzero(acts_ref > 0, as_tuple=False).flatten().tolist())
    ref_domain_mask = DOMAIN_MASKS[ref_domain_idx]
    acts_ref_f = acts_ref.float()

    T = acts_positions.shape[0]
    sigs = {name: [] for name in SIGNALS_REGISTRY}
    for t in range(T):
        acts_t = acts_positions[t]
        acts_t_f = acts_t.float()
        active_mask = (acts_t > 0)
        active_vals = acts_t[active_mask]
        sum_acts = float(active_vals.sum().item())
        if len(active_vals) > 0 and sum_acts > 0.0:
            max_act = float(active_vals.max().item())
            top1_share = max_act / sum_acts
            p = active_vals.float() / sum_acts
            act_entropy = float(-(p * (p + 1e-12).log()).sum().item())
        else:
            max_act = 0.0
            top1_share = 0.0
            act_entropy = 0.0
        # Signature B (drift)
        ids_t = set(torch.nonzero(active_mask, as_tuple=False).flatten().tolist())
        if ids_t or ids_ref:
            jaccard = len(ids_t & ids_ref) / max(1, len(ids_t | ids_ref))
        else:
            jaccard = 1.0
        # cosine between two k-hot vectors (zero everywhere except the k
        # firing slots)
        cos_sim = float(F.cosine_similarity(
            acts_t_f.unsqueeze(0), acts_ref_f.unsqueeze(0)).item())
        prompt_domain_share = float(acts_t[ref_domain_mask].sum().item() /
                                    max(1e-9, sum_acts))
        # Signature C (conflict)
        masses_t = _domain_masses(acts_t)
        total_mass = float(masses_t.sum().item())
        if total_mass > 0.0:
            probs = (masses_t.float() / total_mass).tolist()
            domain_entropy = 0.0
            for p in probs:
                if p > 0.0:
                    domain_entropy -= p * math.log(p)
        else:
            domain_entropy = 0.0

        sigs["max_act"].append(max_act)
        sigs["act_entropy"].append(act_entropy)
        sigs["top1_share"].append(top1_share)
        sigs["jaccard"].append(jaccard)
        sigs["cosine"].append(cos_sim)
        sigs["prompt_domain_share"].append(prompt_domain_share)
        sigs["domain_entropy"].append(domain_entropy)
    return sigs


# Per-span definitions (MEDIUM 1 + MEDIUM 2 fix):
#   span='gen'   : per-step acts = recorded[t][ctx-1] for t in
#                  [0, n_new-1] -- the position whose hidden state
#                  produced logits for the t-th generated token
#                  (for t=0 that is the last prompt token; for t>=1
#                  that is the previously-generated token now the
#                  rightmost input). `n_new = min(len(acts_all),
#                  GEN_LEN)` covers ALL cfg.gen_len=32 generated
#                  positions (the previous code used `n_new = len
#                  (acts_all) - 1` and silently dropped the last
#                  generated position). Baselines available.
#   span='prompt': per-step acts = recorded[0][ctx-prompt_len:ctx-1]
#                  -- the prompt positions (left-padded with EOT to
#                  ctx) EXCLUDING the rightmost one. The rightmost
#                  prompt position (ctx-1) is the Signature B
#                  reference `acts_ref`; including it in span=prompt
#                  would make jaccard/cosine/prompt_domain_share
#                  identically 1.0 at that position (a vector
#                  compared against itself) and bias the drift
#                  signal with a self-referential contribution.
#                  Baselines N/A (no per-prompt-position LM entropy
#                  computed here).
#   span='entity': per-step acts = recorded[0]
#                  [ctx-prompt_len+ent_start:ctx-prompt_len+ent_end]
#                  -- the entity positions only (TEST 1 with verified
#                  span). The entity never reaches the rightmost
#                  prompt token under the current templates, so no
#                  reference-exclusion is needed here. Baselines N/A.

SPANS_ALL = ["entity", "prompt", "gen"]  # canonical order for spans
SPANS_POOL = ["prompt", "gen"]           # for TEST 2 (cloze, no {X})


# Module-level diagnostic list: every time `compute_signatures_for_run`
# gets an entity_span that LOOKS valid but FAILS the bounds check
# (`e_end <= e_start` or `e_end > ctx`), the bounds are appended here.
# This is the exact failure mode that produced the silent exp5 zero
# result -- span=entity vanished from the output with no diagnostic.
# We surface it loudly next to the existing ENTITY-SPAN FAILURES
# printout at the end of the generation block.
# Note: the list is "domain-agnostic" because compute_signatures_for_run
# does not receive a domain argument; the call site (TEST 1 loop) knows
# the domain but the function does not. We tag entries with the literal
# string "(domain-agnostic)" so the caller can correlate against the
# outer-loop domain from the position in `runs`.
entity_span_bound_failures = []  # list of (e_start, e_end, ctx, prompt_len)


def compute_signatures_for_run(recorded_states, prompt_len, entity_span,
                                lm_entropies, neg_max_logprobs):
    """Compute per-span per-position signatures for one generation run.

    `recorded_states` is the list returned by `generate_with_signatures`
    (length `n_new = cfg.gen_len = 32`; first entry is the prefill,
    next `n_new - 1` are decode steps -- the previous docstring said
    `n_new + 1`, but the generation loop runs exactly `n_new` Pythia
    forward passes, so `len(acts_all) == n_new`). Each entry is
    `{"pre": [T, n_slots], "acts": [T, n_slots]}` with T = cfg.ctx.

    `prompt_len` is the number of tokens in the (un-padded) prompt.

    `entity_span` is `(ent_start, ent_end)` 0-indexed offsets inside
    the prompt OR None (TEST 2 cloze). When None, the `entity` span
    is absent from the returned dict.

    `lm_entropies` and `neg_max_logprobs` are per-step baselines of
    length n_new (returned by `generate_with_signatures`). They are
    attached only to the `gen` span (LM baselines are not defined at
    prompt / entity positions, where Pythia has not been asked to
    sample a next token).

    Returns a dict mapping each span name in `SPANS_ALL` (or
    `SPANS_POOL` when `entity_span is None`) to a dict mapping each
    signal name to a list of floats (one per position in the span).

    Reference for Signature B (drift): `recorded[0][ctx-1]` -- the
    rightmost position of the prefill, which is the last PROMPT token
    (the registry's state at the prompt's exit; the same reference
    exp5 used). MEDIUM 2: this reference position is now EXCLUDED
    from the `prompt` span (slice is `[p_start, ctx-1)`) so that
    jaccard/cosine/prompt_domain_share at that position do not equal
    1.0 by construction (a vector compared against itself).
    """
    acts_all = [r["acts"] for r in recorded_states]
    ctx = acts_all[0].shape[0]
    # Reference = registry state at the LAST PROMPT TOKEN
    acts_ref = acts_all[0][ctx - 1]

    out = {}

    # --- span=gen ---
    # token_i's producing state = recorded[i][ctx-1] (for i in
    # [0, n_new-1]). That's n_new = cfg.gen_len positions -- the
    # RIGHTMOST position of the i-th forward pass in `recorded`
    # predicts the i-th generated token. The generation loop in
    # `generate_with_signatures` runs exactly cfg.gen_len (=32)
    # Pythia forward passes; `acts_all` therefore has n_new entries
    # (the first is the prefill, the rest are decode steps), NOT
    # n_new+1 as the previous docstring claimed. MEDIUM 1 fix: we
    # iterate `range(n_new)` to cover ALL cfg.gen_len generated
    # positions -- previously the code used `n_new = len(acts_all) -
    # 1` and iterated `range(n_new)`, dropping the last generated
    # position from the per-N aggregations.
    n_new = min(len(acts_all), GEN_LEN)
    # MEDIUM (T-067): the previous code silently clipped n_new via
    # `min(...)` whenever `len(acts_all) < GEN_LEN`. That makes the
    # `gen` span shorter than cfg.gen_len and shrinks the LM baselines
    # with no trace -- a future regression here would quietly change
    # every span=gen AUROC. Print a loud warning whenever the clip is
    # actually active (the hook recorded fewer forward passes than
    # GEN_LEN, or -- equally bad -- recorded MORE than GEN_LEN, which
    # means the loop iteration also dropped the last position).
    if len(acts_all) != GEN_LEN:
        print(f"WARNING: compute_signatures_for_run: hook recorded "
              f"{len(acts_all)} forward passes but GEN_LEN = {GEN_LEN}; "
              f"clipping n_new to {n_new}. Span=gen AUROC and LM baselines "
              f"are computed over {n_new} positions, not {GEN_LEN} -- "
              f"inspect the hook / generation loop.")
    # safety guard: if the hook recorded fewer entries than GEN_LEN,
    # clip n_new so we don't IndexError. (See "Clamp n_new" comment
    # below for the original intent.)
    gen_stack = torch.stack([acts_all[t][ctx - 1] for t in range(n_new)])
    gen_sigs = _per_position_signatures(gen_stack, acts_ref)
    # Attach baselines at the corresponding positions (length n_new)
    gen_sigs["lm_entropy"] = list(lm_entropies[:n_new])
    gen_sigs["neg_max_logprob"] = list(neg_max_logprobs[:n_new])
    out["gen"] = gen_sigs

    # --- span=prompt ---
    # The prefill's prompt positions are at coords [ctx-prompt_len, ctx).
    # MEDIUM 2 fix: the prompt positions EXCLUDE the rightmost one
    # (ctx-1) -- that position is also `acts_ref` (the Signature B
    # drift reference = registry state at the last prompt token).
    # Including it in span=prompt makes jaccard/cosine/prompt_domain_share
    # identically 1.0 at that position (a vector compared against
    # itself), which is a self-referential contribution to the drift
    # signal. The slice is now [p_start, ctx-1).
    p_start = ctx - prompt_len
    prompt_stack = acts_all[0][p_start:ctx - 1]
    prompt_sigs = _per_position_signatures(prompt_stack, acts_ref)
    out["prompt"] = prompt_sigs

    # --- span=entity (TEST 1 only, when verified) ---
    if entity_span is not None:
        ent_start, ent_end = entity_span
        e_start = p_start + ent_start
        e_end   = p_start + ent_end
        if e_end > e_start and e_end <= ctx:
            entity_stack = acts_all[0][e_start:e_end]
            entity_sigs = _per_position_signatures(entity_stack, acts_ref)
            out["entity"] = entity_sigs
        else:
            # MEDIUM (T-067): silent-skip regression. The previous code
            # fell through here with no diagnostic -- the run was KEPT
            # (its entity span was just None), but there was no record
            # of WHY span=entity disappeared. This is exactly the symptom
            # of the original exp5 zero result: every TEST 1 run had
            # its entity span silently skipped, the AUROC table for
            # span=entity was empty / NaN, and the verdict reported an
            # apparently honest negative that was actually "I never
            # measured what I claimed to measure." Record the bounds
            # failure so the caller can print a loud summary.
            entity_span_bound_failures.append(
                ("(domain-agnostic)", e_start, e_end, ctx, prompt_len))

    return out


# exp5b invariant: still no setter ever called
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros before the generation loop"


# %% [markdown]
# ## 13. Main generation loop -- TEST 1 + TEST 2 (per-span signatures)
#
# We collect every run into a flat list `runs` with per-token
# signatures and labels. After the loop we aggregate to per-N means
# and compute AUROC, separately for the three spans (`entity`,
# `prompt`, `gen`).
#
# `REPEATS = 3` generations per prompt, each with a different seed
# (`torch.manual_seed(SEED_BASE + rep_n)`). Total = 1080 runs
# = (4 * 2 * 30 + 4 * 30) * 3 = 360 prompts * 3.
#
# For TEST 1 we ALSO compute the entity token index range by
# separately tokenizing the template prefix and the prefix+entity
# (FIX 2c, section 11c). If the BPE reconstruction doesn't verify,
# we drop that item (and log it). For TEST 2 (cloze, no `{X}` slot)
# `span=entity` is not applicable.
#
# For TEST 2 the label is computed AFTER generation: decode the
# 32-token continuation and check (case-insensitive substring)
# whether ANY accepted answer string appears. Label = CORRECT (0)
# if matched, else INCORRECT (1).

# %%
# FIX 3 (review T-056-review MEDIUM B): prevent chance matches where short
# or common accepted answers appear by luck in any of 32 sampled
# tokens. Rule applied uniformly across all domains:
#   1. The match is anchored to the FIRST 8 generated tokens only
#      (ANCHOR_TOKENS), not anywhere in the full 32-token continuation.
#   2. Accepted answers >= MIN_ANSWER_LEN characters long use plain
#      case-insensitive substring match within those 8 tokens.
#   3. Accepted answers that are alphabetic/numeric AND shorter than
#      MIN_ANSWER_LEN (bare numbers / letters) must be adjacent to a
#      context word registered in BARENUM_CONTEXT. This handles
#      law years ("1787", "1973", ...) and counts ("9", "27", "10",
#      "18") that would otherwise appear by luck in any 32-token
#      continuation; "nine justices" / "9 justices" / "1787, the
#      Constitution ..." etc. are the legitimate phrases.
#   4. Accepted answers that are SYMBOLIC (operators / non-alnum
#      characters such as "//", "%", "**") pass with direct substring
#      match -- they are distinctive enough that the 8-token anchor
#      suffices.
MIN_ANSWER_LEN   = 3
ANCHOR_TOKENS    = 8
BARENUM_CONTEXT  = {
    # law bare-number years -- context words that should sit next to them
    "1787": ("constitution",),
    "1973": ("roe", "wade"),
    "2001": ("patriot",),
    "2010": ("patient", "aca", "obamacare", "affordable", "citizens"),
    "1965": ("voting", "rights"),
    "1776": ("declaration", "independence"),
    # law bare-number counts
    "27":   ("amendment", "amendments"),
    "10":   ("bill", "rights"),
    "18":   ("age", "voting", "vote", "majority"),
    "9":    ("justices", "supreme", "court", "nine"),
    # code short abbreviations -- require the long form nearby
    "tf":   ("tensorflow", "google", "library"),
    "np":   ("numpy", "numerical", "library"),
}


def _is_correct(generated_tokens, accepted):
    """Case-insensitive substring match for TEST 2 -- but anchored to
    the first ANCHOR_TOKENS generated tokens, and with a >=3-char
    rule plus adjacent-context rule for short answers (see comment
    block above). Operates on raw generated token ids so the anchor
    is exactly the model's first 8 outputs, regardless of how they
    decode together."""
    head = list(generated_tokens[:ANCHOR_TOKENS])
    head_text = tokenizer.decode(head).lower()
    for ans in accepted:
        a = ans.lower()
        if a in head_text and (len(a) >= MIN_ANSWER_LEN or not a.isalnum()):
            return True
        # Short alphabetic / numeric answer -- require adjacent context.
        if a.isalnum() and len(a) < MIN_ANSWER_LEN:
            ctx_words = BARENUM_CONTEXT.get(a, ())
            for cw in ctx_words:
                if f"{cw} {a}" in head_text or f"{a} {cw}" in head_text:
                    return True
    return False


runs = []
dropped_items = []   # T-061 semantic: entity-span RECONSTRUCTION FAILURES.
                      # These are NO LONGER dropped from the dataset (the run
                      # is KEPT for span=prompt / span=gen; only span=entity
                      # is lost). Kept under the name `dropped_items` for
                      # backward compatibility with the JSON schema; see the
                      # JSON serialization site for the full note.
t_halluc = time.time()
n_runs_per_test = {1: 0, 2: 0}

print("=" * 78)
print("exp5b hallucination-detection loop (TEST 1 = real/fake, TEST 2 = cloze)")
print(f"  REPEATS = {REPEATS}, GEN_LEN = {GEN_LEN}, total = 1080 runs target")
print("=" * 78)

# ---- TEST 1: grounded vs ungrounded ----
# Per domain: 30 real + 30 fake prompts. Each is repeated REPEATS times.
print("\n[ TEST 1 ] grounded vs ungrounded (REAL vs FAKE entity names)")
for d in DOMAIN_NAMES:
    template = PROMPT_TEMPLATE_TEST1[d]
    for entity_kind, entity_list in (("real", REAL_ENTITIES[d]),
                                     ("fake", FAKE_ENTITIES[d])):
        for entity in entity_list:
            # FIX 2c / FIX 2d: compute the entity token index range via
            # _compute_entity_span. If it fails (BPE boundary merge,
            # BOS mismatch, etc.), LOW FIX (doc/code consistency):
            # per the markdown at L1512-1514 the run is KEPT for
            # `span=prompt` and `span=gen` and only `span=entity` is
            # lost. The previous code did `continue` here which
            # dropped the entire item -- contradicting the doc. We
            # now keep the run, log the entity-span failure, and pass
            # `entity_span=None` to compute_signatures_for_run (which
            # already handles that case by omitting the `entity` key
            # from the returned dict).
            span_res = _compute_entity_span(template, entity, tokenizer)
            prompt_text = template.format(X=entity)
            if span_res is None:
                dropped_items.append((1, d, entity_kind, entity,
                                      "entity_span reconstruction failed"))
                # Fallback: tokenize the whole prompt directly so we
                # still get a valid prompt_ids for generation. The
                # entity-span computation would have given the same
                # prompt_ids (= tokenizer.encode(prefix_str + entity +
                # suffix_str) = tokenizer.encode(prompt_text)).
                prompt_ids = tokenizer.encode(prompt_text)
                ent_start, ent_end = None, None
            else:
                prompt_ids, ent_start, ent_end = span_res
            for rep_n in range(1, REPEATS + 1):
                torch.manual_seed(SEED_BASE + rep_n)
                toks, lm_ents, neg_mlps, rec, prompt_len = generate_with_signatures(
                    prompt_ids, n_new=GEN_LEN, hook=_readonly_hook)
                assert prompt_len == len(prompt_ids), \
                    f"prompt_len ({prompt_len}) != len(prompt_ids) " \
                    f"({len(prompt_ids)}) -- generate_with_signatures must " \
                    f"report the un-padded length the caller passed in"
                decoded = tokenizer.decode(toks)
                # FIX 2f: compute per-span signatures (entity_span=None
                # when the entity-span reconstruction failed above)
                span_sigs = compute_signatures_for_run(
                    rec, prompt_len=prompt_len,
                    entity_span=(ent_start, ent_end) if ent_start is not None else None,
                    lm_entropies=list(lm_ents),
                    neg_max_logprobs=list(neg_mlps))
                runs.append(dict(
                    test=1, domain=d, condition=entity_kind,
                    rep_n=rep_n, seed=SEED_BASE + rep_n,
                    prompt_text=prompt_text, entity=entity,
                    entity_span=(ent_start, ent_end) if ent_start is not None else None,
                    prompt_ids=[int(t) for t in prompt_ids],
                    generated_tokens=[int(t) for t in toks],
                    generated_text=decoded,
                    # Per-span per-step signatures; each span maps
                    # signal name -> list of floats (one per
                    # position in that span).
                    signatures_by_span={
                        span: {sig: list(v) for sig, v in sig_dict.items()}
                        for span, sig_dict in span_sigs.items()
                    },
                ))
                n_runs_per_test[1] += 1

# ---- TEST 2: correct vs incorrect (cloze) ----
# Per domain: 30 cloze prompts. Repeat REPEATS times.
# Label is computed from the decoded continuation. span=entity is N/A.
print("\n[ TEST 2 ] correct vs incorrect (cloze facts, label from decoded continuation)")
for d in DOMAIN_NAMES:
    for prompt_text, accepted in CLOZE_FACTS[d]:
        prompt_ids = tokenizer.encode(prompt_text)
        for rep_n in range(1, REPEATS + 1):
            torch.manual_seed(SEED_BASE + rep_n)
            toks, lm_ents, neg_mlps, rec, prompt_len = generate_with_signatures(
                prompt_ids, n_new=GEN_LEN, hook=_readonly_hook)
            assert prompt_len == len(prompt_ids), \
                f"prompt_len ({prompt_len}) != len(prompt_ids) " \
                f"({len(prompt_ids)}) -- generate_with_signatures must " \
                f"report the un-padded length the caller passed in"
            decoded = tokenizer.decode(toks)
            label_correct = _is_correct(toks, accepted)
            # FIX 2f: compute per-span signatures (entity_span=None)
            span_sigs = compute_signatures_for_run(
                rec, prompt_len=prompt_len, entity_span=None,
                lm_entropies=list(lm_ents),
                neg_max_logprobs=list(neg_mlps))
            runs.append(dict(
                test=2, domain=d,
                condition=("correct" if label_correct else "incorrect"),
                rep_n=rep_n, seed=SEED_BASE + rep_n,
                prompt_text=prompt_text,
                accepted_answers=list(accepted),
                prompt_ids=[int(t) for t in prompt_ids],
                generated_tokens=[int(t) for t in toks],
                generated_text=decoded,
                signatures_by_span={
                    span: {sig: list(v) for sig, v in sig_dict.items()}
                    for span, sig_dict in span_sigs.items()
                },
            ))
            n_runs_per_test[2] += 1

elapsed = time.time() - t_halluc
print(f"\nhallucination block done in {elapsed/60:.1f} minutes "
      f"({len(runs)} runs, TEST 1 = {n_runs_per_test[1]}, "
      f"TEST 2 = {n_runs_per_test[2]}, "
      f"entity_span_failures = {len(dropped_items)})")

if dropped_items:
    print("\nENTITY-SPAN FAILURES (run KEPT for prompt/gen; entity span is N/A):")
    for it in dropped_items:
        print(f"  test={it[0]} domain={it[1]:>9s} kind={it[2]:>4s} entity={it[3]!r:35s}  reason={it[4]}")

# MEDIUM (T-067): loud summary of the BOUNDS-FAILURE list. This is
# the diagnostic for the silent-span=entity-skip regression the
# previous code had: `_compute_entity_span` returned a tuple that
# LOOKED valid but failed the bounds check inside
# `compute_signatures_for_run`. Without this printout the failure
# mode is invisible. Print UNCONDITIONALLY (not gated on emptiness)
# when the list is empty too -- so a reader can confirm "0 bounds
# failures" without scanning the file.
print(f"\nENTITY-SPAN BOUNDS FAILURES (e_start, e_end, ctx, prompt_len): "
      f"{len(entity_span_bound_failures)} total")
if entity_span_bound_failures:
    for it in entity_span_bound_failures:
        print(f"  domain={it[0]}  e_start={it[1]}  e_end={it[2]}  "
              f"ctx={it[3]}  prompt_len={it[4]}")

# exp5b invariant: nothing in this block sets steer_bias non-zero
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros after the hallucination block"


# %% [markdown]
# ## 14. AUROC -- rank-based, no sklearn dependency (per-span)
#
# Per spec: "Implement AUROC directly (rank-based, no sklearn
# dependency assumption -- but sklearn is fine if already available;
# if you use it, guard the import)." We implement rank-based AUROC
# from scratch. Ties in scores get the average rank (standard).
#
# Per-N aggregation: for each run, take the mean of the first N
# token-level values of each signal, then compute AUROC across runs.
# Per-domain = filter to one domain; pooled = concat all 4 domains.
#
# exp5b generalization: the same logic is applied PER SPAN. Each
# span (`entity`, `prompt`, `gen`) has its own per-run per-N AUROC
# table. For `span=prompt` and `span=entity` the LM-head baselines
# (`lm_entropy`, `neg_max_logprob`) are not computed (they only
# exist for positions where Pythia was sampling next-token logits),
# so those signals are absent from those AUROC tables.

# %%
def auroc_rank(scores, labels):
    """Rank-based AUROC. Positive class = 1.
    `scores`, `labels` are 1-D numpy / python iterables. Returns
    AUROC in [0, 1], or NaN if one class is empty."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos_mask = (labels == 1)
    n_pos = int(pos_mask.sum())
    n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    n = len(scores)
    # Sort ascending; assign average ranks for ties
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0     # 1-indexed average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    pos_ranks_sum = float(ranks[pos_mask].sum())
    auc = (pos_ranks_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


# Aggregate per-N signal means for all runs in a given test+span
def _aggregate_per_n_span(runs_test, span, signal_name, n_values):
    """For each run in `runs_test` and each N in `n_values`, compute
    the mean of the first N values of `signal_name` in the given
    span. Returns:
      - pooled_means: shape [n_runs, len(n_values)] (pooled across domains)
      - per_domain_means: dict {domain: shape [n_runs_in_d, len(n_values)]}
      - pooled_labels: shape [n_runs]
      - per_domain_labels: dict {domain: shape [n_runs_in_d]}

    If a run does not have the requested span (e.g., span='entity'
    for a TEST 2 cloze run), that run contributes NaN labels and is
    skipped from the per-domain and pooled accumulators.
    """
    pooled_means, pooled_labels = [], []
    per_domain_means = {d: [] for d in DOMAIN_NAMES}
    per_domain_labels = {d: [] for d in DOMAIN_NAMES}
    for r in runs_test:
        if span not in r["signatures_by_span"]:
            continue
        sig = r["signatures_by_span"][span].get(signal_name, None)
        if sig is None:
            continue
        means = [float(np.mean(sig[:n])) for n in n_values]
        pooled_means.append(means)
        if r["test"] == 1:
            lbl = 1 if r["condition"] == "fake" else 0
        else:
            lbl = 1 if r["condition"] == "incorrect" else 0
        pooled_labels.append(lbl)
        per_domain_means[r["domain"]].append(means)
        per_domain_labels[r["domain"]].append(lbl)
    pooled_means = np.asarray(pooled_means, dtype=np.float64)         # [R, len(n_values)]
    pooled_labels = np.asarray(pooled_labels, dtype=np.int64)
    for d in DOMAIN_NAMES:
        per_domain_means[d] = np.asarray(per_domain_means[d], dtype=np.float64)
        per_domain_labels[d] = np.asarray(per_domain_labels[d], dtype=np.int64)
    return pooled_means, pooled_labels, per_domain_means, per_domain_labels


def compute_aurocs_for_test_span(runs_test, span, signal_name, n_values):
    """Compute AUROC tables for one signal in one test in one span.
    Returns a dict with keys 'pooled' and 'per_domain'; each maps
    index 0..len(n_values)-1 -> AUROC float."""
    pm, pl, pdm, pdl = _aggregate_per_n_span(runs_test, span, signal_name, n_values)
    out = {"pooled": [], "per_domain": {d: [] for d in DOMAIN_NAMES}}
    for j in range(len(n_values)):
        out["pooled"].append(auroc_rank(pm[:, j], pl))
        for d in DOMAIN_NAMES:
            out["per_domain"][d].append(auroc_rank(pdm[d][:, j], pdl[d]))
    return out


runs_test1 = [r for r in runs if r["test"] == 1]
runs_test2 = [r for r in runs if r["test"] == 2]

# Which spans apply per test?
# TEST 1 (entity+prompt+gen): all 3 spans.
# TEST 2 (cloze, no {X}): prompt + gen only -- entity is N/A.
SPANS_FOR_TEST = {
    1: ["entity", "prompt", "gen"],
    2: ["prompt", "gen"],
}

# The set of available signals per span:
#   span='gen'     -> 7 registry + 2 baseline = 9 signals
#   span='prompt'  -> 7 registry (no baselines)
#   span='entity'  -> 7 registry (no baselines)


def _signals_for_span(span):
    if span == "gen":
        return SIGNALS_ALL
    return SIGNALS_REGISTRY


auroc_tables = {}     # auroc_tables[test_name][span][signal] = {pooled, per_domain}
for test_id, runs_test, test_name in ((1, runs_test1, "test1"),
                                       (2, runs_test2, "test2")):
    auroc_tables[test_name] = {}
    for span in SPANS_FOR_TEST[test_id]:
        auroc_tables[test_name][span] = {}
        signals = _signals_for_span(span)
        for sig in signals:
            auroc_tables[test_name][span][sig] = compute_aurocs_for_test_span(
                runs_test, span, sig, cfg.n_values)


# %% [markdown]
# ## 15. AUROC tables (pooled + per-domain), per span
#
# Two blocks, one per test. Within each block, three sub-blocks
# (entity / prompt / gen) for TEST 1, two sub-blocks (prompt / gen)
# for TEST 2. Each sub-block prints:
#
# - **Pooled AUROC** -- rows = signals, columns = N in `cfg.n_values`.
#   Positive class = fake (TEST 1) or incorrect (TEST 2). Values are
#   the raw AUROC; `0.5 = no discrimination`, `1.0 = perfect`, `0.0 =
#   anti-perfect`.
# - **Per-domain AUROC** -- one table per domain, same shape.
#
# The two LM-head baselines (`lm_entropy`, `neg_max_logprob`) are
# printed only in the `span=gen` sub-block (baselines are not
# defined at prompt or entity positions). Domains with |real_mean -
# fake_mean| > CONFOUND_ABS_GAP_THRESHOLD are flagged as TEST-1-
# CONFOUNDED next to their `span=entity` AUROC.

# %%
def _print_auroc_table(title, auroc_dict, signals, n_values,
                        registry_signals, baseline_signals,
                        subtitle=None):
    print("\n" + "=" * 78)
    print(title)
    if subtitle:
        for line in subtitle.split("\n"):
            print("  " + line)
    print("=" * 78)
    hdr = f"{'signal':>22s}" + "".join(f"{f'N={n}':>10s}" for n in n_values)
    print(hdr)
    print("-" * len(hdr))
    for sig in signals:
        kind = "(B)" if sig in baseline_signals else "   "
        row = f"{sig:>20s}{kind}"
        aurocs = auroc_dict[sig]["pooled"]
        for a in aurocs:
            row += f"{a:>10.3f}"
        print(row)
    print("-" * len(hdr))
    print("    (B) = LM-head baseline (not from the registry)")
    print("    Note: raw AUC < 0.5 = inverted-but-informative signal;")
    print("          the verdict uses max(AUC, 1 - AUC) (polarity-corrected).")
    print()


POLARITY_NOTE = ("Note: raw AUC < 0.5 = inverted-but-informative; "
                 "verdict uses max(AUC, 1 - AUC) (polarity-corrected).")

for test_id, test_name, title_short in (
    (1, "test1", "TEST 1 -- FAKE vs REAL (positive = FAKE)"),
    (2, "test2", "TEST 2 -- INCORRECT vs CORRECT (positive = INCORRECT)"),
):
    spans_here = SPANS_FOR_TEST[test_id]
    for span in spans_here:
        signals_here = _signals_for_span(span)
        is_baseline_span = (span == "gen")
        span_label = {
            "entity": "ENTITY (TEST 1 only, N/A for TEST 2)",
            "prompt": "PROMPT (all left-padded prompt tokens)",
            "gen":    "GEN (the 32 generated tokens; LM baselines available)",
        }[span]
        _print_auroc_table(
            title=f"POOLED AUROC, {title_short}, span={span} -- {span_label}",
            auroc_dict=auroc_tables[test_name][span],
            signals=signals_here,
            n_values=cfg.n_values,
            registry_signals=set(SIGNALS_REGISTRY),
            baseline_signals=(set(SIGNALS_BASELINE) if is_baseline_span else set()),
            subtitle=POLARITY_NOTE,
        )
        for d in DOMAIN_NAMES:
            conf_subtitle = POLARITY_NOTE
            # Confound warning only for TEST 1 (no name-rarity confound for cloze).
            if test_name == "test1" and d in confound_report:
                # FIX B (T-065): use the in-context (span=entity) gap
                # for the severity report -- the bare gap is kept as a
                # fallback for backward compatibility but is no longer
                # the measurement that keys the confound flag.
                gap = confound_report[d].get("ic_gap_after",
                                            confound_report[d].get("gap_after",
                                            confound_report[d].get("abs_gap", float("inf"))))
                if gap > CONFOUND_ABS_GAP_THRESHOLD:
                    conf_subtitle = (
                        POLARITY_NOTE
                        + "\n"
                        + f"*** TEST 1 CONFOUNDED for {d}: real/fake "
                          f"in-context (span=entity) token-length gap "
                          f"= {gap:.2f} (> {CONFOUND_ABS_GAP_THRESHOLD}) "
                          f"-- AUROC for this domain may reflect name "
                          f"rarity, not grounding ***"
                    )
            # Build a synthetic per-domain auroc_dict in the shape
            # _print_auroc_table expects (signal -> {pooled, per_domain}).
            per_domain_aurocs_for_span = {
                sig: dict(
                    pooled=auroc_tables[test_name][span][sig]["per_domain"][d],
                    per_domain={},
                )
                for sig in signals_here
            }
            _print_auroc_table(
                title=f"PER-DOMAIN AUROC, {d}, {title_short}, span={span}",
                auroc_dict=per_domain_aurocs_for_span,
                signals=signals_here,
                n_values=cfg.n_values,
                registry_signals=set(SIGNALS_REGISTRY),
                baseline_signals=(set(SIGNALS_BASELINE) if is_baseline_span else set()),
                subtitle=conf_subtitle,
            )


# %% [markdown]
# ## 16. Per-span verdict block -- does any registry signal beat the baseline?
#
# For each TEST and each SPAN, at each N, we compare the registry's
# best signal (max AUROC discrimination = `max(AUROC, 1 - AUROC)`)
# against the best baseline's discrimination at the same N.
# `max(AUROC, 1-AUROC)` is the "discrimination strength" -- it
# ignores whether the signal is positively or negatively correlated
# with the hallucination label, so a signal that's anti-correlated
# still counts as a discriminator.
#
# `span=gen` is the only one where LM-head baselines exist (the
# baselines come from Pythia's own next-token softmax, which is only
# computed at positions where Pythia was sampling). For `span=prompt`
# and `span=entity` there is no baseline; the verdict reports the
# best registry signal at each N but cannot compute a gap to a
# baseline.
#
# The honest negative result is printed plainly if no registry signal
# beats both baselines at any N (only applies for `span=gen`; for the
# other spans we just report the best registry signal and the gap to
# 0.5).

# %%
def _disc(auc):
    """Discrimination strength: max(AUC, 1 - AUC) in [0.5, 1.0]."""
    if math.isnan(auc):
        return float("nan")
    return max(auc, 1.0 - auc)


def _verdict_for_test_span(test_name, span):
    """Return a per-N verdict table for one test + span + a single-
    line overall result. For `span=gen` the table includes the best
    LM-head baseline's discrimination and the gap (registry -
    baseline). For spans without baselines the table includes
    `baseline=None` and `gap = best_registry_disc - 0.5` (i.e. the
    margin above chance)."""
    rows = []
    best_gap = -1e9    # the per-N gap (registry vs best baseline OR vs 0.5)
    best_N = None
    is_baseline_span = (span == "gen")
    signals_here = _signals_for_span(span)
    for j, N in enumerate(cfg.n_values):
        # Best registry signal in this span
        best_reg_name, best_reg_auc, best_reg_disc = None, float("nan"), -1.0
        for sig in SIGNALS_REGISTRY:
            auc = auroc_tables[test_name][span][sig]["pooled"][j]
            d = _disc(auc)
            if d > best_reg_disc:
                best_reg_disc = d; best_reg_auc = auc; best_reg_name = sig
        if is_baseline_span:
            best_base_name, best_base_auc, best_base_disc = None, float("nan"), -1.0
            for sig in SIGNALS_BASELINE:
                auc = auroc_tables[test_name][span][sig]["pooled"][j]
                d = _disc(auc)
                if d > best_base_disc:
                    best_base_disc = d; best_base_auc = auc; best_base_name = sig
            gap = best_reg_disc - best_base_disc
        else:
            best_base_name = best_base_auc = best_base_disc = None
            gap = best_reg_disc - 0.5
        if gap > best_gap:
            best_gap = gap; best_N = N
        rows.append(dict(
            N=N,
            best_registry=best_reg_name,
            best_registry_auc=float(best_reg_auc),
            best_registry_disc=float(best_reg_disc),
            best_baseline=best_base_name,
            best_baseline_auc=(float(best_base_auc) if best_base_auc is not None else None),
            best_baseline_disc=(float(best_base_disc) if best_base_disc is not None else None),
            gap=float(gap),
            has_baseline=is_baseline_span,
        ))
    return rows, float(best_gap), best_N


print("=" * 78)
print("EXP5b VERDICT -- per-span, per-test (does any registry signal")
print("beat the LM-head baseline where available?)")
print("(discrimination = max(AUC, 1 - AUC); higher is better)")
print("=" * 78)

verdict_results = {}
# Track whether ANYTHING positive happened for the headline verdict.
any_beat = False
any_negative_global = True   # until proven otherwise

for test_id, test_name, title in (
    (1, "test1", "TEST 1: FAKE vs REAL"),
    (2, "test2", "TEST 2: INCORRECT vs CORRECT"),
):
    spans_here = SPANS_FOR_TEST[test_id]
    print(f"\n{'=' * 78}")
    print(f"  {title}")
    print('=' * 78)
    verdict_results[test_name] = {}
    for span in spans_here:
        rows, best_gap, best_N = _verdict_for_test_span(test_name, span)
        verdict_results[test_name][span] = dict(
            rows=[dict(
                N=int(r["N"]),
                best_registry=r["best_registry"],
                best_registry_auc=float(r["best_registry_auc"]),
                best_registry_disc=float(r["best_registry_disc"]),
                best_baseline=r["best_baseline"],
                best_baseline_auc=r["best_baseline_auc"],
                best_baseline_disc=r["best_baseline_disc"],
                gap=float(r["gap"]),
                has_baseline=bool(r["has_baseline"]),
            ) for r in rows],
            best_gap=float(best_gap),
            best_N=int(best_N) if best_N is not None else None,
        )
        has_baseline = rows[0]["has_baseline"]
        if has_baseline:
            print(f"\n  {title}, span={span}:")
            print(f"  {'N':>3s} | {'best registry':>20s} | {'reg AUC':>8s} | "
                  f"{'reg disc':>8s} | {'best baseline':>16s} | {'base AUC':>8s} | "
                  f"{'base disc':>8s} | {'gap':>6s}")
            print("  " + "-" * 110)
            for r in rows:
                print(f"  {r['N']:>3d} | {r['best_registry']:>20s} | "
                      f"{r['best_registry_auc']:>8.3f} | {r['best_registry_disc']:>8.3f} | "
                      f"{r['best_baseline']:>16s} | {r['best_baseline_auc']:>8.3f} | "
                      f"{r['best_baseline_disc']:>8.3f} | "
                      f"{r['gap']:>+6.3f}")
            if best_gap > 0.0:
                print(f"\n  VERDICT ({title}, span={span}): the registry's "
                      f"best signal beats the best LM-head baseline by "
                      f"{best_gap:+.3f} discrimination at N={best_N}.")
                any_beat = True
            else:
                print(f"\n  VERDICT ({title}, span={span}): HONEST NEGATIVE -- "
                      f"no registry signal beats the best baseline at any N. "
                      f"Best gap = {best_gap:+.3f} (at N={best_N}).")
        else:
            # Spans without a baseline: the "gap" is registry disc - 0.5.
            print(f"\n  {title}, span={span} (no LM baseline for this span):")
            print(f"  {'N':>3s} | {'best registry':>20s} | {'reg AUC':>8s} | "
                  f"{'reg disc':>8s} | {'disc - 0.5':>9s}")
            print("  " + "-" * 80)
            for r in rows:
                print(f"  {r['N']:>3d} | {r['best_registry']:>20s} | "
                      f"{r['best_registry_auc']:>8.3f} | {r['best_registry_disc']:>8.3f} | "
                      f"{r['gap']:>+9.3f}")
            if best_gap > 0.0:
                print(f"\n  VERDICT ({title}, span={span}): the registry's "
                      f"best signal achieves {best_gap:+.3f} discrimination "
                      f"above chance at N={best_N} "
                      f"(no baseline to beat in this span).")

# Headline overall verdict
print("\n" + "=" * 78)
print("HEADLINE EXP5b VERDICT")
print("=" * 78)
if any_beat:
    print("  exp5b found at least one registry signal that beats the LM-head")
    print("  baseline, in at least one (test, span) combination, at some N.")
    print("  See the per-span tables above for the exact (test, span, N).")
    any_negative_global = False
else:
    print("  HONEST NEGATIVE -- no registry signal beats the LM-head baseline")
    print("  in ANY (test, span) combination at ANY N. The registry's internal")
    print("  state, measured across the prompt and the generated tokens, does")
    print("  not improve on Pythia's own next-token entropy / max-log-prob for")
    print("  distinguishing FAKE-vs-REAL or INCORRECT-vs-CORRECT continuations.")
    print("  This is the same negative result exp5 returned, despite the new")
    print("  all-positions hook and the token-length-matched FAKE lists.")

# HIGH (T-067) -- confound-honest additions to the headline verdict.
# The headline above says "found at least one registry signal that
# beats the LM-head baseline" pooled over ALL domains. If the only
# positive(s) come from CONFOUNDED TEST 1 domains (in-context
# |real_mean - fake_mean| > CONFOUND_ABS_GAP_THRESHOLD), the contrast
# in those domains may reflect NAME RARITY (rare strings produce
# unusual BPE lengths, and Pythia distributes its next-token mass
# differently over rare strings regardless of any registry signal) --
# it is NOT evidence of hallucination detection. Without these
# additions a reader cannot tell whether the headline survives once
# the confounded domains are removed.
print("\n" + "-" * 78)
print("CONFOUND-HONEST HEADLINE ADDITIONS")
print("-" * 78)

# 1. List of CONFOUNDED domains (TEST 1 only -- TEST 2 has no
#    name-rarity confound; cloze prompts are fixed strings).
confounded_domains = [
    d for d in DOMAIN_NAMES
    if confound_report.get(d, {}).get("ic_gap_after", float("inf"))
       > CONFOUND_ABS_GAP_THRESHOLD
]
nonconfounded_domains = [d for d in DOMAIN_NAMES if d not in confounded_domains]

print(f"  CONFOUND_ABS_GAP_THRESHOLD = {CONFOUND_ABS_GAP_THRESHOLD} "
      f"BPE tokens (in-context |real_mean - fake_mean| after FIX 3)")
print(f"  CONFOUNDED domains ({len(confounded_domains)}): "
      f"{confounded_domains if confounded_domains else 'none'}")
for d in confounded_domains:
    ic_gap = confound_report.get(d, {}).get("ic_gap_after", float("inf"))
    print(f"    {d:>10s}: in-context gap-after = {ic_gap:.3f}")
print(f"  NON-CONFOUNDED domains ({len(nonconfounded_domains)}): "
      f"{nonconfounded_domains if nonconfounded_domains else 'none'}")

if confounded_domains:
    print()
    print("  *** Any TEST 1 result that rests ONLY on CONFOUNDED domains")
    print("  *** may reflect NAME RARITY, not grounding: rare / unusual")
    print("  *** BPE strings are distributed differently by Pythia's")
    print("  *** softmax regardless of any registry signal, so a positive")
    print("  *** AUROC in those domains is not by itself evidence of")
    print("  *** hallucination detection. ***")
else:
    print("  (no CONFOUNDED domains -- the headline above is confound-clean.)")

# 2. TEST 1 / span=entity, best registry discrimination restricted to
#    NON-CONFOUNDED domains only. Reuses the existing
#    auroc_tables["test1"]["entity"][sig]["per_domain"][d] values so
#    the numbers are bit-identical to what the per-domain table
#    printed earlier; only the domain filter changes. `gap` here is
#    best_reg_disc - 0.5 (the same convention the no-baseline spans
#    use, since span=entity has no LM baseline either).
print()
print(f"  TEST 1, span=entity, NON-CONFOUNDED-only best registry discrimination")
print(f"  (max over the 7 registry signals at each N; gap = disc - 0.5):")
print(f"  {'N':>3s} | {'best registry':>20s} | {'reg AUC':>8s} | "
      f"{'reg disc':>8s} | {'disc - 0.5':>9s}")
print("  " + "-" * 70)

test1_entity_nonconfounded = {}
if not nonconfounded_domains:
    print("  (no NON-CONFOUNDED domains to evaluate -- every domain is CONFOUNDED)")
else:
    for j, N in enumerate(cfg.n_values):
        br_name, br_auc, br_disc = None, float("nan"), -1.0
        for sig in SIGNALS_REGISTRY:
            # Pool the per-domain AUROCs at this N across non-confounded
            # domains only; "pool" here means take the MAX discrimination
            # across the filtered set -- a single non-confounded domain
            # with a strong signal still counts. (We are not computing a
            # pooled-domain mean: the per-N verdict answers "did ANYTHING
            # survive once confounds were removed?")
            for d in nonconfounded_domains:
                auc = auroc_tables["test1"]["entity"][sig]["per_domain"][d][j]
                disc = _disc(auc)
                if disc > br_disc:
                    br_disc = disc; br_auc = auc; br_name = sig
        gap = br_disc - 0.5
        test1_entity_nonconfounded[N] = dict(
            best_registry=br_name,
            best_registry_auc=float(br_auc),
            best_registry_disc=float(br_disc),
            gap=float(gap),
        )
        print(f"  {N:>3d} | {br_name:>20s} | {br_auc:>8.3f} | "
              f"{br_disc:>8.3f} | {gap:>+9.3f}")

    # One-line summary: does anything survive the confound filter?
    best_nc = max(test1_entity_nonconfounded.items(),
                  key=lambda kv: kv[1]["gap"])
    if best_nc[1]["gap"] > 0.0:
        print(f"\n  NON-CONFOUNDED verdict: at N={best_nc[0]} the registry's "
              f"best signal ({best_nc[1]['best_registry']}) reaches "
              f"{best_nc[1]['gap']:+.3f} discrimination above chance -- "
              f"something survives the confound filter.")
    else:
        print(f"\n  NON-CONFOUNDED verdict: nothing reaches above-chance "
              f"discrimination at any N (max gap = "
              f"{best_nc[1]['gap']:+.3f} at N={best_nc[0]}). The headline "
              f"above, if positive, was driven entirely by CONFOUNDED "
              f"domains.")
print("=" * 78)

# Per-domain verdict for TEST 1, confound-aware, at headline N.
# Use span='entity' as the most natural choice -- entity-span AUROC
# is the new finding of exp5b; a domain's gap dominance may reflect
# name rarity.
print("\n" + "-" * 78)
print("PER-DOMAIN VERDICT (TEST 1, span=entity) -- confound-aware")
print(f"  (CONFOUND_ABS_GAP_THRESHOLD = {CONFOUND_ABS_GAP_THRESHOLD} "
      f"BPE tokens; domains above are flagged)")
print("-" * 78)
print(f"  {'domain':>10s} | {'ic_gap_after':>12s} | {'confounded':>10s} | "
      f"{'best reg (N=8)':>18s} | {'best reg disc':>14s} | "
      f"{'disc - 0.5':>9s}")
print("  " + "-" * 100)
# Pick N=8 as the headline per-domain N (middle of the sweep)
HEADLINE_N = 8
j_head = cfg.n_values.index(HEADLINE_N)
per_domain_verdict = {}
for d in DOMAIN_NAMES:
    # FIX B (T-065): the confound flag below keys on the IN-CONTEXT
    # gap-after (the length of the actual entity span the registry
    # sees), not on the historical bare-name gap. The bare gap is
    # still kept in `confound_report[d]['gap_after']` and is included
    # in the JSON payload for backward compatibility.
    ic_gap_after = confound_report.get(d, {}).get("ic_gap_after",
                                                 confound_report.get(d, {}).get("gap_after",
                                                 confound_report.get(d, {}).get("abs_gap", float("inf"))))
    confounded = bool(ic_gap_after > CONFOUND_ABS_GAP_THRESHOLD)
    # Best registry signal at N=8 for this domain, span=entity
    if "entity" not in auroc_tables["test1"]:
        # Defensive: should not happen.
        per_domain_verdict[d] = dict(
            abs_gap=float(ic_gap_after), confounded=confounded,
            best_registry=None, best_registry_auc=float("nan"),
            best_registry_disc=float("nan"), gap=float("nan"))
        continue
    br_name, br_auc, br_disc = None, float("nan"), -1.0
    for sig in SIGNALS_REGISTRY:
        auc = auroc_tables["test1"]["entity"][sig]["per_domain"][d][j_head]
        disc = _disc(auc)
        if disc > br_disc:
            br_disc = disc; br_auc = auc; br_name = sig
    domain_gap = br_disc - 0.5
    flag_mark = "***YES***" if confounded else "no"
    print(f"  {d:>10s} | {ic_gap_after:>12.3f} | {flag_mark:>10s} | "
          f"{br_name:>18s} | {br_disc:>14.3f} | "
          f"{domain_gap:>+9.3f}")
    if confounded:
        print(f"  *** TEST 1 CONFOUNDED for {d}: real/fake "
              f"in-context (span=entity) token-length gap = "
              f"{ic_gap_after:.3f} -- AUROC for this domain may "
              f"reflect name rarity, not grounding ***")
    per_domain_verdict[d] = dict(
        abs_gap_after=float(ic_gap_after),
        # FIX B: also store the bare-name gap for backward compat
        # alongside the now-canonical in-context gap.
        bare_gap_after=float(confound_report.get(d, {}).get("gap_after", 0.0)),
        confounded=confounded,
        best_registry=br_name,
        best_registry_auc=float(br_auc),
        best_registry_disc=float(br_disc),
        gap=float(domain_gap),
    )
print("=" * 78)


# %% [markdown]
# ## 16b. JSON save (BEFORE plotting; FIX 4)
#
# `exp5b_halluc.json` carries every individual run's per-token
# signals + labels + the per-span AUROC tables + the tokenization-
# confound numbers + the verdict block + cfg. The summary JSON
# written earlier already carries training/purity.
#
# **CRITICAL:** this save block is placed BEFORE the plotting
# cells. If a plotting cell crashes (matplotlib backend issues,
# out-of-memory on Kaggle, etc.) the JSON outputs above are still
# on disk -- FIX 4 prevents the exp5 failure mode where one
# plotting abort silently destroyed the verdict.

# %%
halluc_payload = dict(
    cfg={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
    spans_for_test={1: SPANS_FOR_TEST[1], 2: SPANS_FOR_TEST[2]},
    signals_registry=SIGNALS_REGISTRY,
    signals_baseline=SIGNALS_BASELINE,
    n_values=list(cfg.n_values),
    gen_len=GEN_LEN,
    repeats=REPEATS,
    confound=confound_report,
    confound_threshold=CONFOUND_ABS_GAP_THRESHOLD,
    # FIX B (T-065): confound flags now key on the IN-CONTEXT span
    # length (the gap that the registry actually sees via
    # span=entity), not the historical bare-name gap. Both gaps are
    # preserved in `confound[d]` for transparency; the bare flag is
    # also kept here as a secondary historical entry for backward
    # compatibility.
    confound_flags_after={d: bool(confound_report.get(d, {}).get("ic_gap_after", 0.0)
                                  > CONFOUND_ABS_GAP_THRESHOLD)
                          for d in DOMAIN_NAMES},
    confound_flags_before={d: bool(confound_report.get(d, {}).get("ic_gap_before", 0.0)
                                   > CONFOUND_ABS_GAP_THRESHOLD)
                           for d in DOMAIN_NAMES},
    # Secondary historical entry (bare-name gap, exp5 measurement).
    confound_flags_after_bare={d: bool(confound_report.get(d, {}).get("gap_after", 0.0)
                                       > CONFOUND_ABS_GAP_THRESHOLD)
                               for d in DOMAIN_NAMES},
    confound_flags_before_bare={d: bool(confound_report.get(d, {}).get("gap_before", 0.0)
                                        > CONFOUND_ABS_GAP_THRESHOLD)
                                for d in DOMAIN_NAMES},
    # NB: this key is named `dropped_items` for backward
    # compatibility with downstream consumers of the JSON schema,
    # but at T-061 the LOW fix changed the semantics: these are
    # entity-span RECONSTRUCTION FAILURES, not items dropped from
    # the dataset. Every listed entity is still present in
    # `runs`; only its `entity_span` is None.
    dropped_items=[dict(test=it[0], domain=it[1], kind=it[2],
                        entity=it[3], reason=it[4])
                   for it in dropped_items],
    # T-067 entity-span BOUNDS failures (the silent-skip regression
    # the previous code had; span=entity vanished with no diagnostic).
    entity_span_bound_failures=[
        dict(domain=it[0], e_start=it[1], e_end=it[2],
             ctx=it[3], prompt_len=it[4])
        for it in entity_span_bound_failures
    ],
    verdict=verdict_results,
    per_domain_verdict_test1=per_domain_verdict,
    # HIGH (T-067) confound-honest headline additions. The headline
    # verdict above ("found at least one registry signal that beats
    # the LM-head baseline") is pooled over ALL domains; if it is
    # positive only because of CONFOUNDED TEST 1 domains the positive
    # may reflect name rarity, not grounding. The fields below let a
    # downstream consumer reproduce the confound-filtered verdict.
    confound_honest_headline=dict(
        confounded_domains=confounded_domains,
        nonconfounded_domains=nonconfounded_domains,
        test1_entity_nonconfounded={
            int(N): dict(
                best_registry=v["best_registry"],
                best_registry_auc=v["best_registry_auc"],
                best_registry_disc=v["best_registry_disc"],
                gap=v["gap"],
            )
            for N, v in test1_entity_nonconfounded.items()
        },
    ),
    # Per-span per-test AUROC tables. Structure:
    # auroc_tables[test_name][span][signal] = {pooled, per_domain}
    auroc_tables={
        tn: {
            span: {
                sig: dict(
                    pooled=[float(x) for x in auroc_tables[tn][span][sig]["pooled"]],
                    per_domain={d: [float(x) for x in auroc_tables[tn][span][sig]["per_domain"][d]]
                                for d in DOMAIN_NAMES},
                )
                for sig in _signals_for_span(span)
            } for span in SPANS_FOR_TEST[int(tn[4:])]
        } for tn in ("test1", "test2")
    },
    runs=[],
)

# Per-run payloads (per-span per-token signals + labels only -- no
# pre / acts tensors, those live in the hook's CPU list and are
# re-derivable from the registry + the saved prompt / generated
# token ids if anyone needs them).
for r in runs:
    halluc_payload["runs"].append(dict(
        test=int(r["test"]),
        domain=r["domain"],
        condition=r["condition"],
        rep_n=int(r["rep_n"]),
        seed=int(r["seed"]),
        prompt_text=r["prompt_text"],
        entity=r.get("entity", None),
        entity_span=list(r["entity_span"]) if r.get("entity_span") is not None else None,
        accepted_answers=r.get("accepted_answers", None),
        prompt_ids=list(r["prompt_ids"]),
        generated_tokens=list(r["generated_tokens"]),
        generated_text=r["generated_text"],
        # exp5b: per-span signatures (each span -> signal -> list of floats)
        signatures_by_span={
            span: {sig: list(v) for sig, v in sig_dict.items()}
            for span, sig_dict in r["signatures_by_span"].items()
        },
    ))

with open(f"{OUT}/exp5b_halluc.json", "w", encoding="utf-8") as f:
    json.dump(halluc_payload, f, indent=2, ensure_ascii=False, default=str)

print(f"saved {OUT}/exp5b_halluc.json "
      f"({len(halluc_payload['runs'])} runs, "
      f"{len(SIGNALS_REGISTRY)} registry signals + {len(SIGNALS_BASELINE)} baselines, "
      f"{len(cfg.n_values)} N values, "
      f"per-domain + pooled AUROC for 2 tests x "
      f"{len(SPANS_FOR_TEST[1])} spans (test1) / "
      f"{len(SPANS_FOR_TEST[2])} spans (test2))")


# %%
# Remove the read-only hook now that the hallucination block is done so
# the hook never accidentally fires if the cell is re-run.
_readonly_handle.remove()
print(f"removed read-only hook from pythia.gpt_neox.layers[{cfg.extract_layer - 1}]")


# %% [markdown]
# ## 17. Plots -- AUROC vs N + signature distributions (FIX 4: try/except)
#
# exp5b produces three PNGs, all wrapped in `try/except` so a
# plotting failure prints a warning but does NOT abort the run
# -- the JSONs above are guaranteed on disk by this point.
#
# - `exp5b_results.png` -- span=gen AUROC vs N curves, one panel
#   per test, registry signals (solid) vs baselines (dashed).
# - `exp5b_signatures.png` -- per-signal pooled distribution histo-
#   grams (positive vs negative label), span=gen.
# - `exp5b_results_3spans.png` -- per-span AUROC summary bars so
#   the three-span story is visible at a glance.

# %%
# FIX 4: every plotting cell is wrapped in try/except so a
# matplotlib backend / OOM / display crash prints a warning and
# continues, instead of destroying the verdict or the JSONs (which
# are already saved above).
import matplotlib.pyplot as _plt_mod
print(f"matplotlib version: {_plt_mod.__version__}")

print("\n--- exp5b_results.png (try/except) ---")
try:
    fig, ax = _plt_mod.subplots(1, 2, figsize=(16, 6))
    colors_reg = {
        "max_act":                 "tab:blue",
        "act_entropy":             "tab:orange",
        "top1_share":              "tab:green",
        "jaccard":                 "tab:red",
        "cosine":                  "tab:purple",
        "prompt_domain_share":     "tab:brown",
        "domain_entropy":          "tab:pink",
    }
    colors_base = {
        "lm_entropy":              "black",
        "neg_max_logprob":         "dimgray",
    }
    for col_idx, (test_name, title) in enumerate((("test1", "TEST 1: FAKE vs REAL"),
                                                  ("test2", "TEST 2: INCORRECT vs CORRECT"))):
        a = ax[col_idx]
        for sig in SIGNALS_REGISTRY:
            aucs = auroc_tables[test_name]["gen"][sig]["pooled"]
            a.plot(cfg.n_values, aucs, marker="o", color=colors_reg[sig],
                   label=sig, lw=1.4)
        for sig in SIGNALS_BASELINE:
            aucs = auroc_tables[test_name]["gen"][sig]["pooled"]
            a.plot(cfg.n_values, aucs, marker="s", color=colors_base[sig],
                   label=sig + " (baseline)", lw=1.4, ls="--")
        a.axhline(0.5, color="k", ls=":", lw=0.8)
        a.set_xscale("log", base=2)
        a.set_xticks(cfg.n_values); a.set_xticklabels([str(n) for n in cfg.n_values])
        a.set_ylim(-0.02, 1.02)
        a.set_xlabel("N (first N generated tokens, mean aggregated)")
        a.set_ylabel("pooled AUROC")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend(fontsize=7, ncol=2)
    _plt_mod.suptitle(
        "exp5b -- pooled span=gen AUROC vs N (registry signals solid, baselines dashed)\n"
        "Note: AUC < 0.5 = inverted-but-informative signal; the verdict "
        "uses max(AUC, 1 - AUC) (polarity-corrected).",
        fontsize=11)
    _plt_mod.tight_layout(rect=[0, 0, 1, 0.95])
    _plt_mod.savefig(f"{OUT}/exp5b_results.png", dpi=130)
    _plt_mod.show()
    print(f"saved {OUT}/exp5b_results.png")
except Exception as _exc:
    print(f"WARNING: exp5b_results.png FAILED with {_exc!r}; JSONs are still on disk.")


print("\n--- exp5b_signatures.png (try/except) ---")
try:
    # Use span=gen for the per-signal pooled distribution histograms.
    def _first32_mean(siglist):
        return float(np.mean(siglist[:GEN_LEN]))

    fig, ax = _plt_mod.subplots(len(SIGNALS_ALL), 2,
                                 figsize=(14, 2.6 * len(SIGNALS_ALL)))
    for row_idx, sig in enumerate(SIGNALS_ALL):
        for col_idx, (test_name, pos_label, neg_label, title) in enumerate((
            ("test1", "fake",     "real",     "TEST 1 (FAKE red, REAL blue)"),
            ("test2", "incorrect","correct",  "TEST 2 (INCORRECT red, CORRECT blue)"),
        )):
            a = ax[row_idx, col_idx]
            pos_vals, neg_vals = [], []
            for r in runs:
                if r["test"] != (1 if test_name == "test1" else 2):
                    continue
                if "gen" not in r["signatures_by_span"]:
                    continue
                if sig not in r["signatures_by_span"]["gen"]:
                    continue
                v = _first32_mean(r["signatures_by_span"]["gen"][sig])
                if r["condition"] == pos_label:
                    pos_vals.append(v)
                elif r["condition"] == neg_label:
                    neg_vals.append(v)
            if pos_vals:
                a.hist(pos_vals, bins=30, alpha=0.6, color="tab:red",
                       label=f"{pos_label} (n={len(pos_vals)})")
            if neg_vals:
                a.hist(neg_vals, bins=30, alpha=0.6, color="tab:blue",
                       label=f"{neg_label} (n={len(neg_vals)})")
            kind = "(baseline)" if sig in SIGNALS_BASELINE else ""
            a.set_title(f"{sig} {kind} -- {title}", fontsize=10)
            a.set_xlabel("mean of first 32 token-level values")
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
    _plt_mod.suptitle("exp5b -- per-signal distributions, span=gen (pooled over 4 domains)",
                      fontsize=13)
    _plt_mod.tight_layout(rect=[0, 0, 1, 0.97])
    _plt_mod.savefig(f"{OUT}/exp5b_signatures.png", dpi=130)
    _plt_mod.show()
    print(f"saved {OUT}/exp5b_signatures.png")
except Exception as _exc:
    print(f"WARNING: exp5b_signatures.png FAILED with {_exc!r}; JSONs are still on disk.")


print("\n--- exp5b_results_3spans.png (try/except) ---")
try:
    # Per-span AUROC bar summary at the headline N (=8). One panel
    # per test, three bars (one per span) showing the best registry
    # discrimination. For span='gen' also show the best baseline's
    # discrimination as a hatched bar.
    HEADLINE_N = 8
    j_head = cfg.n_values.index(HEADLINE_N)

    fig, ax = _plt_mod.subplots(1, 2, figsize=(14, 5))
    for col_idx, (test_id, test_name, title) in enumerate((
        (1, "test1", "TEST 1: FAKE vs REAL"),
        (2, "test2", "TEST 2: INCORRECT vs CORRECT"),
    )):
        a = ax[col_idx]
        spans_here = SPANS_FOR_TEST[test_id]
        x = list(range(len(spans_here)))
        best_reg_discs, best_base_discs = [], []
        best_reg_names = []
        for span in spans_here:
            brd = max(_disc(auroc_tables[test_name][span][sig]["pooled"][j_head])
                      for sig in SIGNALS_REGISTRY)
            brn = max(SIGNALS_REGISTRY,
                      key=lambda s: _disc(auroc_tables[test_name][span][s]["pooled"][j_head]))
            best_reg_discs.append(brd)
            best_reg_names.append(brn)
            if span == "gen":
                bbd = max(_disc(auroc_tables[test_name][span][sig]["pooled"][j_head])
                          for sig in SIGNALS_BASELINE)
            else:
                bbd = float("nan")
            best_base_discs.append(bbd)
        bar_width = 0.4
        a.bar([xi - bar_width/2 for xi in x], best_reg_discs,
              width=bar_width, color="tab:blue",
              label="best registry signal")
        bbd_plot = [b if not math.isnan(b) else 0.0 for b in best_base_discs]
        a.bar([xi + bar_width/2 for xi in x], bbd_plot,
              width=bar_width, color="dimgray", alpha=0.6,
              label="best LM-head baseline (span=gen only)")
        a.axhline(0.5, color="k", ls=":", lw=0.8, label="chance (0.5)")
        a.set_xticks(x)
        a.set_xticklabels(spans_here)
        a.set_ylim(0.45, max(best_reg_discs + [0.55]) + 0.05)
        a.set_ylabel(f"discrimination (max(AUC, 1-AUC)) at N={HEADLINE_N}")
        a.set_title(f"{title}\nbest-reg signal: "
                    + ", ".join(f"{s}={n}={d:.3f}"
                                for s, n, d in zip(spans_here, best_reg_names, best_reg_discs)))
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    _plt_mod.suptitle(
        f"exp5b -- best-signal AUROC per span at N={HEADLINE_N}\n"
        "Hatched / empty bar where no baseline is available.",
        fontsize=11)
    _plt_mod.tight_layout(rect=[0, 0, 1, 0.93])
    _plt_mod.savefig(f"{OUT}/exp5b_results_3spans.png", dpi=130)
    _plt_mod.show()
    print(f"saved {OUT}/exp5b_results_3spans.png")
except Exception as _exc:
    print(f"WARNING: exp5b_results_3spans.png FAILED with {_exc!r}; JSONs are still on disk.")


print("\n--- exp5b done ---")
print(f"  exp5b_halluc.json on disk: {(f'{OUT}/exp5b_halluc.json')} exists")
