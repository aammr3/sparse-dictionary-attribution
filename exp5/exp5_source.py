# %% [markdown]
# # Experiment 5 -- hallucination detection from the registry's internal state
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
# **This experiment (exp5) drops steering entirely and only observes.**
# The forward hook on Pythia's layer-12 block
# (`gpt_neox.layers[11]`) is now **READ-ONLY**: it records, per
# generated position, the registry's `pre` and top-k `acts` WITHOUT
# modifying the layer output. `steer_bias` stays all-zeros for the
# entire run -- exp5 is pure observation of the trained registry's
# internal state at every generation step.
#
# **Two labelled contrasts, three internal signatures, two LM baselines.**
#
# 1. **TEST 1 -- grounded vs ungrounded.** For each domain we embed a
#    list of 30 real, well-known entities (REAL) and 30 invented,
#    plausible-looking entities (FAKE). A prompt template substitutes
#    the entity and Pythia generates 32 new tokens. Label = REAL or
#    FAKE (FAKE = guaranteed-ungrounded; any confident continuation is
#    hallucination).
#
# 2. **TEST 2 -- correct vs incorrect.** 30 cloze facts per domain,
#    each a `(prompt, [accepted answer strings])` pair. Label per
#    generated sample = CORRECT if any accepted string appears
#    (case-insensitive substring) in the decoded continuation, else
#    INCORRECT.
#
# From the registry's recorded `pre` and `acts` at every generated
# position we compute three internal signatures per step:
#
# - **A. concentration / "improvising":** `max_act`, `act_entropy`
#   (entropy over the k firing slots, normalised), `top1_share` =
#   `max_act / sum(acts)`.
# - **B. domain drift:** Reference = registry state at the last prompt
#   token. Per generated token: `jaccard` of top-k slot ids vs
#   reference, `cosine` of the k-hot activation vectors, and
#   `prompt_domain_share` = fraction of activation mass on the
#   reference-domain slots (reference domain = argmax domain-mass of
#   the prompt).
# - **C. conflict:** `domain_entropy` = entropy of the
#   activation-weighted distribution over the 4 slot domains (using
#   `dom_all`).
#
# Plus two **LM-baselines** computed from Pythia's own next-token
# softmax at each step: `lm_entropy` (full softmax entropy, in nats)
# and `neg_max_logprob` (negative log of the softmax-max). These come
# from the LM head, not the registry -- the registry's value-add is
# the gap between its best signature AUROC and the best baseline
# AUROC.
#
# **Evaluation.** For each signature we aggregate over only the FIRST
# N generated tokens (mean over those tokens) for N in
# `[1, 2, 4, 8, 16, 32]`. Then we compute **rank-based AUROC**
# directly (no sklearn dependency assumption) for:
#
# - (a) TEST 1: FAKE (positive) vs REAL (negative).
# - (b) TEST 2: INCORRECT (positive) vs CORRECT (negative).
#
# We report per-domain AUROC and pooled AUROC (over all 4 domains).
#
# **Verdict block.** For each test, at each N, the best registry
# signal's pooled AUROC vs the best baseline's pooled AUROC at the
# same N, with the direction-of-comparison chosen by
# `max(AUC, 1-AUC)` so a signal that anti-correlates still counts as
# a discriminator. If no registry signal beats both baselines at any
# N, we print the honest negative result.
#
# **Outputs**
# - `exp5_model.pt` -- registry state_dict (NOT Pythia).
# - `exp5_summary.json` -- training/purity summary (same schema as
#   exp4c).
# - `exp5_halluc.json` -- every run's per-token signals + labels +
#   the AUROC tables + the tokenization-confound numbers.
# - `exp5_results.png` -- AUROC vs N curves, one panel per test, one
#   line per signal.
# - `exp5_signatures.png` -- signature distributions, pooled real vs
#   fake and correct vs incorrect, one row per signal.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings -> Accelerator -> GPU T4 x2** (or P100)
# 2. **Settings -> Internet -> On**  <- required to download Pythia + data
#
# Expected runtime: ~50-60 minutes total. Training is identical to
# exp3b / exp4c (~19 min). The hallucination block is 360 prompts
# (TEST 1: 4 * 2 * 30 = 240; TEST 2: 4 * 30 = 120) x 3 repeats x 32
# tokens = 1080 generations x 32 tokens -- about 1.5x exp4c's 144-run
# steering sweep, so the additional cost vs exp4c is roughly +10-15
# minutes on a T4.

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
torch.save(registry.state_dict(), f"{OUT}/exp5_model.pt")
print(f"saved registry state_dict to {OUT}/exp5_model.pt "
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
plt.savefig(f"{OUT}/exp5_purity.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp5_purity.png")


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
with open(f"{OUT}/exp5_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
print(f"\nsaved {OUT}/exp5_summary.json")


# %% [markdown]
# ## 9. Read-only forward hook -- records `pre` and top-k `acts` per step
#
# exp5 replaces exp4c's modifying forward hook with a **read-only**
# one. The hook is registered on the same layer
# (`pythia.gpt_neox.layers[cfg.extract_layer - 1]`, the block whose
# output is `hidden_states[cfg.extract_layer]`). On every Pythia
# forward call the hook does exactly three things:
#
# 1. Extracts the LAST position of the layer's output
#   (`h[:, -1:, :]`), which is the position whose hidden state
#   produces the next-token logits in the autoregressive generation
#   loop.
# 2. Computes the registry's `pre = relu(encoder(h - b_dec)) +
#    steer_bias` (with `steer_bias == 0` for the entire run) and
#    top-k `acts = topk(pre, k)` at that last position. Both are
#    detached and moved to CPU for storage.
# 3. **Returns the original layer output unchanged.** No steering
#    patch, no delta, no modification of any kind.
#
# The hook carries a `recorded` list of dicts (one per step). Each
# dict has `{"pre": [n_slots], "acts": [n_slots]}` -- both as
# dense vectors (after top-k, `pre` is zero outside the top-k slots
# and identical to `acts` at the top-k slots, so storing both is
# near-zero overhead). The list is reset by `reset()` between runs.
#
# Both the tuple-form and bare-tensor layer output are handled for
# transformers-version compatibility (Kaggle ships transformers 5.0,
# which returns a bare tensor from `GPTNeoXLayer.forward`).

# %%
class _ReadOnlyHook:
    """Forward hook installed on Pythia's layer-12 block. Records the
    registry's `pre` and top-k `acts` at the LAST position of the
    layer's output (the position whose hidden state produces the
    next-token logits in the generation loop) WITHOUT modifying the
    output. `steer_bias` stays all-zeros for the whole run; the
    `pre + steer_bias` line is kept so the bit-pattern matches what
    `registry.forward()` would compute on the same input."""

    def __init__(self, registry):
        self.registry = registry
        self.recorded = []   # list of {"pre": [n_slots], "acts": [n_slots]}

    def reset(self):
        self.recorded = []

    def __call__(self, module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output          # [B, T, 1024]
        # Compute registry pre + top-k at the LAST position only
        last_h = h[:, -1:, :]                          # [B, 1, 1024]
        reg = self.registry
        pre = F.relu(reg.encoder(last_h - reg.b_dec))  # [B, 1, n_slots]
        pre = pre + reg.steer_bias                     # == pre, steer_bias == 0
        acts = reg._topk(pre, reg.k)                   # [B, 1, n_slots]
        self.recorded.append({
            "pre":  pre.detach().to("cpu", non_blocking=True).squeeze(0).squeeze(0),
            "acts": acts.detach().to("cpu", non_blocking=True).squeeze(0).squeeze(0),
        })
        # READ-ONLY: return the original layer output unchanged.
        return output


_readonly_hook = _ReadOnlyHook(registry)
_readonly_layer = pythia.gpt_neox.layers[cfg.extract_layer - 1]
_readonly_handle = _readonly_layer.register_forward_hook(_readonly_hook)
print(f"registered READ-ONLY hook on pythia.gpt_neox.layers[{cfg.extract_layer - 1}] "
      f"(output is hidden_states[{cfg.extract_layer}]); "
      f"handle={_readonly_handle}")
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must still be all-zeros when the read-only hook is installed"


# %% [markdown]
# ## 10. Sampling generation harness (identical to exp4c)
#
# All generation uses nucleus sampling (`temperature=0.8, top_p=0.9`),
# never greedy / argmax -- project rule. Returns the generated tokens
# AND the per-step LM-head quantities we need for the two baselines
# (`lm_entropy` and `neg_max_logprob`), computed BEFORE sampling so
# the metrics reflect Pythia's belief at the step, not the kept
# subset.
#
# The hook fires on every pythia() call inside the loop, recording
# one entry per step into `_readonly_hook.recorded`. Step t (0-indexed)
# records the registry state at the LAST position of the input at
# that step -- which is the position that produced the (t+1)-th
# generated token (for t = 0, the last position is the last prompt
# token). At the end of the call the recorded list has exactly
# `n_new` entries; the first is the prompt reference (used by
# Signature B for domain drift), the next `n_new - 1` are the per-
# generated-token registry states.

# %%
GEN_LEN             = cfg.gen_len       # 32
GEN_TEMPERATURE     = cfg.gen_temp      # 0.8
GEN_TOP_P           = cfg.gen_top_p     # 0.9
REPEATS             = cfg.repeats       # 3
SEED_BASE           = cfg.seed_base     # 2000
EOT = tokenizer.eos_token_id
if EOT is None:
    EOT = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

print(f"\nGeneration config (identical to exp4c sampling rules):")
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
    recorded_states)`. The read-only hook MUST be installed on the
    layer-12 block; it appends one `{pre, acts}` per step to its
    `recorded` list as a side effect. We `reset()` it on entry so a
    previous run's recordings cannot leak into this one."""
    pythia.eval()
    registry.eval()
    pad_id = EOT if EOT is not None else 0
    ids = [int(t) for t in prompt_ids]
    if len(ids) < cfg.ctx:
        ids = [pad_id] * (cfg.ctx - len(ids)) + ids
    elif len(ids) > cfg.ctx:
        ids = ids[-cfg.ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    generated, lm_entropies, neg_max_logprobs = [], [], []
    if hook is not None:
        hook.reset()
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            o = pythia(x)
        # The hook has fired inside pythia(x). Its recorded state for
        # THIS step corresponds to the LAST position of x, which is the
        # position whose hidden state produced the next-token logits
        # we're about to sample from.
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
    return generated, lm_entropies, neg_max_logprobs, recorded_states


# exp5 invariant: steer_bias is still all-zeros here -- nothing in
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
FAKE_ENTITIES = {
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
# ## 11b. Tokenization-confound check (TEST 1)
#
# For TEST 1 the labels are REAL vs FAKE entity names. If FAKE
# names tokenize to a very different length than REAL names, the
# contrast is confounded by name rarity (rare / unusual strings
# also tend to produce unusual BPE lengths, and Pythia will likely
# distribute its next-token mass differently for rare strings
# regardless of whether the registry flags anything). We report
# the mean / std / absolute gap per domain so the reader can
# judge whether TEST 1 is informative in that domain.

# %%
def _entity_token_length(name):
    """BPE token length of an entity name (Pythia tokenizer)."""
    return len(tokenizer.encode(name))

confound_report = {}
print("=" * 78)
print("TOKENIZATION-CONFOUND CHECK (TEST 1 -- REAL vs FAKE entity names)")
print("(if the abs gap is large the whole REAL-vs-FAKE contrast is")
print(" confounded by name rarity -- Pythia distributes mass differently")
print(" over rare strings regardless of any registry signal)")
print("=" * 78)
for d in DOMAIN_NAMES:
    real_lens = [_entity_token_length(n) for n in REAL_ENTITIES[d]]
    fake_lens = [_entity_token_length(n) for n in FAKE_ENTITIES[d]]
    r_mean, r_std = float(np.mean(real_lens)), float(np.std(real_lens))
    f_mean, f_std = float(np.mean(fake_lens)), float(np.std(fake_lens))
    gap = abs(r_mean - f_mean)
    confound_report[d] = dict(
        real_mean=r_mean, real_std=r_std, n_real=len(real_lens),
        fake_mean=f_mean, fake_std=f_std, n_fake=len(fake_lens),
        abs_gap=gap,
        real_lens=real_lens, fake_lens=fake_lens,
    )
    flag = "  <-- LARGE GAP, contrast is confounded" if gap > 3.0 else ""
    print(f"  {d:>10s}  REAL mean={r_mean:5.2f} std={r_std:4.2f} (n={len(real_lens)})   "
          f"FAKE mean={f_mean:5.2f} std={f_std:4.2f} (n={len(fake_lens)})   "
          f"abs_gap={gap:5.2f}{flag}")
print("=" * 78)


# %% [markdown]
# ## 12. Signature computations
#
# Three registry signatures + two LM-head baselines, computed per
# generated token from the hook's recorded state. All signatures
# are computed in tensor form on CPU (the hook already moves
# `pre` / `acts` to CPU) so the generation GPU buffer pressure is
# unaffected.
#
# **Signature A -- concentration / "improvising"**
# - `max_act` = max value among the k firing slots.
# - `act_entropy` = entropy of the k firing slots after normalising
#   by their sum (so it lives in `[0, log k]`, not `[0, log n_slots]`).
# - `top1_share` = `max_act / sum(acts)`. 1.0 = single-slot spike.
#
# **Signature B -- domain drift** (vs the prompt reference)
# The reference is the registry's `acts` at the LAST PROMPT TOKEN
# (i.e. the entry at index 0 of the recorded list, which the hook
# captured when the input to layer 11 was the left-padded prompt).
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


def compute_signatures_for_run(recorded_states):
    """Compute all per-step signatures for one generation run.
    `recorded_states` is a list of `{"pre": [n_slots], "acts":
    [n_slots]}` (length = GEN_LEN). Returns a dict mapping each
    signal name to a list of floats (length = GEN_LEN)."""
    acts_all = torch.stack([r["acts"] for r in recorded_states], dim=0)   # [GEN_LEN, n_slots]
    # Reference = registry state at the LAST PROMPT TOKEN = acts_all[0]
    acts_ref = acts_all[0]
    # Reference domain = argmax over the 4 domain masses of acts_ref
    ref_domain_masses = _domain_masses(acts_ref)                          # [N_DOMAINS]
    ref_domain_idx = int(torch.argmax(ref_domain_masses).item())
    # Reference top-k slot ids (used by jaccard)
    ids_ref = set(torch.nonzero(acts_ref > 0, as_tuple=False).flatten().tolist())
    ref_domain_mask = DOMAIN_MASKS[ref_domain_idx]                        # bool [n_slots]
    # cosine against reference: convert acts vectors to float for stability
    acts_ref_f = acts_ref.float()

    sigs = {name: [] for name in SIGNALS_ALL}
    for t in range(GEN_LEN):
        acts_t = acts_all[t]
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


# exp5 invariant: still no setter ever called
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros before the generation loop"


# %% [markdown]
# ## 13. Main generation loop -- TEST 1 + TEST 2
#
# We collect every run into a flat list `runs` with per-token
# signatures and labels. After the loop we aggregate to per-N means
# and compute AUROC.
#
# `REPEATS = 3` generations per prompt, each with a different seed
# (`torch.manual_seed(SEED_BASE + rep_n)`). Total = 1080 runs
# = (4 * 2 * 30 + 4 * 30) * 3 = 360 prompts * 3.
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
t_halluc = time.time()
n_runs_per_test = {1: 0, 2: 0}

print("=" * 78)
print("exp5 hallucination-detection loop (TEST 1 = real/fake, TEST 2 = cloze)")
print(f"  REPEATS = {REPEATS}, GEN_LEN = {GEN_LEN}, total = 1080 runs")
print("=" * 78)

# ---- TEST 1: grounded vs ungrounded ----
# Per domain: 30 real + 30 fake prompts. Each is repeated REPEATS times.
print("\n[ TEST 1 ] grounded vs ungrounded (REAL vs FAKE entity names)")
for d in DOMAIN_NAMES:
    template = PROMPT_TEMPLATE_TEST1[d]
    for entity_kind, entity_list in (("real", REAL_ENTITIES[d]),
                                     ("fake", FAKE_ENTITIES[d])):
        for entity in entity_list:
            prompt_text = template.format(X=entity)
            prompt_ids = tokenizer.encode(prompt_text)
            for rep_n in range(1, REPEATS + 1):
                torch.manual_seed(SEED_BASE + rep_n)
                toks, lm_ents, neg_mlps, rec = generate_with_signatures(
                    prompt_ids, n_new=GEN_LEN, hook=_readonly_hook)
                decoded = tokenizer.decode(toks)
                sigs = compute_signatures_for_run(rec)
                sigs["lm_entropy"] = list(lm_ents)
                sigs["neg_max_logprob"] = list(neg_mlps)
                runs.append(dict(
                    test=1, domain=d, condition=entity_kind,
                    rep_n=rep_n, seed=SEED_BASE + rep_n,
                    prompt_text=prompt_text, entity=entity,
                    prompt_ids=[int(t) for t in prompt_ids],
                    generated_tokens=[int(t) for t in toks],
                    generated_text=decoded,
                    # Per-step signatures (list of 32 floats each)
                    signatures={k: list(v) for k, v in sigs.items()},
                ))
                n_runs_per_test[1] += 1

# ---- TEST 2: correct vs incorrect (cloze) ----
# Per domain: 30 cloze prompts. Repeat REPEATS times.
# Label is computed from the decoded continuation.
print("\n[ TEST 2 ] correct vs incorrect (cloze facts, label from decoded continuation)")
for d in DOMAIN_NAMES:
    for prompt_text, accepted in CLOZE_FACTS[d]:
        prompt_ids = tokenizer.encode(prompt_text)
        for rep_n in range(1, REPEATS + 1):
            torch.manual_seed(SEED_BASE + rep_n)
            toks, lm_ents, neg_mlps, rec = generate_with_signatures(
                prompt_ids, n_new=GEN_LEN, hook=_readonly_hook)
            decoded = tokenizer.decode(toks)
            label_correct = _is_correct(toks, accepted)
            sigs = compute_signatures_for_run(rec)
            sigs["lm_entropy"] = list(lm_ents)
            sigs["neg_max_logprob"] = list(neg_mlps)
            runs.append(dict(
                test=2, domain=d,
                condition=("correct" if label_correct else "incorrect"),
                rep_n=rep_n, seed=SEED_BASE + rep_n,
                prompt_text=prompt_text,
                accepted_answers=list(accepted),
                prompt_ids=[int(t) for t in prompt_ids],
                generated_tokens=[int(t) for t in toks],
                generated_text=decoded,
                signatures={k: list(v) for k, v in sigs.items()},
            ))
            n_runs_per_test[2] += 1

elapsed = time.time() - t_halluc
print(f"\nhallucination block done in {elapsed/60:.1f} minutes "
      f"({len(runs)} runs, TEST 1 = {n_runs_per_test[1]}, "
      f"TEST 2 = {n_runs_per_test[2]})")

# exp5 invariant: nothing in this block sets steer_bias non-zero
assert registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must remain all-zeros after the hallucination block"


# %% [markdown]
# ## 14. AUROC -- rank-based, no sklearn dependency
#
# Per spec: "Implement AUROC directly (rank-based, no sklearn
# dependency assumption -- but sklearn is fine if already available;
# if you use it, guard the import)." We implement rank-based AUROC
# from scratch. Ties in scores get the average rank (standard).
#
# Per-N aggregation: for each run, take the mean of the first N
# token-level values of each signal, then compute AUROC across runs.
# Per-domain = filter to one domain; pooled = concat all 4 domains.

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


# Aggregate per-N signal means for all runs in a given test
def _aggregate_per_n(runs_test, signal_name, n_values):
    """For each run in `runs_test` and each N in `n_values`, compute
    the mean of the first N values of `signal_name`. Returns:
      - pooled_means: shape [n_runs, len(n_values)] (pooled across domains)
      - per_domain_means: dict {domain: shape [n_runs_in_d, len(n_values)]}
      - pooled_labels: shape [n_runs]
      - per_domain_labels: dict {domain: shape [n_runs_in_d]}
    """
    pooled_means, pooled_labels = [], []
    per_domain_means = {d: [] for d in DOMAIN_NAMES}
    per_domain_labels = {d: [] for d in DOMAIN_NAMES}
    for r in runs_test:
        sig = r["signatures"][signal_name]
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


def compute_aurocs_for_test(runs_test, signal_name, n_values):
    """Compute AUROC tables for one signal in one test.
    Returns a dict with keys 'pooled' and 'per_domain'; each maps
    n_values -> AUROC float."""
    pm, pl, pdm, pdl = _aggregate_per_n(runs_test, signal_name, n_values)
    out = {"pooled": [], "per_domain": {d: [] for d in DOMAIN_NAMES}}
    for j in range(len(n_values)):
        out["pooled"].append(auroc_rank(pm[:, j], pl))
        for d in DOMAIN_NAMES:
            out["per_domain"][d].append(auroc_rank(pdm[d][:, j], pdl[d]))
    return out


runs_test1 = [r for r in runs if r["test"] == 1]
runs_test2 = [r for r in runs if r["test"] == 2]

auroc_tables = {}
for test_name, runs_test in (("test1", runs_test1), ("test2", runs_test2)):
    auroc_tables[test_name] = {}
    for sig in SIGNALS_ALL:
        auroc_tables[test_name][sig] = compute_aurocs_for_test(
            runs_test, sig, cfg.n_values)


# %% [markdown]
# ## 15. AUROC tables (pooled + per-domain)
#
# Two blocks, one per test. Each block prints two tables:
#
# - **Pooled AUROC** -- rows = signals, columns = N in `cfg.n_values`.
#   Positive class = fake (TEST 1) or incorrect (TEST 2). Values are
#   the raw AUROC; `0.5 = no discrimination`, `1.0 = perfect`, `0.0 =
#   anti-perfect`.
# - **Per-domain AUROC** -- one table per domain, same shape.
#
# The two registry baselines (`lm_entropy`, `neg_max_logprob`) are
# printed at the bottom of each block in a slightly different shade
# so it's easy to see whether any registry signal crosses the
# baseline bar at any N.

# FIX 6 (review T-056-review MEDIUM A): the verdict block below must
# actually consult the per-domain tokenization-confound numbers
# computed in section 11b, instead of just letting them sit in the
# JSON. Threshold chosen: CONFOUND_ABS_GAP_THRESHOLD = 1.0 BPE
# tokens. Domains with |real_mean - fake_mean| > 1.0 token are
# flagged as TEST-1-CONFOUNDED and printed prominently next to
# their AUROC (the contrast in that domain may reflect name rarity
# rather than any grounding signal).
CONFOUND_ABS_GAP_THRESHOLD = 1.0


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

for test_name, title_short in (("test1", "TEST 1 -- FAKE vs REAL (positive = FAKE)"),
                               ("test2", "TEST 2 -- INCORRECT vs CORRECT (positive = INCORRECT)")):
    _print_auroc_table(
        title=f"POOLED AUROC, {title_short}",
        auroc_dict=auroc_tables[test_name],
        signals=SIGNALS_ALL,
        n_values=cfg.n_values,
        registry_signals=set(SIGNALS_REGISTRY),
        baseline_signals=set(SIGNALS_BASELINE),
        subtitle=POLARITY_NOTE,
    )
    for d in DOMAIN_NAMES:
        # FIX 6: per-domain confound warning (TEST 1 only) -- if
        # abs(real_mean - fake_mean) > threshold, the contrast for
        # this domain may reflect name rarity rather than any
        # grounding signal. Flag it next to the AUROC, do NOT drop
        # the domain silently.
        conf_subtitle = POLARITY_NOTE
        if test_name == "test1" and d in confound_report:
            gap = confound_report[d].get("abs_gap", 0.0)
            if gap > CONFOUND_ABS_GAP_THRESHOLD:
                conf_subtitle = (
                    POLARITY_NOTE
                    + "\n"
                    + f"*** TEST 1 CONFOUNDED for {d}: real/fake "
                      f"token-length gap = {gap:.2f} (> "
                      f"{CONFOUND_ABS_GAP_THRESHOLD}) -- AUROC for "
                      f"this domain may reflect name rarity, not "
                      f"grounding ***"
                )
        _print_auroc_table(
            title=f"PER-DOMAIN AUROC, {d}, {title_short}",
            auroc_dict={sig: dict(pooled=auroc_tables[test_name][sig]["per_domain"][d],
                                  per_domain={})
                        for sig in SIGNALS_ALL},
            signals=SIGNALS_ALL,
            n_values=cfg.n_values,
            registry_signals=set(SIGNALS_REGISTRY),
            baseline_signals=set(SIGNALS_BASELINE),
            subtitle=conf_subtitle,
        )


# %% [markdown]
# ## 16. Verdict block -- does any registry signal beat the baseline?
#
# For each test, at each N, we compare the registry's best signal
# (max AUROC discrimination = `max(AUROC, 1 - AUROC)`) against the
# best baseline's discrimination at the same N. `max(AUROC, 1-AUROC)`
# is the "discrimination strength" -- it ignores whether the signal
# is positively or negatively correlated with the hallucination
# label, so a signal that's anti-correlated still counts as a
# discriminator.
#
# The honest negative result is printed plainly if no registry
# signal beats both baselines at any N.

# %%
def _disc(auc):
    """Discrimination strength: max(AUC, 1 - AUC) in [0.5, 1.0]."""
    if math.isnan(auc):
        return float("nan")
    return max(auc, 1.0 - auc)


def _verdict_for_test(test_name):
    """Return a per-N verdict table + a single-line overall verdict.
    Per-N columns: best registry signal, its AUC, its discrimination,
    best baseline signal, its AUC, its discrimination, gap
    (registry - baseline)."""
    rows = []
    overall_best_gap = -1.0
    overall_best_N = None
    for j, N in enumerate(cfg.n_values):
        best_reg_name, best_reg_auc, best_reg_disc = None, -1.0, -1.0
        for sig in SIGNALS_REGISTRY:
            auc = auroc_tables[test_name][sig]["pooled"][j]
            d = _disc(auc)
            if d > best_reg_disc:
                best_reg_disc = d
                best_reg_auc = auc
                best_reg_name = sig
        best_base_name, best_base_auc, best_base_disc = None, -1.0, -1.0
        for sig in SIGNALS_BASELINE:
            auc = auroc_tables[test_name][sig]["pooled"][j]
            d = _disc(auc)
            if d > best_base_disc:
                best_base_disc = d
                best_base_auc = auc
                best_base_name = sig
        gap = best_reg_disc - best_base_disc
        if gap > overall_best_gap:
            overall_best_gap = gap
            overall_best_N = N
        rows.append(dict(
            N=N,
            best_registry=best_reg_name,
            best_registry_auc=best_reg_auc,
            best_registry_disc=best_reg_disc,
            best_baseline=best_base_name,
            best_baseline_auc=best_base_auc,
            best_baseline_disc=best_base_disc,
            gap=gap,
        ))
    return rows, overall_best_gap, overall_best_N


print("=" * 78)
print("EXP5 VERDICT -- does the registry beat the LM-head baseline?")
print("(discrimination = max(AUC, 1 - AUC); higher is better)")
print("=" * 78)

verdict_results = {}
for test_name, title in (("test1", "TEST 1: FAKE vs REAL"),
                         ("test2", "TEST 2: INCORRECT vs CORRECT")):
    rows, best_gap, best_N = _verdict_for_test(test_name)
    verdict_results[test_name] = dict(
        rows=[dict(
            N=int(r["N"]),
            best_registry=r["best_registry"],
            best_registry_auc=float(r["best_registry_auc"]),
            best_registry_disc=float(r["best_registry_disc"]),
            best_baseline=r["best_baseline"],
            best_baseline_auc=float(r["best_baseline_auc"]),
            best_baseline_disc=float(r["best_baseline_disc"]),
            gap=float(r["gap"]),
        ) for r in rows],
        best_gap=float(best_gap),
        best_N=int(best_N) if best_N is not None else None,
        # FIX 6: per-domain TEST 1 confound summary (TEST 2 has no
        # confound report -- only TEST 1 is confounded by name
        # rarity). Attached to verdict_results[test_name] under the
        # "per_domain" key.
        per_domain=(per_domain_verdict if test_name == "test1" else {}),
    )
    print(f"\n  {title}:")
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
        print(f"\n  VERDICT ({title}): the registry's best signal "
              f"beats the best LM-head baseline by "
              f"{best_gap:+.3f} discrimination at N={best_N}.")
    else:
        print(f"\n  VERDICT ({title}): HONEST NEGATIVE -- no registry "
              f"signal beats the best baseline at any N. Best gap = "
              f"{best_gap:+.3f} (at N={best_N}).")

# FIX 6: per-domain verdict for TEST 1 that consults the
# tokenization-confound numbers. The pooled verdict above treats
# all 4 domains equally; if one domain's REAL/FAKE names tokenize
# to very different lengths, its AUROC may just reflect name
# rarity. Flag such domains here; do NOT drop them silently.
print("\n" + "-" * 78)
print("PER-DOMAIN VERDICT (TEST 1) -- confound-aware")
print(f"  (CONFOUND_ABS_GAP_THRESHOLD = {CONFOUND_ABS_GAP_THRESHOLD} "
      f"BPE tokens; domains above are flagged)")
print("-" * 78)
print(f"  {'domain':>10s} | {'abs_gap':>7s} | {'confounded':>10s} | "
      f"{'best reg (N=8)':>18s} | {'best reg disc':>14s} | "
      f"{'best base (N=8)':>18s} | {'best base disc':>15s} | "
      f"{'gap':>6s}")
print("  " + "-" * 130)
# Pick N=8 as the headline per-domain N (middle of the sweep)
HEADLINE_N = 8
j_head = cfg.n_values.index(HEADLINE_N)
per_domain_verdict = {}
for d in DOMAIN_NAMES:
    gap = confound_report.get(d, {}).get("abs_gap", 0.0)
    confounded = bool(gap > CONFOUND_ABS_GAP_THRESHOLD)
    # Best registry signal at N=8 for this domain
    br_name, br_auc, br_disc = None, float("nan"), float("nan")
    for sig in SIGNALS_REGISTRY:
        auc = auroc_tables["test1"][sig]["per_domain"][d][j_head]
        disc = _disc(auc)
        if disc > br_disc:
            br_disc = disc; br_auc = auc; br_name = sig
    bb_name, bb_auc, bb_disc = None, float("nan"), float("nan")
    for sig in SIGNALS_BASELINE:
        auc = auroc_tables["test1"][sig]["per_domain"][d][j_head]
        disc = _disc(auc)
        if disc > bb_disc:
            bb_disc = disc; bb_auc = auc; bb_name = sig
    domain_gap = br_disc - bb_disc
    flag_mark = "***YES***" if confounded else "no"
    print(f"  {d:>10s} | {gap:>7.2f} | {flag_mark:>10s} | "
          f"{br_name:>18s} | {br_disc:>14.3f} | "
          f"{bb_name:>18s} | {bb_disc:>15.3f} | "
          f"{domain_gap:>+6.3f}")
    if confounded:
        print(f"  *** TEST 1 CONFOUNDED for {d}: real/fake "
              f"token-length gap = {gap:.2f} -- AUROC for this "
              f"domain may reflect name rarity, not grounding ***")
    per_domain_verdict[d] = dict(
        abs_gap=float(gap),
        confounded=confounded,
        best_registry=br_name,
        best_registry_auc=float(br_auc),
        best_registry_disc=float(br_disc),
        best_baseline=bb_name,
        best_baseline_auc=float(bb_auc),
        best_baseline_disc=float(bb_disc),
        gap=float(domain_gap),
    )
print("=" * 78)


# %% [markdown]
# ## 17. Plots -- AUROC vs N + signature distributions

# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
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
        aucs = auroc_tables[test_name][sig]["pooled"]
        a.plot(cfg.n_values, aucs, marker="o", color=colors_reg[sig],
               label=sig, lw=1.4)
    for sig in SIGNALS_BASELINE:
        aucs = auroc_tables[test_name][sig]["pooled"]
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
plt.suptitle(
    "exp5 -- pooled AUROC vs N (registry signals solid, baselines dashed)\n"
    "Note: AUC < 0.5 = inverted-but-informative signal; the verdict "
    "uses max(AUC, 1 - AUC) (polarity-corrected).",
    fontsize=11)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{OUT}/exp5_results.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp5_results.png")


# %%
# exp5_signatures.png: per-signal distribution plot.
# Rows = signals, columns = tests. Each panel shows the pooled
# (over all domains) histogram of the mean-of-first-32 value of the
# signal, split by label.
def _first32_mean(siglist):
    return float(np.mean(siglist[:GEN_LEN]))

fig, ax = plt.subplots(len(SIGNALS_ALL), 2, figsize=(14, 2.6 * len(SIGNALS_ALL)))
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
            v = _first32_mean(r["signatures"][sig])
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
plt.suptitle("exp5 -- per-signal distributions (pooled over 4 domains)",
             fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(f"{OUT}/exp5_signatures.png", dpi=130)
plt.show()
print(f"saved {OUT}/exp5_signatures.png")


# %% [markdown]
# ## 18. JSON output
#
# `exp5_halluc.json` carries every individual run's per-token
# signals + labels + the AUROC tables + the tokenization-confound
# numbers + the verdict block + cfg. The summary JSON written
# earlier already carries training/purity.

# %%
halluc_payload = dict(
    cfg={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
    confound=confound_report,
    # FIX 6: explicit confound-flag summary at top level for easy
    # programmatic lookup. Maps domain -> bool "abs gap exceeded
    # CONFOUND_ABS_GAP_THRESHOLD?".
    confound_threshold=CONFOUND_ABS_GAP_THRESHOLD,
    confound_flags={d: bool(confound_report.get(d, {}).get("abs_gap", 0.0)
                            > CONFOUND_ABS_GAP_THRESHOLD)
                    for d in DOMAIN_NAMES},
    verdict=verdict_results,
    auroc_tables={
        tn: {
            sig: dict(
                pooled=[float(x) for x in auroc_tables[tn][sig]["pooled"]],
                per_domain={d: [float(x) for x in auroc_tables[tn][sig]["per_domain"][d]]
                            for d in DOMAIN_NAMES},
            )
            for sig in SIGNALS_ALL
        } for tn in ("test1", "test2")
    },
    n_values=list(cfg.n_values),
    signals_registry=SIGNALS_REGISTRY,
    signals_baseline=SIGNALS_BASELINE,
    runs=[],
)

# Per-run payloads (per-token signals + labels only -- no pre /
# acts tensors, those live in the hook's CPU list and are
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
        accepted_answers=r.get("accepted_answers", None),
        prompt_ids=list(r["prompt_ids"]),
        generated_tokens=list(r["generated_tokens"]),
        generated_text=r["generated_text"],
        signatures={k: list(v) for k, v in r["signatures"].items()},
    ))

with open(f"{OUT}/exp5_halluc.json", "w", encoding="utf-8") as f:
    json.dump(halluc_payload, f, indent=2, ensure_ascii=False, default=str)

print(f"saved {OUT}/exp5_halluc.json "
      f"({len(halluc_payload['runs'])} runs, "
      f"{len(SIGNALS_ALL)} signals, {len(cfg.n_values)} N values, "
      f"per-domain + pooled AUROC for 2 tests)")


# %%
# Remove the read-only hook now that the hallucination block is done so
# the hook never accidentally fires if the cell is re-run.
_readonly_handle.remove()
print(f"removed read-only hook from pythia.gpt_neox.layers[{cfg.extract_layer - 1}]")
