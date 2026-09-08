# %% [markdown]
# # Experiment 2 — Steering/control experiment (Phase F)
#
# **Foundation:** exp1h is the "winning" training run. This notebook is a
# byte-identical copy of `exp1/exp1h_source.py` at the training level —
# same `n_slots=4096, k=16, reserved_per_domain=256`, same four domains
# (medicine / law / code / literature), same `supervision_anneal_end=1000`,
# same `SEED = 2024`, same 6000 training steps, same alpha schedule, same
# purity diagnostics, same `reserved_block` classification check with
# `pred_dist` — and *nothing* in that pipeline is changed. We train fresh
# in this notebook (no external checkpoint loads), exactly like every prior
# exp0/exp1 run.
#
# **What exp2 adds (the only delta vs exp1h):** once training and the
# exp1h-style eval block finish, we attach a small **steering
# mechanism** to `ConceptRegistry` and run a fully-automatic suite of
# generative interventions on the *frozen, just-trained* model. Two
# questions, one mechanism:
#
# 1. **Can we BOOST a domain?** If we add a positive bias to a reserved
#    domain's block (`[d*256:(d+1)*256]` of the 4096-slot registry), does
#    greedy generation start producing tokens characteristic of that
#    domain's vocabulary — without breaking fluency?
# 2. **Can we SUPPRESS a domain?** If we subtract a bias from a reserved
#    domain's block while sampling from a real 50-token prefix drawn from
#    that domain's eval corpus, does the generator drift AWAY from that
#    domain's vocabulary?
#
# **Mechanism (one new tensor on the registry):**
#
# - `ConceptRegistry.steer_bias` — a `[n_slots]` tensor, default all
#   zeros, registered as a non-persistent buffer so it travels with
#   `state_dict` plumbing but never gets a gradient and never perturbs
#   training. On every forward pass we add `steer_bias` to the
#   *pre-topk* slot scores right before the top-k selection, so a
#   positive value can push a slot INTO the active set and a negative
#   value can push it OUT. With `steer_bias = 0` (its default at train
#   time and during the exp1h-style eval block) the registry's forward
#   pass is bit-equivalent to exp1h. A small `set_steer_bias()` setter
#   and a `steering()` context manager handle the per-run swap; we
#   always restore to zero between runs to guarantee no leakage.
#
# **Domain vocab lists:** before any steering run, we build a
# "distinctive vocabulary" set per domain — the top ~150 tokens by
# frequency ratio
# `freq_in_d / (freq_in_other_three_domains + eps)` — using the same
# training corpora exp1h already loaded. We then reuse these vocab sets
# as the per-domain target vocabulary for the BOOST/SUPPRESS metrics.
#
# **Steering runs (no human in the loop):**
#
# - **BOOST** direction: for each domain `d`, for strength in
#   `[0, 5, 10, 20, 40]`, set `steer_bias[d*256:(d+1)*256] = +strength`,
#   generate **200 tokens** greedily from a neutral 1-token prompt (just
#   the `<|eot|>` boundary token), and record three metrics:
#   `domain_d_vocab_hit_rate` (fraction of generated tokens inside d's
#   vocab list), `repetition_rate` (fraction of generated bigrams that
#   are repeats of an earlier bigram in the same generation), and
#   `avg_entropy` (mean entropy in **nats** of the softmax next-token
#   distribution over the 200 steps).
# - **SUPPRESS** direction: for each domain `d`, for strength in
#   `[0, 5, 10, 20, 10000]`, set `steer_bias[d*256:(d+1)*256] = -strength`,
#   generate 200 tokens greedy from a real **50-token prefix** drawn
#   uniformly from d's eval corpus, and record the same three metrics.
#   We expect `domain_d_vocab_hit_rate` to drop vs the `strength=0`
#   baseline as suppression grows.
# - **No leakage:** `steer_bias` is reset to all-zeros between every
#   run. A baseline (`strength = 0`) run is included in *both* directions
#   so we can compare boosted / suppressed to the same model with no
#   intervention.
#
# **Automatic collapse-point detector.** We auto-flag the first strength
# (per domain, per direction) where `repetition_rate > 0.5` OR
# `avg_entropy < 1.0` as **"fluency broken here"** in the printed output.
# No human judgment required — if boosting collapses to a loop, we mark
# it. This lets us say e.g. "boost code is healthy through +20 but
# collapses at +40" purely from the metrics.
#
# **Final verdict block:** for each domain and direction, we report
# whether boosting raised `domain_d_vocab_hit_rate` (monotonically-ish,
# before the collapse point) and whether suppressing lowered it relative
# to the `strength=0` baseline. Same three-tier "✅ / 🟡 / 🟡" style as
# the exp1h verdict block.
#
# **Outputs:**
# - `exp2_model.pt` — the freshly trained model.
# - `exp2_steering_results.json` — every metric per (domain, direction,
#   strength), plus the per-domain vocab lists.
# - `exp2_results.png` — extends the exp1h-style plot with two extra
#   axes showing the boost / suppress curves.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings → Accelerator → GPU T4 x2** (or P100)
# 2. **Settings → Internet → On**  ← required to download data
#
# Expected runtime: ~70–110 minutes total (training unchanged from
# exp1h: ~60–90 min; steering adds ~10–20 min for 4×2×(5+5)=40
# generation runs of 200 tokens each plus the JSON / verdict).

# %%
import os, math, time, json, random, contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 2024
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEVICE)
if DEVICE == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("!! no GPU detected — go to Settings > Accelerator > GPU")

OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."
os.makedirs(OUT, exist_ok=True)


# %%
class CFG:
    # --- model ---
    vocab_size    = 8192
    ctx           = 256
    d_model       = 256
    n_layers      = 6
    n_heads       = 4
    dropout       = 0.0

    # --- registry ---
    registry_layer = 3       # after which block to place it (0-indexed)
    n_slots        = 4096    # 16x expansion (unchanged from exp0d)
    k              = 16      # how many slots stay active per token (unchanged from exp0d)

    # --- reserved slots / supervision (new in exp1) ---
    #   256 slots x 4 domains = 1024 reserved (25% of n_slots=4096)
    #   slots [0:256]   = medicine   (domain index 0)
    #   slots [256:512] = law        (domain index 1)
    #   slots [512:768] = code       (domain index 2)
    #   slots [768:1024]= literature (domain index 3)
    #   slots [1024:4096] remain free / unsupervised (same as exp0d's whole registry)
    reserved_per_domain = 256
    # exp1f: lambda_supervision is the PEAK / starting value of an
    # annealed schedule, NOT a single scalar. See supervision_weight_at().
    lambda_supervision  = 0.2
    # exp1f: supervision anneals to zero by this step (separate from
    # alpha_start_step, which stays 1500). Sits between exp1e's 400-step
    # window and exp1d's 1500-step window. exp1e's 400-step window gave
    # a healthy final_dead_frac=11.3% (near exp1c's 6.9%) but a
    # reserved_block classification_accuracy of exactly 25.0% — the
    # 4-class chance floor — and WORSE than exp1c's unsupervised 31.3%.
    # The 25.0% acc on a balanced 480-per-domain eval is suspiciously
    # close to the score a "predict one class every time" strategy would
    # produce; the diag below (pred_dist) makes this concrete. 1000 is
    # longer than exp1e's 400 but still well under exp1d's 1500, testing
    # whether the 25.0% acc was a genuine drop vs. a class-collapse that
    # more exposure would fix.
    supervision_anneal_end = 1000

    # --- data ---
    tokens_per_domain = 15_000_000
    tokenizer_docs    = 20_000

    # --- training ---
    batch_size    = 48
    steps         = 6000
    lr            = 6e-4
    lr_registry   = 3e-3     # registry needs a higher LR
    warmup        = 200
    grad_clip     = 1.0

    # --- alpha schedule (force info to flow through the registry) ---
    alpha_start_step = 1500   # before this, the registry only learns
    alpha_full_step  = 4000   # after this, it's forced at 100%

    # --- losses ---
    lambda_recon      = 1.0
    lambda_auxk       = 0.03
    k_aux             = 32
    dead_after        = 800        # a slot that hasn't fired in this many steps = dead

    # --- eval ---
    eval_every        = 500
    eval_batches      = 40
    eval_loss_batches = 10

cfg = CFG()
print(json.dumps({k: v for k, v in vars(CFG).items() if not k.startswith("_")},
                 indent=2, ensure_ascii=False))


# %% [markdown]
# ## 1. Data — four distant domains
#
# Same contrast-style setup as exp0d but with **four** domains that span
# distinctly different registers:
#
# - **medicine** — PubMed abstracts (biomedical prose)
# - **law** — US legislative bill text (legal English)
# - **code** — Python source code (programming register)
# - **literature** — Project Gutenberg English literature (prose fiction / non-fiction)
#
# Each domain carries its own `extract` callable because `medicine`'s
# `context` field is a dict, not a flat string — the lambda uniformizes that.

# %%
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

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
assert N_DOMAINS == 4, "exp1 is hard-coded to 4 domains"

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


# %%
TOK_PATH = f"{OUT}/tokenizer.json"

if os.path.exists(TOK_PATH):
    tok = Tokenizer.from_file(TOK_PATH)
    print("tokenizer loaded from file")
else:
    print("training tokenizer on a mix of all 4 domains...")
    t0 = time.time()

    def mixed_iter():
        gens = {k: stream_texts(v, limit=cfg.tokenizer_docs // N_DOMAINS)
                for k, v in DOMAINS.items()}
        for k, g in gens.items():
            for t in g:
                yield t

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=cfg.vocab_size,
        special_tokens=["<|eot|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    tok.train_from_iterator(mixed_iter(), trainer=trainer)
    tok.save(TOK_PATH)
    print(f"done in {time.time()-t0:.0f}s — vocab={tok.get_vocab_size()}")

EOT = tok.token_to_id("<|eot|>")


# %%
def build_corpus(domain, spec, target_tokens):
    """Turns domain texts into a single token array."""
    path = f"{OUT}/{domain}.npy"
    if os.path.exists(path):
        arr = np.load(path)
        print(f"{domain}: loaded {len(arr):,} tokens")
        return arr

    t0 = time.time()
    buf, total = [], 0
    for text in stream_texts(spec):
        ids = tok.encode(text).ids
        buf.append(np.array(ids + [EOT], dtype=np.uint16))
        total += len(ids) + 1
        if total >= target_tokens:
            break
    arr = np.concatenate(buf)[:target_tokens]
    np.save(path, arr)
    if len(arr) < target_tokens * 0.95:
        print(f"WARNING: {domain} produced only {len(arr):,} of {target_tokens:,} requested tokens "
              f"-- the source was exhausted. Domains will be imbalanced.")
    print(f"{domain}: {len(arr):,} tokens in {time.time()-t0:.0f}s")
    return arr

corpora = {d: build_corpus(d, s, cfg.tokens_per_domain) for d, s in DOMAINS.items()}

# train/eval split
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
    """4-way balanced: 12 rows per domain in DOMAINS dict order
    (medicine, law, code, literature). Returns (x, y, domain_labels)."""
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
# ## 2. Soft supervision loss (module-level helper)
#
# Operates on the `acts` tensor already produced by `ConceptRegistry.forward`,
# so the registry itself stays unchanged. The activation mass over each of
# the 4 reserved 256-slot blocks is treated as a per-domain "logit", and a
# plain `F.cross_entropy` against the sequence's true domain label provides
# the auxiliary gradient signal. Raw non-negative masses are passed directly
# as logits — `cross_entropy` applies log_softmax internally, and manually
# pre-logging near-zero sums would be numerically unstable.

# %%
def supervision_loss(acts, domain_labels, reserved_per_domain, n_domains=4):
    """Cross-entropy between each domain's reserved-block activation mass and the
    true domain label. acts: [B, T, n_slots]. domain_labels: [B] (one label per
    sequence — every row in a training batch belongs entirely to one domain, so
    the label applies to the whole sequence). Soft/encouraging, not
    architecturally forcing: this is just an auxiliary loss term."""
    seq_mass = acts.sum(dim=1)                                 # [B, n_slots]
    block_mass = torch.stack(
        [seq_mass[:, d*reserved_per_domain:(d+1)*reserved_per_domain].sum(-1)
         for d in range(n_domains)], dim=-1)                   # [B, n_domains]
    return F.cross_entropy(block_mass.float(), domain_labels)


# %% [markdown]
# ## 3. The registry layer — heart of the experiment (unchanged from exp0d)
#
# **exp2 addition:** `steer_bias` (a non-persistent `[n_slots]` buffer,
# default all zeros) is added to the *pre-topk* slot scores every
# forward pass. With `steer_bias = 0` (the default at train time and
# throughout the exp1h-style eval block) this is a bit-exact no-op —
# the registry's training-time forward behavior is identical to exp1h.
# A `set_steer_bias()` setter and a `steering()` context manager give
# the steering code a clean way to swap a bias in for a generation run
# and restore it (to zero, in our case) afterwards.

# %%
class ConceptRegistry(nn.Module):
    """TopK sparse dictionary. Spreads a dense representation across numbered slots."""

    def __init__(self, d_model, n_slots, k, k_aux):
        super().__init__()
        self.n_slots, self.k, self.k_aux = n_slots, k, k_aux
        self.encoder = nn.Linear(d_model, n_slots, bias=True)
        self.decoder = nn.Linear(n_slots, d_model, bias=False)
        self.b_dec   = nn.Parameter(torch.zeros(d_model))

        # init decoder as transpose of encoder (standard practice for SAEs)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True) + 1e-8)

        # counter: last step each slot fired (used to detect dead slots)
        self.register_buffer("last_fired", torch.zeros(n_slots, dtype=torch.long))

        # exp2: per-slot additive bias applied to the pre-topk scores
        # every forward pass. Default all zeros -> bit-equivalent to
        # the exp1h forward at training / eval time. Non-persistent so
        # it travels with the module's device-plumbing but is not
        # saved into state_dict and never gets a gradient.
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
        `with` block, restore the previous value on exit. Use
        `with model.registry.steering(bias):` around the generation
        calls you want to steer; we additionally force a final
        `clear_steer_bias()` so a stray bias can never leak into the
        next run even if an exception is raised mid-block."""
        prev = self.set_steer_bias(bias)
        try:
            yield
        finally:
            self.set_steer_bias(prev)
            self.clear_steer_bias()

    def forward(self, h, step=None, dead_after=None):
        pre  = F.relu(self.encoder(h - self.b_dec))
        # exp2: add the per-slot steering bias BEFORE top-k selection.
        # At training / standard-eval time `steer_bias` is identically
        # zero, so this addition is mathematically a no-op. We keep it
        # unconditional so the code path is identical between the
        # exp1h-style eval block and the steering runs below; a single
        # `.zero_()` call between runs is the leakage guard.
        pre = pre + self.steer_bias
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


# %% [markdown]
# ## 4. The model — small transformer with the registry injected
#
# `forward` now accepts an optional `domain_labels` argument. When provided
# (i.e. during training), it computes `supervision_loss` on the registry's
# already-computed `acts` tensor and returns it alongside the existing losses.
# When `domain_labels` is None (eval/inference), this term is skipped exactly
# like `aux_loss` already is outside training.

# %%
class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.ln1 = nn.LayerNorm(c.d_model)
        self.attn = nn.MultiheadAttention(c.d_model, c.n_heads,
                                          dropout=c.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(c.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(c.d_model, 4 * c.d_model), nn.GELU(),
            nn.Linear(4 * c.d_model, c.d_model), nn.Dropout(c.dropout),
        )

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPTWithRegistry(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.wte = nn.Embedding(c.vocab_size, c.d_model)
        self.wpe = nn.Embedding(c.ctx, c.d_model)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.n_layers)])
        self.ln_f = nn.LayerNorm(c.d_model)
        self.head = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.head.weight = self.wte.weight            # weight tying

        # F11: _init must run BEFORE ConceptRegistry construction so the
        # registry's tied encoder/decoder-transpose init is preserved.
        self.apply(self._init)
        self.registry = ConceptRegistry(c.d_model, c.n_slots, c.k, c.k_aux)
        self.register_buffer("mask", torch.triu(
            torch.full((c.ctx, c.ctx), float("-inf")), diagonal=1))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None, alpha=0.0, step=None,
                want_acts=False, domain_labels=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        mask = self.mask[:T, :T]

        acts_out, recon_loss, aux_loss = None, x.new_zeros(()), x.new_zeros(())
        sup_loss = x.new_zeros(())

        for i, blk in enumerate(self.blocks):
            x = blk(x, mask)
            if i == self.c.registry_layer:
                recon, acts, aux_loss = self.registry(
                    x, step=step, dead_after=self.c.dead_after)
                # target is detached: the registry learns to reconstruct,
                # without us pushing the model to simplify its representation
                # just to make the registry's job easier
                recon_loss = F.mse_loss(recon.float(), x.detach().float())
                x = x + alpha * (recon - x)
                if want_acts:
                    acts_out = acts.detach()
                if domain_labels is not None and acts is not None:
                    sup_loss = supervision_loss(
                        acts.float(), domain_labels,
                        self.c.reserved_per_domain, n_domains=N_DOMAINS)

        logits = self.head(self.ln_f(x))
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      targets.reshape(-1))
        return dict(logits=logits, lm_loss=lm_loss, recon_loss=recon_loss,
                    aux_loss=aux_loss, supervision_loss=sup_loss, acts=acts_out)


model = TinyGPTWithRegistry(cfg).to(DEVICE)
n_model = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("registry"))
n_reg   = sum(p.numel() for n, p in model.named_parameters() if n.startswith("registry"))
print(f"model params     : {n_model/1e6:.2f}M")
print(f"registry params  : {n_reg/1e6:.2f}M")


# %% [markdown]
# ## 5. Training

# %%
reg_params  = [p for n, p in model.named_parameters() if n.startswith("registry")]
base_params = [p for n, p in model.named_parameters() if not n.startswith("registry")]
opt = torch.optim.AdamW([
    {"params": base_params, "lr": cfg.lr,          "name": "base"},
    {"params": reg_params,  "lr": cfg.lr_registry, "name": "registry"},
], betas=(0.9, 0.95), weight_decay=0.1)

scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

def alpha_at(step):
    if step < cfg.alpha_start_step: return 0.0
    if step >= cfg.alpha_full_step: return 1.0
    span = cfg.alpha_full_step - cfg.alpha_start_step
    return (step - cfg.alpha_start_step) / span

def supervision_weight_at(step):
    """Linear anneal from cfg.lambda_supervision (peak) down to 0 by
    cfg.supervision_anneal_end -- exp1f uses 1000 (between exp1e's 400 and exp1d's 1500).
    exp1e's 400-step window produced a healthy final_dead_frac=11.3% but a reserved_block
    classification_accuracy of exactly 25.0% (the 4-class chance floor), worse than exp1c's
    unsupervised 31.3%. We suspect the predictor collapsed to always-predict-one-class,
    which scores exactly 25.00% on a balanced 4-way eval -- hence the pred_dist diagnostic
    added to reserved_block_accuracy() below. 1000 steps sits between 400 and 1500 to
    probe whether more exposure yields real spread of predictions, while still keeping
    imprint exposure well under exp1d's full 1500-step window. Supervision naturally
    converges to near-zero loss within ~200-300 steps in every prior run, but a longer
    window may also be needed for the predictor to actually separate classes (not just
    drive the loss down). Held at 0 for the rest of training (including through the
    entire alpha ramp, same as exp1d/exp1e).
    """
    if step >= cfg.supervision_anneal_end:
        return 0.0
    return cfg.lambda_supervision * (1.0 - step / cfg.supervision_anneal_end)

def lr_scale(step):
    if step < cfg.warmup: return step / max(1, cfg.warmup)
    p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
    return 0.1 + 0.45 * (1 + math.cos(math.pi * min(p, 1.0)))

@torch.no_grad()
def eval_loss(alpha):
    model.eval()
    out = {}
    for d in DOMAINS:
        tot = 0.0
        for _ in range(cfg.eval_loss_batches):
            x, y = get_batch(d, "eval", cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                tot += model(x, y, alpha=alpha)["lm_loss"].item()
        out[d] = tot / cfg.eval_loss_batches
    model.train()
    return out

# %%
hist = []
t0 = time.time()
model.train()

for step in range(1, cfg.steps + 1):
    alpha = alpha_at(step)
    sup_weight = supervision_weight_at(step)
    for g in opt.param_groups:
        base_lr = cfg.lr if g["name"] == "base" else cfg.lr_registry
        g["lr"] = base_lr * lr_scale(step)

    x, y, dl = get_mixed_batch(cfg.batch_size)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        o = model(x, y, alpha=alpha, step=step, domain_labels=dl)
        loss = (o["lm_loss"]
                + cfg.lambda_recon       * o["recon_loss"]
                + cfg.lambda_auxk        * o["aux_loss"]
                + sup_weight             * o["supervision_loss"])

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt); scaler.update()

    if step % 100 == 0 or step == 1:
        dead = model.registry.dead_fraction(step, cfg.dead_after)
        lf = model.registry.last_fired
        live_mask = (step - lf) <= cfg.dead_after
        reserved_live = live_mask[:N_DOMAINS*cfg.reserved_per_domain].float().mean().item()
        free_live     = live_mask[N_DOMAINS*cfg.reserved_per_domain:cfg.n_slots].float().mean().item()
        el = time.time() - t0
        print(f"step {step:5d} | alpha {alpha:.2f} | lm {o['lm_loss'].item():.3f} "
              f"| recon {o['recon_loss'].item():.4f} "
              f"| sup {o['supervision_loss'].item():.3f} "
              f"| sup_w {sup_weight:.3f} "
              f"| dead {dead*100:4.1f}% "
              f"| res_live {reserved_live*100:4.1f}% "
              f"| free_live {free_live*100:4.1f}% "
              f"| {el:.0f}s")
        hist.append(dict(step=step, alpha=alpha, lm=o["lm_loss"].item(),
                         recon=o["recon_loss"].item(),
                         sup=o["supervision_loss"].item(),
                         sup_w=sup_weight, dead=dead))

    if step % cfg.eval_every == 0:
        ev = eval_loss(alpha)
        print(f"   >> eval: " + "  ".join(f"{d}={v:.3f}" for d, v in ev.items()))

print(f"\ntraining done in {(time.time()-t0)/60:.1f} minutes")
torch.save(model.state_dict(), f"{OUT}/exp2_model.pt")


# %% [markdown]
# ## 6. Slot activity collection — restricted to reserved vs free regions
#
# For each of the 4 domains: how often did each slot fire? Then we split the
# 4096-slot space into:
#
# - **reserved** = `[0 : 4*reserved_per_domain]` = `[0:1024]` (4 blocks of 256)
# - **free**     = `[4*reserved_per_domain : n_slots]` = `[1024:4096]`

# %%
RESERVED_END = N_DOMAINS * cfg.reserved_per_domain  # 1024

@torch.no_grad()
def collect_activity(n_batches):
    """Collects: how often each slot (and each raw dim) fired per domain."""
    model.eval()
    slot_cnt = {d: torch.zeros(cfg.n_slots, device=DEVICE) for d in DOMAINS}
    dim_cnt  = {d: torch.zeros(cfg.d_model, device=DEVICE) for d in DOMAINS}
    # for interpreting slots: which tokens activate each one most
    tok_score = torch.zeros(cfg.n_slots, cfg.vocab_size, device=DEVICE)

    hidden = {}
    h_handle = model.registry.register_forward_hook(
        lambda m, i, o: hidden.__setitem__("h", i[0]))

    for d in DOMAINS:
        for _ in range(n_batches):
            x, y = get_batch(d, "eval", cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                o = model(x, alpha=1.0, want_acts=True)
            acts = o["acts"].float().flatten(0, 1)          # [B*T, n_slots]
            slot_cnt[d] += (acts > 0).float().sum(0)

            # same sparsity applied to raw embedding (fair comparison)
            h = hidden["h"].float().flatten(0, 1)           # [B*T, d_model]
            topv, topi = torch.topk(h.abs(), cfg.k, dim=-1)
            dim_cnt[d].scatter_add_(0, topi.flatten(),
                                    torch.ones_like(topi.flatten(), dtype=torch.float))

            flat_tok = x.flatten()
            tok_score.index_add_(1, flat_tok, acts.t())

    h_handle.remove()
    model.train()
    return slot_cnt, dim_cnt, tok_score

slot_cnt, dim_cnt, tok_score = collect_activity(cfg.eval_batches)


# %%
def purity_stats(counts, min_activity=50, n_domains=4):
    """4-way purity over a per-domain slot/dim count dict.
    purity_i = max_d counts[d][i] / sum_d counts[d][i]  for slot i (live slots only).
    """
    stack = torch.stack([counts[d] for d in DOMAIN_NAMES])   # [n_domains, N]
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
        # extra info retained for downstream (a) reserved-block analysis
        total=total.cpu().numpy(),
    )

# Region-restricted count dicts
def _restrict(counts, sl):
    return {d: counts[d][sl] for d in DOMAINS}

# Slice helpers: reserved [0:RESERVED_END], free [RESERVED_END:n_slots]
sl_reserved = slice(0, RESERVED_END)
sl_free     = slice(RESERVED_END, cfg.n_slots)

slot_res_cnt = _restrict(slot_cnt, sl_reserved)
slot_free_cnt = _restrict(slot_cnt, sl_free)

P_res_slots = purity_stats(slot_res_cnt)
P_free_slots = purity_stats(slot_free_cnt)
P_all_slots  = purity_stats(slot_cnt)
P_dims       = purity_stats(dim_cnt)

gap_res  = P_res_slots["mean"]  - P_dims["mean"]
gap_free = P_free_slots["mean"] - P_dims["mean"]
gap_all  = P_all_slots["mean"]  - P_dims["mean"]

print("="*72)
print(f"{'':24} {'reserved':>12} {'free':>12} {'whole':>12} {'raw dims':>12}")
print("-"*72)
print(f"{'slot range':24} {'[0:1024]':>12} {'[1024:4096]':>12} {'[0:4096]':>12} {'d_model=256':>12}")
print(f"{'live units':24} {P_res_slots['n_live']:>12d} {P_free_slots['n_live']:>12d} "
      f"{P_all_slots['n_live']:>12d} {P_dims['n_live']:>12d}")
print(f"{'mean purity':24} {P_res_slots['mean']:>12.3f} {P_free_slots['mean']:>12.3f} "
      f"{P_all_slots['mean']:>12.3f} {P_dims['mean']:>12.3f}")
print(f"{'median purity':24} {P_res_slots['median']:>12.3f} {P_free_slots['median']:>12.3f} "
      f"{P_all_slots['median']:>12.3f} {P_dims['median']:>12.3f}")
print(f"{'purity > 0.80':24} {P_res_slots['frac_above_80']:>11.1%} "
      f"{P_free_slots['frac_above_80']:>11.1%} "
      f"{P_all_slots['frac_above_80']:>11.1%} "
      f"{P_dims['frac_above_80']:>11.1%}")
print(f"{'purity > 0.95':24} {P_res_slots['frac_above_95']:>11.1%} "
      f"{P_free_slots['frac_above_95']:>11.1%} "
      f"{P_all_slots['frac_above_95']:>11.1%} "
      f"{P_dims['frac_above_95']:>11.1%}")
print("="*72)
print(f"gap vs raw dims        : reserved {gap_res:+.3f}  "
      f"free {gap_free:+.3f}  whole {gap_all:+.3f}")


# %% [markdown]
# ## 7. Reserved-block classification accuracy
#
# Same `block_mass` computation as the supervision loss, but on held-out eval
# batches. `argmax` over the 4 per-domain block masses gives the predicted
# domain; compare against the true domain label of each eval sequence. This is
# the most direct signal of whether supervision worked.

# %%
@torch.no_grad()
def reserved_block_accuracy(n_batches):
    model.eval()
    correct = 0
    total = 0
    # exp1f diagnostic: count how many argmax predictions fall into each of
    # the 4 domain classes. exp1e scored exactly 25.0% (= 480/1920), which
    # is what an "always-predict-one-class" strategy gets on a perfectly
    # balanced 4-way eval -- a strong hint the predictor collapsed rather
    # than genuinely spreading guesses. We accumulate this in the SAME loop
    # that computes accuracy (no separate pass): pred_count_per_class[d] is
    # the total number of eval sequences the model assigned to domain d.
    pred_count_per_class = torch.zeros(N_DOMAINS, dtype=torch.long, device=DEVICE)
    for d_idx, d in enumerate(DOMAIN_NAMES):
        for _ in range(n_batches):
            x, _ = get_batch(d, "eval", cfg.batch_size)
            with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE=="cuda")):
                o = model(x, alpha=1.0, want_acts=True)
            acts = o["acts"].float()                          # [B, T, n_slots]
            seq_mass = acts.sum(dim=1)                        # [B, n_slots]
            block_mass = torch.stack(
                [seq_mass[:, k*cfg.reserved_per_domain:(k+1)*cfg.reserved_per_domain].sum(-1)
                 for k in range(N_DOMAINS)], dim=-1)          # [B, n_domains]
            pred = block_mass.argmax(dim=-1)
            true = torch.full((x.size(0),), d_idx,
                              dtype=torch.long, device=x.device)
            # bincount into the diagnostic counter (reuses the same pred tensor)
            pred_count_per_class += torch.bincount(pred, minlength=N_DOMAINS)
            correct += (pred == true).sum().item()
            total   += x.size(0)
    model.train()
    return correct / total, pred_count_per_class.cpu().tolist()

clf_acc, pred_dist = reserved_block_accuracy(cfg.eval_batches)
pred_dist_str = "  ".join(f"{DOMAIN_NAMES[i]}={pred_dist[i]}" for i in range(N_DOMAINS))
print(f"\nreserved-block classification accuracy (eval): {clf_acc*100:.1f}%")
print(f"   pred_dist: {pred_dist_str}   (total={sum(pred_dist)}, balanced=1920)")


# %% [markdown]
# ## 8. Reading the slots — the eyeball proof, four domains
#
# For each domain, print the highest-purity slots from the **reserved**
# block (the supervised region) and from the **free** region (if any free
# slots reach high purity — the unsupervised-discovery story).

# %%
# (a) RESERVED BLOCK — per-domain purity within the 1024 reserved slots
c = torch.stack([slot_cnt[d] for d in DOMAIN_NAMES])        # [4, n_slots]
total_all = c.sum(0)
pur_all = torch.where(total_all > 0,
                      c.max(0).values / total_all.clamp(min=1),
                      torch.zeros_like(total_all))
dom_all = c.argmax(0)

print("\n========== RESERVED BLOCK [0:1024] — top slots per domain ==========")
for d_idx, d in enumerate(DOMAIN_NAMES):
    block_lo = d_idx  * cfg.reserved_per_domain
    block_hi = (d_idx + 1) * cfg.reserved_per_domain
    # only look inside the reserved block assigned to this domain
    mask = (dom_all == d_idx) & (total_all > 200) & (pur_all > 0.70) \
           & (torch.arange(len(total_all), device=total_all.device) >= block_lo) \
           & (torch.arange(len(total_all), device=total_all.device) <  block_hi)
    idx = torch.nonzero(mask).flatten()
    if len(idx) == 0:
        print(f"\n--- {d} (reserved {block_lo}-{block_hi-1}): not enough pure slots")
        continue
    idx = idx[total_all[idx].argsort(descending=True)][:5]
    print(f"\n--- {d} (reserved {block_lo}-{block_hi-1}) ---")
    for s in idx.tolist():
        top_tok = tok_score[s].topk(12).indices.tolist()
        words = [repr(tok.decode([t]))[1:-1] for t in top_tok]
        print(f"  slot {s:4d} | purity {pur_all[s]:.2f} | fires {int(total_all[s]):6,} "
              f"| {' · '.join(words[:10])}")

# (b) FREE REGION — same per-domain survey but only for slots >= RESERVED_END
print("\n========== FREE REGION [1024:4096] — top slots per domain ==========")
mask_free = (torch.arange(len(total_all), device=total_all.device) >= RESERVED_END)
any_free_pure = False
for d_idx, d in enumerate(DOMAIN_NAMES):
    mask = (dom_all == d_idx) & (total_all > 200) & (pur_all > 0.70) & mask_free
    idx = torch.nonzero(mask).flatten()
    if len(idx) == 0:
        print(f"\n--- {d} (free): not enough pure slots")
        continue
    any_free_pure = True
    idx = idx[total_all[idx].argsort(descending=True)][:5]
    print(f"\n--- {d} (free region) ---")
    for s in idx.tolist():
        top_tok = tok_score[s].topk(12).indices.tolist()
        words = [repr(tok.decode([t]))[1:-1] for t in top_tok]
        print(f"  slot {s:4d} | purity {pur_all[s]:.2f} | fires {int(total_all[s]):6,} "
              f"| {' · '.join(words[:10])}")
if not any_free_pure:
    print("\n(no free-region slots reached purity > 0.70)")


# %% [markdown]
# ## 9. Steering — domain vocab lists, generation harness, BOOST / SUPPRESS sweeps
#
# All training-time forward behavior is unchanged: at this point
# `model.registry.steer_bias` is still identically zero, so every
# generation call below starts from the exact exp1h-trained model. We
# swap the bias only inside the per-run `with model.registry.steering(bias):`
# block and force a `clear_steer_bias()` between runs, so there is no
# way for one run's bias to leak into the next.

# %%
N_VOCAB_PER_DOMAIN  = 150     # top-N distinctive tokens per domain
BOOST_STRENGTHS     = [0, 5, 10, 20, 40]
SUPPRESS_STRENGTHS  = [0, 5, 10, 20, 10000]
GEN_LEN             = 200     # tokens generated per steering run
PROMPT_LEN_SUPPRESS = 50      # prefix length for the SUPPRESS direction
COLLAPSE_REP        = 0.5     # repetition_rate > this -> flagged
COLLAPSE_ENT        = 1.0     # avg_entropy   < this -> flagged (nats)
COLLAPSE_MIN_STRENGTH_FOR_BOOST   = 1   # baseline (0) trivially "fluent"
COLLAPSE_MIN_STRENGTH_FOR_SUPPRESS = 1   # we never flag the no-bias baseline

# Sanity-check the assumption: at this point steer_bias should still be zeros.
assert model.registry.steer_bias.abs().sum().item() == 0.0, \
    "steer_bias must be all-zeros at the start of the steering block"


# %% [markdown]
# ### 9a. Domain vocabulary lists (top ~150 tokens by frequency ratio)
#
# `vocab[d]` = top N_VOCAB_PER_DOMAIN tokens by
# `freq_in_d / (freq_in_other_3_domains + eps)`, restricted to tokens
# that appear at least `MIN_FREQ_IN_DOMAIN` times in d's training
# corpus (so we don't pick up rare one-off tokens). We compute these
# vocab sets once and reuse them as the per-domain target vocabulary
# for the BOOST / SUPPRESS metrics below.

# %%
MIN_FREQ_IN_DOMAIN = 10

@torch.no_grad()
def build_domain_vocab(n_tokens=N_VOCAB_PER_DOMAIN, min_freq=MIN_FREQ_IN_DOMAIN):
    """Per-domain distinctive-vocabulary set. freq_in_d / sum_other
    + small eps to avoid divide-by-zero. Low-frequency tokens in d
    are filtered out by `min_freq`."""
    vocab_counts = {}
    for d in DOMAIN_NAMES:
        arr = data[d]["train"].astype(np.int64)
        counts = torch.bincount(torch.from_numpy(arr),
                                minlength=cfg.vocab_size).to(DEVICE).float()
        vocab_counts[d] = counts
    eps = 1.0
    out = {}
    for d in DOMAIN_NAMES:
        others = sum(vocab_counts[o] for o in DOMAIN_NAMES if o != d)
        ratio  = vocab_counts[d] / (others + eps)
        ratio  = ratio * (vocab_counts[d] >= min_freq).float()
        top    = torch.topk(ratio, n_tokens).indices.cpu().tolist()
        out[d] = top
    return out

domain_vocab      = build_domain_vocab()
domain_vocab_sets = {d: set(v) for d, v in domain_vocab.items()}

print("Per-domain distinctive-vocab sizes:")
for d in DOMAIN_NAMES:
    sample = [tok.decode([t]) for t in domain_vocab[d][:12]]
    print(f"  {d:>10s}  ({len(domain_vocab[d])} tokens)  sample: {sample}")


# %% [markdown]
# ### 9b. Greedy generation harness + the three steering metrics
#
# All steering runs share the same generation function and the same
# three metric definitions:
#
# - `domain_d_vocab_hit_rate` — fraction of generated tokens that
#   belong to `domain_vocab_sets[d]`.
# - `repetition_rate` — fraction of generated bigrams that are
#   repeats of an earlier bigram *in the same generation*. A
#   repeated token alone doesn't count — we want the *loop*
#   signal (e.g. "the the the").
# - `avg_entropy` — mean entropy in **nats** of the softmax
#   next-token distribution over the 200 generation steps.

# %%
@torch.no_grad()
def generate(prompt_ids, n_new=GEN_LEN):
    """Greedy generation from `prompt_ids` (1-D list/array of token ids).
    Pads on the LEFT with `EOT` if the prompt is shorter than
    `cfg.ctx`; truncates on the LEFT if longer. Returns
    `(generated_tokens, entropies_per_step)`. Steering is the caller's
    responsibility — wrap this in `with model.registry.steering(bias):`
    to apply a bias during generation."""
    model.eval()
    device = next(model.parameters()).device
    pad_id = EOT if EOT is not None else 0
    ids = [int(t) for t in prompt_ids]
    if len(ids) < cfg.ctx:
        ids = [pad_id] * (cfg.ctx - len(ids)) + ids
    elif len(ids) > cfg.ctx:
        ids = ids[-cfg.ctx:]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    generated, entropies = [], []
    for _ in range(n_new):
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
            o = model(x, alpha=1.0)
        logits = o["logits"][0, -1].float()
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum().item()
        entropies.append(ent)
        nxt = int(logits.argmax(dim=-1).item())
        generated.append(nxt)
        x = torch.cat([x[:, 1:],
                       torch.tensor([[nxt]], device=device)], dim=1)
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


def first_collapse(results_for_domain, min_strength=1):
    """Auto-detect the first strength where fluency breaks: repetition_rate
    above COLLAPSE_REP OR avg_entropy below COLLAPSE_ENT. Returns the
    strength or None. The baseline (`strength == 0`) is never flagged."""
    for r in results_for_domain:
        s = r["strength"]
        if s < min_strength:
            continue
        if (r["rep"] > COLLAPSE_REP) or (r["ent"] < COLLAPSE_ENT):
            return s
    return None


# %% [markdown]
# ### 9c. BOOST sweep — push a domain's reserved block UP
#
# For each domain `d`, set `steer_bias[d*256:(d+1)*256] = +strength`
# for `strength` in `[0, 5, 10, 20, 40]`. Generate 200 tokens from a
# neutral 1-token prompt (`[EOT]` only — the natural "between documents"
# boundary token). We do NOT seed on a domain-specific prefix for BOOST;
# if boosting a domain's reserved block actually steers generation,
# it should push the model toward d's vocabulary *from scratch*.

# %%
boost_results = {}

print("=" * 78)
print("BOOST sweep  (neutral prompt = [EOT], n_gen=200, strengths=[0,5,10,20,40])")
print("=" * 78)
print(f"{'domain':>10s}  {'strength':>8s}  {'hit%':>6s}  {'rep%':>6s}  {'ent(nats)':>10s}  {'n_tok':>6s}  note")
print("-" * 78)
for d_idx, d in enumerate(DOMAIN_NAMES):
    model.registry.clear_steer_bias()              # leakage guard
    rows = []
    for strength in BOOST_STRENGTHS:
        bias = torch.zeros(cfg.n_slots, device=DEVICE)
        if strength != 0:
            lo = d_idx * cfg.reserved_per_domain
            hi = (d_idx + 1) * cfg.reserved_per_domain
            bias[lo:hi] = strength
        with model.registry.steering(bias):
            toks, ents = generate([EOT], n_new=GEN_LEN)
        hit = domain_vocab_hit_rate(toks, domain_vocab_sets[d])
        rep = repetition_rate(toks)
        ent = avg_entropy(ents)
        broken = (rep > COLLAPSE_REP) or (ent < COLLAPSE_ENT)
        rows.append(dict(direction="boost", domain=d, strength=strength,
                         hit=hit, rep=rep, ent=ent,
                         broken=bool(broken),
                         n_tokens=len(toks)))
        note = ""
        if broken and strength >= COLLAPSE_MIN_STRENGTH_FOR_BOOST:
            note = "  ** fluency broken here **"
        print(f"{d:>10s}  {strength:>8d}  {hit*100:5.1f}%  {rep*100:5.1f}%  "
              f"{ent:10.3f}  {len(toks):6d}{note}")
    model.registry.clear_steer_bias()              # leakage guard
    boost_results[d] = rows
    boost_results.setdefault("_collapse", {})[d] = first_collapse(
        rows, min_strength=COLLAPSE_MIN_STRENGTH_FOR_BOOST)
print("-" * 78)
print("(any row tagged ** fluency broken here ** has "
      "repetition_rate > 0.5 or avg_entropy < 1.0; the first such "
      "strength per domain is the auto-flagged collapse point.)")


# %% [markdown]
# ### 9d. SUPPRESS sweep — push a domain's reserved block DOWN
#
# For each domain `d`, sample a real **50-token prefix** from d's own
# eval corpus (a fresh seed per domain; the same prefix is reused across
# the strength sweep so the comparison to the `strength=0` baseline is
# apples-to-apples). Set
# `steer_bias[d*256:(d+1)*256] = -strength` for `strength` in
# `[0, 5, 10, 20, 10000]` and generate 200 tokens. We expect
# `domain_d_vocab_hit_rate` to drop relative to the `strength=0`
# baseline as suppression grows.

# %%
suppress_results = {}

print("=" * 78)
print("SUPPRESS sweep  (50-token prefix from own eval corpus, "
      "n_gen=200, strengths=[0,5,10,20,10000])")
print("=" * 78)
print(f"{'domain':>10s}  {'strength':>8s}  {'hit%':>6s}  {'rep%':>6s}  "
      f"{'ent(nats)':>10s}  {'n_tok':>6s}  note")
print("-" * 78)
for d_idx, d in enumerate(DOMAIN_NAMES):
    model.registry.clear_steer_bias()              # leakage guard
    # sample a 50-token prefix from d's eval corpus (one prefix, reused)
    eval_arr = data[d]["eval"]
    start = int(np.random.randint(0, max(1, len(eval_arr) - PROMPT_LEN_SUPPRESS - 1)))
    prefix = [int(t) for t in eval_arr[start:start + PROMPT_LEN_SUPPRESS].tolist()]
    rows = []
    for strength in SUPPRESS_STRENGTHS:
        bias = torch.zeros(cfg.n_slots, device=DEVICE)
        if strength != 0:
            lo = d_idx * cfg.reserved_per_domain
            hi = (d_idx + 1) * cfg.reserved_per_domain
            bias[lo:hi] = -strength
        with model.registry.steering(bias):
            toks, ents = generate(prefix, n_new=GEN_LEN)
        hit = domain_vocab_hit_rate(toks, domain_vocab_sets[d])
        rep = repetition_rate(toks)
        ent = avg_entropy(ents)
        broken = (rep > COLLAPSE_REP) or (ent < COLLAPSE_ENT)
        rows.append(dict(direction="suppress", domain=d, strength=strength,
                         hit=hit, rep=rep, ent=ent,
                         broken=bool(broken),
                         n_tokens=len(toks)))
        note = ""
        if broken and strength >= COLLAPSE_MIN_STRENGTH_FOR_SUPPRESS:
            note = "  ** fluency broken here **"
        print(f"{d:>10s}  {strength:>8d}  {hit*100:5.1f}%  {rep*100:5.1f}%  "
              f"{ent:10.3f}  {len(toks):6d}{note}")
    model.registry.clear_steer_bias()              # leakage guard
    suppress_results[d] = rows
    suppress_results.setdefault("_collapse", {})[d] = first_collapse(
        rows, min_strength=COLLAPSE_MIN_STRENGTH_FOR_SUPPRESS)
print("-" * 78)
print("(suppression at strength=10000 is intentionally extreme; for some "
      "domains it will collapse fluency before hitting the requested depth, "
      "and the collapse-detector flags it on the spot.)")


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 5, figsize=(28, 4.5))

steps = [h["step"] for h in hist]
ax[0].plot(steps, [h["lm"] for h in hist], label="LM loss")
ax0b = ax[0].twinx()
ax0b.plot(steps, [h["alpha"] for h in hist], "r--", alpha=.6, label="alpha")
ax0b.plot(steps, [h["sup"] for h in hist], "g:", alpha=.6, label="supervision")
ax0b.set_ylabel("alpha / supervision loss", color="r")
ax[0].set_title("language-model loss + alpha + sup loss"); ax[0].set_xlabel("step")

ax[1].plot(steps, [h["dead"]*100 for h in hist], color="darkorange")
ax[1].set_title("dead-slot fraction %"); ax[1].set_xlabel("step")

bins = np.linspace(0.25, 1.0, 31)
ax[2].hist(P_dims["values"],      bins=bins, alpha=.4, label="raw dims",    color="gray")
ax[2].hist(P_all_slots["values"], bins=bins, alpha=.5, label="all slots",   color="teal")
ax[2].hist(P_res_slots["values"], bins=bins, alpha=.6, label="reserved",    color="crimson")
ax[2].hist(P_free_slots["values"],bins=bins, alpha=.5, label="free region", color="steelblue")
ax[2].axvline(0.25, color="k", ls=":", lw=1, label="uniform floor (0.25)")
ax[2].set_title("purity distribution — regions vs raw")
ax[2].set_xlabel("purity"); ax[2].legend(loc="upper left", fontsize=8)

# exp2 additions — axes 3 and 4 are the steering curves
DOMAIN_COLORS = {"medicine": "tab:blue",  "law": "tab:orange",
                 "code":     "tab:green", "literature": "tab:purple"}
ax[3].set_title("BOOST: hit-rate on d's vocab vs strength")
ax[3].set_xlabel("strength"); ax[3].set_ylabel("domain_d_vocab_hit_rate")
for d in DOMAIN_NAMES:
    rs = boost_results[d]
    ax[3].plot([r["strength"] for r in rs],
               [r["hit"]    for r in rs],
               "-o", color=DOMAIN_COLORS[d], label=d)
ax[3].set_xscale("symlog", linthresh=1)
ax[3].axhline(0.25, color="k", ls=":", lw=1)
ax[3].legend(loc="best", fontsize=8)

ax[4].set_title("SUPPRESS: hit-rate on d's vocab vs strength")
ax[4].set_xlabel("strength"); ax[4].set_ylabel("domain_d_vocab_hit_rate")
for d in DOMAIN_NAMES:
    rs = suppress_results[d]
    ax[4].plot([r["strength"] for r in rs],
               [r["hit"]    for r in rs],
               "-o", color=DOMAIN_COLORS[d], label=d)
ax[4].set_xscale("symlog", linthresh=1)
ax[4].axhline(0.25, color="k", ls=":", lw=1)
ax[4].legend(loc="best", fontsize=8)

plt.tight_layout()
plt.savefig(f"{OUT}/exp2_results.png", dpi=130)
plt.show()


# %% [markdown]
# ## 9. Training verdict + summary
#
# Three-tier verdict, based on the two clearest signals:
#
# - **Reserved classification accuracy** — direct measure of whether the
#   supervision signal successfully steered the reserved blocks.
# - **Free-region gap vs raw dims** — direct measure of whether unsupervised
#   specialization still emerged in the free region, the way it did in exp0d.
#
# (These thresholds are a first reasonable guess — not the same hard
# pass/fail bar exp0d hit. They are printed as guidance text, not gates.)
#
# **Exposure confound (per-domain step count):** Because exp1 uses 4-way
# batching at the same `batch_size=48` and `steps=6000` as exp0d, each
# domain now gets ~12 rows/step instead of exp0d's 24 rows/step — roughly
# half the per-domain gradient exposure. So if the free-region gap comes
# back weaker than exp0d's +0.222, that confound (less exposure) cannot be
# cleanly distinguished from supervision having competed away free-region
# specialization; a weaker gap is not automatically evidence against the
# free-discovery mechanism — it may simply need more steps. This is why
# the `steps=6000` budget is a starting point, not a final answer; we will
# iterate cheaply (exp1 -> exp1b -> exp1c -> ...) the same way exp0 ->
# exp0b -> exp0c -> exp0d converged.
#
# **Trunk-gradient confound (supervision is not detached):** The
# `supervision_loss` gradient flows through `acts -> encoder -> x` into
# transformer layers 0-3 (it is *not* detached from the trunk), so the
# supervision signal can in principle reshape the very trunk
# representation that free-region purity is measured on. This is a
# deliberate design choice — the supervision signal *should* be able to
# shape the trunk, that's the mechanism — but it means free-region purity
# reflects trunk-representation drift + registry-slot competition for
# capacity together, not registry-only effects in isolation.

# %%
if clf_acc > 0.70 and gap_free > 0.10:
    verdict = ("\u2705 both mechanisms work: supervision successfully steers reserved blocks, "
               "and unsupervised specialization still emerges in the free region.")
    nxt = "next: try a second-pass refinement (e.g. increase reserved_per_domain, or train longer)."
elif clf_acc > 0.70 and gap_free <= 0.10:
    verdict = ("\U0001F7E1 supervision works, but unsupervised free-region specialization is "
               "weaker with 4 domains than it was with 2 in exp0.")
    nxt = ("consider: lowering lambda_supervision to leave more gradient for the free region, "
           "or shrinking reserved_per_domain so more slots stay free.")
else:
    verdict = ("\U0001F7E1 supervision signal is weak \u2014 consider raising lambda_supervision "
               "or reserved_per_domain before concluding.")
    nxt = ("double-check: (1) domain_labels wired correctly through get_mixed_batch / forward, "
           "(2) lambda_supervision not accidentally 0, (3) reserved_per_domain indexing matches "
           "DOMAINS order.")

print(verdict)
print("next step:", nxt)


# %% [markdown]
# ## 10. Steering verdict (exp2)
#
# A separate verdict block for the steering runs. Two questions, both
# answered automatically from the recorded metrics:
#
# 1. **BOOST direction.** For each domain `d`, did
#    `domain_d_vocab_hit_rate` rise (monotonically-ish, before the
#    collapse point) as we increased the positive bias on d's
#    reserved block? A "monotonically-ish rise" means the baseline
#    hit-rate is the smallest of the non-collapse strengths (or tied
#    for smallest).
# 2. **SUPPRESS direction.** For each domain `d`, did
#    `domain_d_vocab_hit_rate` drop relative to the `strength=0`
#    baseline as we increased the magnitude of the negative bias?
#
# Collapse points are reported separately, per domain / direction,
# from the auto-detector above.

# %%
def _hit_at(rows, strength):
    for r in rows:
        if r["strength"] == strength:
            return r["hit"]
    return None

# Per-domain BOOST verdict
boost_verdict = {}
for d in DOMAIN_NAMES:
    rows = boost_results[d]
    collapse = boost_results["_collapse"][d]
    # use the strengths up to (but not including) the first collapse,
    # or all strengths if no collapse was flagged
    if collapse is None:
        ok_rows = rows
        note = "no collapse"
    else:
        ok_rows = [r for r in rows if r["strength"] < collapse]
        note = f"collapse at strength={collapse}"
    if len(ok_rows) >= 2:
        baseline_hit = _hit_at(ok_rows, 0)
        max_hit      = max(r["hit"] for r in ok_rows)
        rose         = (max_hit is not None and baseline_hit is not None
                        and max_hit >= baseline_hit - 1e-9)
    else:
        rose = False
        baseline_hit = _hit_at(rows, 0) if rows else None
        max_hit      = baseline_hit
    boost_verdict[d] = dict(
        baseline_hit=baseline_hit,
        max_non_collapse_hit=max_hit,
        rose_monotonically=bool(rose),
        collapse_strength=collapse,
        note=note,
    )

# Per-domain SUPPRESS verdict
suppress_verdict = {}
for d in DOMAIN_NAMES:
    rows = suppress_results[d]
    collapse = suppress_results["_collapse"][d]
    baseline_hit = _hit_at(rows, 0)
    # for SUPPRESS, "fell" = the hit-rate at the LARGEST non-collapse
    # strength is less than the strength=0 baseline.
    if collapse is None:
        ok_rows = rows
        note = "no collapse"
    else:
        ok_rows = [r for r in rows if r["strength"] < collapse]
        note = f"collapse at strength={collapse}"
    if len(ok_rows) >= 2 and baseline_hit is not None:
        min_hit = min(r["hit"] for r in ok_rows if r["strength"] > 0) \
            if any(r["strength"] > 0 for r in ok_rows) else baseline_hit
        fell   = (min_hit < baseline_hit - 1e-9)
    else:
        fell   = False
        min_hit = baseline_hit
    suppress_verdict[d] = dict(
        baseline_hit=baseline_hit,
        min_non_collapse_hit=min_hit,
        fell_vs_baseline=bool(fell),
        collapse_strength=collapse,
        note=note,
    )

print("=" * 78)
print("STEERING VERDICT  (per domain, BOOST and SUPPRESS)")
print("=" * 78)
for d in DOMAIN_NAMES:
    bv = boost_verdict[d]
    sv = suppress_verdict[d]
    boost_str  = ("raised hit-rate" if bv["rose_monotonically"]
                  else "did NOT cleanly raise hit-rate")
    suppress_str = ("lowered hit-rate vs baseline" if sv["fell_vs_baseline"]
                    else "did NOT cleanly lower hit-rate vs baseline")
    print(f"  {d:>10s}  BOOST  : {boost_str:34s}  "
          f"(baseline={bv['baseline_hit']:.3f}, "
          f"max_healthy={bv['max_non_collapse_hit']:.3f}, "
          f"{bv['note']})")
    print(f"  {d:>10s}  SUPPRESS: {suppress_str:34s}  "
          f"(baseline={sv['baseline_hit']:.3f}, "
          f"min_healthy={sv['min_non_collapse_hit']:.3f}, "
          f"{sv['note']})")
print("=" * 78)

# One-line summary verdict, same three-tier style as the training block
n_boost_ok   = sum(1 for d in DOMAIN_NAMES if boost_verdict[d]["rose_monotonically"])
n_suppress_ok = sum(1 for d in DOMAIN_NAMES if suppress_verdict[d]["fell_vs_baseline"])
print()
if n_boost_ok >= 3 and n_suppress_ok >= 3:
    steering_verdict = ("\u2705 steering works in both directions on most domains: "
                        "boosting a reserved block raises hit-rate on its "
                        "domain's vocab, and suppressing it lowers it.")
elif n_boost_ok >= 3 or n_suppress_ok >= 3:
    steering_verdict = ("\U0001F7E1 steering works in one direction but not the other: "
                        "see the per-domain table above for which domains moved "
                        "and which did not.")
else:
    steering_verdict = ("\U0001F7E1 steering did not cleanly move hit-rate in either "
                        "direction; reserved blocks may not have separated the "
                        "domains strongly enough to support generative steering.")
print(steering_verdict)


# %%
summary = dict(
    verdict=verdict,
    steering_verdict=steering_verdict,
    reserved_block=dict(
        classification_accuracy=clf_acc,
        mean=P_res_slots["mean"],
        median=P_res_slots["median"],
        live=P_res_slots["n_live"],
        above_80=P_res_slots["frac_above_80"],
        gap_vs_raw=gap_res,
    ),
    free_region=dict(
        mean=P_free_slots["mean"],
        median=P_free_slots["median"],
        live=P_free_slots["n_live"],
        above_80=P_free_slots["frac_above_80"],
        gap_vs_raw=gap_free,
    ),
    whole_registry=dict(
        mean=P_all_slots["mean"],
        median=P_all_slots["median"],
        live=P_all_slots["n_live"],
        above_80=P_all_slots["frac_above_80"],
        gap_vs_raw=gap_all,
    ),
    raw_dims=dict(mean=P_dims["mean"], median=P_dims["median"]),
    final_dead_frac=hist[-1]["dead"],
    final_lm_loss=hist[-1]["lm"],
    final_supervision_loss=hist[-1]["sup"],
    config={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
)
with open(f"{OUT}/exp2_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))


# %% [markdown]
# ## 11. Steering JSON dump — `exp2_steering_results.json`
#
# The per-run metrics, the per-domain vocab lists, and the per-domain
# verdicts — everything needed to re-draw the steering curves or
# re-score the auto collapse-detector offline. Token ids are ints
# (not the underlying strings) so the JSON stays compact and
# tokenizer-independent.

# %%
steering_payload = dict(
    config=dict(
        n_vocab_per_domain=N_VOCAB_PER_DOMAIN,
        min_freq_in_domain=MIN_FREQ_IN_DOMAIN,
        gen_len=GEN_LEN,
        prompt_len_suppress=PROMPT_LEN_SUPPRESS,
        boost_strengths=BOOST_STRENGTHS,
        suppress_strengths=SUPPRESS_STRENGTHS,
        collapse_rep_threshold=COLLAPSE_REP,
        collapse_entropy_threshold=COLLAPSE_ENT,
    ),
    vocab={d: [int(t) for t in v] for d, v in domain_vocab.items()},
    boost={
        d: [dict(strength=int(r["strength"]),
                 hit=float(r["hit"]),
                 rep=float(r["rep"]),
                 ent=float(r["ent"]),
                 broken=bool(r["broken"]),
                 n_tokens=int(r["n_tokens"]))
            for r in rows]
        for d, rows in boost_results.items() if d != "_collapse"
    },
    suppress={
        d: [dict(strength=int(r["strength"]),
                 hit=float(r["hit"]),
                 rep=float(r["rep"]),
                 ent=float(r["ent"]),
                 broken=bool(r["broken"]),
                 n_tokens=int(r["n_tokens"]))
            for r in rows]
        for d, rows in suppress_results.items() if d != "_collapse"
    },
    collapse_points=dict(
        boost={d: (None if boost_results["_collapse"][d] is None
                   else int(boost_results["_collapse"][d]))
               for d in DOMAIN_NAMES},
        suppress={d: (None if suppress_results["_collapse"][d] is None
                      else int(suppress_results["_collapse"][d]))
                  for d in DOMAIN_NAMES},
    ),
    verdict=dict(
        training=verdict,
        steering=steering_verdict,
        per_domain_boost={d: dict(
            baseline_hit=float(boost_verdict[d]["baseline_hit"]) if boost_verdict[d]["baseline_hit"] is not None else None,
            max_non_collapse_hit=float(boost_verdict[d]["max_non_collapse_hit"]) if boost_verdict[d]["max_non_collapse_hit"] is not None else None,
            rose_monotonically=bool(boost_verdict[d]["rose_monotonically"]),
            collapse_strength=(None if boost_verdict[d]["collapse_strength"] is None
                               else int(boost_verdict[d]["collapse_strength"])),
            note=str(boost_verdict[d]["note"]),
        ) for d in DOMAIN_NAMES},
        per_domain_suppress={d: dict(
            baseline_hit=float(suppress_verdict[d]["baseline_hit"]) if suppress_verdict[d]["baseline_hit"] is not None else None,
            min_non_collapse_hit=float(suppress_verdict[d]["min_non_collapse_hit"]) if suppress_verdict[d]["min_non_collapse_hit"] is not None else None,
            fell_vs_baseline=bool(suppress_verdict[d]["fell_vs_baseline"]),
            collapse_strength=(None if suppress_verdict[d]["collapse_strength"] is None
                               else int(suppress_verdict[d]["collapse_strength"])),
            note=str(suppress_verdict[d]["note"]),
        ) for d in DOMAIN_NAMES},
    ),
)
with open(f"{OUT}/exp2_steering_results.json", "w", encoding="utf-8") as f:
    json.dump(steering_payload, f, indent=2, ensure_ascii=False)
print(f"\nwrote {OUT}/exp2_steering_results.json  "
      f"(boost runs: {sum(len(v) for v in steering_payload['boost'].values())}, "
      f"suppress runs: {sum(len(v) for v in steering_payload['suppress'].values())})")

print("\nALL EXP2 OUTPUT FILES:")
print(f"  {OUT}/exp2_model.pt")
print(f"  {OUT}/exp2_steering_results.json")
print(f"  {OUT}/exp2_results.png")
