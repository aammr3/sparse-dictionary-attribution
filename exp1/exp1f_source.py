# %% [markdown]
# # Experiment 1f — Mid-length supervision-anneal window (linear ramp to zero by step 1000)
#
# **The single question:** exp1d (1500-step anneal) kept a healthy
# `final_dead_frac = 29.8%` but slot collapse was still on the same
# alpha-crossing schedule as exp1/exp1b, so we cut the exposure window
# sharply in exp1e — 400 steps, ~27% of exp1d's window — and got a
# nearly-healthy `final_dead_frac = 11.3%` (close to exp1c's 6.9%).
# But exp1e's `reserved_block` classification accuracy came in at
# **exactly 25.0%** — the 4-class chance floor on a balanced 1920-sample
# eval, and *worse* than exp1c's unsupervised 31.3%. We suspect the
# predictor collapsed to always-predict-one-class (which scores
# exactly 25.00% on a balanced set) rather than genuinely spreading
# guesses. exp1f sits between exp1e (400) and exp1d (1500) at **1000
# steps**, testing whether more exposure gets the predictor to spread
# its guesses (and recover some accuracy) without re-introducing the
# imprint-heavy collapse. We also add a `pred_dist` diagnostic to
# `reserved_block_accuracy` so a future "exactly 25.0%" can be ruled
# in or out as a class-collapse on the spot.
#
# **The mechanism, layered on top of exp1d:**
#
# - 4 domains (unchanged from exp1/exp1b/exp1d): medicine / law / code / literature
# - Slot index space: `[0:256]` reserved for medicine, `[256:512]` reserved for
#   law, `[512:768]` reserved for code, `[768:1024]` reserved for literature,
#   `[1024:4096]` stays free / unsupervised
# - The reserved-range architecture is kept structurally identical to exp1d:
#   `reserved_per_domain = 256`, the `acts -> supervision_loss(.)` computation
#   still runs every step (so the per-100-step log still reports `sup`), and
#   the reserved-block classification accuracy check still runs at the end.
# - What changes vs exp1d is the supervision-anneal window length. In exp1d
#   the weight decayed linearly from `cfg.lambda_supervision = 0.2` (peak)
#   at step 0 down to `0.0` at `cfg.alpha_start_step` (1500). In exp1e the
#   window was cut to 400 steps; in **exp1f** it sits at **1000 steps**
#   (between exp1e's 400 and exp1d's 1500), independent of
#   `cfg.alpha_start_step` which stays at 1500 and is **not touched**.
#   Supervision is on only for the first 1000 steps and is held at exactly
#   `0.0` for the remaining 5000 steps, including the entire alpha ramp
#   from step 1500 to step 4000.
# - Free-region analysis (`[1024:4096]`) is the same purity-gap test as
#   exp1d/exp1c, just restricted to that subset.
#
# **Why a mid-length window — the prior five Kaggle runs:**
#
# | run   | supervision           | final_dead_frac | reserved live | clf acc |
# |-------|-----------------------|-----------------|---------------|---------|
# | exp1  | constant lambda=0.2    | 44.9%           | 74/1024       | 99.6%   |
# | exp1b | constant lambda=0.05   | 48.2%           | 93/1024       | 99.8%   |
# | exp1c | constant lambda=0.0    |  6.9%           | 677/1024      | 31.3% (chance) |
# | exp1d | anneal 0.2->0 by 1500  | 29.8%           | 92/1024       | 68.8%   |
# | exp1e | anneal 0.2->0 by  400  | 11.3%           | ~677?/1024    | 25.0% (chance, suspected collapse) |
#
# **The exp1e anomaly.** exp1e produced a *healthy* `final_dead_frac`
# (11.3%, close to exp1c's 6.9%) but a `reserved_block` classification
# accuracy of exactly **25.0%** — the 4-class chance floor on a balanced
# 1920-sample eval, and *worse* than exp1c's unsupervised 31.3%. On a
# perfectly balanced 4-way classification problem, a predictor that
# always picks the same class scores *exactly* 25.00% by construction.
# So exp1e's number is what you would expect if the predictor had
# collapsed to "predict-one-class-every-time" rather than genuinely
# spreading its guesses. The diagnostic added in exp1f
# (`pred_dist`, printed alongside the accuracy) makes this concrete:
# it counts how many eval argmax predictions fall into each of the 4
# domain classes. If exp1e's `pred_dist` showed e.g. `[480, 480, 480, 480]`
# averaged across the 4 true domains, that confirms uniform collapse to
# the most-predicted class; if instead it showed e.g. `[1920, 0, 0, 0]`
# (all predictions on one class for every true domain), that confirms
# total class-collapse to a single output.
#
# **Why 1000, not just "stay at 400" or "go back to 1500".** exp1e's
# 400-step window appears to plant enough *imprint* to keep slots
# healthy (collapses tied to alpha, not to early exposure) but not
# enough *signal* to get the predictor to spread its guesses. exp1d's
# 1500-step window did both — but at the cost of `final_dead_frac`
# climbing back up to 29.8%. 1000 sits between the two: longer than the
# ~200–300-step natural convergence point of `supervision_loss` (so the
# predictor has more time to actually separate the four classes, not
# just minimize the loss), but still well under exp1d's full 1500-step
# exposure. The hypothesis: the 25.0% acc was a genuine drop on a
# *spread* predictor with too-little exposure, not a class-collapse, in
# which case 1000 should recover accuracy while keeping `dead_frac`
# close to exp1e/ex1c levels. If instead 1000 also lands near 25% with a
# similarly-flat `pred_dist`, then the underlying issue is something
# else (e.g. 400 steps is already past the useful-exposure point and
# extra steps just add noise).
#
# **Hoped-for outcome:** `final_dead_frac` landing between exp1e's
# 11.3% and exp1d's 29.8% (closer to exp1e), with classification
# accuracy back above chance (i.e. *not* exactly 25.0%) — ideally
# somewhere between exp1e's 25% (collapsed?) and exp1d's 68.8%, with a
# `pred_dist` that is no longer flat or single-class dominated.
#
# **Foundation:** This builds directly on exp1d (and ultimately on exp0d,
# the validated four-run Kaggle baseline (`n_slots=4096, k=16`) which
# finished at mean purity **0.805** and purity gap **+0.222** over raw
# embedding dims — a clean pass of the original `mean > 0.80 AND gap > 0.10`
# bar). Everything in exp1d's working architecture stays exactly as-is:
# alpha curriculum, TopK sparse coding, AuxK dead-slot revival, decoder
# column renorm every 200 steps, fp32 loss casting, optimizer LR grouping,
# the modern `torch.amp.GradScaler("cuda", ...)` API, and all eleven F1–F11
# fixes from the prior review cycle.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings → Accelerator → GPU T4 x2** (or P100)
# 2. **Settings → Internet → On**  ← required to download data
#
# Expected runtime: ~60–90 minutes total.
#
# ---
# **Variant note (exp1f):** exp1f is the mid-length-anneal-window variant
# of exp1d/exp1e. All three turn supervision off before the alpha ramp
# (which begins at `cfg.alpha_start_step = 1500` and is unchanged from
# exp1d). exp1d's supervision weight decayed linearly from
# `lambda_supervision = 0.2` (the same peak as exp1) down to `0.0`
# across the first **1500** steps. exp1e cut that window to **400**
# steps — close to the natural ~200–300-step convergence point of
# `supervision_loss` — and got `final_dead_frac = 11.3%` (healthy, near
# exp1c's 6.9%) but a `reserved_block` classification accuracy of
# exactly **25.0%**, the 4-class chance floor on a balanced 1920-sample
# eval, and *worse* than exp1c's unsupervised 31.3%. We suspect the
# predictor collapsed to "always predict the same class" (which scores
# exactly 25.00% on a balanced eval) rather than genuinely spreading its
# guesses. exp1f tests a third point: **1000 steps** — between exp1e's
# 400 and exp1d's 1500 — on the hypothesis that exp1e's 25.0% was a
# genuine drop on a *spread* predictor with too little exposure, not a
# class-collapse. exp1f also adds the `pred_dist` diagnostic to
# `reserved_block_accuracy` (printed alongside the accuracy) so any
# future "exactly 25.0%" can be ruled in or out as a class-collapse on
# the spot: a uniform `[480, 480, 480, 480]` `pred_dist` confirms
# uniform collapse; `[1920, 0, 0, 0]` (or any single-class
# concentration) confirms total class-collapse. Everything else is
# unchanged from exp1d/exp1e: same four domains, same `n_slots=4096,
# k=16`, `reserved_per_domain=256`, same 6000 steps, same alpha
# schedule (the new `supervision_anneal_end = 1000` is *separate from*
# `alpha_start_step = 1500` and does not touch the alpha ramp), same
# other losses, same eval logic, same `res_live` / `free_live` /
# `sup_w` logging.

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
torch.save(model.state_dict(), f"{OUT}/exp1f_model.pt")


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


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(18, 4))

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

plt.tight_layout()
plt.savefig(f"{OUT}/exp1f_results.png", dpi=130)
plt.show()


# %% [markdown]
# ## 9. Verdict + summary
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


# %%
summary = dict(
    verdict=verdict,
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
with open(f"{OUT}/exp1f_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
