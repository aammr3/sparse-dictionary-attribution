# %% [markdown]
# # Experiment 1 — Full specialization across 4 domains with named reserved slot ranges
#
# **The single question:** If we reserve named blocks of registry slots for
# specific domains (medicine, law, code, literature) and add a *soft* auxiliary
# loss that encourages those reserved blocks to light up on their assigned
# domain — while leaving the rest of the registry free to specialize
# unsupervised, exactly the way exp0d did — does the supervised signal
# successfully steer the reserved blocks, AND does unsupervised specialization
# still emerge on its own in the free region?
#
# **The mechanism, layered on top of exp0d:**
#
# - 4 domains (was 2 in exp0d): medicine / law / code / literature
# - Slot index space: `[0:256]` reserved for medicine, `[256:512]` reserved for
#   law, `[512:768]` reserved for code, `[768:1024]` reserved for literature,
#   `[1024:4096]` stays free / unsupervised
# - A *soft* cross-entropy auxiliary loss: for each training sequence, the
#   activation mass over the 4 reserved blocks is passed through
#   `F.cross_entropy` against the sequence's true domain label. Weight is
#   `lambda_supervision = 0.2` — encouraging, not architecturally forcing.
# - Free-region analysis (`[1024:4096]`) is the exact same purity-gap test as
#   exp0d, just restricted to that subset. If the gap vs raw dims is still
#   positive there, unsupervised specialization survived the new competition.
#
# **Foundation:** This builds directly on exp0d, the validated four-run Kaggle
# baseline (`n_slots=4096, k=16`) which finished at mean purity **0.805** and
# purity gap **+0.222** over raw embedding dims — a clean pass of the original
# `mean > 0.80 AND gap > 0.10` bar. Everything in exp0d's working architecture
# stays exactly as-is: alpha curriculum, TopK sparse coding, AuxK dead-slot
# revival, decoder column renorm every 200 steps, fp32 loss casting, optimizer
# LR grouping, the modern `torch.amp.GradScaler("cuda", ...)` API, and all
# eleven F1–F11 fixes from the prior review cycle.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings → Accelerator → GPU T4 x2** (or P100)
# 2. **Settings → Internet → On**  ← required to download data
#
# Expected runtime: ~60–90 minutes total.
#
# ---
# **Variant note:** exp0d proved that 4096 slots / k=16 with two domains
# reaches clean specialization (mean 0.805, gap +0.222). This experiment asks
# the next question: does that success transfer to four distant domains, and
# can a *soft* per-domain hint steer reserved slot blocks without destroying
# the free-region's emergent specialization? The thresholds in the verdict
# logic below are a first reasonable guess — they are not the same hard
# pass/fail bar exp0d hit.

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
    lambda_supervision  = 0.2

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
    for g in opt.param_groups:
        base_lr = cfg.lr if g["name"] == "base" else cfg.lr_registry
        g["lr"] = base_lr * lr_scale(step)

    x, y, dl = get_mixed_batch(cfg.batch_size)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        o = model(x, y, alpha=alpha, step=step, domain_labels=dl)
        loss = (o["lm_loss"]
                + cfg.lambda_recon       * o["recon_loss"]
                + cfg.lambda_auxk        * o["aux_loss"]
                + cfg.lambda_supervision * o["supervision_loss"])

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt); scaler.update()

    if step % 100 == 0 or step == 1:
        dead = model.registry.dead_fraction(step, cfg.dead_after)
        el = time.time() - t0
        print(f"step {step:5d} | alpha {alpha:.2f} | lm {o['lm_loss'].item():.3f} "
              f"| recon {o['recon_loss'].item():.4f} "
              f"| sup {o['supervision_loss'].item():.3f} "
              f"| dead {dead*100:4.1f}% "
              f"| {el:.0f}s")
        hist.append(dict(step=step, alpha=alpha, lm=o["lm_loss"].item(),
                         recon=o["recon_loss"].item(),
                         sup=o["supervision_loss"].item(), dead=dead))

    if step % cfg.eval_every == 0:
        ev = eval_loss(alpha)
        print(f"   >> eval: " + "  ".join(f"{d}={v:.3f}" for d, v in ev.items()))

print(f"\ntraining done in {(time.time()-t0)/60:.1f} minutes")
torch.save(model.state_dict(), f"{OUT}/exp1_model.pt")


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
            correct += (pred == true).sum().item()
            total   += x.size(0)
    model.train()
    return correct / total

clf_acc = reserved_block_accuracy(cfg.eval_batches)
print(f"\nreserved-block classification accuracy (eval): {clf_acc*100:.1f}%")


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
plt.savefig(f"{OUT}/exp1_results.png", dpi=130)
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
with open(f"{OUT}/exp1_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
