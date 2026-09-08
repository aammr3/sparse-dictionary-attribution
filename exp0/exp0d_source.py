# %% [markdown]
# # Experiment Zero — Do the registry slots specialize?
#
# **The single question:** If we drop a sparse dictionary layer into a small
# transformer and train it on two distant domains (Python code + Wikipedia text),
# will the slots split cleanly across the two domains — or will they blend?
#
# **Crucially:** This experiment runs **without any supervision**. Nobody tells
# the model "these slots are for code". If specialization happens, it happens
# on its own — and that is the foundation the rest of the project rests on. If
# it doesn't, there's no point continuing.
#
# **The decisive metric:** We compare registry-slot purity against raw-embedding
# purity at exactly the same sparsity. If the registry is clearly purer, the
# mechanism is doing real work.
#
# ---
# ### Required Kaggle settings before running
# 1. **Settings → Accelerator → GPU T4 x2** (or P100)
# 2. **Settings → Internet → On**  ← required to download data
#
# Expected runtime: 50–80 minutes total.
#
# ---
# **Variant d note:** This is exp0c re-run with `n_slots` doubled again, from
# 2048 to 4096, and `k` held at 16 (the value that has performed best across
# all three prior runs). It comes after a three-run trend on real Kaggle T4
# GPUs at 6000 steps: exp0 (n_slots=1024, k=16) reached mean purity 0.766 with
# gap +0.187; exp0b (n_slots=1024, k=8) collapsed to mean purity 0.708 and
# gap +0.089 — the lower-`k` hypothesis was rejected, because giving the
# registry less per-token capacity ballooned `recon_loss` and poisoned both
# language quality and specialization; exp0c (`n_slots` doubled to 2048, `k`=16) then
# recovered and improved on every metric, reaching mean purity 0.786 and
# gap +0.197, sitting only 0.014 short of the 0.80 mean-purity success
# threshold (`mean > 0.80 and gap > 0.10`). The capacity-confirmation trend
# is the highest-probability lever left, so this run asks one question: does
# continuing it push mean purity decisively past 0.80? Everything else
# (domains, steps, alpha schedule, losses, eval logic, output filenames) is
# identical to exp0c. This will run in a fresh Kaggle session, so no filename
# collisions with the prior runs.

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
    n_slots        = 4096    # 16x expansion (doubled again from exp0c's 2048)
    k              = 16      # how many slots stay active per token (restored to baseline value)

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
    lambda_recon = 1.0
    lambda_auxk  = 0.03
    k_aux        = 32
    dead_after   = 800        # a slot that hasn't fired in this many steps = dead

    # --- eval ---
    eval_every    = 500
    eval_batches  = 40
    eval_loss_batches = 10

cfg = CFG()
print(json.dumps({k: v for k, v in vars(CFG).items() if not k.startswith("_")},
                 indent=2, ensure_ascii=False))


# %% [markdown]
# ## 1. Data — two domains as far apart as possible
#
# Python code vs. raw Wikipedia. We picked them because if the separation doesn't
# show up here, it won't show up anywhere.

# %%
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

DOMAINS = {
    "code": dict(path="codeparrot/codeparrot-clean-valid", name=None,
                 split="train", field="content"),
    "wiki": dict(path="Salesforce/wikitext", name="wikitext-103-raw-v1",
                 split="train", field="text"),
}

def stream_texts(spec, limit=None):
    """Reads texts from HuggingFace via streaming (no full dataset download)."""
    ds = load_dataset(spec["path"], spec["name"], split=spec["split"], streaming=True)
    n = 0
    for ex in ds:
        t = ex[spec["field"]]
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
    print("training tokenizer on a mix of both domains...")
    t0 = time.time()

    def mixed_iter():
        gens = {k: stream_texts(v, limit=cfg.tokenizer_docs // 2)
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
    """Half code, half wiki per batch — keeps the two domains balanced."""
    half = bs // 2
    xc, yc = get_batch("code", "train", half)
    xw, yw = get_batch("wiki", "train", bs - half)
    return torch.cat([xc, xw]), torch.cat([yc, yw])


# %% [markdown]
# ## 2. The registry layer — heart of the experiment

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
# ## 3. The model — small transformer with the registry injected

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

    def forward(self, idx, targets=None, alpha=0.0, step=None, want_acts=False):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)
        mask = self.mask[:T, :T]

        acts_out, recon_loss, aux_loss = None, x.new_zeros(()), x.new_zeros(())

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

        logits = self.head(self.ln_f(x))
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      targets.reshape(-1))
        return dict(logits=logits, lm_loss=lm_loss, recon_loss=recon_loss,
                    aux_loss=aux_loss, acts=acts_out)


model = TinyGPTWithRegistry(cfg).to(DEVICE)
n_model = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("registry"))
n_reg   = sum(p.numel() for n, p in model.named_parameters() if n.startswith("registry"))
print(f"model params     : {n_model/1e6:.2f}M")
print(f"registry params  : {n_reg/1e6:.2f}M")


# %% [markdown]
# ## 4. Training

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

    x, y = get_mixed_batch(cfg.batch_size)
    with torch.autocast("cuda", dtype=torch.float16, enabled=(DEVICE == "cuda")):
        o = model(x, y, alpha=alpha, step=step)
        loss = (o["lm_loss"]
                + cfg.lambda_recon * o["recon_loss"]
                + cfg.lambda_auxk  * o["aux_loss"])

    opt.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    scaler.step(opt); scaler.update()

    if step % 100 == 0 or step == 1:
        dead = model.registry.dead_fraction(step, cfg.dead_after)
        el = time.time() - t0
        print(f"step {step:5d} | alpha {alpha:.2f} | lm {o['lm_loss'].item():.3f} "
              f"| recon {o['recon_loss'].item():.4f} | dead {dead*100:4.1f}% "
              f"| {el:.0f}s")
        hist.append(dict(step=step, alpha=alpha, lm=o["lm_loss"].item(),
                         recon=o["recon_loss"].item(), dead=dead))

    if step % cfg.eval_every == 0:
        ev = eval_loss(alpha)
        print(f"   >> eval: " + "  ".join(f"{d}={v:.3f}" for d, v in ev.items()))

print(f"\ntraining done in {(time.time()-t0)/60:.1f} minutes")
torch.save(model.state_dict(), f"{OUT}/exp0_model.pt")


# %% [markdown]
# ## 5. The decisive measurement — slot purity
#
# For each slot: how often did it fire on code vs. wiki?
#
# `purity = max(code, wiki) / (code + wiki)`
#
# - `0.5` = fully random (slot fires on both equally)
# - `1.0` = perfect specialization
#
# **Fair comparison:** We apply the exact same metric to the raw embedding
# dimensions at the same sparsity (same `k`). If the registry isn't clearly
# purer, the dictionary added nothing.

# %%
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
def purity_stats(counts, min_activity=50):
    c = torch.stack([counts["code"], counts["wiki"]])   # [2, N]
    total = c.sum(0)
    live = total > min_activity
    pur = c.max(0).values[live] / total[live]
    return dict(
        n_live=int(live.sum()),
        n_total=len(total),
        mean=pur.mean().item(),
        median=pur.median().item(),
        frac_above_80=(pur > 0.80).float().mean().item(),
        frac_above_95=(pur > 0.95).float().mean().item(),
        values=pur.cpu().numpy(),
    )

P_slots = purity_stats(slot_cnt)
P_dims  = purity_stats(dim_cnt)

print("="*62)
print(f"{'':22} {'registry slots':>16} {'raw dims':>16}")
print("-"*62)
print(f"{'live units':22} {P_slots['n_live']:>10}/{P_slots['n_total']:<5} "
      f"{P_dims['n_live']:>10}/{P_dims['n_total']:<5}")
print(f"{'mean purity':22} {P_slots['mean']:>16.3f} {P_dims['mean']:>16.3f}")
print(f"{'median purity':22} {P_slots['median']:>16.3f} {P_dims['median']:>16.3f}")
print(f"{'purity > 0.80':22} {P_slots['frac_above_80']:>15.1%} {P_dims['frac_above_80']:>15.1%}")
print(f"{'purity > 0.95':22} {P_slots['frac_above_95']:>15.1%} {P_dims['frac_above_95']:>15.1%}")
print("="*62)


# %%
gap = P_slots["mean"] - P_dims["mean"]
print(f"\nmean-purity gap: {gap:+.3f}\n")

if P_slots["mean"] > 0.80 and gap > 0.10:
    verdict = "✅ pass — slots specialized on their own, and the registry is clearly purer than the raw embedding."
    nxt = "move on to experiment 1: 4 domains + full dictionary + named reserved ranges."
elif P_slots["mean"] > 0.70 and gap > 0.05:
    verdict = "🟡 weak-positive signal — there is specialization, but it's not decisive."
    nxt = "try: lower k (8), more slots (2048), or longer training before judging."
else:
    verdict = "❌ fail — slots did not specialize in a useful way."
    nxt = ("walk through this checklist: (1) is the dead-slot fraction high? "
           "(2) did alpha ramp up too fast and the model collapse? (3) is recon_loss too high? "
           "if all three are fine, the idea needs fundamental rethinking before any further work.")

print(verdict)
print("next step:", nxt)


# %%
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 3, figsize=(16, 4))

steps = [h["step"] for h in hist]
ax[0].plot(steps, [h["lm"] for h in hist], label="LM loss")
ax0b = ax[0].twinx()
ax0b.plot(steps, [h["alpha"] for h in hist], "r--", alpha=.6, label="alpha")
ax0b.set_ylabel("alpha", color="r")
ax[0].set_title("language-model loss + alpha schedule"); ax[0].set_xlabel("step")

ax[1].plot(steps, [h["dead"]*100 for h in hist], color="darkorange")
ax[1].set_title("dead-slot fraction %"); ax[1].set_xlabel("step")

bins = np.linspace(0.5, 1.0, 26)
ax[2].hist(P_dims["values"],  bins=bins, alpha=.6, label="raw dims", color="gray")
ax[2].hist(P_slots["values"], bins=bins, alpha=.7, label="registry slots", color="teal")
ax[2].axvline(0.5, color="k", ls=":", lw=1)
ax[2].set_title("purity distribution — registry vs. raw")
ax[2].set_xlabel("purity"); ax[2].legend()

plt.tight_layout()
plt.savefig(f"{OUT}/exp0_results.png", dpi=130)
plt.show()


# %% [markdown]
# ## 6. Reading the slots — the "aha" moment
#
# For each of the most active slots: the tokens that activate it most. You
# should see clearly code-flavored slots (`def`, `import`, `self`, `return`)
# and clearly prose-flavored slots — that's the eyeball proof, not just a
# number.

# %%
c = torch.stack([slot_cnt["code"], slot_cnt["wiki"]])
total = c.sum(0)
pur_all = torch.where(total > 0, c.max(0).values / total.clamp(min=1), torch.zeros_like(total))
dom_all = c.argmax(0)

for label, dom_id in [("code slots", 0), ("prose slots", 1)]:
    mask = (dom_all == dom_id) & (total > 200) & (pur_all > 0.85)
    idx = torch.nonzero(mask).flatten()
    if len(idx) == 0:
        print(f"\n### {label}: not enough pure slots")
        continue
    idx = idx[total[idx].argsort(descending=True)][:6]

    print(f"\n{'='*58}\n### {label}\n{'='*58}")
    for s in idx.tolist():
        top_tok = tok_score[s].topk(12).indices.tolist()
        words = [repr(tok.decode([t]))[1:-1] for t in top_tok]
        print(f"slot {s:4d} | purity {pur_all[s]:.2f} | fires {int(total[s]):6,} "
              f"| {' · '.join(words[:10])}")


# %%
summary = dict(
    verdict=verdict,
    slots=dict(mean=P_slots["mean"], median=P_slots["median"],
               live=P_slots["n_live"], above_80=P_slots["frac_above_80"]),
    raw_dims=dict(mean=P_dims["mean"], median=P_dims["median"]),
    gap=gap,
    final_dead_frac=hist[-1]["dead"],
    final_lm_loss=hist[-1]["lm"],
    config={k: v for k, v in vars(CFG).items() if not k.startswith("_")},
)
with open(f"{OUT}/exp0_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
