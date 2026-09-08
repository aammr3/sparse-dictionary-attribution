# 2. Setup

## 2.1 Model and dictionary

**Base model.** Pythia-410M [14] (EleutherAI), frozen throughout. Every parameter has
`requires_grad = False`, the model is held in evaluation mode, and the optimizer
is constructed over dictionary parameters only. The base model's next-token
cross-entropy was measured on held-out data at intervals during training and
remained stable, confirming that no parameter drift occurred.

**Dictionary.** A TopK sparse dictionary ("registry") over residual-stream
activations at layer 12 of 24:

```
pre   = ReLU( W_enc (h − b_dec) )
acts  = TopK( pre, k )
recon = W_dec acts + b_dec
```

with `n_slots = 4096`, `k = 16`, and an auxiliary top-`k_aux = 32` reconstruction
term (weight 0.03) to revive slots that have stopped firing. Decoder columns are
unit-normalized at initialization and renormalized periodically during training.
The training objective is reconstruction only — no domain labels enter it.

The expansion factor is 4 (4096 slots over 1024 dimensions). This is
substantially lower than is typical in recent dictionary work, and we treat it as
a scope limitation rather than a design choice (§7).

**Domains.** Four text corpora: medicine (PubMedQA), law (BillSum), code
(CodeParrot), literature (Project Gutenberg). Domains define the concept
partition against which slot specialization is measured; they are not used as
training signal.

## 2.2 Measuring specialization

For each slot we count firings per domain over held-out text and define

```
purity(slot) = max_d count[d] / Σ_d count[d]
```

so a slot firing only within one domain has purity 1.0 and a slot firing equally
across four has purity 0.25.

The comparison of interest is against the model's own hidden dimensions. To be
meaningful, this must be made **at matched sparsity**: raw-dimension purity is
computed over the top-16 dimensions per token, the same budget the dictionary
operates under. Comparing a sparse representation against a dense one inflates
the apparent gap and we do not report such comparisons.

## 2.3 Intervention

Steering biases slot scores *before* selection, so that the bias can change which
slots are selected, and injects the difference between the two decodes as a
residual-stream update at the same layer:

```
delta   = decode( TopK(pre + bias) ) − decode( TopK(pre) )
h'      = h + gain · delta
```

At `bias = 0` this is a bit-exact no-op, verified by the fact that purity and
training metrics are identical to the runs without a steering hook installed.

**Magnitude.** Every intervention row reports

```
delta_ratio = ‖ gain · delta ‖ / ‖ h ‖
```

the norm of the injected update relative to the residual stream it is added to,
averaged over token positions. §4.1 explains why we regard reporting this as
mandatory rather than optional.

## 2.4 Generation and fluency gating

Generation uses sampling (`temperature = 0.8`, `top_p = 0.9`). Greedy decoding is
not used anywhere in this work; §4.2 explains why. The nucleus filter includes
the token that crosses the probability threshold rather than excluding it.

Every generated continuation is scored for repetition rate and mean token-level
entropy, and rows exceeding a repetition threshold or falling below an entropy
threshold are flagged as degenerate. §4.6 documents a case where a metric was
maximized by exactly such degenerate output, and §5.2 records the consequence.

## 2.5 Metrics

- **Concept-defined rate** — frequency of terms selected as distinctive to a
  domain by relative frequency against the other three corpora.
- **Feature-defined rate** — frequency of the tokens the intervened slots
  themselves most strongly fire on. Introduced as a mismatch diagnostic (§4.3),
  and restricted to that role after §4.6.
- **AUROC**, polarity-corrected as `max(AUC, 1 − AUC)`, for the detection
  experiments. Polarity correction is necessary because several registry signals
  are inverted-but-informative; it is applied identically to registry signals and
  to baselines, and the permutation test in §3.5 recomputes the maximum over
  signals inside every shuffle so that the correction cannot inflate the reported
  significance.

## 2.6 Detection experiment

For the hallucination experiments the steering bias is identically zero
throughout and the hook is read-only; this is asserted at six points in the
execution path. Signals are recorded over three spans — the entity tokens
specifically, the full prompt, and the generated tokens — with the entity span
located by re-encoding the template prefix, prefix-plus-entity, and the full
prompt, and verifying that the reconstruction agrees.

LM-head baselines (next-token entropy and negative max log-probability) are
computed from the prefill at every context position, from the full softmax before
temperature and nucleus filtering, and sliced into the same spans under two
alignments: at the same positions as the registry signal, and shifted by one so
as to characterize the same tokens. The verdict is required to clear **the
stronger of the two**, so that the choice of alignment cannot favour the
dictionary.

## 2.7 Reproducibility of the runs reported here

Training in this configuration is bit-reproducible on the hardware used (NVIDIA
T4). Across ten independent sessions the purity gap was identical to sixteen
significant figures and per-slot firing counts were identical as integers. All
runs used a single fixed training seed; consequently seed-to-seed variance was
never measured, and no claim in this paper rests on stability across seeds
(§3.2, §5.2 R4).
