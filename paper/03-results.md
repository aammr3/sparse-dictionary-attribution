# 3. Results

This section reports the findings that survive the adversarial audit described
in §5. Four claims that appeared in earlier internal versions of this work were
withdrawn or restricted by that audit; they are listed in §5.2 rather than here,
and the restrictions are carried into the statements below.

Every number is traceable to a run log; references are given inline.

---

## 3.1 Slots specialize without supervision

Trained with a reconstruction objective alone — no domain labels, no auxiliary
classification term — registry slots separate by domain far more cleanly than
the model's own hidden dimensions do at the same sparsity.

| | registry slots | raw hidden dims |
|---|---|---|
| mean purity | **0.805** | 0.583 |
| gap | **+0.222** | — |

The comparison is made at matched sparsity: the raw-dimension purity is computed
over the top-16 dimensions per token, the same budget the registry is given.
Without this matching the comparison is not meaningful, since a dense
representation is being compared against a sparse one.

The effect is monotone in dictionary capacity, over three successive
configurations at fixed sparsity `k = 16`:

| slots | 1024 | 2048 | 4096 |
|---|---|---|---|
| mean purity | 0.766 | 0.786 | **0.805** |

**A hypothesis we rejected.** Our prior expectation was the opposite of what we
found. We expected that *reducing* the number of simultaneously active slots
would force cleaner specialization, and tested `k = 8` at fixed capacity. Every
measured quantity degraded: mean purity fell to 0.708, the purity gap fell from
0.187 to 0.089, and language-model loss worsened. Increasing capacity, not
increasing sparsity pressure, is what produced specialization in our setting.

---

## 3.2 The mechanism transfers to a frozen pretrained model, and improves with scale

We repeated the procedure on frozen Pythia models, training only the registry on
mid-network activations while every model parameter was held fixed
(`requires_grad = False`, verified; cross-entropy of the base model measured
stable throughout training).

| | Pythia-160M | Pythia-410M |
|---|---|---|
| purity gap over raw dims | +0.288 | **+0.317** |
| dead slots at end of training | 5.1% | **0.5%** |
| live slots | 2377 / 4096 | **3527 / 4096** |

Both the separation and the utilization of the dictionary improve with model
scale. Two data points do not establish a trend, and we do not claim one; we
report the direction and note that it is consistent with the expectation that
richer representations admit cleaner sparse decomposition.

**Reproducibility.** Training in this regime is bit-reproducible on the hardware
used. Across **ten independent sessions spanning seven distinct codebases**, the
reported purity gap was identical to sixteen significant figures
(`0.31747859716415405`), as were per-slot firing counts as integers. We note
this explicitly because an earlier phase of this work — in which the dictionary
was trained *jointly* with a from-scratch transformer — was strongly
non-reproducible at fixed seed, and we had incorrectly carried that expectation
forward. See §5.2, R4.

A caveat follows directly from the same fact: because the seed was held fixed at
a single value throughout, **seed-to-seed variance in this regime has never been
measured.** No claim in this paper depends on stability across seeds.

---

## 3.3 Writing into slots changes generation, monotonically in the injected magnitude

Biasing a domain's purest slots before top-k selection, and injecting the
resulting difference of decodes as a residual-stream update, produces a monotone
dose–response relationship between the relative magnitude of the update and the
rate of targeted vocabulary in the continuation:

| `delta_ratio` | 0% | 2.8% | 6.1% | 7.7% | 9.0% | 11.4% |
|---|---|---|---|---|---|---|
| targeted-vocabulary rate | 0.0% | 1.3% | 16.0% | 22.8% | 38.2% | 41.7% |

Monotonicity across six points is the relevant evidence here, not any single
value; individual cell values carry substantial variance (§5.2, R1).

Fluency was checked on every row via repetition rate and token-level entropy,
under sampling-based decoding. The operating region in which the intervention
produces measurable movement without degrading fluency is approximately
`delta_ratio ∈ [6%, 8%]`; below roughly 1% the intervention is not detectable
(§4.1), and above roughly 10% generation degrades.

---

## 3.4 Suppression is structurally saturated

The same mechanism applied in the negative direction — biasing a domain's slots
*downward* so they fall out of the top-k selection — does not produce a
comparable intervention. Measured at identical settings:

| direction | maximum achieved `delta_ratio` |
|---|---|
| promotion | **12.29%** |
| suppression | **≤ 0.36%** |

a factor of **34**, and the suppression figure does not increase with strength.
The mechanism is straightforward: removing k slots from a top-k selection
promotes the next k in rank order, and their reconstruction is nearly
equivalent, so the difference between the two decodes remains small no matter
how negative the bias becomes.

This is a limitation of the intervention mechanism, not of the dictionary. Any
application requiring suppression rather than promotion needs a different
mechanism — for example forcing activations to zero after selection rather than
biasing scores before it.

---

## 3.5 The registry's internal signature at entity tokens exceeds the model's own uncertainty signal

We tested whether the registry's internal state predicts ungroundedness *before
generation*, using a read-only hook (steering bias identically zero throughout,
asserted at six points in the code) on a frozen Pythia-410M.

The contrast is between prompts naming real entities and prompts naming invented
ones, where the invented case is ungrounded by construction. Signals are computed
over three spans — the entity tokens, the full prompt, and the generated tokens —
and compared against the model's own LM-head baselines (next-token entropy and
negative max log-probability) computed at the **same positions**, in two
alignments, with the verdict required to clear the stronger of them.

**Automated verdict.** At the entity span, the best registry signal exceeds the
strongest of four LM-head baselines by **+0.139** discrimination
*(logs27.txt:1121)*. At the generated-token span the margin is +0.047, close to
the noise floor — consistent with §4.4, where measuring only at generated tokens
returned a complete null.

**Independent entity-level analysis.** Because the entity-span signals are read
from the deterministic prefill, they are bit-identical across generation repeats
— verified for all 240 entities. The effective independent unit is therefore the
entity, not the run, and any interval computed over runs would be too narrow by a
factor of √3. Re-analysing at the entity level (30 real vs 30 invented per
domain), aggregating each signal over the whole entity span so that no threshold
on span length is selected post hoc:

| domain | best registry signal | disc. | strongest baseline | disc. | gap | 95% CI | perm. *p* | signals beating baseline |
|---|---|---|---|---|---|---|---|---|
| **literature** | `prompt_domain_share` | **0.837** | `neg_max_logprob_state` | 0.636 | **+0.201** | **[+0.020, +0.344]** | **0.0015** | **5 / 7** |
| law | `max_act` | 0.750 | `lm_entropy_token` | 0.623 | +0.127 | [−0.070, +0.267] | 0.0285 | 2 / 7 |
| code | `max_act` | 0.653 | `neg_max_logprob_state` | 0.713 | −0.060 | [−0.149, +0.128] | 0.93 | 0 / 7 |
| ~~medicine~~ | `prompt_domain_share` | 0.958 | `neg_max_logprob_token` | 0.744 | +0.213 | [+0.081, +0.331] | 0.0005 | *excluded — confounded* |

The permutation test recomputes the maximum over all seven registry signals
*inside every label shuffle*, so the reported *p*-value already accounts for the
post-hoc selection of the best signal.

**Confound control.** Medicine is excluded by an automatic gate, not by
inspection. Real disease names are common English words that tokenize to a single
in-context BPE token (` asthma`, ` stroke`, ` migraine`); invented proper nouns
cannot. The in-context token-length gap between the real and invented lists is
0.767 for medicine against a 0.5 threshold, versus 0.133, 0.167 and 0.033 for the
other three domains. The experiment prints the flag and reports the restricted
result separately *(logs27.txt:1186–1208)*.

**What this supports, and what it does not.** In one of three non-confounded
domains the result is robust: the confidence interval excludes zero, the
permutation *p* survives Bonferroni correction across the three domains
(α = 0.0167), and five of seven independent signals clear the strongest
baseline. Law is suggestive but not robust. Code fails, consistent with its
behaviour throughout this work: its purest slots fire on punctuation and fire an
order of magnitude less often than other domains'.

We do not claim a general hallucination detector. We claim a measured instance,
in one domain, in which a frozen model's sparse-dictionary state at the moment it
reads an entity is more informative about ungroundedness than the model's own
next-token uncertainty at the same positions.

**Second test.** A cloze-style correct-versus-incorrect test was included and
returned no usable signal: Pythia-410M produced zero exact-match correct answers
across all 360 runs under sampling, leaving a single label class. The
experiment's degeneracy check detected this and reported an explicit honest
negative rather than propagating undefined AUROC values *(logs27.txt:1161,
1173)*.
