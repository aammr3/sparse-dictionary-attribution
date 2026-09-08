# 5. An Adversarial Audit of Our Own Claims

The pitfalls in §4 were found during development, each by adding a measurement
that had been missing. Two of them, however — the floored control (§4.5) and the
collapse-rewarding metric (§4.6) — were not found that way. They were found only
after the results had been recorded as successes, by deliberately attempting to
destroy them.

We describe that process, because it is the part of this work we consider most
reusable, and we report its outcome without softening: **four of our own
recorded claims did not survive it.**

---

## 5.1 Method

Before committing to a further experiment, we submitted the design *and the
established results it rested on* to five independent adversarial reviews, each
assigned a distinct axis and each instructed to attempt refutation rather than
confirmation:

| axis | task |
|---|---|
| **prior art** | find published work that removes the novelty of the proposed contribution |
| **external source** | verify our reading of the paper we were responding to, from its source, and flag any over-reading |
| **implementation** | verify every claim we made about our own code, with line numbers, from the code |
| **design** | attack the experimental design; quantify anything that would make its result uninterpretable |
| **independent replication of the above** | a second executor, given the same brief with the relevant code injected |

Two properties of this setup mattered more than the number of reviewers.

First, **the reviews covered our established results, not only the new design.**
It is normal to review new code; it is less common to re-open conclusions that
have already been recorded as settled. Three of the four claims withdrawn below
came from that re-opening.

Second, **reviewers were required to cite evidence they had themselves
retrieved** — a line number, a log row, a quoted passage — rather than to
summarize our description of it. Two of the four withdrawals were found by
recomputing figures directly from our own run logs, which we had summarized
correctly but compared incorrectly.

---

## 5.2 What did not survive

### R1 — "The result replicated across three seeds"

We had recorded a targeted-vocabulary rate of 22.8% ± 2.7%, described as a
replication of an earlier run's 23.5%.

Recomputation from the logs shows the two figures were measured **at different
intervention strengths**. At the same strength (2.1868):

| run | targeted-vocabulary rate |
|---|---|
| earlier run | **23.5%** |
| later run, three repeats | 15.5%, 16.5%, 16.0% → **16.0% ± 0.5%** |

*(logs22.txt:278–280)*

A 7.5 percentage-point difference between two bit-identical registries measured
at the same dose. Pooling the four available draws gives σ̂ ≈ 3.8 pp — the
reported ±0.5 understates the true dispersion by roughly a factor of eight, which
is the expected behaviour of a standard deviation estimated from three samples
(2 degrees of freedom; SD(s) ≈ 0.52σ).

Further, the three "repeats" varied only the *generation* seed. The training seed
was fixed at a single value in every run reported in this work. **No seed
replication was performed.**

**Withdrawn.** The number is reported in §3.3 as a point on a dose–response
curve, with the curve's monotonicity — not any single cell — as the evidence.

---

### R2 — "The control returned exactly zero, therefore the effect is specific"

Discussed in §4.5. Two independent problems.

The control's value was `0.0` with standard deviation `0.0` in 18 of 18 runs, so
the difference statistic was numerically identical to the raw targeted value and
carried no additional information.

And the control slots were themselves *other domains' target slots* — medicine's
controls included code's fourth-ranked, law's second-ranked and literature's
seventh-ranked target slots. The comparison therefore tests cross-domain leakage.
It is a real test with a real answer, but it is passed by any dictionary with
distinct per-domain directions, including a randomized one, and so cannot support
a claim about learned structure.

**Restricted, not withdrawn.** We now state the result as specificity *against
another domain's slots at matched purity and matched magnitude*, and not as
evidence that a learned dictionary is required. §6.5 sets out the additional
controls that would be needed for the stronger claim.

---

### R3 — "The cleanest separation in the project"

A 52.2% versus 1.7% separation in the law domain, on the feature-defined metric.

The same metric produces its largest value anywhere in our data — 66.7
percentage points in the code domain — on a generation with repetition 95.8% and
entropy 0.75 nats, flagged as collapsed in all three repeats. The metric counts
the tokens the boosted slots fire on; for law and code these are function words
and punctuation; a degenerate repetition of such a token scores near-perfectly.

**Withdrawn from primary analysis.** The metric is retained only as a
mismatch diagnostic (§4.3), always reported alongside repetition and entropy.

---

### R4 — "Training is not reproducible even at fixed seed"

We had carried this forward as an established property of the pipeline, and had
been reading single-run figures as draws from a wide distribution accordingly.

It is false for the regime in which the results of this paper were obtained.
Across ten independent sessions and seven distinct codebases, the purity gap was
identical to sixteen significant figures and per-slot firing counts were
identical as integers.

The non-reproducibility we had observed (final dead-slot fraction ranging 11% to
48% at fixed seed) belonged to an **earlier architecture that no longer exists**
in this pipeline: a dictionary trained *jointly* with a from-scratch transformer
over 6000 steps. Training only a dictionary on a fully frozen pretrained model is
a far better-conditioned problem, and in our runs it was deterministic.

**Withdrawn.** The correction cuts both ways: we had been applying an
inapplicable caution, and simultaneously had never measured the seed variance
that does matter. §3.2 states this.

---

### R5 — The decision rule's false-positive rate was uncontrolled

Not a single claim but the rule that generated several. Our automatic verdict
declared a "clean window" where the targeted condition exceeded both its baseline
and its control by more than twice the pooled standard deviation, with n = 3 per
group.

With n = 3, that threshold is equivalent to `t > 2.45` on 4 degrees of freedom —
a two-sided **p ≈ 0.070** — and it was applied across 4 domains × 2 metrics × 6
strengths = **48 tests with no correction**. The expected number of spurious
clean windows under the null is **3.4**, and the family-wise error rate, if any
single window is treated as a success, is **0.97**.

Our run reported clean windows in three of four domains. That count is fully
consistent with noise under this rule.

**Consequence.** Verdicts produced by that rule are treated as exploratory
throughout this paper. The one confirmatory claim we retain (§3.5) uses a
pre-specified aggregation, an entity-level independent unit, a permutation test
that recomputes the maximum over signals inside each shuffle, and Bonferroni
correction across domains.

---

## 5.3 What survived, and why the distinction matters

Six findings survived unchanged (§3.1–3.5, and the suppression-saturation
measurement). They share a property that the withdrawn claims did not: each rests
either on a **monotone trend across several points** or on a **quantity measured
identically across many independent runs**, rather than on a single cell compared
against a single reference.

The withdrawn claims, by contrast, were all of the form *"this number is large
and this other number is small."* That form is exactly what a floored control, an
unmatched dose, an uncorrected multiple comparison, or a collapse-rewarding
metric will produce for free.

---

## 5.4 A note on cost and on what this section is for

The audit cost approximately one working session and no additional model training
or GPU runs, though it consumed a substantial number of language-model API calls,
as the reviews were model-assisted. It removed four
claims, one of which we had described in our own records as the cleanest result
in the project.

We report it in full for two reasons. The first is simply that the claims were
wrong and the record should say so. The second is that the *distribution* of what
it caught is informative: the four false negatives in §4 were each found by
adding a measurement, but **both false positives were found only by attacking
results we already believed.** No amount of additional instrumentation would have
surfaced them, because the instruments were reporting correctly — a control
really was returning zero, and a metric really was returning 66.7 points. What
was wrong was the inference drawn from those readings.

We suggest that a deliberate adversarial pass over *settled* results, not only
over new code, belongs in the standard practice for causal-specificity claims,
and that its cost is low relative to the cost of publishing a result that a
reviewer will overturn.
