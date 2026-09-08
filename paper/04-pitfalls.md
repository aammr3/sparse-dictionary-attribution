# 4. Six Measured Pitfalls

We report six evaluation failures encountered while building a sparse-registry
steering and hallucination-detection pipeline on a frozen Pythia-410M. Each one
individually produced either a false negative (an effect we wrongly reported as
absent) or a false positive (a number we wrongly reported as evidence). All six
were caught, and all are quantified from our own runs; log line references are
given so that each claim is traceable.

We present them in the order a practitioner is likely to meet them: first the
failures that corrupt the *measurement apparatus*, then those that corrupt the
*conclusion drawn from it*.

---

## 4.1 An uncalibrated intervention magnitude produces a false negative

**Symptom.** Our first steering experiment on the frozen model biased the two
purest slots per domain across a strength grid of `[0, 0.5, 1, 2, 4] × θ`, where
`θ` was the empirical top-k selection threshold. Across all four domains and all
strengths, the targeted-vocabulary rate was flat at approximately zero. Fluency
was healthy throughout (repetition 9–32%, entropy 4.2–5.2 nats), so the null
could not be attributed to degenerate generation.

**The conclusion we drew, and it was wrong.** We recorded the result as a
negative: *writing into registry slots does not steer a frozen pretrained model
in any measurable direction.* The reasoning appeared sound — the intervention
was applied, the model kept generating fluently, and nothing moved.

**What was actually happening.** We had never measured how large the
intervention was *relative to the signal it was competing with*. We added a
diagnostic, `delta_ratio`, defined as the norm of the injected residual-stream
update over the norm of the residual stream itself:

```
delta_ratio = || gain · (decode(topk(pre + bias)) − decode(topk(pre))) || / || h ||
```

The failed experiment had been injecting a perturbation of roughly **0.2% of the
residual stream norm**. The intervention was not ineffective; it was *inaudible*.
When we widened the intervention to eight slots per domain and swept a gain
factor, the same architecture, the same model and the same slots produced a
monotone dose–response curve:

| `delta_ratio` | 0% | 2.8% | 6.1% | 7.7% | 9.0% | 11.4% |
|---|---|---|---|---|---|---|
| targeted-vocabulary rate | 0.0% | 1.3% | 16.0% | 22.8% | 38.2% | 41.7% |

The effect had been there the whole time, below the threshold of measurement.

**Fix.** Report `delta_ratio` (or an equivalent relative-magnitude measure) for
every intervention row, and calibrate the strength grid so that it spans the
region where the quantity is on the order of a few percent. In our setting the
usable window was roughly 6–8%; below ~1% nothing is measurable, and above ~10%
fluency degrades.

> **Transferable lesson.** A null steering result is uninterpretable unless the
> intervention magnitude is reported. "We intervened and nothing happened" and
> "we did not intervene enough to be detectable" are indistinguishable without
> it, and the second is the more common of the two.

*Evidence: exp4 (all-zero sweep) vs exp4b (`delta_ratio` diagnostic and gain
sweep).*

---

## 4.2 Greedy decoding manufactures the collapse you are trying to detect

**Symptom.** Our collapse detector flagged *every* non-zero intervention
strength, in every domain, in both intervention directions, as having broken
fluency. There was no strength small enough to escape it.

**The conclusion we drew, and it was wrong.** We recorded that the intervention
mechanism destroys generation coherence at any dose, and therefore that no clean
operating window exists.

**What was actually happening.** Generation used deterministic argmax decoding.
Small language models are well known to fall into repetition loops under greedy
decoding, and ours did — *before any intervention at all*. At `strength = 0.0`,
with the steering bias identically zero and the intervention path a bit-exact
no-op, the measured repetition rate was:

```
BOOST,    strength = 0.0 :  rep = 77.4%   (all four domains, shared neutral prompt)
SUPPRESS, strength = 0.0 :  rep = 69.8% – 88.9%
```
*(logs16.txt:240–258 and 271–289)*

Our collapse threshold was 50%. The unintervened baseline was already at 77%.
The detector was not measuring the effect of our intervention; it was reporting
a property of our decoding strategy.

**Fix.** Replace greedy decoding with sampling (`temperature = 0.8`,
`top_p = 0.9`). Under the corrected decoder the same baseline condition measured
5.0–18.6% repetition, and across all 48 intervention rows — every domain, every
strength, both directions — **zero rows were flagged as collapsed**.

**Cost.** Five runs were contaminated before this was found, and two results we
had reported as positive had to be withdrawn, because the old detector's
threshold logic had silently exempted the lowest non-zero strengths from the
collapse check.

> **Transferable lesson.** Measure your control condition first. If the
> unintervened baseline already fails your quality gate, the gate is measuring
> your pipeline, not your intervention.

---

## 4.3 Metric mismatch: the intervention worked, and we were looking elsewhere

**Symptom.** After the magnitude fix, the intervention moved the targeted
metric in one domain and left it at exactly zero in the other three.

**The conclusion we drew, and it was partly wrong.** We recorded that steering
works for one domain and fails for the rest, and began looking for properties of
the failing domains that would explain it.

**What was actually happening.** Our metric counted the rate of *distinctive*
domain vocabulary — terms selected by relative frequency against the other three
corpora (e.g. `subparagraph`, `Secretary` for law). But inspection of the slots
we were boosting showed that the purest slots in those domains fire on
*function words and punctuation* (`the`, `(`, `shall`, `this` for law;
`:`, `)`, `.`, newline for code). Boosting such a slot changes the register and
formatting of the text, not its rare content words.

Adding a second metric — the rate of the tokens that *the boosted slots
themselves* most strongly fire on — revealed movement of up to tens of
percentage points in domains where the original metric had been flat at zero.
The intervention had been working in three of four domains; the metric had been
measuring the wrong thing in two of them.

**Fix.** Report both a *concept-defined* metric and a *feature-defined* metric.
Their divergence is itself informative: it localizes the mismatch between what
the feature encodes and what the evaluator counts.

> **Transferable lesson.** When an intervention shows no effect, the first
> question is not "why is the mechanism failing" but "is the metric measuring
> the quantity the mechanism actually moves."

**A caution we impose on ourselves.** The feature-defined metric turned out to
have a serious failure mode of its own, described in §4.6. It should be reported
alongside the concept-defined metric as a diagnostic, never used alone as
evidence of success.

---

## 4.4 Measuring at the wrong token span yields a total null — and the baseline is the clue

**Symptom.** Our hallucination-detection experiment returned AUROC ≈ 0.5 for
every registry signal, at every value of N, on both tests. A complete null.

**Why this null was diagnosable.** The experiment included a mandatory baseline:
the frozen model's own next-token entropy and negative max log-probability, read
from its LM head on the same runs. **That baseline was also at 0.5.**

This is the key observation. Had the registry been at chance while the model's
own uncertainty signal discriminated, the correct conclusion would have been
that the registry carries no information the model does not already expose. But
both were at chance simultaneously — and the model's own uncertainty is known to
carry *some* signal about unfamiliar entities. Two independent measurement
instruments returning chance at the same time indicates that the quantity is not
present *where both are looking*.

**What was actually happening.** Our read-only hook recorded only the final
position of each forward pass, which meant we were measuring at the last prompt
token and at the generated tokens. But the moment at which a model can be said
to encounter an entity it does not know occurs while it *reads that entity*,
during the prefill. The entity tokens were never measured at all.

**Fix.** Record all positions of every forward pass and compute signals over
three explicit spans: the entity tokens, the full prompt, and the generated
tokens. Under the corrected measurement, on the same model and the same
registry, polarity-corrected discrimination at the entity span reached 0.65–0.84
in the non-confounded domains, while the generated-token span remained at chance
— exactly as before.

**Result.** The final automated verdict, comparing against the stronger of four
LM-head baselines computed at the same positions, reported a margin of **+0.139**
at the entity span *(logs27.txt:1121)*.

> **Transferable lesson.** Always include a baseline you expect to work. Its
> failure is not a nuisance — it is the most informative single bit in a null
> result. A registry-only null is ambiguous; a *joint* null localizes the problem
> to the measurement position rather than the instrument.

---

## 4.5 A control pinned at the metric floor carries no information

**Symptom.** Our strongest reported result was a comparison against a
purity-matched control: eight slots drawn from *other* domains, greedily matched
to the target slots' measured purity (mean absolute purity gap 0.000 for three
of four domains) and driven at a matched injected magnitude. The control
returned 0.0% ± 0.0% at every strength and in every repeat, against 22.8% ± 2.7%
for the targeted slots. We reported this as decisive.

**What an adversarial audit of our own result found.** The control's value was
`0.0` with standard deviation `0.0` in **18 of 18 runs**. It is therefore
*numerically identical* to the raw targeted measurement:

```
specificity_gap = hit(target) − hit(control) = hit(target) − 0 ≡ hit(target)
```

The control added reassurance, not information. Any comparison it licenses is a
comparison the raw number already licenses.

**A second problem, structural rather than numerical.** Inspection of the
selected control slot identifiers showed that they were themselves *other
domains' target slots*: medicine's controls included code's fourth-ranked target
slot, law's second, and literature's seventh. The question the control actually
answers is therefore *"if I steer toward a different domain, do this domain's
words appear?"* — a test of cross-domain leakage. It is a legitimate question,
and the answer is informative. But it does not license the claim we attached to
it, because **any dictionary with distinct per-domain directions passes it,
including a randomized one.**

**Fix.** Two changes. First, use a metric with dynamic range in both directions
(for example the log-odds of domain-vocabulary mass in the next-token
distribution) so that a control can be measurably below, at, or above the target.
Second, report multiple controls that rule out different hypotheses; a single
control conflates them (see §6.5).

> **Transferable lesson.** A control that cannot move is not a control. Before
> reporting a control-based claim, check the control's variance: if it is
> identically zero, the metric is floored and the control is decorative.

**Revised claim.** We now state this result as: *the effect is specific to the
targeted domain's slots as against another domain's slots at matched purity and
matched magnitude* — and explicitly not as evidence that the effect requires a
learned dictionary.

---

## 4.6 A metric that rewards fluency collapse

**Symptom.** The feature-defined metric introduced in §4.3 produced the single
largest specificity margin anywhere in our data: **66.7 percentage points** in
the code domain, with a t-statistic of 38.

**What was actually happening.** That row's generation had
`repetition = 95.8%` and `entropy = 0.75 nats` — near-deterministic emission of a
single token, flagged as collapsed in all three repeats. The feature-defined
vocabulary for code's purest slots consists of punctuation and whitespace
(`:`, `)`, `.`, newline). A generation that degenerates into repeating one such
token scores near-perfectly on a metric that counts exactly those tokens.

The metric was not measuring successful steering. It was measuring collapse, and
rewarding it.

**Why this matters beyond one row.** The result we had described in our own
records as the cleanest in the project — a 52.2% versus 1.7% separation in the
law domain — is produced by the same metric, on slots whose feature vocabulary
is likewise function words and punctuation. We withdraw it from our primary
analysis.

**Independent confirmation.** This conclusion was reached from a single
collapsed row. A later experiment (§6.6.3) tested it on entirely new data, with
the metric held fixed to the targeted domain's vocabulary across four arms. In
one domain a *random direction* scored 42.3 against the targeted arm's 20.7; in
another, a purity-matched control equalled the targeted arm. A metric that ranks
a null above the treatment cannot serve as evidence of steering, and the
withdrawal stands on independent grounds.

**Fix.** Gate every metric on a fluency criterion computed on the same
generation, and report the collapse rate per condition as a co-primary outcome.
Where two conditions have different collapse propensities — which is precisely
the case when comparing a trained dictionary against a randomized one — a
collapse-rewarding metric cannot be used for comparison at all.

> **Transferable lesson.** Ask of every metric: what score does a degenerate
> output receive? If the answer is "a high one," the metric is unusable for any
> comparison in which the conditions differ in their tendency to degenerate.

---

## 4.7 Summary

| # | Pitfall | Direction of error | Quantified evidence |
|---|---|---|---|
| 4.1 | Uncalibrated intervention magnitude | **False negative** | injected 0.2% of residual norm; effect appears from ~6% |
| 4.2 | Greedy decoding | **False negative** | baseline repetition 77.4% at zero intervention |
| 4.3 | Metric mismatch | **False negative** | 3 of 4 domains moved on the feature-defined metric only |
| 4.4 | Wrong measurement span | **False negative** | AUROC 0.5 including the baseline; 0.65–0.84 at the entity span |
| 4.5 | Control pinned at the floor | **False positive** | control 0.0 ± 0.0 in 18/18; gap ≡ raw value |
| 4.6 | Metric rewards collapse | **False positive** | 66.7 pp margin at entropy 0.75, repetition 95.8% |

Four of the six produced false negatives — results we would have published as
"the mechanism does not work." Two produced false positives — results we did
publish internally as successes and later withdrew. We note that the false
negatives were each caught by adding a *measurement*, while the false positives
were caught only by an adversarial review of results we already believed
(§5).

> **The single most reusable item in this section** is the diagnostic in §4.4:
> when a null result includes a baseline you expected to work, the joint failure
> localizes the problem to *where* you are measuring, not to *what* you are
> measuring with. That observation converted a complete null into the strongest
> positive result in our pipeline.
