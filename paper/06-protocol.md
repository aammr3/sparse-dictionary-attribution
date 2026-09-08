# 6. A Control Protocol for Causal-Specificity Claims

§4.5 and §5.2 established that our own control, despite being carefully
constructed, licensed a weaker claim than we attached to it. This section sets
out what we think a control for a causal-specificity claim has to do, organized
around what each control class rules out.

The framing is deliberately modest: we propose a protocol and identify the two
components of it that we did not find in use in this literature. §6.6 reports the result of
running four of its five controls, and §6.5 and §6.7 state what remains untested
and why.

---

## 6.1 The claim being controlled

A causal-specificity claim has the form:

> *Intervening on feature set F changes behaviour B, and the change is
> attributable to F rather than to the act of intervening.*

The second clause is where the work is. It decomposes into a sequence of
progressively stronger negations, and a single control cannot establish more than
one of them.

---

## 6.2 Control taxonomy

| control | rules out | used in |
|---|---|---|
| no intervention | "nothing moves anyway" | most steering work |
| random direction, magnitude-matched | **"any vector of this size steers"** | established practice [9] |
| randomized dictionary (frozen / soft-frozen decoder) | **"learning the dictionary is unnecessary"** | recent sanity-check work [1] |
| **other concepts' features, quality-matched** | **"any good feature steers"** | **not found in this literature** |
| **shuffled decoder** (same columns, permuted assignment) | **"any learned column steers"** | **not found in this literature** |

The first three negate hypotheses about the **mechanism**: whether anything
happens, whether direction matters, whether training matters. The last two negate
hypotheses about **attribution**: whether *this* feature, as opposed to any
comparable one, is responsible.

Neither group substitutes for the other. A result that clears a random-direction
control has shown that direction matters; it has not shown that the *particular*
direction is the concept it is labelled with.

---

## 6.3 The unclaimed component: quality-matched controls

A search across the dictionary and steering literature returned no instance of a
control set selected by **matching on a measured feature-quality score**. We
claim only that we did not find one in this literature; permutation- and
matching-based controls are long established elsewhere, including in saliency-map
sanity checks.
Existing controls are either constructed nulls (random directions, permuted
signs, subspace projections) or ablated training variants. Controls drawn from
*real learned features chosen to be comparable on a measured axis* appear not to
have been used.

The construction is simple. For each target feature, select from the pool of
features belonging to other concepts the one whose measured quality is closest,
greedily and without replacement, and record the per-feature matching gap so that
match quality is auditable from the log alone. In our runs this produced a mean
absolute purity gap of 0.000 for three of four domains.

**And it is not sufficient on its own.** Two failures observed in our own
implementation must be designed out:

1. **The metric must have dynamic range in both directions.** Our control
   returned exactly the metric's floor in every run, which made the difference
   statistic numerically identical to the raw targeted value (§4.5). A control
   that cannot move is decorative. A metric such as the log-odds of
   domain-vocabulary mass in the next-token distribution admits movement in both
   directions and makes a floored control impossible.

2. **Matching must cover the variables known to be causal.** We matched on purity
   and used firing frequency only as a tie-breaker, leaving per-slot frequency
   gaps in the thousands — while our own analysis had identified firing rarity as
   the mechanism behind one domain's failure to steer. A variable known to drive
   the outcome cannot be left unmatched. We propose matching jointly on quality
   and on log firing frequency.

---

## 6.4 Proposed protocol

1. **Report intervention magnitude** for every row, relative to the signal
   intervened upon, and **match it across conditions** to a tight tolerance.
   Compare dose–response *curves* rather than a single operating point selected
   after seeing the outcome. Our own verdict rule admitted a 35% magnitude
   mismatch, which at the measured slope of the dose–response curve is worth up
   to 21 percentage points of the metric — larger than the effect being measured.

2. **Use a two-sided metric** with range in both directions, and gate it on a
   fluency criterion computed on the same generation. Report collapse rate per
   condition as a co-primary outcome (§4.6).

3. **Report at least three controls**: magnitude-matched random direction,
   quality-matched other-concept features, and shuffled decoder. They rule out
   different things and the reader needs all three.

4. **Match on every variable known to be causal**, not only the headline one.

5. **Vary the training seed, not only the generation seed**, and report the
   seed-level variance separately. Our own repeats varied only the generation
   seed and were reported as replication (§5.2, R1).

6. **Pre-specify one primary endpoint** and correct the rest for multiple
   comparisons. Our unadjusted rule had a family-wise error rate of 0.97 across
   48 tests (§5.2, R5).

7. **Diversify the prompt.** Our steering measurements used a single degenerate
   context, leaving prompt variance entirely unmeasured.

---

## 6.5 What remains untested, and two objections we cannot yet answer

We have since run four of the five controls in §6.2; the results are in §6.6.
What remains untested is the randomized-dictionary arm, for the reason given in
§6.7, and the two-sided metric of §6.3, which is not implemented. The protocol is
therefore partly validated: its control taxonomy has been exercised and produced
a result the single-control version could not license, while its dose-matching
requirement was measurably violated in our own run (§6.6.2) — which is itself
evidence that the requirement needs enforcing rather than assuming.

Two objections are on record and we do not consider them answered:

**Magnitude may be the wrong matching variable.** Recent work [10] argues that
matching the norm of a perturbation fixes its Fisher norm while leaving its
alignment with the behavioural gradient free — so norm-matched directions "need
not be comparably dosed" in the sense a specificity audit requires. If this is
right, `delta_ratio` matching is necessary but not sufficient, and an
off-target-KL criterion should be reported alongside it.

**Quality may be the wrong matching axis.** Work [11] separating "input features"
(which reflect input activations) from "output features" (which causally
influence output tokens) reports substantial steering gains from selecting on the
latter. Purity, as we compute it, is an input-side score. A reviewer is entitled
to ask why controls should be matched on an input-side quantity when the claim
being controlled is about output-side effect. We think the answer is that
matching should be on the axis used to *select* the targets — and we selected on
purity — but this is an argument, not a result.
