# Attributing Steering Effects to Specific Dictionary Slots

*A control protocol for a frozen language model, and what survived auditing it*

**Amr Elsaeed**
Independent Researcher
`amrelsaeed225@gmail.com`

September 2026

---

**Code and data availability.** All experiment sources, generated notebooks, and
the complete run logs for every experiment reported here (29 runs) are available
at: <https://github.com/aammr3/sparse-dictionary-attribution>. The research log
documenting all 81 recorded events, including the four withdrawn claims of §5, is
included in that repository.

---

## Abstract

When writing into a sparse dictionary's slots changes a frozen language model's
output, what licenses the claim that the *concept* those slots encode caused the
change? We give a control protocol for that question, run it, and report what it
licenses and what it does not.

The question is live because the field currently has no agreed way to determine
whether a given dictionary's handles are meaningful: recent work
reports that randomized dictionary variants match trained ones on standard
evaluations [1], an audit of a widely used benchmark finds several of its metrics
unfit for use [2], and automated interpretability metrics have been shown not to
distinguish trained from randomly initialized models [3]. The diagnosis is better
developed than the remedy.

**Our main result.** On a frozen Pythia-410M with a TopK concept registry (4096
slots, k = 16) over four text domains, a targeted intervention rises monotonically
to 41.7% on a concept-defined metric while **three independent controls return
exactly zero at every strength**: a magnitude-matched random direction,
quality-matched features belonging to other concepts, and a **shuffled-decoder**
control that biases the same slots through the same learned columns under a
permuted assignment. The last of these is the informative one, and we did not
find it in use in this literature. Subject to a dose mismatch we report rather
than suppress (18–29% against a 5% tolerance), the pattern is most consistent
with the effect arising from the specific slot-to-column correspondence rather
than from slot quality or from the columns merely being learned.

**Why the controls are the contribution.** Getting to a result we could defend
required removing several we could not. We document **six evaluation pitfalls**, each
quantified from our own runs, that individually produced a false negative or a
false positive in our results: uncalibrated intervention magnitude, greedy
decoding artifacts, metric mismatch, measurement at the wrong token span, a
control pinned at the metric floor, and a metric that rewards fluency collapse.
Four produced effects we recorded as absent; two produced numbers we recorded as
evidence and later withdrew.

We then subject our own settled results to an adversarial audit, which **overturns
or restricts four recorded claims**, including the one our internal record
described as the cleanest result in the project. We report what survives, and
note that the two false positives could not have been found by adding
instrumentation — the instruments were reporting correctly; the inference drawn
from them was wrong.

We organize the control landscape by what each class rules out, and identify two
constructions we did not find used in this literature: controls matched on a
*measured feature-quality score*, and the shuffled-decoder control above
(permutation controls are long established in other settings). We further report
that the feature-defined metric ranks a random direction *above* the intended
intervention in one domain — an independent confirmation of the sixth pitfall,
on new data.

This is not a novelty claim. Where prior work anticipates our findings we say so.
Its contribution is that the failure modes are measured rather than hypothesized,
and reported by the people who fell into them.

---

**Keywords:** sparse autoencoders · activation steering · evaluation methodology ·
causal specificity · controls · negative results

---

## Contributions

1. **An attribution result** (§6.6): a steering effect that survives three
   independent controls at zero, including a shuffled-decoder control we did not
   find in use in this literature — reported with the dose mismatch that limits it.
2. **Six quantified evaluation pitfalls** from the same pipeline (§4), each with
   the reasoning that produced the wrong conclusion, the measurement that revealed
   it, and a transferable rule.
3. **An adversarial audit of our own settled results** (§5), reported in full,
   which overturned or restricted four claims and whose selectivity is itself
   diagnostic.
4. **A control taxonomy** organized by what each class rules out (§6.2), with two
   constructions we did not find used in this literature (§6.3), four of five run
   empirically (§6.6).
5. **A diagnostic for null results** (§4.4): when a null includes a baseline you
   expected to work, the joint failure localizes the problem to the measurement
   position rather than the instrument. Applying it converted a complete null into
   the strongest positive result in our pipeline.
6. **Results that survive the audit** (§3), stated with the restrictions it
   imposed — including a frozen model's dictionary state at entity-read time
   discriminating real from invented entities better than the model's own
   next-token uncertainty at the same positions, in one of three non-confounded
   domains, at n = 30 per class and a single training seed.
