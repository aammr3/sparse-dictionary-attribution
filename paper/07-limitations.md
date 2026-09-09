# 7. Limitations

We state these at the level of detail we would want from a paper we were
reviewing. Several of them are load-bearing enough that a reader should treat the
corresponding claims as provisional.

## 7.1 Scope

**One model, one size.** Pythia-410M throughout, with a single earlier comparison
point at 160M. The scale trend we report in §3.2 rests on two points and is
reported as a direction, not a trend.

**One dictionary architecture.** TopK only. We did not test JumpReLU, BatchTopK,
gated variants, transcoders or crosscoders.

**Expansion factor 4.** 4096 slots over 1024 dimensions. Much recent dictionary
work operates at expansion 32 or higher, where random dictionaries are
measurably more expressive. Conclusions drawn at expansion 4 do not transfer
upward, and in particular any comparison against randomized-dictionary baselines
would be biased in our favour at this expansion factor. We do not make such a
comparison for that reason.

**Four domains, one language.** Domains are coarse (medicine, law, code,
literature) and English-only. Slot "concepts" are only as fine-grained as the
partition used to measure them.

## 7.2 The most important limitation

**We measured domain vocabulary, not correctness.**

Every steering result in this paper measures the rate at which domain-associated
tokens appear in the continuation. At no point did we measure whether the
model's answers became more *accurate*. We have no evidence that intervening on
a domain's slots improves task performance, and it would be a misreading of this
work to conclude that it does.

The intervention demonstrably changes register, vocabulary and formatting. Whether
it changes what the model knows or gets right is untested here.

## 7.3 Statistical limitations

**Seed variance is unmeasured.** All runs used a single fixed training seed. The
training is bit-reproducible in this regime (§3.2), which means our repeats
carried no information about seed-level variability. No claim rests on stability
across seeds, and none should be read as doing so.

**Small samples in the detection experiment.** Thirty real and thirty invented
entities per domain. Confidence intervals are correspondingly wide, and the
result that survives correction does so in one domain of three.

**The exploratory verdict rule.** The automatic rule used through most of this
work corresponds to an uncorrected p ≈ 0.070 applied 48 times (§5.2, R5). Every
verdict it produced is treated as exploratory. Only the analysis in §3.5 is
confirmatory, and it is pre-specified, entity-level, permutation-tested with the
maximum recomputed inside each shuffle, and Bonferroni-corrected.

**One domain excluded by confound gate.** Medicine, the numerically strongest
domain in the detection experiment, is excluded automatically because real
disease names are common single-token English words and invented proper nouns
cannot be. This is the correct action but it removes the strongest data.

## 7.4 Measurement and implementation caveats

**No attention mask on the padded prefill.** Generation uses left padding with an
end-of-text token and no attention mask, so the model attends to pad positions.
This is applied identically across all conditions and both to registry signals
and to baselines, so it is not a differential bias — but it is a departure from
best practice and we state it rather than leave it implicit.

**Decoder renormalization is periodic, not continuous.** Decoder columns are
renormalized at intervals during training rather than every step, and the saved
model is one optimizer step past the last renormalization. No decoder-norm
diagnostic was logged. For the results reported here this affects nothing we
measure, but it would need fixing before any cross-condition comparison that
depends on injected magnitude being comparable by construction.

**Control arms were not equally dosed.** In the four-arm experiment of §6.6
the injected magnitude differed between arms by 18–29% at the same nominal
strength, against a 5% tolerance, in every cell. At the measured dose–response
slope this is worth of order 11 percentage points of the metric. It does not
account for the separation we report (41.7 against 0.0), but it invalidates fine
comparisons among the controls and would make a smaller effect uninterpretable.
Per-arm calibration to a target `delta_ratio`, rather than a shared nominal
strength, is required.

**The feature-defined metric is retained only as a diagnostic.** §4.6 shows it is
maximized by degenerate output. It appears in this paper as evidence of metric
mismatch and nowhere as evidence of successful steering.

## 7.5 Relation to prior work

This work is an independent replication of established findings, with tighter
controls in some places and, as §5 documents, looser ones in others than we had
believed. It is not a novelty claim.

Two prior results in particular anticipate parts of it: sparse-dictionary
directions that detect whether a model recognizes an entity, reported with causal
interventions we did not perform; and a decomposition of causal inertness into
read- and write-directional components, which is the mechanism our own reasoning
in §6.2 appeals to. Both predate this work and we cite them as such (§8).

The one component we could not find precedent for is the quality-matched control
of §6.3, together with the shuffled-decoder control of §6.2. Both were run
(§6.6); the quality-matched control, taken alone, returns the metric floor and
carries no information, while the shuffled-decoder control is the one that
licenses the attribution claim we make.

## 7.6 A note on the research log

The repository includes the project's working research log, in Arabic
(`research-log.ar.md`), with an English translation alongside it
(`research-log.en.md`); the Arabic file is the original and governs where the
two differ, and the translation is provided for accessibility only. The log is
an unedited development diary, written informally and in real time, not a polished
document; it is included as evidence of the process described in §5, including
the four withdrawn claims. Executor and reviewer roles in it are referred to by
neutral labels rather than by product name, because the incidents recorded there
are single-sample, time-bound, and in several cases attributable to transport or
tooling rather than to any model's behaviour.

## 7.7 What would change our conclusions

- A run at expansion 32 could invalidate any comparison we make against
  randomized-dictionary results.
- Measuring seed-level variance could widen every interval reported here.
- Implementing the two-sided metric of §6.3 could show that our surviving
  specificity result is smaller than reported, since the floored metric cannot
  express a control performing *worse* than chance.
- A test of task accuracy under steering could show that the intervention we
  characterize as control is cosmetic. In the closest published setting the
  outcome was worse than cosmetic: [15] reports accuracy degrading under
  dictionary steering that had cleared a random-direction control of the kind
  we use (§8.4). We have not run such a test.
