# 6.6 Running the taxonomy: results

§6.2 set out five control classes and what each rules out, and §6.5 noted that
two of them had no data in our pipeline. We ran them. This section reports the
result, which strengthens one claim, settles a second, and introduces a
measured caveat that applies to all of them.

**Design.** The identical registry (training is bit-reproducible in this regime,
§3.2) with four arms per cell — domain × strength × repeat:

| arm | construction |
|---|---|
| `targeted` | bias the domain's eight purest slots |
| `control` | bias eight purity-matched slots belonging to *other* domains |
| `random_dir` | bypass the dictionary entirely: inject a random unit vector scaled to the norm the `targeted` arm injected **in that same cell**, redrawn independently per repeat |
| `shuffled_dec` | bias the **same target slot ids**, but decode through a permuted slot-to-column assignment of the same decoder matrix |

288 runs. The encoder and top-k selection are untouched in every arm; only the
write path differs.

---

## 6.6.1 Result on the concept-defined metric

Medicine, the one domain in which this metric has dynamic range:

| strength | `targeted` | `control` | `random_dir` | `shuffled_dec` |
|---|---|---|---|---|
| 0.00 | 0.0 | 0.0 | 0.0 | 0.0 |
| 1.64 | 1.3 | 0.0 | 0.0 | 0.0 |
| 2.19 | **16.0** | 0.0 | 0.0 | 0.0 |
| 2.73 | **22.8** | 0.0 | 0.0 | 0.0 |
| 3.28 | **38.2** | 0.0 | 0.0 | 0.0 |
| 4.37 | **41.7** | 0.0 | 0.0 | 0.0 |

All three controls return zero at every strength, while the targeted arm rises
monotonically to 41.7%.

**The `shuffled_dec` result is the informative one.** That arm biases the *same
slot identifiers*, at approximately the same injected magnitude, through the
*same set of decoder columns* — differing only in which column each slot writes
through. It returns zero. Slot quality and the columns being
learned are therefore not sufficient on their own to produce the effect; what
distinguishes the targeted arm is **the specific slot-to-column
correspondence**. Intervention magnitude is not fully excluded — see the dose
mismatch in §6.6.2.

This is the strongest attribution our pipeline supports. It is stronger than the
claim we made before running these arms (§5.2, R2), which the purity-matched
control alone could not license — and weaker than it would be had the arms been
equally dosed.

**Literature** shows the same ordering at much smaller magnitude (targeted
3.5–3.8 against controls 0.0–0.8). **Law and code** sit at the metric floor in
all four arms and are uninformative here, consistent with §4.3.

---

## 6.6.2 A measured caveat: the arms are not equally dosed

Our own dose-match check failed on **every cell**, at a 5% tolerance:

```
medicine 2.7335 : gap 19.1%   targeted 7.708% | control 7.479% | random_dir 6.233% | shuffled_dec 6.651%
law      1.6401 : gap 29.0%
...                            all cells, gaps 18%–29%
```

The targeted arm consistently injects a larger perturbation than the controls at
the same nominal strength. At the dose–response slope measured in §3.3 — roughly
7.9 points of the metric per point of `delta_ratio` — a 19% relative gap at
`delta_ratio ≈ 7.7%` corresponds to a bias of order **11 percentage points**.

We report this rather than suppress it. It does not account for a separation of
41.7 against 0.0, but it is large enough that a smaller effect measured this way
would be uninterpretable, and it invalidates any fine comparison *among* the
three controls.

**Consequence for the protocol.** Matching the nominal strength across arms is
not sufficient; each arm must be calibrated to a target `delta_ratio`
individually. Our §6.4 already required a tolerance of 0.05; this run shows that
requirement is not automatically satisfied by construction, and must be enforced
by per-arm calibration rather than assumed.

---

## 6.6.3 The feature-defined metric fails, independently

§4.6 withdrew the feature-defined metric from primary analysis on the grounds
that it rewards degenerate output, reasoning from a single domain in an earlier
run. This experiment tests that conclusion on entirely new data, and confirms it.

At strength 2.19, with the metric fixed to the *targeted* domain's slot
vocabulary in all four arms:

| domain | `targeted` | `control` | `random_dir` | `shuffled_dec` |
|---|---|---|---|---|
| medicine | 30.0 | **31.2** | 14.8 | 1.0 |
| literature | 20.7 | 3.7 | **42.3** | 24.8 |
| code | 88.7 | 34.7 | 11.8 | **70.7** |

In medicine the purity-matched control **equals** the targeted arm. In literature
the *random direction* **exceeds** it. In code the shuffled decoder comes close
to it.

The one apparent anomaly in the concept-defined table resolves the same way:
`code / random_dir / 4.37` averaged 9.7% because a single repeat returned 29.0%
with repetition 86.4% and entropy 1.156 nats — flagged as collapsed.

A metric on which a random direction outscores the intended intervention in a
whole domain cannot be used as evidence of steering. We had reached that
conclusion by inspection of one collapsed row; it now rests on an independent
run in which the metric ranks a null above the treatment.

---

## 6.6.4 Completed taxonomy

| control | rules out | our result |
|---|---|---|
| no intervention | "nothing moves anyway" | ✅ 0.0 at `strength = 0` |
| random direction, magnitude-matched | "any vector of this size steers" | ✅ **ruled out** (0.0) |
| other concepts' features, quality-matched | "any good feature steers" | ✅ **ruled out** (0.0) |
| **shuffled decoder** | **"any learned column steers"** | ✅ **ruled out** (0.0) |
| randomized dictionary | "learning the dictionary is unnecessary" | ✗ not run (§6.7) |

**What we can now claim.** On the concept-defined metric, in one domain, and
subject to the dose mismatch of §6.6.2, the pattern is most consistent with the
steering effect arising from the specific slot-to-decoder-column correspondence
rather than from slot quality or from the columns merely being learned. Because
the controls were systematically *under*-dosed relative to the targeted arm, we
cannot fully separate magnitude from correspondence on this data.

**What we still cannot claim.** That a *trained* dictionary is required. That
question needs the randomized-dictionary arm, which we did not run for the reason
given in §6.7.

**Standing limitations on the above.** Dose mismatch of 18–29% across arms;
one domain with dynamic range on the metric; three repeats per cell; a single
training seed.

---

## 6.7 Why we did not run the randomized-dictionary arm

The natural fifth arm — a dictionary trained with a randomly initialized and
constrained decoder — was designed and then dropped before execution, for a
reason we state because it may save others the run.

Under a cosine constraint of τ = 0.8 to a random initialization, a decoder column
can carry at most 60% of its magnitude along the direction an unconstrained
column would learn. At the dose–response slope of §3.3, that amplitude penalty
alone predicts a gap of roughly 13 percentage points between arms — before any
question of dictionary quality enters. Our design would then have been unable to
distinguish "learning matters" from "the constrained arm was quieter," and at a
single training seed per arm its minimum detectable effect exceeded the effect it
predicted.

Separately, our expansion factor of 4 makes random dictionaries measurably less
expressive than at the expansion factors used in the work reporting that
randomized baselines match trained ones. A negative result for the randomized arm
at expansion 4 would not transfer upward, and we would not have been entitled to
present it as a response to that work.

The arm remains the right experiment; it needs an amplitude-matched null arm,
multiple training seeds, and a higher expansion factor to be interpretable.
