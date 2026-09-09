# 8. Related Work

We group prior work by its relation to this paper: work that motivates it, work
that anticipates its findings, and work that objects to its proposed method. We
place the anticipating work first and prominently, because two of those results
substantially overlap with ours and it would be misleading to bury them.

---

## 8.1 Work that anticipates our findings

**Entity-recognition directions and hallucination [4].**
Sparse-dictionary directions at entity tokens have been shown to encode whether a
model can recall facts about that entity — a form of self-knowledge about its own
capabilities — and to be causally relevant, in that intervening on them can induce
refusal on known entities or hallucinated attributes on unknown ones.

Our §3.5 recovers the detection half of this independently, on a different model,
with a baseline comparison at the same token positions that we do not believe has
been reported in that form. We do not recover the causal half: we performed no
intervention in the detection experiment. **This prior result is stronger than
ours and predates it**, and we cite it as the primary reference for the finding
rather than presenting §3.5 as new.

**Read- versus write-directional causal inertness [8].**
A decomposition of feature causal inertness by direction — separating whether a
feature can be *read from* versus *written through* — has been reported, together
with the observation that a large fraction of features passing a correlational
recovery criterion are causally inert. This is the same mechanism our §6.2
reasoning appeals to when distinguishing controls that test the encoder path from
those that test the decoder path. We did not originate that distinction.

---

## 8.2 Work that motivates this paper

**Randomized dictionary baselines [1].**
Recent sanity-check work constructs dictionary variants with randomized and
frozen encoders or decoders and reports that the soft-frozen variant matches
fully-trained dictionaries on interpretability, sparse probing and causal editing
at matched sparsity, concluding that dictionaries in their present form do not
reliably decompose model internals.

We note two properties of that comparison relevant to reading it, both visible in
the reported tables. First, the three headline pairs correspond to *different*
baseline variants — the causal-editing figure to the soft-frozen decoder, the
sparse-probing figure to a frozen decoder — so each is a maximum over five
variants presented as a single result. Second, the causal-editing metric — the RAVEL
disentanglement score [6] — has a floor: a variant with zero explained variance
and chance-level probing still scores approximately 0.485. Read against that floor, the headline comparison of
0.73 against 0.72 becomes 0.24 against 0.23, and in two of three model-layer
settings the fully-trained dictionary is itself close to the floor. This does not
contradict the paper's conclusion — the trained model's above-floor margin is
indeed largely matched — but it substantially changes the effect size a reader
takes away, and it is not stated in the paper's limitations.

**Benchmark reliability [2, 7].**
An audit of a widely used dictionary benchmark suite [7] finds that two of its
metrics fail multiple evaluation lenses at their canonical settings and should
not be used, recommends a third, and explicitly does not propose replacements.
Separately, automated interpretability metrics have been shown not to distinguish
trained from randomly initialized transformers [3].

Together these establish the gap this paper addresses: there is no accepted
causal discriminator for dictionary quality, and the diagnosis is better developed
than the remedy.

---

## 8.3 Work on controls and intervention magnitude

**Magnitude-matched controls [9].**
Matching a control perturbation to the same residual-stream distortion as the
intervention is established practice, and has been used to show that steering
collapse is direction-pattern-dependent rather than magnitude-dependent. Our
`delta_ratio` reporting (§4.1) is an instance of this practice, not a
contribution; we report it because our failure to apply it produced a false
negative, not because the idea is new.

**Structural matching of controls [13].**
Matched-random-control protocols that hold structural properties constant while
varying only concept alignment have been used in attribution-graph work, with the
explicit warning that without such matching the labelled-versus-random gap is
easy to misstate. This is the closest precedent we found for the *philosophy* of
§6.3, though applied to a different object.

**Specificity as side-effect measurement [12].**
Work on inference-time intervention defines specificity as the preservation of
unrelated behaviour — general capability, related-behaviour retention, robustness
under distribution shift. This is a different quantity from the attribution
specificity of §6.1: it asks whether the intervention broke anything else, not
whether the effect is attributable to the intervened feature. That work does not
employ matched controls at equivalent magnitude, which is the gap §6.3 addresses.

**Feature selection for steering [11].**
Separating features that reflect input activations from those that causally
influence output tokens, and selecting on the latter, has been reported to
improve steering substantially. This is the basis of the objection we record in
§6.5: purity is an input-side score, and a reviewer may reasonably ask why
controls should be matched on it.

---

## 8.4 Work on steering and task performance

Inference-time intervention along truth-correlated directions has been shown to
improve accuracy on a truthfulness benchmark substantially [5], establishing that
activation-level intervention can change task performance and not only style.

We flag this specifically because it marks the boundary of our own claims: we did
not measure accuracy under intervention (§7.2), and the existence of that prior
result means the question is neither open nor answered by us.

A more recent result runs the other way, and bears directly on our controls.
Replicating the entity-recognition feature pattern of [4] on a 27B
reasoning-tuned model, [15] reports a single sparse-dictionary latent reaching
0.814 AUROC on known-versus-unknown classification, and then steers on those
features: top-K ablation at K = 200, 0.3% of the dictionary, produces a 4–8σ
effect against a random-K null *while making the model worse* — the
incorrect-answer rate on known entities rises from 62% to 77% and the
correct-answer rate falls from 8% to zero. The paper characterises what the
intervention found as "a hallucination-induction circuit, not a calibration
knob," and states that significance against a random-K null is "necessary but
not sufficient evidence for a calibration mechanism."

We take that as a constraint on our own strongest result rather than as a
finding about it. The setting differs — [15] ablates 200 features on a
reasoning-tuned 27B model, we promote 8 on a 410M base model — so its numbers
do not transfer. What transfers is the inference: §6.6 rests on three controls
returning zero, one of them a norm-matched random direction, and [15] is a
worked case in which clearing exactly that control accompanied a *loss* of
accuracy. Our controls license an attribution to specific decoder columns
(§6.6). They do not license the further claim that steering those columns is
useful, and on the evidence of [15] that further claim would need its own
accuracy measurement before anyone made it.

---

## 8.5 Positioning

Relative to this literature, the present work is an independent replication with
a documented failure catalogue. Its contributions are the six measured pitfalls
(§4), the adversarial audit of its own results (§5), the control taxonomy (§6.2),
and two control constructions we did not find in use in this literature (§6.3) — of which
the shuffled-decoder control produced the attribution result in §6.6 that the
quality-matched control alone could not license.
