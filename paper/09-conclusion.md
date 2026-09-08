# 9. Conclusion

We set out to build a numbered concept registry inside a frozen language model and
to test whether its internal state could be used for control and for diagnosis.
The mechanism works, within limits we can now state precisely. But the more
durable output of the work turned out to be the record of how often, and how
quietly, it appeared to work when it did not.

**What we can state.** Slots in a sparse dictionary trained on a frozen model's
activations specialize by domain without supervision, and the separation improves
with model scale. Writing into a domain's slots changes generation monotonically
in the injected magnitude, with a usable window of roughly 6–8% of the residual
stream norm. In one domain, on a concept-defined metric, that effect survives
three independent controls — a magnitude-matched random direction, quality-matched
features from other domains, and a shuffled decoder that preserves the learned
columns but permutes their assignment — each of which returns exactly zero. Subject
to the dose mismatch of 18–29% we report in §6.6.2, that pattern points to the
specific slot-to-column correspondence as the locus of the effect. And a frozen
model's dictionary state, read while it processes an entity name, discriminates
real from invented entities better than the model's own next-token uncertainty at
the same positions, in one of three non-confounded domains.

**What we cannot state.** That any of this improves accuracy: we measured domain
vocabulary throughout and never measured correctness. That suppression is
available: it saturates structurally, by a factor of 34. That the results extend
beyond a single 410M model, one dictionary architecture, an expansion factor of 4,
and a single training seed. That a *trained* dictionary is required: the arm that
would answer it was designed, costed, and dropped because its outcome was
predictable from geometry alone.

**What we would emphasize to a practitioner.** Four of our six pitfalls produced
false negatives, and each was found by adding a measurement that had been absent —
the injected magnitude, the baseline's own behaviour, a second metric, the
positions being read. The remaining two produced false positives, and neither
could have been found that way. The control really was returning zero; the metric
really was returning 66.7 points. What was wrong was the inference, and only a
deliberate attempt to destroy a result we already believed surfaced it.

That asymmetry is the thing we would carry to the next project. Instrumentation
protects against the failure to see an effect. Nothing but adversarial review
protects against seeing one that is not there — and in our case that review cost one
working session and no additional model training or GPU runs — though it did
consume a substantial number of language-model API calls, since the reviews were
themselves model-assisted — against results that had already been recorded as
settled.

**A closing note on the shape of the claim.** The strongest sentence in this paper
is narrow: one domain, one metric, one model, three controls at zero. We arrived
at it by removing four broader sentences that we had preferred. We think that
trade is the correct one, and that a literature which reported it more often would
be easier to build on.
