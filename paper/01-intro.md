# 1. Introduction

Sparse dictionaries trained on the activations of a frozen language model are
now a standard tool for obtaining interpretable, and putatively causal, handles
on model internals. The premise is that writing into a dictionary's slots
constitutes an intervention on a *concept*, and that the resulting behavioural
change can be attributed to that concept.

**This paper is about the second half of that premise.** Demonstrating that an
intervention changes behaviour is routine. Demonstrating that the change is
attributable to the intervened feature — rather than to the act of intervening,
to the magnitude injected, or to the feature merely being a learned one — requires
controls that rule out each of those alternatives separately. We set out what
those controls are, run four of them, and report the attribution they license.

That premise is currently under pressure. Recent work reports that randomized
dictionary variants match fully-trained ones on interpretability scores, sparse
probing, and causal editing, and concludes that dictionaries in their present
form "do not reliably decompose models' internal mechanisms" [1]. A separate audit
of a widely used benchmark suite [7] finds that several of its metrics "should not
be used to evaluate SAEs," and proposes no replacement [2]. A third line of work
finds that automated interpretability metrics fail to distinguish trained from
randomly initialized transformers at all [3].

The common thread is not that dictionaries do not work. It is that **the field
does not currently have an agreed way to tell whether a given dictionary's
handles are meaningful**, and the metrics in use are too weak to separate
learning from the exploitation of random structure at scale.

## What this paper contributes

We do not resolve that question. We report, from a complete pipeline built to
answer a different question, a set of concrete failure modes that lie between a
correct implementation and a correct conclusion — and a self-audit that removed
four of our own results.

Specifically:

1. **Six measured evaluation pitfalls** (§4), each quantified from our own runs.
   Four produced false negatives — effects we recorded as absent that were
   present. Two produced false positives — numbers we recorded as evidence that
   were artifacts. Each is presented with the reasoning that led us to the wrong
   conclusion, the measurement that revealed it, and the transferable rule.

2. **An adversarial audit of our own settled results** (§5), which overturned or
   restricted four recorded claims, including the one we had described in our
   internal record as the cleanest result in the project. We report the audit's
   method, its findings, and — because it is diagnostic — the observation that
   the two false positives could not have been caught by adding instrumentation,
   only by attacking conclusions we already believed.

3. **A control taxonomy and a proposed protocol** (§6), organized around what
   each control class actually rules out. We identify two constructions for which
   we did not find in use in this literature — controls matched on a *measured feature-quality
   score*, and a *shuffled-decoder* control that preserves the learned columns
   but permutes their assignment — run four of the five classes (§6.6), and show
   from our own data why the quality-matched control alone carries no
   information.

4. **The results that survive** (§3), stated with the restrictions the audit
   imposed. The strongest of these is that a frozen model's sparse-dictionary
   state, read at the moment it processes an entity name, discriminates real
   from invented entities better than the model's own next-token uncertainty at
   the same positions — in one of three non-confounded domains.

## What this paper is not

It is not a novelty claim. The mechanisms we build on are established, and where
prior work anticipates our findings we say so explicitly (§8) — including one
paper that reports entity-recognition directions of the kind we recover in §3.5 [4],
and another that separates causal read- from write-inertness along the same lines
as our own mechanistic reasoning [8].

It is a case study, on a single 410M-parameter model, four text domains, and one
dictionary architecture. Its value, if it has any, is that the failure modes are
measured rather than hypothesized, and that they are reported by the people who
fell into them.

## Why report failures at this level of detail

Each pitfall in §4 cost us at least one full experimental run, and in one case
five. Two of them produced results we recorded internally as successes and later
withdrew. None of them are exotic: uncalibrated intervention magnitude, greedy
decoding, a metric that does not measure what the mechanism moves, measurement at
the wrong token positions, a control that cannot vary, and a metric that rewards
degenerate output.

We expect these to be common precisely because they are individually easy to
miss and collectively invisible in published work — a paper that fell into any
of them would report either a clean null or a clean effect, and in neither case
would the underlying failure be visible to a reader.
