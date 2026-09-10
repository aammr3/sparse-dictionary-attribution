# Sparse-Dictionary Steering on a Frozen LM — Experiments, Logs, and a Failure Record

Code, notebooks, and complete run logs for the paper:

> **Attributing Steering Effects to Specific Dictionary Slots**
> *A control protocol for a frozen language model, and what survived auditing it*
> Amr Elsaeed · [`paper/attributing-steering-effects.pdf`](paper/attributing-steering-effects.pdf)

---

## What this is

A TopK sparse concept dictionary ("registry") trained on the residual-stream
activations of a **fully frozen Pythia-410M**, across four text domains. The
project asked three questions in sequence: do slots specialize without
supervision, does writing into them steer generation, and does their internal
state predict hallucination before it happens.

The answers are in the paper. Its central result is an **attribution**: a steering
effect that survives three independent controls — a magnitude-matched random
direction, quality-matched features from other concepts, and a shuffled decoder
that biases the same slots through the same learned columns under a permuted
assignment — each returning exactly zero.

Getting to a defensible version of that result meant discarding several
indefensible ones. So this repository also contains **29 complete run logs**,
including the runs behind results we later withdrew, and a research log of all 79
events — decisions, errors, and four claims an adversarial audit overturned.

---

## Headline results

| | |
|---|---|
| Slots specialize without supervision | purity **0.805** vs **0.583** for raw dimensions at matched sparsity |
| Transfers to a frozen pretrained model, improves with scale | gap **+0.288** (160M) → **+0.317** (410M) |
| Steering is monotone in injected magnitude | `delta_ratio` 0→11.4% ⇒ target rate 0→41.7% |
| Effect survives **three** independent controls | random direction, quality-matched features, shuffled decoder — **all exactly 0.0**, in the one domain where the metric has range; the arms were not equally dosed (§6.6.2) |
| Dictionary state at entity-read time beats the model's own uncertainty | **+0.139** over the strongest of four LM-head baselines — **in 1 of 3 non-confounded domains** (§3.5) |
| Suppression saturates structurally | promotion 12.29% vs suppression **≤ 0.36%** — a factor of 34 |

**And what it does not show:** we measured domain vocabulary, never accuracy.
Nothing here demonstrates that steering makes answers more correct.

---

## Repository layout

```
exp0/    from-scratch toy model — does specialization happen at all?
exp1/    reserved slot ranges + supervision  (supervision collapses the dictionary)
exp2/    steering on the toy model  (greedy decoding artifact found here)
exp3/    transfer to frozen Pythia-160M
exp3b/   transfer to frozen Pythia-410M
exp4/    first steering attempt on the frozen model — negative, and wrongly so
exp4b/   delta_ratio diagnostic — explains exp4's null
exp4c/   repeats + purity-matched control
exp5/    hallucination detection — total null, including the baseline
exp5b/   measurement moved to entity tokens
exp5c/   dual-alignment LM-head baselines + confound gate
exp9/    four-arm control taxonomy

logs*.txt  29 raw Kaggle run logs, unedited (at repository root)
paper/   paper sources (markdown), references (prose + BibTeX), built PDF
research-log.ar.md   full research log — 79 numbered events, in Arabic (the original)
research-log.en.md   English translation of the research log
```

Each `expN/` contains a `*_source.py`, a generated `*_kaggle.ipynb`, and the
notebook generator. The notebooks are self-contained: they download the model
and datasets, train, evaluate, and print their own verdict.

`exp0/exp0_source.py` and its notebook carry Arabic comments; every later
experiment is in English. This is deliberate and left as-is: that file is the
version that actually produced `logs.txt`, and translating it now would break
the correspondence between the published source and the log it generated.

---

## Reproducing

Every experiment runs end-to-end in a single Kaggle notebook session.

```
Accelerator : GPU T4 x2   (P100 is not supported — sm_60 fails)
Internet    : On          (downloads Pythia + datasets from HuggingFace)
Runtime     : ~20 min (exp3b) to ~47 min (exp9)
```

Upload `expN/expN_kaggle.ipynb` via **File → Import Notebook**, run all.

**Training was bit-identical across our ten Kaggle T4 sessions**, spanning seven
distinct codebases: the purity gap matched to sixteen significant figures
(`0.31747859716415405`) and per-slot firing counts matched as integers. Runs on
other GPU microarchitectures or driver versions may differ in low-order digits. All runs used a single fixed training seed, so
seed-to-seed variance is *unmeasured* — see §7.3 of the paper.

---

## The raw results payload

The hallucination-detection experiment's full output — all 1080 runs, the
per-span AUROC tables, the confound report and the verdicts — is attached to
the [v1.0 release](https://github.com/aammr3/sparse-dictionary-attribution/releases/tag/v1.0)
as `exp5c_halluc.json` (15 MB). It is not committed, because it would sit in
the history of every clone.

It is the payload behind §3.5: the entity-level reanalysis was run on it
locally, on CPU. Read `label_counts` before the numbers — TEST 2 is degenerate
(360 positive, zero negative) and medicine is flagged as confounded.

---

## The logs

`logs*.txt` (29 files, repository root) are raw and unedited, including the failed runs. They are the
evidence base for the paper's §4 and §5; the numbers quoted in those sections are
traceable to lines in them.

Runs worth reading if you only read a few:

| log | what it shows |
|---|---|
| `logs16.txt` | greedy decoding: **77.4% repetition at zero intervention** (§4.2) |
| `logs20.txt` | the false negative — steering "fails" at a 0.2% injected magnitude (§4.1) |
| `logs21.txt` | the same setup at 6.18% — the effect appears (§4.1) |
| `logs22.txt` | the run whose headline result the audit later restricted (§5.2) |
| `logs23.txt` | AUROC ≈ 0.5 for everything **including the baseline** (§4.4) |
| `logs27.txt` | the corrected detection result, `+0.139` (§3.5) |
| `logs28.txt` | four-arm control taxonomy (§6.6) |

---

## The research log

[`research-log.ar.md`](research-log.ar.md) is a chronological record of the
whole project — 79 numbered events, in Arabic. It contains the reasoning behind
each decision, the 16 code and data errors caught before they corrupted a
result, and the adversarial audit that withdrew four claims we had recorded as
settled. An English translation is provided alongside it as
[`research-log.en.md`](research-log.en.md); the Arabic file is the original and
the one the paper's §7.6 refers to.

It is included deliberately. The paper argues that two of its six pitfalls could
only be caught by attacking results one already believes; this file is what that
looked like in practice.

---

## Known limitations

- One model (Pythia-410M), one dictionary architecture (TopK), **expansion factor 4**
- Four coarse English domains
- **Domain vocabulary measured; accuracy never measured**
- Suppression does not work — saturates by a factor of 34
- The code domain fails throughout, for a measured reason (its purest slots fire
  on punctuation, an order of magnitude less often than other domains')
- Seed variance unmeasured; single training seed throughout
- Control arms in `exp9` were not equally dosed (18–29% gaps against a 5%
  tolerance) — reported in §6.6.2 rather than suppressed

---

## Citation

```bibtex
@misc{elsaeed2026attributing,
  title     = {Attributing Steering Effects to Specific Dictionary Slots},
  author    = {Elsaeed, Amr},
  year      = {2026},
  publisher = {figshare},
  doi       = {10.6084/m9.figshare.33516322},
  url       = {https://doi.org/10.6084/m9.figshare.33516322},
  note      = {Preprint. Code and logs:
               https://github.com/aammr3/sparse-dictionary-attribution}
}
```

---

## License

Code — the `exp*/` sources, the generated notebooks and the notebook generators:
**MIT**, see [LICENSE](LICENSE).

Everything else — the paper in `paper/`, the raw run logs `logs*.txt`, and the
research log in both languages: **CC BY 4.0**. Reuse it with attribution.

The point of publishing the logs and the log was that they could be checked;
having no licence at all would have made checking them legally awkward.

---

## Contact

Amr Elsaeed — `amrelsaeed225@gmail.com`

Corrections are welcome, particularly to §5 and §6. Four claims in this work were
withdrawn after adversarial review; if something else here does not hold, it
should be said.
