# Running Experiment One (Exp1) on Kaggle 🚀

A quick, practical, step-by-step guide to running the first experiment for the built-in Concept Registry on Kaggle for free, extending to 4 domains and testing named reserved ranges under partial supervision.

---

## 1. The goal of the experiment (in two or three lines)

Experiment one (Exp1) builds on the result achieved in experiment zero (where the sparse registry specialized automatically between two domains with no supervision, at a purity of 0.805 and a gap of 0.222).
In Exp1 we widen the scope to **4 distant domains** (medicine, law, code, literature) and test a new mechanism: reserving **1024 slots** out of 4096 (split into 4 equal blocks, 256 slots per domain) with a light steering supervision (a Soft Auxiliary Supervision Loss) to encourage them to specialize in their named domain, while the remaining **3072 slots** are left entirely free with no supervision so the automatic discovery can replicate. The goal is to demonstrate the registry's ability to combine supervised steering by name with free concept discovery at the same time.

---

## 2. Running it on Kaggle, in detail

The experiment is designed to run in a single session (expected time: **60 to 90 minutes**) on Kaggle's free GPUs.

1. **Create the notebook:**
   - Sign in to your [Kaggle](https://www.kaggle.com) account.
   - From the sidebar or the home page, click **`+ Create`** then choose **`New Notebook`**.
2. **Upload the experiment code:**
   - From the notebook's top menu choose **`File`** -> **`Import Notebook`** (or `Upload Notebook`).
   - Upload the [`exp1_kaggle.ipynb`]() file directly (or paste the [`exp1_source.py`]() code into the notebook's cells).
3. **Configure the settings (Settings panel in the right sidebar):**
   - **Accelerator:** choose **`GPU T4 x2`** **exclusively** (do not use `GPU P100`: the `sm_60` architecture is unsupported in the PyTorch version used at the time these experiments were run — August 2026).
   - **Internet:** set it to **`On`** ⚡ *(mandatory for downloading the dataset from HuggingFace and training the tokenizer)*.
   - **Persistence:** leave it at the default (`No persistence`).
4. **Start training:**
   - Click **`Run All`** in the top toolbar, or click **`Save Version`** and choose **`Run & Save All (Commit)`** to run the whole notebook in the background even if you close the browser.
   - Watch the printed steps to confirm the loss is falling, `alpha` is ramping, the reserved-range classification accuracy is progressing, and the dead-slot fraction is dropping.

---

## 3. Expected outputs in `/kaggle/working`

These files are saved automatically in the working directory at the end of the run:

| File name | Type | What it is and its role in the experiment |
|---|---|---|
| `tokenizer.json` | JSON | The trained tokenizer (Byte-level BPE, Vocab=8192) over the mixture of all four domains. |
| `medicine.npy` | NumPy Binary | The tokenized medical text array (uint16), for pulling batches straight from memory. |
| `law.npy` | NumPy Binary | The tokenized legal text array (uint16). |
| `code.npy` | NumPy Binary | The tokenized code text array (uint16). |
| `literature.npy` | NumPy Binary | The tokenized literary/novel text array (uint16). |
| `exp1_model.pt` | PyTorch Weights | The full model weights after training (the base model + the 4096-slot `ConceptRegistry` layer). |
| `exp1_results.png` | PNG image | A chart showing: (1) the LM loss trajectory against the `alpha` ramp, (2) the reserved-range accuracy and the dead-slot percentage, (3) a comparison of the purity distribution between the registry and the raw dimensions. |
| `exp1_summary.json` | JSON | The full final report: the automated `verdict`, the three headline numbers (reserved accuracy, free-region purity, and whole-registry purity), and the experiment's settings. |

---

## 4. If this happens... then the problem is this (troubleshooting)

| Problem or error message | Likely cause | Quick fix |
|---|---|---|
| `device: cpu` or a "no GPU" warning | The notebook is set to CPU. | Open Settings in the sidebar and choose `GPU T4 x2`. |
| `ConnectionError` / the dataset fails to download | Internet access is disabled in the notebook. | In Settings in the sidebar, make sure **`Internet: On`** and re-run the cell. |
| `CUDA out of memory` (OOM) | Excess memory use from repeated cells in the session. | Click `Restart Session`. If it recurs, reduce `batch_size` in the `CFG` from 48 to 32. |
| GPU quota exhausted / the session drops | The 30 weekly hours are used up, or the session stopped for idleness. | Run the notebook via `Save Version -> Run & Save (Commit)` to avoid browser-disconnection problems, or switch to another Kaggle account. |
| The data-preparation step is slow | HuggingFace text streaming depends on server speed. | Normal for the first 5–10 minutes; once the four `.npy` files are created and saved, training proceeds at full speed straight from disk. |

---

## 5. How to read the experiment's result (`exp1_summary.json`)

Unlike experiment zero, which relied on a single purity metric, experiment one's report produces **three headline numbers**:

1. **Reserved-block domain accuracy:**
   - Measures how well the four reserved slot blocks (256 slots per domain = 1024 slots in total) can predict and identify the text's true domain.
   - It represents the supervised-steering signal and confirms that the ranges reserved by name actually responded to the domain assigned to them.
2. **Free-region purity & gap:**
   - Measures the mean purity and the $\text{gap}$ against the raw representation across the remaining 3072 slots left entirely unsupervised.
   - It directly replicates the unsupervised-discovery result achieved in Exp0, but in a wider space with 4 domains.
3. **Whole-registry purity & gap:**
   - Measures the overall mean purity across all 4096 slots, allowing a direct like-for-like comparison with the earlier Exp0/Exp0d results (Exp0 reached a purity of 0.766 / a gap of 0.187, and Exp0d reached 0.805 / 0.222).

### An example of the resulting `exp1_summary.json`:

```json
{
  "verdict": "✅ pass — the reserved ranges specialized by name, and the free region discovered concepts automatically at high purity.",
  "reserved_blocks": {
    "accuracy": 0.892,
    "medicine_acc": 0.910,
    "law_acc": 0.875,
    "code_acc": 0.940,
    "literature_acc": 0.843
  },
  "free_region": {
    "mean_purity": 0.785,
    "gap": 0.195,
    "live_slots": 3020
  },
  "whole_registry": {
    "mean_purity": 0.812,
    "gap": 0.222,
    "total_live": 4035
  },
  "final_dead_frac": 0.015,
  "final_lm_loss": 3.28
}
```

### How do you judge the result?

| Indicator | The expected positive range | What it means scientifically and practically |
|---|---|---|
| **Reserved-range accuracy (`reserved_blocks.accuracy`)** | $> 0.80$ | The light supervision succeeded in steering the reserved ranges toward their named specialization without disruption. |
| **Free-region purity & gap (`free_region`)** | `mean > 0.75`<br>and `gap > 0.10` | The unsteered slots are still able to organize and discover concepts on their own in a 4-domain space. |
| **Whole-registry purity & gap (`whole_registry`)** | `mean > 0.75`<br>and `gap > 0.15` | The architecture as a whole is coherent and clearly purer than the raw representation, and comparable with the Exp0 results. |

> [!WARNING]
> **An important methodological caution:**
> The interpretation thresholds given here are **a preliminary, reasoned estimate** (unlike experiment zero, whose thresholds were calibrated across 4 actual repeated runs), and they are not yet established facts.
> **So always read the detailed `verdict` text printed in the log** for the precise caveats and notes, and avoid relying blindly on any single number in isolation.

---

> **💡 The eyeball check (interpreting the reserved and free slots):**
> At the end of the run the code prints the most activating tokens for each reserved block (medical, legal, programming and literary terms) as well as the most prominent free-region slots. Checking those token lists directly gives you immediate, concrete confirmation of how well the concepts were sorted.
