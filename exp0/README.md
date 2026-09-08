# Running Experiment Zero (Exp0) on Kaggle 🚀

A quick, practical, step-by-step guide to running the core validation experiment for the built-in Concept Registry on Kaggle for free.

---

## 1. The goal of the experiment (in two lines)

We test whether a Sparse Concept Registry layer embedded inside a small Transformer (~10M parameters) can sort and specialize its slots automatically between two distant domains (Python code vs Wikipedia text) **without any prior supervision**.
If the registry's slots come out clearly purer than the raw representation dimensions, the proof of concept has succeeded and we can build the rest of the architecture with confidence; if not, the idea needs revisiting from the ground up.

---

## 2. Running it on Kaggle, in detail

The experiment is designed to run in a single session (expected time: **50 to 80 minutes**) on Kaggle's free GPUs.

1. **Create the notebook:**
   - Sign in to your [Kaggle](https://www.kaggle.com) account.
   - From the sidebar or the home page, click **`+ Create`** then choose **`New Notebook`**.
2. **Upload the experiment code:**
   - From the notebook's top menu choose **`File`** -> **`Import Notebook`** (or `Upload Notebook`).
   - Upload the [`exp0_kaggle.ipynb`](exp0_kaggle.ipynb) file directly (or paste the [`exp0_source.py`](exp0_source.py) code into the notebook's cells).
3. **Configure the settings (Settings panel in the right sidebar):**
   - **Accelerator:** choose **`GPU T4 x2`** **exclusively** (do not use `GPU P100`: the `sm_60` architecture is unsupported in the PyTorch version used at the time these experiments were run — August 2026).
   - **Internet:** set it to **`On`** ⚡ *(mandatory for downloading the dataset from HuggingFace and training the tokenizer)*.
   - **Persistence:** leave it at the default (`No persistence`).
4. **Start training:**
   - Click **`Run All`** in the top toolbar, or click **`Save Version`** and choose **`Run & Save All (Commit)`** to run the whole notebook in the background even if you close the browser.
   - Watch the printed steps to confirm the loss is falling, `alpha` is ramping up, and the dead-slot fraction is dropping.

---

## 3. Expected outputs in `/kaggle/working`

These files are saved automatically in the working directory at the end of the run:

| File name | Type | What it is and its role in the experiment |
|---|---|---|
| `tokenizer.json` | JSON | The trained tokenizer (Byte-level BPE, Vocab=8192) over a mixture of code and wiki text. |
| `code.npy` | NumPy Binary | The tokenized Python code array (15 million uint16 tokens), for pulling batches straight from memory. |
| `wiki.npy` | NumPy Binary | The tokenized Wikipedia text array (15 million uint16 tokens), to speed up training. |
| `exp0_model.pt` | PyTorch Weights | The full model weights after 6000 training steps (the base model + the `ConceptRegistry` layer). |
| `exp0_results.png` | PNG image | A 3-panel chart: (1) the LM loss trajectory against the `alpha` ramp, (2) the dead-slot percentage, (3) a histogram comparing the purity distribution of the registry against the raw dimensions. |
| `exp0_summary.json` | JSON | The full final report: the automated `verdict`, the mean purity, the `gap`, the dead fraction, and the CFG settings. |

---

## 4. If this happens... then the problem is this (troubleshooting)

| Problem or error message | Likely cause | Quick fix |
|---|---|---|
| `device: cpu` or a "no GPU" warning | The notebook is set to CPU. | Open Settings in the sidebar and choose `GPU T4 x2`. |
| `ConnectionError` / the dataset fails to download | Internet access is disabled in the notebook. | In Settings in the sidebar, make sure **`Internet: On`** and re-run the cell. |
| `CUDA out of memory` (OOM) | Excess memory use from repeated cells in the session. | Click `Restart Session`. If it recurs, reduce `batch_size` in the `CFG` from 48 to 32. |
| GPU quota exhausted / the session drops | The 30 weekly hours are used up, or the session stopped for idleness. | Run the notebook via `Save Version -> Run & Save (Commit)` to avoid browser-disconnection problems, or switch to another Kaggle account. |
| The data-preparation step is slow | HuggingFace text streaming depends on server speed. | Normal for the first 5–10 minutes; once the `.npy` files are created and saved, training proceeds at full speed straight from disk. |

---

## 5. How to read the experiment's result (`exp0_summary.json`)

In the final section of the code, each slot's purity is measured:
$$\text{purity} = \frac{\max(\text{code\_acts}, \text{wiki\_acts})}{\text{code\_acts} + \text{wiki\_acts}}$$
- **0.50**: completely random (the slot fires on both domains at the same rate).
- **1.00**: pure, perfect specialization on a single domain.

And the gap is computed between the registry's mean purity and the raw representation dimensions' mean purity (at the same sparsity $k=16$):
$$\text{gap} = \text{mean\_purity}(\text{slots}) - \text{mean\_purity}(\text{raw\_dims})$$

### The success and failure thresholds actually programmed into the code:

```json
// An example of the resulting exp0_summary.json
{
  "verdict": "✅ pass — slots specialized on their own, and the registry is clearly purer than the raw embedding.",
  "slots": { "mean": 0.842, "median": 0.887, "live": 1014, "above_80": 0.792 },
  "raw_dims": { "mean": 0.618, "median": 0.605 },
  "gap": 0.224,
  "final_dead_frac": 0.011,
  "final_lm_loss": 3.14
}
```

| Status and symbol | The numeric conditions (from the code) | What it means scientifically and practically | Next step |
|---|---|---|---|
| **✅ pass** | `mean > 0.80`<br>and `gap > 0.10` | The slots specialized on their own with high purity, and the registry is clearly purer than the raw representation by more than 0.10. | Move straight on to **experiment 1** (4 domains + a full dictionary + named reserved ranges). |
| **🟡 weak positive signal** | `mean > 0.70`<br>and `gap > 0.05` | There is specialization and separation, but it isn't decisive. | Try adjusting the hyper-parameters before judging: a lower $k$ (8), or more slots (2048), or longer training. |
| **❌ fail** | Below the conditions above | The slots did not specialize usefully and the registry added no purity. | Work through this diagnostic order:<br>1. Is the dead-slot fraction `final_dead_frac` high?<br>2. Did `alpha` ramp too fast and the model collapse?<br>3. Is `recon_loss` very high?<br>*(If all of those are fine, the idea needs a fundamental rethink.)* |

---

> **💡 The eyeball check (slot interpretation):**
> In the last cell, the code prints the top tokens that activate the most active slots. If you see explicitly code-like slots (things like `def`, `import`, `self`, `return`) and explicitly general-text slots, that is the direct, concrete evidence that the slots sorted themselves out and captured the nature of each domain.
