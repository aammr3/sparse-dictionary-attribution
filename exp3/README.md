# Running Experiment Three (Exp3) on Kaggle 🚀
### Phase G — testing specialization on the representations of a real pretrained model (Pythia-160m)

A quick, practical, step-by-step guide to running the first phase-G experiment for the Sparse Concept Registry on Kaggle for free, to check whether the automatic-specialization phenomenon replicates on the representations of a real frozen model.

---

## 1. The goal of the experiment (in a brief paragraph)

Experiment three (Exp3), the first of the **Phase G** experiments, tests: does the automatic specialization of sparse-registry slots (which we demonstrated in Exp0 on a small model trained from scratch) replicate when applied to the **real intermediate representations of a fully frozen pretrained model (EleutherAI/pythia-160m)**?
The notebook loads `pythia-160m` fully frozen (`eval()` with no modification to its weights whatsoever), and extracts the residual stream activations from the middle sixth layer (`hidden_states[6]`, 768 dimensions) on the fly as batches flow through memory (without saving them to disk), across **4 diverse domains** (medicine, law, code and literature — the same HuggingFace sources as in Exp1). **Only a sparse registry** is trained (a TopK sparse dictionary with `n_slots=4096`, `k=16`, plus the `AuxK` dead-slot revival mechanism) to approximate and reconstruct those activations, with no language-modelling loss (no LM loss) and no modification to Pythia. The success criterion is comparing the registry slots' purity against the purity of Pythia's raw representation dimensions (the same Exp0 test at the same sparsity level $k=16$), while printing Pythia's next-token loss as a sanity check.

---

## 2. Running it on Kaggle, in detail

The experiment is designed to run in a single session (expected time: **15 to 25 minutes**) on Kaggle's free GPUs.

1. **Create the notebook:**
   - Sign in to your [Kaggle](https://www.kaggle.com) account.
   - From the sidebar or the home page, click **`+ Create`** then choose **`New Notebook`**.
2. **Upload the experiment code:**
   - From the notebook's top menu choose **`File`** -> **`Import Notebook`** (or `Upload Notebook`).
   - Upload the [`exp3_kaggle.ipynb`](exp3_kaggle.ipynb) file directly (or paste the [`exp3_source.py`](exp3_source.py) code into the notebook's cells).
3. **Configure the settings (Settings panel in the right sidebar):**
   - **Accelerator:** choose **`GPU T4 x2`** exclusively ⚠️ *(important: the `sm_60` (P100) architecture is unsupported in the PyTorch version used at the time these experiments were run — August 2026; use T4 x2)*.
   - **Internet:** set it to **`On`** ⚡ *(mandatory for downloading the `EleutherAI/pythia-160m` weights and the four datasets from HuggingFace, and for installing the libraries)*.
   - **Persistence:** leave it at the default (`No persistence`).
4. **Start training:**
   - Make sure you run the first cell to install/verify the `transformers` library (`pip install -q transformers`).
   - Click **`Run All`** in the top toolbar, or click **`Save Version`** and choose **`Run & Save All (Commit)`** to run the whole notebook safely in the background.
   - Watch the console output to confirm the reconstruction loss (`recon_loss`) is falling, the dead-slot fraction (`dead_frac`) is dropping, and training is stable.

---

## 3. Expected runtime and what the console output looks like

- **Expected time:** roughly **15 to 25 minutes** in total on a T4 (given the lighter data budget of ~3 million tokens per domain and ~3000-4000 steps).

### The sequence of output during the run:

1. **Initialization, hardware check and model download:**
   ```text
   device: cuda
   gpu: Tesla T4
   Loading frozen EleutherAI/pythia-160m and tokenizer...
   Pythia-160m loaded: 12 layers, hidden_size=768 (All parameters FROZEN).
   ```
2. **Data streaming and tokenization:**
   ```text
   Streaming datasets from HuggingFace (medicine, law, code, literature)...
   Tokenizing ~3M tokens per domain using Pythia's GPTNeoX tokenizer...
   Data ready in memory.
   ```
3. **Training-loop output (printed periodically every 100 or 200 steps):**
   ```text
   step   200/3500 | loss: 0.0412 (recon: 0.0385, auxk: 0.0027) | dead: 0.4% | 18.2 step/s
   step   400/3500 | loss: 0.0278 (recon: 0.0259, auxk: 0.0019) | dead: 0.2% | 18.5 step/s
   ...
   step  3500/3500 | loss: 0.0115 (recon: 0.0108, auxk: 0.0007) | dead: 0.1% | 18.4 step/s
   ```
   - You should see a continuous, smooth decline in `recon_loss`.
   - The dead-slot fraction `dead_frac` should stay very low (< 1%) thanks to the `AuxK dead-slot revival` mechanism.

---

## 4. Expected outputs in `/kaggle/working`

The following files are saved automatically in the output folder at the end of the run:

| File name | Type | What it is and its role in the experiment |
|---|---|---|
| `exp3_model.pt` | PyTorch Weights | The weights of the sparse `ConceptRegistry` layer only (4096 slots), without Pythia's frozen weights. |
| `exp3_results.png` | PNG image | A chart showing: (1) the decline of `recon_loss` and `auxk_loss`, (2) the dead-slot percentage, (3) a histogram comparing the purity distribution of the registry slots against Pythia's raw dimensions. |
| `exp3_summary.json` | JSON | The full final report: the automated `verdict`, the mean and median purity, the purity `gap`, the fraction of active slots above 80% and 95%, and Pythia's sanity-check loss value (`pythia_heldout_ce`). |

---

## 5. How to read the result and the final table (`exp3_summary.json`)

At the end of the experiment, each registry slot's purity is measured across the four domains:
$$\text{purity} = \frac{\max(\text{domain\_acts})}{\sum_{d=1}^{4} \text{domain\_acts}}$$
- **0.25**: a perfectly uniform random distribution across the four domains (the slot is not specialized).
- **1.00**: pure, absolute specialization on a single domain.

And to keep the comparison fair, the purity of Pythia's 768 raw representation dimensions is measured **at the same sparsity level $k=16$** (taking the top 16 dimensions per token):
$$\text{gap} = \text{mean\_purity}(\text{registry\_slots}) - \text{mean\_purity}(\text{raw\_pythia\_dims})$$

### An example of the `exp3_summary.json` file and the printed report:

```json
{
  "verdict": "✅ PASSED — Sparse registry specialized autonomously on real frozen Pythia activations (gap > 0.10).",
  "registry_slots": {
    "mean_purity": 0.835,
    "median_purity": 0.872,
    "live_slots": 4060,
    "purity_above_80": 0.784,
    "purity_above_95": 0.362
  },
  "raw_pythia_dims": {
    "mean_purity": 0.582,
    "median_purity": 0.565
  },
  "gap": 0.253,
  "final_dead_frac": 0.008,
  "final_recon_loss": 0.0108,
  "pythia_heldout_ce": 3.42
}
```

### The criteria for judging the result:

| Status and symbol | The numeric conditions (from the code) | What it means scientifically and practically |
|---|---|---|
| **✅ Passed** | `mean > 0.75`<br>and `gap > 0.10` | The registry slots specialized automatically with high purity on real Pythia representations, and decisively outperformed the raw dimensions by more than 0.10. |
| **🟡 Partial positive signal** | `mean > 0.65`<br>and `gap > 0.05` | There is noticeable differentiation between the domains, but it needs tuning of the capacity or of the extraction layer (`hidden_states`). |
| **❌ Failed** | Below the conditions above | The slots did not specialize sufficiently, or they matched Pythia's raw dimensions. |

> **💡 The sanity check (Pythia sanity check):**
> The `pythia_heldout_ce` value is Pythia-160m's next-token loss on held-out test data; it should be stable and normal (~3.2 to 3.6) to confirm the data feeding is sound and the model is frozen.

> **💡 The eyeball check (slot interpretability):**
> In the closing cell the code prints the top tokens that activate the most specialized slots in each domain (medical terms like `syndrome`, `patient`; legal terms like `statute`, `amendment`; programming words like `def`, `async`, `return`; and literary constructions). Seeing these words gives you direct qualitative evidence of the conceptual sorting.

---

## 6. If this happens... then the problem is this (troubleshooting)

| Problem or error message | Likely cause | Quick fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'transformers'` | The first installation cell wasn't run. | Make sure you execute the notebook's first cell: `!pip install -q transformers`, then re-run the cells. |
| `ConnectionError` / Pythia or the dataset fails to download | Internet access is disabled in the notebook. | In the **Settings** menu in the right sidebar, change **`Internet`** to **`On`** then re-run the notebook. |
| `device: cpu` or a "no GPU" warning | The notebook is running on CPU, or the wrong accelerator type was chosen. | Open **Settings** and choose **`GPU T4 x2`** exclusively. Do not use P100 (the sm_60 architecture is unsupported). |
| `CUDA out of memory` (OOM) | Objects accumulated in memory from earlier sessions. | Click **`Restart Session`**. The default settings (`batch_size=24`, `ctx=256`) are designed to use less than 6GB of VRAM on a T4. |
| An error when running on a P100 | The `sm_60` (P100) architecture is unsupported in the PyTorch version used at the time these experiments were run. | Switch to the **`GPU T4 x2`** accelerator and the experiment will run smoothly straight away. |
| GPU quota exhausted / the session drops | The weekly hours are used up, or the session stopped for idleness during interactive work. | Run the notebook via **`Save Version` -> `Run & Save All (Commit)`** so it runs automatically in the background with no need to keep the browser open. |
| The first data-preparation stage is slow | HuggingFace server speed during streaming. | Normal, and only for the first 2-3 minutes while the datasets are pulled and the text tokenized; training then completes very quickly. |
