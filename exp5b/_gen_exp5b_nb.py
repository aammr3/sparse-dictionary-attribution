"""Generate exp5b/exp5b_kaggle.ipynb from exp5b/exp5b_source.py.

Splits the source on `# %%` markers:
  * `# %% [markdown]` blocks become markdown cells (the leading `# ` is stripped).
  * plain `# %%` blocks become code cells.

Output: a valid .ipynb (nbformat=4, nbformat_minor=5) with kernelspec
python3 and language_info python, matching the structure of
exp5/exp5_kaggle.ipynb.

Self-contained: no project-external imports. Read the source, split,
write JSON. No jupytext / no external converters.

**FIX 1 self-check (review T-056-review, from exp5's crash):** after
writing the notebook we parse the resulting JSON and verify that
every top-level `NAME = ...` assignment in the .py source appears
in at least one CODE cell of the .ipynb. The exp5 crash was caused
by `CONFOUND_ABS_GAP_THRESHOLD = 1.0` being stranded inside a
`# %% [markdown]` block (which the notebook generator turns into
markdown and never executes), so the constant never existed at
runtime. The self-check prevents that class of bug from re-appearing:
every top-level constant is verified to live in a CODE cell.
"""
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "exp5b_source.py"))
DST = os.path.normpath(os.path.join(HERE, "exp5b_kaggle.ipynb"))

# Top-level Python identifier pattern: match identifiers that start at
# column 0 (no leading whitespace) and look like a constant
# assignment (capitals/digits/underscores, then optional whitespace,
# then an `=` that is NOT `==`).
TOPLEVEL_NAME_ASSIGN_RE = re.compile(
    r"^([A-Z_][A-Z0-9_]*)\s*=[^=]"
)


def split_cells(text):
    """Yield (cell_type, source_lines) tuples in order."""
    lines = text.splitlines()
    cells = []
    cur_type = None
    cur = []
    for line in lines:
        if line.startswith("# %% [markdown]"):
            if cur_type is not None:
                cells.append((cur_type, cur))
            cur_type = "markdown"
            cur = []
        elif line.startswith("# %%"):
            if cur_type is not None:
                cells.append((cur_type, cur))
            cur_type = "code"
            cur = []
        else:
            if cur_type is None:
                continue
            cur.append(line)
    if cur_type is not None and cur:
        cells.append((cur_type, cur))
    return cells


def strip_md_hashes(source_lines):
    """Markdown cells: drop a single leading `# ` (or `#`) from each line so
    the cell renders as a heading/paragraph instead of a literal `#`."""
    out = []
    for ln in source_lines:
        if ln.startswith("# "):
            out.append(ln[2:])
        elif ln == "#":
            out.append("")
        else:
            out.append(ln)
    return out


def to_source_array(lines):
    """Notebook `source` is a list of single-line strings; each line must end
    with `\n` except the last. We add a trailing newline to every non-empty
    line for clean rendering."""
    arr = []
    for i, ln in enumerate(lines):
        if i < len(lines) - 1:
            arr.append(ln + "\n")
        else:
            arr.append(ln)
    if arr and not arr[-1].endswith("\n"):
        arr[-1] = arr[-1] + "\n"
    return arr


def find_top_level_assignments(source_text):
    """Return a list of (line_no_1_indexed, NAME) for every top-level
    `NAME = ...` assignment in the source. Top-level means starts at
    column 0 (no indentation). The pattern matches the standard
    UPPERCASE-with-underscores constant naming convention used
    throughout this project.

    Multi-line assignments starting at column 0 (e.g. `NAME = (\\n
    ...)`) are picked up by the leading line; subsequent lines do
    NOT match because they are indented. This is intentional -- the
    self-check enforces "every top-level constant has at least one
    CODE-cell definition line."""
    out = []
    for ln_idx, line in enumerate(source_text.splitlines(), start=1):
        if line.startswith((" ", "\t")):
            continue
        m = TOPLEVEL_NAME_ASSIGN_RE.match(line)
        if m:
            out.append((ln_idx, m.group(1)))
    return out


def collect_code_cell_text(nb):
    """Concatenate the source text of every CODE cell in `nb` into a
    single string for `re.search` lookups."""
    chunks = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            chunks.append("".join(cell["source"]))
    return "\n".join(chunks)


def self_check_notebook(source_text, nb):
    """FIX 1 self-check. Returns (n_assignments, n_missing, missing_list)
    where `missing_list` is the list of NAME strings that do not appear
    in any CODE cell of `nb`. Raises AssertionError if the list is
    non-empty so the caller can fail loudly."""
    pairs = find_top_level_assignments(source_text)
    names = [n for (_, n) in pairs]
    code_text = collect_code_cell_text(nb)
    missing = []
    for n in names:
        # Match NAME = at the start of any line in a CODE cell (after
        # optional leading whitespace). The re.MULTILINE flag makes `^`
        # match at every line start.
        if not re.search(r"^" + re.escape(n) + r"\s*=", code_text, re.MULTILINE):
            missing.append(n)
    return len(names), missing


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        text = f.read()
    cells = split_cells(text)
    nb_cells = []
    for ctype, lines in cells:
        if ctype == "markdown":
            src_lines = strip_md_hashes(lines)
        else:
            src_lines = lines
        nb_cells.append({
            "cell_type": ctype,
            "metadata": {},
            "source": to_source_array(src_lines),
            "execution_count": None,
            "outputs": [],
        })
    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {DST}  ({len(nb_cells)} cells)")

    # FIX 1 self-check: every top-level NAME = assignment in the .py
    # source must also appear in at least one CODE cell of the .ipynb.
    n_assignments, missing = self_check_notebook(text, nb)
    print(f"self-check: {n_assignments} top-level assignments in source")
    if missing:
        print(f"SELF-CHECK FAILED: {len(missing)} top-level assignment(s) "
              f"missing from CODE cells:")
        for n in missing:
            print(f"  MISSING: {n}")
        raise SystemExit(1)
    print("self-check PASS: every top-level NAME = assignment present in a CODE cell.")


if __name__ == "__main__":
    main()
