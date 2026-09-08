"""Generate exp4/exp4_kaggle.ipynb from exp4/exp4_source.py.

Splits the source on `# %%` markers:
  * `# %% [markdown]` blocks become markdown cells (the leading `# ` is stripped).
  * plain `# %%` blocks become code cells.

Output: a valid .ipynb (nbformat=4, nbformat_minor=5) with kernelspec
python3 and language_info python, matching the structure of
exp3b/exp3b_kaggle.ipynb and the older self-generated notebooks in
this project.

Self-contained: no project-external imports. Read the source, split,
write JSON. No jupytext / no external converters.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "exp4_source.py"))
DST = os.path.normpath(os.path.join(HERE, "exp4_kaggle.ipynb"))


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


if __name__ == "__main__":
    main()
