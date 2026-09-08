"""Generate exp5c/exp5c_kaggle.ipynb from exp5c/exp5c_source.py.

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
import ast
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "exp5c_source.py"))
DST = os.path.normpath(os.path.join(HERE, "exp5c_kaggle.ipynb"))

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


def self_check_cell_order(nb):
    """FIX 2 self-check. Return a list of
    (name, used_in_cell_idx, defined_in_cell_idx) tuples for every
    top-level name that a CODE cell uses before some later CODE cell
    defines it. Empty list means the notebook is order-safe.

    Rules:
      * Keep only `cell_type == "code"` cells, in nb order.
      * `ast.parse` each cell source; skip any cell that raises SyntaxError.
      * `first_def[name]` is the first CODE-cell index that DEFINES `name`,
        where a definition is a MODULE-LEVEL `ast.FunctionDef` /
        `ast.AsyncFunctionDef` / `ast.ClassDef`, the target Name of a
        MODULE-LEVEL `ast.Assign`, or a module-level `ast.Import` /
        `ast.ImportFrom` binding (`asname` if present, else the first
        dotted component).
      * Per code cell, collect ONLY module-level `ast.Name` nodes in
        `Load` context. DO NOT descend into `ast.FunctionDef`,
        `ast.AsyncFunctionDef`, `ast.ClassDef`, `ast.Lambda`,
        `ast.ListComp`, `ast.SetComp`, `ast.DictComp`,
        `ast.GeneratorExp` -- this prevents comprehension-local
        names (e.g. `c` in the `model_tag` join) from being reported.
      * Report a name only when `first_def[name] > using_cell_index`.
    """
    code_cells = [
        (i, cell) for i, cell in enumerate(nb["cells"])
        if cell["cell_type"] == "code"
    ]

    _SKIP_INTO = (
        ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef,
        ast.Lambda, ast.ListComp, ast.SetComp, ast.DictComp,
        ast.GeneratorExp,
    )

    def _register_definition(stmt, cell_idx):
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            first_def.setdefault(stmt.name, cell_idx)
            return
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                bind = alias.asname if alias.asname else alias.name.split(".")[0]
                first_def.setdefault(bind, cell_idx)
            return
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                bind = alias.asname if alias.asname else alias.name.split(".")[0]
                first_def.setdefault(bind, cell_idx)
            return
        if isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    first_def.setdefault(tgt.id, cell_idx)
                elif isinstance(tgt, (ast.Tuple, ast.List)):
                    for elt in tgt.elts:
                        if isinstance(elt, ast.Name):
                            first_def.setdefault(elt.id, cell_idx)

    def _load_names_in(stmt):
        """Yield Name-id strings in Load context from `stmt`, without
        descending into nested scopes/comprehensions."""
        if isinstance(stmt, _SKIP_INTO):
            return
        if isinstance(stmt, ast.Name) and isinstance(stmt.ctx, ast.Load):
            yield stmt.id
        for child in ast.iter_child_nodes(stmt):
            yield from _load_names_in(child)

    first_def = {}
    parsed = []
    for cell_idx, cell in code_cells:
        src = "".join(cell["source"])
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        parsed.append((cell_idx, tree))
        for stmt in tree.body:
            _register_definition(stmt, cell_idx)

    offenders = []
    for cell_idx, tree in parsed:
        seen = set()
        for stmt in tree.body:
            for name in _load_names_in(stmt):
                if name in seen:
                    continue
                seen.add(name)
                def_idx = first_def.get(name)
                if def_idx is not None and def_idx > cell_idx:
                    offenders.append((name, cell_idx, def_idx))
    return offenders


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

    # FIX 2 self-check (T-073): every top-level name used by a CODE cell
    # must be defined in an EARLIER CODE cell. Guards against
    # `NameError: name '_disc' is not defined`-style crashes where a
    # CODE cell uses a helper that a LATER CODE cell defines.
    n_code_cells = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    order_offenders = self_check_cell_order(nb)
    print(f"order-check: {n_code_cells} code cells scanned")
    if order_offenders:
        for name, used_idx, def_idx in order_offenders:
            print(f"  USE-BEFORE-DEF: {name} used in code cell #{used_idx + 1} "
                  f"but defined in #{def_idx + 1}")
        raise SystemExit(1)
    print("order-check PASS: every top-level name is defined before it is used.")


if __name__ == "__main__":
    main()
