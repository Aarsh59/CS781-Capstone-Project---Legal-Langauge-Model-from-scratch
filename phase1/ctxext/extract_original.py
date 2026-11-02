#!/usr/bin/env python3
"""Copy the exact Phase-1 data-build code out of capstone_phase1_arch12.ipynb (cells [6], [17], [19]) into p1/,
so the replay runs the same code that built train.bin (plumbing; declare it)."""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
nb = json.load(open(os.path.join(HERE, "..", "notebooks", "capstone_phase1_arch12.ipynb")))
src = lambda i: "".join(nb["cells"][i]["source"])
c6, c17, c19 = src(6), src(17), src(19)
assert c6.startswith("%%writefile a1_tokenizer.py") and "def sample_texts_random_seek" in c17 and "train.bin" in c19
open(os.path.join(HERE, "p1", "a1_tokenizer.py"), "w").write("\n".join(c6.splitlines()[1:]) + "\n")
defs = c17[c17.index("def normalize_ws"):]                       # helper definitions only (no downloads)
open(os.path.join(HERE, "p1", "cell17_defs.py"), "w").write(defs)
open(os.path.join(HERE, "p1", "cell19.py"), "w").write(c19)
print("extracted: a1_tokenizer.py", len(c6), "| cell17 defs", len(defs), "| cell19", len(c19))
