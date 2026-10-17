"""Produce a separate notebook arm; change only the admission-hook cell."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def build(source: Path, destination: Path, *, token_quantum=65536):
    if source.resolve() == destination.resolve():
        raise ValueError("The frozen baseline must not be overwritten")
    if type(token_quantum) is not int or token_quantum < 0:
        raise ValueError("Invalid token_quantum")
    raw = source.read_bytes()
    baseline = json.loads(raw)
    candidate = json.loads(raw)
    indices = []
    for index, cell in enumerate(candidate["cells"]):
        code = cell.get("source", "")
        code = "".join(code) if isinstance(code, list) else code
        if "_fp_mod.install_d(_fp_ta, _fp_solver)" in code:
            indices.append(index)
    if len(indices) != 1:
        raise RuntimeError("Expected exactly one active D-prime installation cell")
    index = indices[0]
    module = (HERE / "no_action_handover.py").read_text()
    code = candidate["cells"][index]["source"]
    code = "".join(code) if isinstance(code, list) else code
    if "_arc35_handover_source" in code:
        raise RuntimeError("Candidate already contains this experiment")
    code += ("\n# Single optional experiment: bounded no-action admission lease.\n"
             "# Entire completed reasoning and sandbox state remain intact.\n"
             f"_arc35_handover_source = {module!r}\n"
             "_arc35_handover_module = _fp_types.ModuleType('arc35_no_action_handover')\n"
             "_fp_sys.modules['arc35_no_action_handover'] = _arc35_handover_module\n"
             "exec(compile(_arc35_handover_source, 'arc35_no_action_handover.py', 'exec'), "
             "_arc35_handover_module.__dict__)\n"
             f"_arc35_handover_stats = _arc35_handover_module.install(_fp_ta, token_quantum={token_quantum})\n"
             "print('No-action admission lease:', _arc35_handover_stats.snapshot(), flush=True)\n")
    ast.parse(code)
    candidate["cells"][index]["source"] = code
    changed = [i for i, (a, b) in enumerate(zip(baseline["cells"], candidate["cells"])) if a != b]
    if changed != [index]:
        raise AssertionError("Unexpected notebook cell difference")
    candidate.setdefault("metadata", {})["arc35_no_action_handover"] = {
        "token_quantum": token_quantum, "enabled": token_quantum > 0,
        "changed_cell_indices": changed,
        "baseline_notebook_sha256": hashlib.sha256(raw).hexdigest(),
        "module_sha256": hashlib.sha256(module.encode()).hexdigest(),
        "score_gain_established": False, "speed_gain_established": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(candidate, indent=1) + "\n")
    assert source.read_bytes() == raw
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--token-quantum", type=int, default=65536)
    options = parser.parse_args()
    print(build(options.source, options.destination, token_quantum=options.token_quantum))
