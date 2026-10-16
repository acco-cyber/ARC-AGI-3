# ARC-AGI-3

Experimental notebooks for the 2026 Kaggle ARC-AGI-3 competition. The current
engineering target is an official score above 40; no such result is claimed.

## Current candidate

[arc3-target40-engineered.ipynb](Notebook/arc3-target40-engineered.ipynb) retains
the Franzen/Affectify D-prime model and scheduling stack while fixing startup
handoff, final-token accounting, per-game validation and server cleanup.
See [engineering notes](docs/ARC40_ENGINEERING.md) and
[execution evidence](runs/2026-10-16T1225KST/).

Default Save & Run is a three-game, three-minute integration smoke after model
startup. It produces diagnostics only. Set `ARC40_RUN_MODE=validation` before the
first cell to evaluate all 25 public games for approximately 121 solver minutes.
The public validation gate requires every game to have real inference evidence
and mean/partition mean/bootstrap lower bound strictly above 40. A passing public
run still does not establish a hidden competition score.

Use the exact `NvidiaRtxPro6000` 96GB accelerator, internet disabled, the upstream
image and attached inputs listed in
[arc40-kernel-metadata.json](Notebook/arc40-kernel-metadata.json).
The real competition rerun uses the live gateway and a 532-minute absolute soft
deadline from notebook entry. Smoke runs write no placeholder submission file.

## Local checks

```bash
python -m pip install ipython nbformat kaggle
PYTHONPATH=scripts python -m unittest discover -s tests -v
python scripts/build_arc40.py
python scripts/run_arc40_once.py --receipt run-request.json
```

The Kaggle SDK reads credentials from its usual configuration or
`KAGGLE_API_TOKEN`; credentials are not stored in this repository. Build creates
a temporary ignored `launch/arc40` directory. The last command requests one
explicit GPU session with a 30-minute wall limit; it does not submit to competition.

## Established evidence

The account's observed official public best is 27.22. A recovered older offline
run on October 15 scored 42.8303 over 25 public games, but its public partition mean
was 35.1660 and bootstrap lower 31.2720, so its own validation gate failed. Source
hashes, exact printed aggregates and approximate reconstructed per-game rows
are preserved separately; this is not evidence for a new official 40 result.

## Attribution

Based on [Daniel Franzen's solution](https://github.com/da-fr/arc-agi-3-solution),
[Tufa Labs' Duck harness](https://github.com/Tufalabs/duck-harness),
[Affectify's public notebook](https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub),
and the [official starter](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter).
Third-party code and weights retain their respective licenses.
