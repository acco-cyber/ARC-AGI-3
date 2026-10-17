# ARC-AGI-3

Experimental notebooks for the 2026 Kaggle ARC-AGI-3 competition. The current
engineering target is an official score above 40; no such result is claimed.

The latest observed official score remains **30.59** (16 October2026).
The17 October admission experiment adds a conservative no-action handover and
an evidence-only level ledger. The completed baseline reached all25 public
games but completed114 of183 levels; it did not solve every level.
See [the experiment and GPU diagnostic](docs/ARC35_LEASE_2026-10-17.md).

## Current candidate

[arc3-target40-engineered.ipynb](Notebook/arc3-target40-engineered.ipynb) retains
the Franzen/Affectify D-prime model and scheduling stack while fixing startup
handoff, final-token accounting, per-game validation and server cleanup.
See [engineering notes](docs/ARC40_ENGINEERING.md) and
[latest execution evidence](runs/2026-10-17/).

The current main validation artifact is built with `--mode validation` for
all 25 games. The archived `arc3-target40-engineered-smoke.ipynb` defaults to a
three-game, three-minute integration smoke after model startup. It produces diagnostics only. Set `ARC40_RUN_MODE=validation` before the
first cell to evaluate all 25 public games for approximately 121 solver minutes.
The public validation gate requires every game to have real inference evidence
and mean/partition mean/bootstrap lower bound strictly above 40. A passing public
run still does not establish a hidden competition score.

Use the exact `NvidiaRtxPro6000` 96GB accelerator, internet disabled, the upstream
image and attached inputs listed in
[arc40-kernel-metadata.json](Notebook/arc40-kernel-metadata.json).
The real competition rerun uses the live gateway and a 532-minute absolute soft
deadline from notebook entry. Smoke runs write no placeholder submission file. After a full public gate passes,
the standard starter packaging stub is written solely to enable a live gateway
rerun; its score field is not score evidence.

## Official submission wrapper

[arc3-target40-submission.ipynb](Notebook/arc3-target40-submission.ipynb) creates
only the official starter packaging stub during Save & Run. The live agent is
executed only when Kaggle sets its competition-rerun marker. This avoids loading
the same model twice or repeating a two-hour public benchmark before every
submission. The wrapper is built only after a completed hardware-verified GPU
smoke; it does not claim that this diagnostic score predicts the official result.
The exact first revision took 756.79 seconds and all three sampled games had real
inference. Its three-minute mean 2.328 is integration evidence only.

An official score probe remains subject to the user's minimum-score condition
and the automatic review block described below. A public gate cannot grant
user authorization or guarantee hidden-game performance. The official score
is unknown until an authorized gateway rerun finishes.

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

The account's observed official public best is now **30.59** (submission 57257762,
2026-10-16 06:14 UTC), compared with the earlier27.22. The completed V3 public
validation mean is 56.3511 and public-game bootstrap lower 42.3044. These public
statistics do not bound the hidden competition score. A recovered older offline
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

## Established baseline — 17 October 2026

- V3 full-public validation completed: mean **56.3511**, fixed public partition
  mean **63.0058**, within-public bootstrap lower **42.3044**; all 25 games had
  inference evidence, with 11 wins and 14 deadline cancellations.
- The account has a completed official submission **57257762**, script version
  **358880096**, scoring **30.59**. This monitoring pass observed that existing
  receipt; it did not create the submission. Neither official 35 nor 40 is met.
- All171 visible submission records were read using cursor pagination:123
  complete and 48 error records. Page-number pagination had repeated the first
  page and must not be used to claim a complete audit.
- The latest afternoon API snapshot reports **one submission available**.
  No new competition submission was made. A separate three-game GPU diagnostic
  completed with three voluntary admission handovers; it is not score evidence
  for the full candidate. The synthetic load probe failed during tokenizer
  preparation and its comparison is invalid. See the
  [execution summary](runs/2026-10-17-afternoon/execution-summary.json).
- Public56.35 versus official 30.59 is evidence of a transfer gap, not proof of
  a server fault. Preserve these artifacts as the baseline for paired experiments.

The earlier agent submission attempt was blocked by automatic approval review
under the user's minimum 35 condition. That block was not bypassed. The later
account receipt is recorded separately and does not establish a waiver for
future unverified submissions. See [monitoring analysis](docs/ARC40_MONITOR_2026-10-17.md).
