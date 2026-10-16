# ARC3 harness engineering — 16 October 2026

The experimental target is an official score above 40. No official result or
speed improvement is claimed for this candidate. The previous account best
observed through Kaggle is 27.22.

Six parallel reviews examined harness correctness, runtime provisioning,
evaluation, inference efficiency, public solutions, and models. The pinned
Franzen/Affectify model, prompts, INT4/MTP recipe, FP8 KV cache, recurrent cache,
128 Ki context and ten-stream scheduling are retained.

## Concrete changes

- Restore the competition startup handoff after the original 12-minute deadline.
  The extra five-minute health wait applies only to offline runs, allowing the
  existing warmup RESET to reach the gateway while the backend finishes loading.
- Preserve the NVIDIA driver search path from the historically successful run.
  Check Python 3.12, one RTX PRO 6000 and at least 90000 MiB before installation;
  record a JSON failure with a ten-second hardware probe timeout.
- Register the server as an owned process group immediately after launch. Stop
  its workers on benchmark completion, exceptions or interpreter shutdown,
  including an IPython cell error. Escalate from TERM to KILL within bounded time.
- Count setup/model loading against an absolute competition soft deadline at
  532 minutes from notebook entry, leaving eight minutes below the nine-hour cap.
- Preserve unbilled final generated tokens on cancellation/no-action turns.
  Eight tests exercise actual extracted upstream solver and GameRun methods;
  five fail against the unmodified baseline and all pass with the patch.
- Check exact public game identities, normalized terminal states and inference
  for every game. Missing/duplicate/wrong IDs, nonfinite scores, impossible
  completion scores and invalid runtime evidence invalidate public validation.

## GPU modes

The default `ARC40_RUN_MODE=smoke` selects three public games deterministically
and runs 180 seconds of gameplay after startup. It measures integration only:
its report can never pass the public score gate and it writes no packaging file.
Set `ARC40_RUN_MODE=validation` before the first cell for all 25 public games,
approximately 121 minutes of solver time, and strict checks above 40 for the
mean, fixed public partition mean and one-sided game-bootstrap lower bound.
The partition uses already public games, so it is not an untouched test set.
The bootstrap represents variation within these public games, not hidden tasks.

A real competition rerun automatically selects the live gateway and preserves
the original scheduler/inference setup. Actual scored output belongs to the
competition gateway. A smoke run creates no packaging file. Only after the full public gate passes
does validation create the official starter schema stub required for a live
rerun, plus explicit metadata that the stub is not score evidence.

Full request snapshots are opt-in through `ARC40_SAVE_REQUEST_LOGS=1`. Measured
runs use minimal framework diagnostics, release the GPU, then save compact
full-precision benchmark JSON. Bulky HTML and frame serialization are excluded.

## Validation and provenance

Run `PYTHONPATH=scripts python -m unittest discover -s tests -v` for CPU checks.
Run `python scripts/build_arc40.py` to regenerate the self-contained notebook
and temporary launch directory; IPython and nbformat are build-time tools.
Notebook magics and top-level await are compiled; nbformat validation is separate.
No GPU inference is implied by these checks.

`Notebook/arc40-kernel-metadata.json` names the exact RTX PRO 6000, offline
inputs and the current upstream image returned by Kaggle's metadata API.
`run_arc40_once.py` requests one 30-minute-bounded smoke session without fallback.
It reads credentials through the SDK and writes only nonsecret execution evidence.

The recovered October 15, 05:16–07:28 KST run of the older notebook performed
real inference on all 25 public games: mean 42.8302747, public partition mean
35.1660395 and bootstrap lower 31.2719977. Its own gate failed. Those are historical
local metrics, not this revision's performance and not an official score. Exact
printed aggregates, source hashes and reconstructed rounded rows are preserved
separately in `runs/2026-10-16T1225KST/historical-public-evidence.json`.

## Sources and attribution

- https://www.kaggle.com/code/dfranzen/arc-agi-3-milestone-2-solution
- https://www.kaggle.com/code/shiiin9/affectify-arc-31-54-in-a-single-sub
- https://github.com/da-fr/arc-agi-3-solution
- https://github.com/Tufalabs/duck-harness
- https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter

No public downloadable single-GPU Kaggle result above 40 was verified by these
reviews. TWIN, PRO-LONG, Kepler and Memento executable-model results are research
references; their frontier-API/public-demo scores are not offline Kaggle evidence.
Changing models, quantization, prompts or cache policy requires a measured A/B run.

## Version scope

The first explicit GPU smoke request uses notebook SHA256
`6b24ac50b4e2f19f777cd4409df53790623b480f350311b1b569d357911e026c`
(source in commit `e1dfde9`). The later revision changes diagnostic persistence,
request-log defaults, stale-output cleanup and gated packaging. It preserves
model/prompt/inference configuration. Do not label a later source hash as GPU
tested on the strength of the earlier request; each receipt records its hash.

## Measured smoke and official score probe

The completed V1 Kaggle GPU smoke used RTX PRO6000 97887MiB, Python3.12.13,
server readiness after511seconds and total reported elapsed756.789seconds.
All three selected games had real inference, finite scores and valid terminal
cancellation. Their three-minute mean was2.328; this is not an official score
or a full-public performance estimate. The exact output files are in
`runs/2026-10-16T1225KST/smoke-v1`.

The separate `arc3-target40-submission.ipynb` wrapper reuses this technical
evidence and the CPU-reviewed engineering changes. Commit mode makes the
official starter schema stub without loading the model. The competition marker
activates the real agent automatically. This is a prepared official score
probe targeting above40, not a claimed public-gate pass or a score guarantee.
Its submission is blocked by the review condition described below.
The wrapper's manifest explicitly records full_public_validation_passed=false.

`python scripts/build_arc40.py --mode validation` selects the optional full
public experiment. It is not rerun automatically for every packaging operation.

## Submission block and ongoing validation

The official submission action was rejected by automatic approval review: the
existing user condition requires at least35 verification before consuming quota,
while the current three-minute smoke is only2.33 and does not meet that condition.
No workaround or repeat official submission was attempted. The unaffected
all25 public validation was launched as Kaggle version3 with a9000-second cap.
Its code hash is recorded in `full-validation-request.json`; score remains
unknown until it finishes. V2 packaging remains saved, but it is not an accepted
official submission.
