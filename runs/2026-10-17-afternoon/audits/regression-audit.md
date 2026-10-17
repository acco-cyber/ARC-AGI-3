# Independent regression audit — 17 October 2026

Scope: read-only audit of ARC-AGI-3 candidate, production wrapper, embedded source patch and saved official/public evidence. No GPU, API credential access, submission or repository mutation.

## Result

No new blocking regression found that explains official 30.59 or warrants changing the model/runtime stack. Competition marker handling agrees between wrapper and candidate. The live path executes the implementation, uses live environments and preserves the 532-minute absolute wall deadline from notebook entry. Startup warmup restores a gateway action while model readiness may still be pending. Server launch is registered as an owned process group. Full-public and official scores remain distinct; public validation cannot guarantee hidden performance.

This is source review, not independent authentication of Kaggle's exact submitted bytes or hidden execution trace.

## One confirmed residual edge case

`tests/fixtures/duck-source/ARC3-Inference/inference/framework/solver.py`:

- `_run_games` cancellation sets stop, drains for at most `cancel_drain_timeout_s` (production default 120 seconds), then calls `_finish_remaining`.
- `_finish_remaining` calls `game.finish_game()` with no unbilled generated-token delta while an executor worker may still be alive.
- The worker's later `_finish_if_needed` skips a run whose `final_score` is already set, permanently omitting the tail usage.

CPU reproduction executes the actual three solver methods, the actual session finalizer and real TAAF `GameRun`/`finish_game`, extracted via AST to avoid GPU dependencies. It uses a 0.01-second drain only to reach the timeout branch quickly. A worker with 150 reported tokens and a 100-token action baseline is cancelled; forced finish records 0 final tokens, and later worker finalization leaves 0 instead of 50. See `reproduce_cancellation_accounting.py` and `cancellation-repro.json`.

This establishes an accounting risk only when a worker outlives the cancellation drain. It does not establish that the official run hit that branch, change its score, or justify another GPU benchmark by itself.

## Recommended treatment

Keep this separate from the sole gameplay hypothesis. Add a cancellation integration test (the existing tests only exercise session finalization directly). A robust future fix needs synchronized active-session finalization or explicit incomplete-usage evidence, with idempotent accounting when a worker returns late. Merely calling `finish_game()` twice cannot recover usage because finalization is deliberately idempotent. No speculative concurrency patch is proposed here.
