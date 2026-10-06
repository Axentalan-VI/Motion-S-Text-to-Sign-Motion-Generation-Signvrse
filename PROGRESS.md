# Progress

## Current status

Finished. The competition closed on 2026-05-10 **without a selected
submission**, so the entry is unranked among 130 teams. One submission scored
**0.433** on the competition metric.

The pipeline generates 6 RVQ token layers decoded by a frozen RVQ-VAE, with a
MoMask masked transformer and T2M-GPT ensembled under classifier-free guidance.

## Last session (2026-10-06)

- Corrected the README, which claimed the competition was still running.
- Added `pytest.ini`: `scripts/test_*.py` are manual smoke scripts that load
  real checkpoints, so a bare `pytest` spent 126 seconds importing torch to
  report "no tests ran". It now takes 0.1 s.

## Open issues

- **No submission was selected before the deadline**, so a scored entry exists
  but no ranking does. That is a pure process loss: the work was done.
- **No automated tests.** The four `scripts/test_*.py` files are smoke scripts
  run by hand against checkpoints, not a suite.
- No local metric was recorded, so 0.433 has nothing to compare against.

## Next steps (prioritized)

1. If revisited: select a submission as soon as one scores, rather than at the
   end. An unselected submission scores nothing.
2. Add real tests for the RVQ round-trip - that the decoder reconstructs tokens
   the encoder produced - which is the invariant everything else rests on.

## Decisions & rationale

- The RVQ-VAE is frozen and only the token models are trained, because the
  decoder is the part that is expensive and the part least likely to be
  improved on this compute budget (2026-05).
