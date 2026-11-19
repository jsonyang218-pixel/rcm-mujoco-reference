# State-attention v1 + Plan-2 dev10 evaluation

- Result: FAIL
- Success: 0/10
- Final error mean: 3458.685 um
- Final error median: 4308.960 um
- Best per episode: 34.009 um
- Encoder settled at exit: 0/10
- Safety-limited steps: 0
- Paired vs frozen v1 first 10: 2 improved, 8 worsened
- Delta final-error mean vs v1: +3107.522 um

The automatic 20-target pipeline comparison was rejected because the candidate contains episode indices 0-9 while the frozen baseline contains 0-19. This report uses the exact target-ID-matching first 10 records from the frozen baseline.

This is simulation-only and does not establish hardware or clinical accuracy.
