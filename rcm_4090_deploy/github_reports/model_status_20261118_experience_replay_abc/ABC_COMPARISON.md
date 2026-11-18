# Experience Replay A/B/C

| Metric | A original | B static | C balanced |
|---|---:|---:|---:|
| Successes | 0/20 | 0/20 | 0/20 |
| Final mean um | 1087.778 | 207.290 | 1486.514 |
| Final median um | 883.749 | 141.257 | 1637.206 |
| Final P90 um | 2196.504 | 510.406 | 2661.735 |
| Final maximum um | 2466.592 | 677.360 | 2915.647 |
| Minimum error mean um | 63.988 | 63.011 | 84.064 |
| Safety-limited steps | 0 | 0 | 0 |
| Encoder settled at exit | 0/20 | 0/20 | 0/20 |

## Paired final-error improvements

- `B` vs `A`: 18 improved, 2 worsened, mean improvement +880.488 um.
- `C` vs `A`: 6 improved, 14 worsened, mean improvement -398.736 um.
- `C` vs `B`: 3 improved, 17 worsened, mean improvement -1279.224 um.
