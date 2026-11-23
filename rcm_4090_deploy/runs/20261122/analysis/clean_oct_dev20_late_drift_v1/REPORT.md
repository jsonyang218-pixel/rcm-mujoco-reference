# clean OCT dev20 late drift analysis

Offline analysis only; original results unchanged.

Same five targets have identical first1000 states/errors/actions/OCT and evaluation contracts.
Their mean endpoint error grows from 45.845300 um to 112.102668 um.
All20 mean after1000=55.072375 um, after1600=104.285155 um; 18/20 worsen.
Additional15 final mean=101.679318 um.
Last600 invalid OCT frames=0, safety limited steps=0.

This establishes late drift, not a proven attribution to any single component. Mean action, net motion and A active no-motion fractions are in summary.json; source hashes and exact prefix checks included.
Saved axis_diagnostics shows large late A contributions; column norms are not additive errors.
Changing the stop rule to select the best ground-truth error would alter evaluation/control and is not performed.
