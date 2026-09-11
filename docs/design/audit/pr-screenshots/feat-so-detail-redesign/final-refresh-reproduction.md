# final refresh timing diagnostic comparison

Frozen baseline: 12b01b6. Exact new LAYOUT-020 failure cells reproduced: 22/22. Canonical matrices are unchanged; all diagnostic samples remain separate. No measurement code, threshold, fixture, application code, or response latency changed.

| Screen | Variant | Canonical before CLS | Current CLS | Diagnostic CLS values (FAIL marked) | Failure reproduced |
|---|---|---:|---:|---|---|
| S-01 | 1440-light | 0.0011 | 0.2893 | 1: 0.2892 FAIL; 2: 0.2893 FAIL; 3: 0.2219 | Yes |
| S-01 | 1440-dark | 0 | 0.2958 | 1: 0.001; 2: 0.1101; 3: 0.2957 FAIL | Yes |
| S-02 | 390-light | 0.0011 | 0.4003 | 1: 0.4003 FAIL; 2: 0.4003 FAIL; 5: 0.0011 | Yes |
| S-02 | 1440-dark | 0.0047 | 0.2581 | 1: 0.0047; 2: 0.2281; 5: 0.0046; 6: 0.3157 FAIL | Yes |
| S-02b | 390-light | 0 | 0.4003 | 5: 0.4003 FAIL | Yes |
| S-02b | 1440-light | 0 | 0.3158 | 5: 0.0047; 6: 0.3451 FAIL | Yes |
| S-03 | 390-light | 0 | 0.4003 | 1: 0.4075 FAIL; 2: 0.4075 FAIL | Yes |
| S-03 | 390-dark | 0.0011 | 0.4075 | 1: 0.4075 FAIL; 2: 0; 3: 0.4075 FAIL | Yes |
| S-04 | 390-dark | 0.0011 | 0.4075 | 1: 0.0011; 2: 0.4003 FAIL; 3: 0.4003 FAIL | Yes |
| S-04 | 1440-dark | 0.0012 | 0.2902 | 1: 0.2975 FAIL; 2: 0; 3: 0.2975 FAIL; 4: 0.1105 | Yes |
| S-05 | 390-dark | 0.0011 | 0.4003 | 5: 0.0011; 6: 0.4075 FAIL | Yes |
| S-06 | 390-light | 0.0011 | 0.4003 | 1: 0.0011; 2: 0.4075 FAIL | Yes |
| S-10 | 390-light | 0.0011 | 0.4003 | 5: 0.4075 FAIL | Yes |
| S-11 | 390-dark | 0.0011 | 0.4003 | 1: 0.0011; 2: 0.0011; 3: 0.4003 FAIL | Yes |
| S-12 | 1440-dark | 0.224 | 0.2892 | 1: 0.2892 FAIL; 2: 0.2893 FAIL | Yes |
| S-13 | 1440-dark | 0.0011 | 0.2893 | 1: 0.2892 FAIL; 2: 0.2893 FAIL | Yes |
| S-16 | 390-light | 0.0011 | 0.5736 | 1: 0.5606 FAIL; 2: 0.0011 | Yes |
| S-16 | 390-dark | 0.0011 | 0.5606 | 1: 0.5736 FAIL; 2: 0.5142 FAIL | Yes |
| S-17 | 390-dark | 0.0606 | 0.5736 | 1: 0.5736 FAIL; 2: 0.5606 FAIL | Yes |
| S-18 | 390-light | 0.0606 | 0.5606 | 1: 0.5606 FAIL; 2: 0.5736 FAIL | Yes |
| S-55 | 1440-light | 0.001 | 0.2892 | 1: 0.001; 2: 0.2892 FAIL; 3: 0; 4: 0.1101 | Yes |
| S-55 | 1440-dark | 0 | 0.2892 | 1: 0.2892 FAIL; 2: 0.2893 FAIL; 3: 0.2958 FAIL; 4: 0.001 | Yes |

20/22 current new failure cells have byte-identical settled anchor geometry to their canonical baseline counterparts.

22/22 current new failure cells have at least one frozen-baseline diagnostic FAIL with byte-identical settled anchor geometry to the current capture.

Geometry comparison includes anchors before/after, moved/vanished counts, maximum displacement, and the full worst-anchor list. Changed geometry and each diagnostic sample are recorded in the companion JSON; a reproduced failure is not automatically presented as identical geometry.
