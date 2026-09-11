# Supplemental refresh latency experiment

Both snapshots add 200 ms API latency immediately before refresh measurement. Initial setup is unchanged. No checks, thresholds, or screen recipes were changed.

| Cell | Before 1 CLS / status | Before 2 CLS / status | After 1 CLS / status | After 2 CLS / status |
|---|---|---|---|---|
| S-01 1440-dark | 0.2015 / WARN | 0.2016 / WARN | 0.2016 / WARN | 0.2016 / WARN |
| S-01 1440-light | 0.2015 / WARN | 0.2016 / WARN | 0.2016 / WARN | 0.2016 / WARN |
| S-01 390-dark | 0.1632 / WARN | 0.156 / WARN | 0.156 / WARN | 0.156 / WARN |
| S-01 390-light | 0.156 / WARN | 0.156 / WARN | 0.156 / WARN | 0.156 / WARN |
| S-02 1440-dark | 0.2278 / WARN | 0.2278 / WARN | 0.258 / FAIL | 0.2284 / WARN |
| S-02 1440-light | 0.228 / WARN | 0.228 / WARN | 0.2281 / WARN | 0.258 / FAIL |
| S-02 390-dark | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL |
| S-02 390-light | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL |
| S-54 1440-dark | 0.2015 / WARN | 0.2016 / WARN | 0.2016 / WARN | 0.2016 / WARN |
| S-54 1440-light | 0.2015 / WARN | 0.2016 / WARN | 0.2016 / WARN | 0.2016 / WARN |
| S-54 390-dark | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL |
| S-54 390-light | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL | 0.4003 / FAIL |
