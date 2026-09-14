# Production Runs review pass 2

| Rule | Before | After failures |
|---|---|---:|
| STATUS-002 | N/A — absent page | 0 |
| STATUS-004 | N/A — absent page | 0 |
| STATUS-005 | N/A — absent page | 0 |
| STATUS-006 | N/A — absent page | 0 |
| STATUS-007 | N/A — absent page | 0 |
| STATUS-008 | N/A — absent page | 0 |
| STATUS-010 | N/A — absent page | 0 |
| STATUS-011 | N/A — absent page | 4 |

1440-light: PASS — factory date, week races, tabs, quantities, hooks, errors, create/edit/cancel, full replacement/clear/preserve, all completion states, authentication, no SO writes.

1440-dark: PASS — factory date, week races, tabs, quantities, hooks, errors, create/edit/cancel, full replacement/clear/preserve, all completion states, authentication, no SO writes.

390-light: PASS — factory date, week races, tabs, quantities, hooks, errors, create/edit/cancel, full replacement/clear/preserve, all completion states, authentication, no SO writes.

390-dark: PASS — factory date, week races, tabs, quantities, hooks, errors, create/edit/cancel, full replacement/clear/preserve, all completion states, authentication, no SO writes.

68 captures checked with the existing shared STATUS implementation. Baseline is an absent page; zero is not claimed for it.

## Failures

```
coverage/1440/light STATUS-011: [{"text":"Existing link; remaining quantity unavailable.","count":2,"paths":["div#dialog-content > div.coverage-item > p","div#dialog-content > div.coverage-item > p"]}]
```

```
coverage/1440/dark STATUS-011: [{"text":"Existing link; remaining quantity unavailable.","count":2,"paths":["div#dialog-content > div.coverage-item > p","div#dialog-content > div.coverage-item > p"]}]
```

```
coverage/390/light STATUS-011: [{"text":"Existing link; remaining quantity unavailable.","count":2,"paths":["div#dialog-content > div.coverage-item > p","div#dialog-content > div.coverage-item > p"]}]
```

```
coverage/390/dark STATUS-011: [{"text":"Existing link; remaining quantity unavailable.","count":2,"paths":["div#dialog-content > div.coverage-item > p","div#dialog-content > div.coverage-item > p"]}]
```
