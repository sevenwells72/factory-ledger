# Factory Ledger — data-health baseline (read-only audit)

Generated 2026-08-25 against production via pgbouncer 6543, per-query `BEGIN TRANSACTION READ ONLY`. Epoch = 2026-08-11 (bulk backfill boundary; migration_backfill_039 ran 2026-08-11 14:32 UTC). All windows are by `business_date` (authoritative ET business day). 'Live' rows = `created_at_source = 'database'`; 'backfilled' = `created_at_source IN ('migration_backfill_039','legacy_unverified')`.

## 0. Posted transaction counts (denominators)

| window | type | posted txns | backfilled |
|---|---|---|---|
| last 7 d | adjust | 2 | 0 |
| last 7 d | make | 7 | 0 |
| last 7 d | pack | 24 | 0 |
| last 7 d | receive | 2 | 0 |
| last 7 d | ship | 18 | 0 |
| last 30 d | adjust | 88 | 10 |
| last 30 d | make | 49 | 27 |
| last 30 d | pack | 100 | 47 |
| last 30 d | receive | 18 | 12 |
| last 30 d | ship | 89 | 42 |
| post-epoch | adjust | 77 | 0 |
| post-epoch | make | 22 | 0 |
| post-epoch | pack | 53 | 0 |
| post-epoch | receive | 6 | 0 |
| post-epoch | ship | 46 | 0 |

## 3a. Entry lag (occurred_at → created_at), LIVE rows only

| window | type | n | median h | p90 h | % same-day | % >24 h | % |lag|<2 min |
|---|---|---|---|---|---|---|---|
| last 7 d | adjust | 2 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 7 d | make | 7 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 7 d | pack | 24 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 7 d | receive | 2 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 7 d | ship | 18 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 30 d | adjust | 78 | 75.8 | 75.8 | 16.7 | 83.3 | 16.7 |
| last 30 d | make | 22 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 30 d | pack | 53 | 0.0 | 0.0 | 96.2 | 3.8 | 96.2 |
| last 30 d | receive | 6 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| last 30 d | ship | 47 | 0.0 | 75.8 | 85.1 | 14.9 | 85.1 |
| post-epoch | adjust | 77 | 75.8 | 75.8 | 16.9 | 83.1 | 16.9 |
| post-epoch | make | 22 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| post-epoch | pack | 53 | 0.0 | 0.0 | 96.2 | 3.8 | 96.2 |
| post-epoch | receive | 6 | 0.0 | 0.0 | 100.0 | 0.0 | 100.0 |
| post-epoch | ship | 46 | 0.0 | 75.8 | 87.0 | 13.0 | 87.0 |

### 3a-ii. Lag distribution, live rows post-epoch (0.5 h bins — a single dominant bin means occurred_at is defaulted at entry time and lag is NOT measurable)

| lag bin (h) | txns |
|---|---|
| 0.0 | 132 |
| 76.0 | 72 |

## 3b. Backfill rate by type, weekly from Jul 1 (week = Monday start)

| week | type | txns | backfilled | % backfilled |
|---|---|---|---|---|
| 2026-06-29 | adjust | 2 | 2 | 100.0 |
| 2026-06-29 | make | 4 | 4 | 100.0 |
| 2026-06-29 | pack | 13 | 13 | 100.0 |
| 2026-06-29 | receive | 1 | 1 | 100.0 |
| 2026-06-29 | ship | 11 | 11 | 100.0 |
| 2026-07-06 | adjust | 10 | 10 | 100.0 |
| 2026-07-06 | make | 13 | 13 | 100.0 |
| 2026-07-06 | pack | 24 | 24 | 100.0 |
| 2026-07-06 | receive | 5 | 5 | 100.0 |
| 2026-07-06 | ship | 16 | 16 | 100.0 |
| 2026-07-13 | adjust | 10 | 10 | 100.0 |
| 2026-07-13 | make | 11 | 11 | 100.0 |
| 2026-07-13 | pack | 19 | 19 | 100.0 |
| 2026-07-13 | receive | 3 | 3 | 100.0 |
| 2026-07-13 | ship | 20 | 20 | 100.0 |
| 2026-07-20 | adjust | 14 | 13 | 92.9 |
| 2026-07-20 | make | 14 | 14 | 100.0 |
| 2026-07-20 | pack | 25 | 25 | 100.0 |
| 2026-07-20 | receive | 5 | 5 | 100.0 |
| 2026-07-20 | ship | 12 | 12 | 100.0 |
| 2026-07-27 | adjust | 1 | 0 | 0.0 |
| 2026-07-27 | make | 10 | 10 | 100.0 |
| 2026-07-27 | pack | 21 | 21 | 100.0 |
| 2026-07-27 | receive | 1 | 1 | 100.0 |
| 2026-07-27 | ship | 17 | 16 | 94.1 |
| 2026-08-03 | adjust | 10 | 10 | 100.0 |
| 2026-08-03 | make | 15 | 15 | 100.0 |
| 2026-08-03 | pack | 23 | 23 | 100.0 |
| 2026-08-03 | receive | 11 | 11 | 100.0 |
| 2026-08-03 | ship | 22 | 22 | 100.0 |
| 2026-08-10 | adjust | 68 | 0 | 0.0 |
| 2026-08-10 | make | 12 | 2 | 16.7 |
| 2026-08-10 | pack | 26 | 3 | 11.5 |
| 2026-08-10 | receive | 1 | 0 | 0.0 |
| 2026-08-10 | ship | 21 | 4 | 19.0 |
| 2026-08-17 | adjust | 9 | 0 | 0.0 |
| 2026-08-17 | make | 10 | 0 | 0.0 |
| 2026-08-17 | pack | 24 | 0 | 0.0 |
| 2026-08-17 | receive | 4 | 0 | 0.0 |
| 2026-08-17 | ship | 20 | 0 | 0.0 |
| 2026-08-24 | make | 2 | 0 | 0.0 |
| 2026-08-24 | pack | 6 | 0 | 0.0 |
| 2026-08-24 | receive | 1 | 0 | 0.0 |
| 2026-08-24 | ship | 9 | 0 | 0.0 |

## 3c. Entry bursts (live rows, last 30 d by entry day; burst = txns entered ≤5 min apart)

| entry day (ET) | txns | bursts | biggest burst | % in biggest | >80% in one burst |
|---|---|---|---|---|---|
| 2026-08-11 | 9 | 3 | 6 | 66.7 |  |
| 2026-08-12 | 13 | 3 | 6 | 46.2 |  |
| 2026-08-13 | 12 | 3 | 8 | 66.7 |  |
| 2026-08-14 | 13 | 5 | 4 | 30.8 |  |
| 2026-08-17 | 96 | 8 | 74 | 77.1 |  |
| 2026-08-18 | 10 | 3 | 6 | 60.0 |  |
| 2026-08-19 | 17 | 8 | 4 | 23.5 |  |
| 2026-08-20 | 11 | 6 | 3 | 27.3 |  |
| 2026-08-21 | 7 | 3 | 4 | 57.1 |  |
| 2026-08-24 | 17 | 3 | 10 | 58.8 |  |
| 2026-08-25 | 1 | 1 | 1 | 100.0 | **FLAG** |

Flagged days: 2026-08-25

## 3d. Production-day coverage (weekdays post-epoch)

| weekday | dow | make txns | first make entered (ET) | gap |
|---|---|---|---|---|
| 2026-08-11 | Tue | 2 | 2026-08-11 15:36:38.900020 |  |
| 2026-08-12 | Wed | 2 | 2026-08-12 16:35:47.400118 |  |
| 2026-08-13 | Thu | 4 | 2026-08-13 15:39:10.013512 |  |
| 2026-08-14 | Fri | 2 | 2026-08-14 13:25:47.025966 |  |
| 2026-08-17 | Mon | 2 | 2026-08-17 16:35:40.753115 |  |
| 2026-08-18 | Tue | 3 | 2026-08-18 16:30:20.864585 |  |
| 2026-08-19 | Wed | 2 | 2026-08-19 15:54:26.626403 |  |
| 2026-08-20 | Thu | 2 | 2026-08-20 16:01:45.859027 |  |
| 2026-08-21 | Fri | 1 | 2026-08-21 14:03:44.189777 |  |
| 2026-08-24 | Mon | 2 | 2026-08-24 16:18:36.773199 |  |
| 2026-08-25 | Tue | 0 | — | **NO MAKE** |

### 3a-iii. Live txns post-epoch with lag > 24 h (who/what they are)

| txn | type | business_date | lag h | operator_id | notes (first 48 ch) |
|---|---|---|---|---|---|
| 1950 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1954 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Low Carb under-pack 20 |
| 1955 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Low Carb under-pack 20 |
| 1956 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Cranberry under-pack 1 |
| 1957 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB #1 Bulk 1875 vs ledger |
| 1960 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | YTD billed 8016 cs minus  |
| 1961 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | YTD billed ~915 minus rec |
| 1962 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Low Carb 964 cs alloca |
| 1963 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Cranberry 283 cs. forc |
| 1964 | pack | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Original (#9) Bulk per |
| 1965 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB Original (#9) Bulk per |
| 1966 | pack | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB #1 Bulk 1,875 lb |
| 1967 | ship | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | QB #1 Bulk 1,875 lb. forc |
| 1968 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1969 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1970 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1971 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1972 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1973 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1974 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1975 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1976 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1977 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1978 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1979 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1980 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1981 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1982 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1983 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1984 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1985 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1986 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1987 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1988 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1989 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1990 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1991 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1992 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1993 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1994 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1995 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1996 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1997 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1998 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 1999 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2000 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2001 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2002 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2003 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2004 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2005 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2006 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2007 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2008 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2009 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2010 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2011 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2012 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2013 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2014 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2015 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2016 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2017 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2018 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2019 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2020 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2021 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2022 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2023 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2024 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2025 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |
| 2026 | adjust | 2026-08-14 | 75.8 | inv-recon-2026-08-17 | INV-RECON-2026-08-17 | physical-count-2026-08-14 |

## 3e. Ship linkage — customer and order linkage split by evidence type (`shipment_lines` rows are written mechanically on every /ship commit, so they are shown separately and NOT counted as an SO link)

| window | ship txns | lb | % customer (n) | % customer (lb) | % order_reference | % SO link (n) | % SO link (lb) | % shipment row | % order_ref OR SO |
|---|---|---|---|---|---|---|---|---|---|
| last 7 d | 18 | 16990.0000 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 |
| last 30 d | 89 | 208054.5600 | 100.0 | 100.0 | 7.9 | 92.1 | 74.0 | 100.0 | 100.0 |
| post-epoch | 46 | 109212.5000 | 100.0 | 100.0 | 13.0 | 87.0 | 57.3 | 100.0 | 100.0 |

### 3e-ii. Ship txns with neither order_reference nor SO link, last 30 d

| txn id | business_date | customer_name | lb |
|---|---|---|---|
| _no rows_ | | | |

## 3f. Receipt linkage (expected_receipt_id read from base `transactions` — the ledger view predates migration 041 and does not expose it)

| window | receive txns | % supplier | % BOL/ref | % supplier+BOL | % matched to expected receipt |
|---|---|---|---|---|---|
| last 7 d | 2 | 100.0 | 100.0 | 100.0 | 0.0 |
| last 30 d | 18 | 100.0 | 100.0 | 100.0 | 0.0 |
| post-epoch | 6 | 100.0 | 100.0 | 100.0 | 0.0 |

`expected_receipts` rows ever created: **0** (0 open) — receipts cannot match what was never entered.

## 3g. Trace completeness — batches (make txns) last 30 d whose consumed ingredient lots ALL resolve to a posted receive with a supplier name

| batches | fully resolved | % |
|---|---|---|
| 49 | 0 | 0.0 |

### 3g-ii. Unresolved ingredient lots (consumed last 30 d, no supplier-bearing receive)

| lot_code | product | entry_source | on-hand lb | age days |
|---|---|---|---|---|
| JUL 23 2026 | Batch BS Dark Chocolate Granola 350 | production_output | 0.0000 | 33 |
| 26-01-30-FOUND-003 | BS Banana Bites – Small | found_inventory | 578.8100 | 207 |
| 26-01-30-FOUND-014 | BS Peanut Butter Chips | found_inventory | 65.8200 | 207 |
| 26-02-06-FOUND-006 | Cinnamon Ground | found_inventory | 105.0000 | 200 |
| 6012 | Coconut Fancy Desiccated | found_inventory | 2800.0000 | 103 |
| 6013 | Coconut Flake Desiccated | found_inventory | 6750.0000 | 103 |
| 6020 | Coconut Flake Desiccated | found_inventory | 0.0000 | 103 |
| 25120 | Coconut Macaroon Desiccated | found_inventory | 1013.0000 | 103 |
| 6020 | Coconut Medium Desiccated | found_inventory | 7000.0000 | 103 |
| 26-02-03-FOUND-002 | Flavor – Almond | found_inventory | 0.0000 | 203 |
| 26-02-03-FOUND-016 | Salt | found_inventory | 0.0000 | 203 |

## 3h. Lot hygiene

### 3h-i. Stale fractional lots: on-hand > 0, age > 30 d, and < 0.5 batch (batch products) or < 1 case (finished)

| product | type | lots | total lb |
|---|---|---|---|
| Batch Coconut Sweetened Fancy | batch | 4 | 422.8 |
| Batch Coconut Sweetened Flake | batch | 4 | 250.0 |
| Granola Fruit Nut Batch | batch | 1 | 169.0 |
| Classic Granola 25 LB | finished | 1 | 5.0 |

### 3h-ii. Negative lot balances (posted-only)

| lot_code | product | type | balance lb |
|---|---|---|---|
| _no rows_ | | | |

### 3h-iii. Product families with >1 distinct lot-code format (format = letter/digit-run signature; family = parent batch product for finished goods, else product)

| family | formats | one example per format |
|---|---|---|
| Almonds – Slivered | 2 | `26-02-16-DUTC-004`; `08226` |
| BS Oats GF | 2 | `26-01-30-FOUND-012`; `121125` |
| Batch BS Almond Butter Granola 350 (family) | 2 | `B26-0305-001`; `SW2607593` |
| Batch BS Dark Chocolate Granola 350 | 2 | `B26-0202-001`; `JUL 23 2026` |
| Batch BS Dark Chocolate Granola 350 (family) | 2 | `MAR 20 2026`; `SW2607890` |
| Batch BS Hazelnut Butter Granola 350 | 3 | `FEB 19 2026`; `B26-0304-002`; `26-05-06-HAZE-001` |
| Batch BS Peanut Butter Banana Granola (family) | 2 | `SW2607708`; `26-02-10-FOUND-005` |
| Batch Classic Chocolate Chip Granola #9 | 2 | `APR 24 2026`; `2026-04-20` |
| Batch Classic Granola #9 | 7 | `ABR 23 2026`; `26-02-05-FOUND-006`; `2026-04-15`; `B26-0204-001`; `FEB-02-26`; `FEB 10 2026-02`; `JUL15 2026` |
| Batch Classic Granola #9 (family) | 4 | `ABR 15 2026`; `2026-04-13`; `26-02-05-FOUND-007`; `B26-0304-001` |
| Batch Coconut Sweetened Fancy | 3 | `APR 27 2026`; `APR 10 2026 Lot`; `2026-04-20` |
| Batch Coconut Sweetened Fancy (family) | 4 | `APR 27 2026`; `APR 10 2026 Lot`; `26-02-12-FOUND-014`; `2026-04-20` |
| Batch Coconut Sweetened Flake | 7 | `ABR 24 2026`; `B26-0212-001`; `26-05-04-COCO-001`; `2026-04-14`; `FEB-03-2026`; `020526`; `TEST-WATER-3` |
| Batch Coconut Sweetened Flake (family) | 8 | `ABR 21 2026`; `STAGED-FEESERS-67476`; `2026-04-13`; `26-02-04-VINN-001`; `LOT FEB 24 2026`; `FEB-03-2026`; `FEB 16 2026 |
| Batch Coconut Sweetened Medium | 2 | `APR 27 2026`; `2026-04-17` |
| Batch Coconut Sweetened Medium (family) | 3 | `AGO 19 2026`; `26-02-12-FOUND-015`; `2026-04-17` |
| Batch Coconut Toasted Sweetened Flake | 3 | `APR 24 2026`; `APR 10 2026 Lot`; `2026-04-17` |
| Batch Coconut Toasted Sweetened Flake (family) | 3 | `APR 24 2026`; `2026-04-20`; `26-02-12-FOUND-016` |
| Batch SS Chocolate Chip Granola #2 | 5 | `APR 13 2026`; `APR 10 2026 Lot`; `26-05-04-SSCHOC-001`; `26-02-25`; `B26-0212-002` |
| Batch SS Chocolate Chip Granola #2 (family) | 4 | `BB021627`; `BB022427 Lot`; `FEB 13 2026`; `26-02-25` |
| Batch SS Classic Granola #9 (Kosher Ignition) (family) | 2 | `AUG 11 2026`; `BULK-#9-YTD` |
| Batch SS Cranberry Granola #3 | 3 | `BB081727`; `AUG 06 2026`; `26-05-05-CRAN-001` |
| Batch SS Low Carb Chocolate Chip Granola #8 | 3 | `26-05-04-LOWCARB-001`; `B26-0302-003`; `JUN 25 2026` |
| Batch SS Low Carb Chocolate Chip Granola #8 (family) | 2 | `B26-0302-003`; `BB070827` |
| Batch SS Low Carb Original Granola #7 | 3 | `26-05-05-LOWCARB-002`; `B26-0302-002`; `JUN 24 2026` |
| Batch SS Low Carb Original Granola #7 (family) | 2 | `B26-0302-002`; `BB070827` |
| Batch SS Original Granola #1 | 4 | `BB082717`; `APR 13 2026`; `2026-04-17`; `B26-0302-001` |
| Batch SS Original Granola #1 (family) | 4 | `BB022427`; `BULK-#1-YTD`; `APR 24 2026`; `2026-04-17` |
| Batch Vanilla Crisp Granola #16(no almonds) | 2 | `26-05-05-VAN-001`; `APR 27 2026` |
| Coconut Chips Desiccated | 2 | `25133`; `26-02-06-FOUND-007` |
| Coconut Flake Desiccated | 2 | `6012`; `26-01-30-FRAN-001` |
| Coconut Macaroon Desiccated | 2 | `25120`; `26-02-03-FOUND-001` |
| Coconut Medium Desiccated | 3 | `24289`; `VINA202524`; `26-02-05-FOUND-005` |
| Graham Cracker Crumbs – 10 LB | 2 | `601141`; `26-02-06-CREA-001` |
| Granola Fruit Nut Batch | 2 | `FN-0219`; `JUN 29 2026` |
| Granola Fruit Nut Batch (family) | 2 | `FN-0219`; `JUN 29 2026` |
| Kookies & Kreme – 10 LB | 2 | `26-02-10-CREA-002`; `601262` |
| Oats – Gluten Free | 2 | `26-02-25-DUTC-001`; `251121N` |
| Sprinkles Chocolate 25 LB | 2 | `26-03-12-FOUN-002`; `25114` |
| Sprinkles Rainbow 10 LB | 2 | `26-02-16-EURO-001`; `26054` |
| Sprinkles Rainbow 25 LB | 2 | `26-02-27-EURO-001`; `26054` |
| Sweetened Flake Coconut | 2 | `26-01-30-PH-001`; `FEB-04-2026` |

### 3h-iv. Lot codes containing non-English month tokens (ENE/ABR/AGO/DIC/SET)

| lot_code | product | tokens |
|---|---|---|
| AGO 19 2026 | Coconut Sweetened Medium CNS 10 LB | AGO |
| ABR 23 2026 | CQ Coconut Sweetened Flake 10 LB | ABR |
| ABR 15 2026 | Granola Classic 25 LB | ABR |
| ABR 24 2026 | Granola Chocolate Chip 25 LB | ABR |
| ABR 24 2026 | Batch Coconut Sweetened Flake | ABR |
| ABR 21 2026 | CQ Coconut Sweetened Flake 10 LB | ABR |
| ABR 22 2026 | CQ Coconut Sweetened Flake 10 LB | ABR |
| ABR 23 2026 | CQ Granola 10 LB | ABR |
| ABR 23 2026 | Batch Classic Granola #9 | ABR |
| ABR 30 2026 | CQ Coconut Sweetened Flake 10 LB | ABR |

Near-duplicates differing only by month spelling:

| Spanish-month code | English twin | product (ES) | product (EN) |
|---|---|---|---|
| ABR 23 2026 | APR 23 2026 | CQ Coconut Sweetened Flake 10 LB | Batch Coconut Sweetened Flake |
| ABR 23 2026 | APR 23 2026 | CQ Coconut Sweetened Flake 10 LB | CQ Coconut Sweetened Flake 10 LB |
| ABR 24 2026 | APR 24 2026 | Granola Chocolate Chip 25 LB | Batch Coconut Toasted Sweetened Flake |
| ABR 24 2026 | APR 24 2026 | Granola Chocolate Chip 25 LB | Batch Classic Chocolate Chip Granola #9 |
| ABR 24 2026 | APR 24 2026 | Granola Chocolate Chip 25 LB | Batch SS Original Granola #1 |
| ABR 24 2026 | APR 24 2026 | Granola Chocolate Chip 25 LB | Granola Honey Nut 25 LB |
| ABR 24 2026 | APR 24 2026 | Granola Chocolate Chip 25 LB | Coconut Toasted Sweetened Flake CNS 25 LB |
| ABR 24 2026 | APR 24 2026 | Batch Coconut Sweetened Flake | Batch Coconut Toasted Sweetened Flake |
| ABR 24 2026 | APR 24 2026 | Batch Coconut Sweetened Flake | Batch Classic Chocolate Chip Granola #9 |
| ABR 24 2026 | APR 24 2026 | Batch Coconut Sweetened Flake | Batch SS Original Granola #1 |
| ABR 24 2026 | APR 24 2026 | Batch Coconut Sweetened Flake | Granola Honey Nut 25 LB |
| ABR 24 2026 | APR 24 2026 | Batch Coconut Sweetened Flake | Coconut Toasted Sweetened Flake CNS 25 LB |
| ABR 22 2026 | APR 22 2026 | CQ Coconut Sweetened Flake 10 LB | Batch Coconut Sweetened Flake |
| ABR 23 2026 | APR 23 2026 | CQ Granola 10 LB | Batch Coconut Sweetened Flake |
| ABR 23 2026 | APR 23 2026 | CQ Granola 10 LB | CQ Coconut Sweetened Flake 10 LB |
| ABR 23 2026 | APR 23 2026 | Batch Classic Granola #9 | Batch Coconut Sweetened Flake |
| ABR 23 2026 | APR 23 2026 | Batch Classic Granola #9 | CQ Coconut Sweetened Flake 10 LB |
| ABR 30 2026 | APR 30 2026 | CQ Coconut Sweetened Flake 10 LB | CQ Coconut Sweetened Flake 10 LB |
| ABR 30 2026 | APR 30 2026 | CQ Coconut Sweetened Flake 10 LB | Batch SS Chocolate Chip Granola #2 |

## 3i. Lot-code collisions (same code, >1 lot record)

Total colliding codes: **159**

| code | lots | products |
|---|---|---|
| JUN 29 2026 | 14 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 892 ; Batch Classic Granola #9 [batch] lot 896 ; Batch SS Original Granola #1 [batch] lot 897 ; Batch V |
| MAY 12 2026 | 11 | Batch SS Original Granola #1 [batch] lot 580 ; Batch Setton Cinnamon Almond Granola #14 [batch] lot 581 ; Batch Classic Granola #9 [batch] lot 582 ; Batch Grano |
| APR 27 2026 | 10 | Batch Vanilla Crisp Granola #16(no almonds) [batch] lot 476 ; Batch SS Original Granola #1 [batch] lot 478 ; Batch Setton Cinnamon Almond Granola #14 [batch] lo |
| JUN 02 2026 | 10 | Batch Classic Granola #9 [batch] lot 726 ; Batch Coconut Sweetened Fancy [batch] lot 727 ; Batch Coconut Sweetened Medium [batch] lot 728 ; Batch Coconut Sweete |
| JUL 27 2026 | 9 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1062 ; Batch Classic Granola #9 [batch] lot 1063 ; Batch Coconut Sweetened Flake [batch] lot 1065 ; Coc |
| JUN 30 2026 | 9 | Batch Setton Cocoa Crunch Granola #13 [batch] lot 910 ; Batch Setton Cinnamon Almond Granola #14 [batch] lot 911 ; Batch Classic Granola #9 [batch] lot 912 ; Gr |
| MAY 21 2026 | 9 | Batch Classic Granola #9 [batch] lot 662 ; Batch Coconut Sweetened Fancy [batch] lot 664 ; Batch Coconut Sweetened Medium [batch] lot 665 ; Coconut Sweetened Me |
| AUG 03 2026 | 8 | Batch Classic Chocolate Chip Granola #9 [batch] lot 1099 ; Batch Classic Granola #9 [batch] lot 1100 ; Granola Classic 25 LB [finished] lot 1103 ; Batch Coconut |
| AUG 04 2026 | 8 | Batch Classic Granola #9 [batch] lot 1114 ; Batch Coconut Sweetened Medium [batch] lot 1116 ; Batch Coconut Sweetened Flake [batch] lot 1117 ; Batch Coconut Swe |
| JUL 21 2026 | 8 | Batch Classic Granola #9 [batch] lot 1021 ; Batch Vanilla Crisp Granola #16(no almonds) [batch] lot 1022 ; Granola Classic 25 LB [finished] lot 1024 ; Batch Coc |
| JUN 08 2026 | 8 | Batch SS Chocolate Chip Granola #2 [batch] lot 755 ; Batch SS Original Granola #1 [batch] lot 756 ; Batch Coconut Sweetened Flake [batch] lot 758 ; Batch Coconu |
| 2026-04-17 | 7 | Batch SS Original Granola #1 [batch] lot 422 ; Batch Coconut Sweetened Medium [batch] lot 424 ; Batch Coconut Toasted Sweetened Flake [batch] lot 425 ; Coconut  |
| APR 28 2026 | 7 | Batch Classic Granola #9 [batch] lot 490 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 491 ; Batch SS Chocolate Chip Granola #2 [batch] lot 492 ; Batch  |
| AUG 13 2026 | 7 | Batch Classic Granola #9 [batch] lot 1164 ; Batch Coconut Sweetened Fancy [batch] lot 1165 ; Batch Coconut Sweetened Medium [batch] lot 1166 ; Batch Coconut Swe |
| JUL 13 2026 | 7 | Coconut Sweetened Medium UNIPRO 10 LB [finished] lot 984 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 985 ; Batch Coconut Sweetened Flake [batch] lot 9 |
| JUL 22 2026 | 7 | Batch Classic Chocolate Chip Granola #9 [batch] lot 1037 ; Batch Coconut Toasted Sweetened Flake [batch] lot 1038 ; Batch Coconut Sweetened Fancy [batch] lot 10 |
| MAY 28 2026 | 7 | Batch Classic Granola #9 [batch] lot 698 ; Batch Coconut Sweetened Flake [batch] lot 699 ; Coconut Sweetened Flake UNIPRO 10 LB [finished] lot 700 ; Coconut Swe |
| AUG 11 2026 | 6 | Batch Classic Granola #9 [batch] lot 1148 ; Batch Coconut Sweetened Flake [batch] lot 1154 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1156 ; Granola SS C |
| FEB 16 2026 | 6 | Batch SS Original Granola #1 [batch] lot 181 ; Batch Vanilla Crisp Granola #16(no almonds) [batch] lot 182 ; Batch Coconut Sweetened Flake [batch] lot 183 ; CQ  |
| FEB 18 2026 | 6 | Batch Vanilla Crisp Granola #16(no almonds) [batch] lot 195 ; Batch SS Chocolate Chip Granola #2 [batch] lot 196 ; Batch Coconut Sweetened Flake [batch] lot 197 |
| JUL 01 2026 | 6 | Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 923 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 924 ; Batch Classic Granola #9 [batch] l |
| JUL 07 2026 | 6 | Batch Classic Granola #9 [batch] lot 950 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 951 ; Granola Classic 25 LB [finished] lot 952 ; Granola Crunchy  |
| JUN 19 2026 | 6 | Batch Coconut Sweetened Flake [batch] lot 838 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 839 ; Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 8 |
| MAR 18 2026 | 6 | Batch Classic Granola #9 [batch] lot 328 ; Batch BS Dark Chocolate Granola 350 [batch] lot 332 ; Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 333 ;  |
| MAR 26 2026 | 6 | Batch Coconut Sweetened Medium [batch] lot 366 ; Batch Coconut Sweetened Fancy [batch] lot 367 ; Coconut Sweetened Medium CNS 10 LB [finished] lot 368 ; Coconut |
| MAY 11 2026 | 6 | Batch Classic Chocolate Chip Granola #9 [batch] lot 572 ; Batch Coconut Sweetened Flake [batch] lot 575 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] l |
| MAY 13 2026 | 6 | Batch Coconut Sweetened Flake [batch] lot 593 ; Batch Classic Granola #9 [batch] lot 594 ; Granola Classic 25 LB [finished] lot 597 ; CQ Coconut Sweetened Flake |
| MAY 14 2026 | 6 | Batch SS Original Granola #1 [batch] lot 628 ; Batch SS Cranberry Granola #3 [batch] lot 629 ; Batch Classic Granola #9 [batch] lot 630 ; Batch Coconut Sweetene |
| MAY 18 2026 | 6 | Batch SS Chocolate Chip Granola #2 [batch] lot 642 ; Batch Coconut Sweetened Flake [batch] lot 644 ; Coconut Sweetened Flake CNS 10 LB [finished] lot 646 ; CQ C |
| MAY 19 2026 | 6 | Batch Classic Granola #9 [batch] lot 648 ; Batch Coconut Sweetened Flake [batch] lot 649 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 651 ; Granola Crunchy |
| APR 10 2026 LOT | 5 | Batch Coconut Toasted Sweetened Flake [batch] lot 391 ; Batch Coconut Sweetened Fancy [batch] lot 392 ; Batch SS Chocolate Chip Granola #2 [batch] lot 394 ; Coc |
| APR 13 2026 | 5 | Batch SS Chocolate Chip Granola #2 [batch] lot 398 ; Batch Classic Granola #9 [batch] lot 399 ; Batch SS Original Granola #1 [batch] lot 400 ; Batch Coconut Swe |
| APR 24 2026 | 5 | Batch Classic Chocolate Chip Granola #9 [batch] lot 462 ; Batch SS Original Granola #1 [batch] lot 463 ; Batch Coconut Toasted Sweetened Flake [batch] lot 468 ; |
| APR 29 2026 | 5 | Batch Classic Granola #9 [batch] lot 502 ; Batch Coconut Sweetened Flake [batch] lot 504 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 506 ; Granola Classic |
| AUG 05 2026 | 5 | Batch SS Original Granola #1 [batch] lot 1126 ; Batch SS Chocolate Chip Granola #2 [batch] lot 1127 ; Batch Coconut Sweetened Flake [batch] lot 1128 ; Coconut S |
| AUG 06 2026 | 5 | Batch SS Chocolate Chip Granola #2 [batch] lot 1133 ; Batch SS Cranberry Granola #3 [batch] lot 1134 ; Batch Coconut Sweetened Flake [batch] lot 1135 ; Coconut  |
| AUG 10 2026 | 5 | Batch Classic Granola #9 [batch] lot 1143 ; Batch Coconut Sweetened Flake [batch] lot 1145 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1147 ; CQ Granola 1 |
| AUG 12 2026 | 5 | Batch Coconut Sweetened Flake [batch] lot 1157 ; Batch Classic Granola #9 [batch] lot 1158 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1162 ; Granola Clas |
| AUG 14 2026 | 5 | Batch SS Chocolate Chip Granola #2 [batch] lot 1175 ; Batch Coconut Sweetened Flake [batch] lot 1177 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1179 ; Gr |
| BB070827 | 5 | Granola SS Original Low Carb 12x10 OZ Case [finished] lot 961 ; Granola SS Chocolate Chip Low Carb 12x10 OZ Case [finished] lot 962 ; Granola SS Cranberry 12x10 |
| FEB 06 2026 | 5 | Batch Classic Granola #9 [batch] lot 100 ; Batch Coconut Sweetened Flake [batch] lot 102 ; CQ Granola 10 LB [finished] lot 178 ; Granola Classic 25 LB [finished |
| FEB 09 2026 | 5 | Batch Classic Granola #9 [batch] lot 113 ; Batch Coconut Sweetened Flake [batch] lot 118 ; Granola Classic 25 LB [finished] lot 255 ; Granola Crunchy CNS 10 LB  |
| FEB 19 2026 | 5 | Batch Setton Cocoa Crunch Granola #13 [batch] lot 207 ; Batch BS Hazelnut Butter Granola 350 [batch] lot 208 ; Granola Cocoa Vibes 25 LB [finished] lot 209 ; Ba |
| FEB 24 2026 | 5 | Batch SS Chocolate Chip Granola #2 [batch] lot 225 ; Batch Coconut Sweetened Medium [batch] lot 228 ; Batch Coconut Sweetened Flake [batch] lot 229 ; Coconut Sw |
| JUL 08 2026 | 5 | Batch Classic Chocolate Chip Granola #9 [batch] lot 959 ; Batch SS Chocolate Chip Granola #2 [batch] lot 960 ; Batch Coconut Sweetened Fancy [batch] lot 964 ; B |
| JUL 09 2026 | 5 | Coconut Sweetened Medium UNIPRO 10 LB [finished] lot 970 ; Coconut Sweetened Flake CNS 10 LB [finished] lot 971 ; Batch SS Chocolate Chip Granola #2 [batch] lot |
| JUL 20 2026 | 5 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1016 ; Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 1017 ; Batch SS Chocolate Chip Granola  |
| JUN 01 2026 | 5 | Batch Coconut Sweetened Flake [batch] lot 721 ; Batch Classic Granola #9 [batch] lot 722 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 723 ; CQ Granola 10 L |
| JUN 09 2026 | 5 | Batch Classic Granola #9 [batch] lot 789 ; Batch SS Original Granola #1 [batch] lot 790 ; Batch Coconut Sweetened Flake [batch] lot 792 ; Coconut Sweetened Flak |
| JUN 10 2026 | 5 | Batch Classic Chocolate Chip Granola #9 [batch] lot 798 ; Batch Coconut Sweetened Flake [batch] lot 799 ; Coconut Sweetened Flake UNIPRO 10 LB [finished] lot 80 |
| JUN 22 2026 | 5 | Batch SS Chocolate Chip Granola #2 [batch] lot 847 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 848 ; Coconut Toasted Sweetened Flake CNS 25 LB [finish |
| JUN 24 2026 | 5 | Batch SS Original Granola #1 [batch] lot 870 ; Batch SS Cranberry Granola #3 [batch] lot 871 ; Batch SS Low Carb Original Granola #7 [batch] lot 872 ; Batch Coc |
| JUN 25 2026 | 5 | Batch SS Low Carb Chocolate Chip Granola #8 [batch] lot 877 ; Batch Classic Granola #9 [batch] lot 878 ; Batch Coconut Sweetened Flake [batch] lot 879 ; CQ Coco |
| MAR 11 2026 | 5 | Batch Classic Granola #9 [batch] lot 293 ; Batch Coconut Sweetened Flake [batch] lot 296 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 298 ; CQ Granola 10 L |
| MAR 12 2026 | 5 | Batch SS Original Granola #1 [batch] lot 304 ; Batch Classic Granola #9 [batch] lot 305 ; Granola Honey Nut 25 LB [finished] lot 306 ; Batch Coconut Sweetened F |
| MAR 24 2026 | 5 | Granola Classic 25 LB [finished] lot 356 ; Granola Crunchy CNS 10 LB Case [finished] lot 358 ; Batch Classic Granola #9 [batch] lot 359 ; Granola Wheat Free 25  |
| MAY 05 2026 | 5 | CQ Coconut Sweetened Flake 10 LB [finished] lot 547 ; Granola Classic 25 LB [finished] lot 552 ; Granola Crunchy CNS 10 LB Case [finished] lot 554 ; Granola Set |
| MAY 20 2026 | 5 | Batch Classic Granola #9 [batch] lot 655 ; Batch Coconut Sweetened Flake [batch] lot 656 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 658 ; Coconut Sweeten |
| 2026-04-20 | 4 | Batch Classic Chocolate Chip Granola #9 [batch] lot 431 ; Batch Coconut Sweetened Fancy [batch] lot 432 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] l |
| 2026-04-21 | 4 | Batch Classic Granola #9 [batch] lot 442 ; Batch Coconut Sweetened Flake [batch] lot 443 ; Coconut Sweetened Flake CNS 25 LB [finished] lot 444 ; CQ Coconut Swe |
| AUG 18 2026 | 4 | Batch SS Chocolate Chip Granola #2 [batch] lot 1295 ; Batch SS Original Granola #1 [batch] lot 1296 ; Batch Coconut Sweetened Flake [batch] lot 1297 ; CQ Coconu |
| AUG 20 2026 | 4 | Batch Classic Granola #9 [batch] lot 1310 ; Batch Coconut Sweetened Flake [batch] lot 1313 ; Coconut Sweetened Flake CNS 10 LB [finished] lot 1314 ; CQ Coconut  |
| BB081727 | 4 | Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 1298 ; Granola SS Original 12x10 OZ Case [finished] lot 1302 ; Batch SS Cranberry Granola #3 [batch] lot  |
| FEB 10 2026 | 4 | Batch Classic Granola #9 [batch] lot 125 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 129 ; Batch Coconut Sweetened Flake [batch] lot 131 ; CQ Coconut  |
| FEB 17 2026 | 4 | Batch Vanilla Crisp Granola #16(no almonds) [batch] lot 186 ; Batch Coconut Sweetened Flake [batch] lot 187 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 19 |
| FEB 25 2026 | 4 | Batch Coconut Sweetened Fancy [batch] lot 239 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 240 ; Batch Coconut Sweetened Flake [batch] lot 241 ; Coconu |
| JUL 06 2026 | 4 | Batch Classic Granola #9 [batch] lot 942 ; CQ Granola 10 LB [finished] lot 945 ; Batch Coconut Sweetened Flake [batch] lot 946 ; CQ Coconut Sweetened Flake 10 L |
| JUL 10 2026 | 4 | Batch Classic Granola #9 [batch] lot 978 ; Batch Coconut Toasted Sweetened Flake [batch] lot 979 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 981  |
| JUL 17 2026 | 4 | Batch Classic Granola #9 [batch] lot 1009 ; Batch Coconut Toasted Sweetened Flake [batch] lot 1011 ; Granola Wheat Free 25 LB [finished] lot 1015 ; Granola Clas |
| JUL 28 2026 | 4 | Batch Classic Granola #9 [batch] lot 1071 ; Batch Coconut Sweetened Flake [batch] lot 1072 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1074 ; CQ Granola 1 |
| JUL 30 2026 | 4 | Batch Classic Granola #9 [batch] lot 1083 ; Batch Coconut Sweetened Flake [batch] lot 1084 ; CQ Granola 10 LB [finished] lot 1088 ; CQ Coconut Sweetened Flake 1 |
| JUN 03 2026 | 4 | Batch Classic Chocolate Chip Granola #9 [batch] lot 733 ; Batch Setton Cocoa Crunch Granola #13 [batch] lot 734 ; Batch Coconut Sweetened Flake [batch] lot 735  |
| JUN 15 2026 | 4 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 816 ; Batch Coconut Sweetened Flake [batch] lot 817 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 8 |
| JUN 18 2026 | 4 | Batch SS Chocolate Chip Granola #2 [batch] lot 832 ; Batch Coconut Sweetened Flake [batch] lot 833 ; Batch Coconut Toasted Sweetened Flake [batch] lot 834 ; CQ  |
| JUN 23 2026 | 4 | Batch SS Chocolate Chip Granola #2 [batch] lot 861 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 862 ; Batch Coconut Sweetened Flake [batch] lot 864 ; C |
| JUN 28 2026 | 4 | Batch Classic Granola #9 [batch] lot 887 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 888 ; Batch Coconut Toasted Sweetened Flake [batch] lot 889  |
| MAR 03 2026 | 4 | Batch Classic Granola #9 [batch] lot 261 ; CQ Granola 10 LB [finished] lot 308 ; Batch Coconut Sweetened Flake [batch] lot 381 ; Coconut Sweetened Flake UNIPRO  |
| MAR 09 2026 | 4 | Batch BS Dark Chocolate Granola 350 [batch] lot 279 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 280 ; Batch Coconut Sweetened Flake [batch] lot 282 ;  |
| MAR 10 2026 | 4 | Batch Classic Granola #9 [batch] lot 287 ; Batch Coconut Sweetened Flake [batch] lot 288 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 291 ; CQ Granola 10 L |
| MAR 25 2026 | 4 | Batch Coconut Sweetened Flake [batch] lot 362 ; Batch Coconut Sweetened Fancy [batch] lot 363 ; Coconut Sweetened Flake UNIPRO 10 LB [finished] lot 364 ; Coconu |
| MAR 31 2026 | 4 | Batch Coconut Sweetened Fancy [batch] lot 385 ; Batch Coconut Sweetened Flake [batch] lot 386 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 387 ; Coconu |
| MAY 01 2026 | 4 | Batch Setton Cocoa Crunch Granola #13 [batch] lot 512 ; Batch Coconut Toasted Sweetened Flake [batch] lot 518 ; Batch SS Chocolate Chip Granola #2 [batch] lot 5 |
| MAY 08 2026 | 4 | Batch Coconut Toasted Sweetened Flake [batch] lot 567 ; Batch Classic Granola #9 [batch] lot 568 ; Granola Classic 25 LB [finished] lot 596 ; Granola Crunchy CN |
| MAY 26 2026 | 4 | Batch SS Chocolate Chip Granola #2 [batch] lot 671 ; Batch Coconut Sweetened Flake [batch] lot 675 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 67 |
| MAY 27 2026 | 4 | Batch Classic Granola #9 [batch] lot 686 ; Batch Classic Chocolate Chip Granola #9 [batch] lot 687 ; Batch Coconut Sweetened Flake [batch] lot 688 ; CQ Coconut  |
| 2026-04-15 | 3 | Batch Classic Granola #9 [batch] lot 412 ; Batch Coconut Sweetened Flake [batch] lot 413 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 415 |
| 26-01-29-DUTVF-001 | 3 | Almonds – Sliced [ingredient] lot 2 ; Almonds – Diced [ingredient] lot 3 ; Sugar – Invert(Cream) [ingredient] lot 4 |
| ABR 23 2026 | 3 | Batch Classic Granola #9 [batch] lot 456 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 459 ; CQ Granola 10 LB [finished] lot 516 |
| AUG 17 2026 | 3 | Batch SS Chocolate Chip Granola #2 [batch] lot 1290 ; Batch Coconut Sweetened Flake [batch] lot 1292 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1294 |
| AUG 21 2026 | 3 | Batch Classic Granola #9 [batch] lot 1318 ; Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 1321 ; Granola Classic 25 LB [finished] lot 1326 |
| FEB 13 2026 | 3 | Batch SS Chocolate Chip Granola #2 [batch] lot 167 ; Batch Coconut Sweetened Flake [batch] lot 169 ; Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 200 |
| FEB 20 2026 | 3 | Batch Coconut Sweetened Flake [batch] lot 218 ; Batch BS Hazelnut Butter Granola 350 [batch] lot 219 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 221 |
| FEB 26 2026 | 3 | Batch Classic Chocolate Chip Granola #9 [batch] lot 245 ; Batch Classic Granola #9 [batch] lot 246 ; Batch SS Chocolate Chip Granola #2 [batch] lot 247 |
| FEB-03-2026 | 3 | Batch Classic Granola #9 [batch] lot 68 ; Batch Coconut Sweetened Flake [batch] lot 71 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 77 |
| JUL 02 2026 | 3 | CQ Granola 10 LB [finished] lot 936 ; Batch Coconut Sweetened Flake [batch] lot 937 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 939 |
| JUL 23 2026 | 3 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1046 ; Batch Coconut Toasted Sweetened Flake [batch] lot 1048 ; Batch BS Dark Chocolate Granola 350 [ba |
| JUL 24 2026 | 3 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1052 ; Batch Coconut Toasted Sweetened Flake [batch] lot 1054 ; Batch BS Dark Chocolate Granola 350 [ba |
| JUL 26 2026 | 3 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1058 ; Coconut Toasted Sweetened Flake CNS 10 LB [finished] lot 1060 ; Batch Coconut Toasted Sweetened  |
| JUN 12 2026 | 3 | Batch Coconut Toasted Sweetened Flake [batch] lot 811 ; Batch Coconut Sweetened Flake [batch] lot 812 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 815 |
| JUN 17 2026 | 3 | Batch SS Chocolate Chip Granola #2 [batch] lot 825 ; Batch Coconut Sweetened Flake [batch] lot 826 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 828 |
| JUN 26 2026 | 3 | Batch Coconut Toasted Sweetened Flake [batch] lot 882 ; Batch Classic Granola #9 [batch] lot 883 ; CQ Granola 10 LB [finished] lot 894 |
| MAR 16 2026 | 3 | Batch Classic Granola #9 [batch] lot 317 ; CQ Granola 10 LB [finished] lot 321 ; Granola Crunchy CNS 10 LB Case [finished] lot 357 |
| MAR 27 2026 | 3 | Batch Coconut Sweetened Medium [batch] lot 374 ; Coconut Sweetened Medium UNIPRO 10 LB [finished] lot 377 ; Granola Wheat Free 25 LB [finished] lot 434 |
| MAY 07 2026 | 3 | Batch BS Hazelnut Butter Granola 350 [batch] lot 562 ; Batch Coconut Sweetened Flake [batch] lot 563 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 565 |
| MAY 15 2026 | 3 | Batch SS Chocolate Chip Granola #2 [batch] lot 639 ; Batch Coconut Toasted Sweetened Flake [batch] lot 640 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished |
| MAY 29 2026 | 3 | Batch Coconut Sweetened Flake [batch] lot 705 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 707 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 719 |
| 2026-04-13 | 2 | CQ Coconut Sweetened Flake 10 LB [finished] lot 408 ; Granola Classic 25 LB [finished] lot 435 |
| 2026-04-14 | 2 | Batch Coconut Sweetened Flake [batch] lot 407 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 409 |
| 2026-04-16 | 2 | Batch Coconut Sweetened Flake [batch] lot 418 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 420 |
| 25114 | 2 | Sprinkles Chocolate 10 LB [finished] lot 654 ; Sprinkles Chocolate 25 LB [finished] lot 741 |
| 25120 | 2 | Coconut Macaroon Desiccated [ingredient] lot 616 ; Coconut Medium Desiccated [ingredient] lot 620 |
| 26-02-06-CREA-001 | 2 | Graham Cracker Crumbs – 50 LB [ingredient] lot 93 ; Graham Cracker Crumbs – 10 LB [finished] lot 450 |
| 26-02-10-CREA-002 | 2 | Kookies & Kreme – 25 LB [finished] lot 123 ; Kookies & Kreme – 10 LB [finished] lot 486 |
| 26-02-25 | 2 | Batch SS Chocolate Chip Granola #2 [batch] lot 237 ; Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 267 |
| 26-04-16-UNKN-001 | 2 | Graham Cracker Crumbs – 50 LB [ingredient] lot 411 ; Graham Cracker Crumbs – 10 LB [finished] lot 485 |
| 26-06-02-CREA-001 | 2 | Kookies & Kreme – 25 LB [finished] lot 724 ; Kookies & Kreme – 10 LB [finished] lot 1101 |
| 26054 | 2 | Sprinkles Rainbow 10 LB [finished] lot 598 ; Sprinkles Rainbow 25 LB [finished] lot 641 |
| 6012 | 2 | Coconut Fancy Desiccated [ingredient] lot 610 ; Coconut Flake Desiccated [ingredient] lot 612 |
| 6020 | 2 | Coconut Flake Desiccated [ingredient] lot 613 ; Coconut Medium Desiccated [ingredient] lot 617 |
| 6036 | 2 | Coconut Flake Desiccated [ingredient] lot 614 ; Coconut Fancy Desiccated [ingredient] lot 635 |
| ABR 24 2026 | 2 | Granola Chocolate Chip 25 LB [finished] lot 465 ; Batch Coconut Sweetened Flake [batch] lot 466 |
| APR 14 2026 | 2 | Batch SS Chocolate Chip Granola #2 [batch] lot 405 ; Batch SS Original Granola #1 [batch] lot 406 |
| APR 23 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 457 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 496 |
| APR 30 2026 | 2 | Batch SS Chocolate Chip Granola #2 [batch] lot 508 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 510 |
| AUG 07 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 1140 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1142 |
| AUG 24 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 1330 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 1332 |
| B26-0302-002 | 2 | Batch SS Low Carb Original Granola #7 [batch] lot 252 ; Granola SS Original Low Carb 12x10 OZ Case [finished] lot 268 |
| B26-0302-003 | 2 | Batch SS Low Carb Chocolate Chip Granola #8 [batch] lot 254 ; Granola SS Chocolate Chip Low Carb 12x10 OZ Case [finished] lot 271 |
| B26-0304-001 | 2 | Batch Classic Granola #9 [batch] lot 265 ; Granola Classic 25 LB [finished] lot 269 |
| B26-0305-001 | 2 | Batch BS Almond Butter Granola 350 [batch] lot 270 ; BS Almond Butter Granola – 6x7 OZ Case [finished] lot 281 |
| BB030327 | 2 | Granola SS Original 12x10 OZ Case [finished] lot 262 ; Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 263 |
| BB041327 | 2 | Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 410 ; Granola SS Original 12x10 OZ Case [finished] lot 416 |
| BB061527 | 2 | Granola SS Original 12x10 OZ Case [finished] lot 820 ; Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 831 |
| BB081027 | 2 | Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 1159 ; Granola SS Original 12x10 OZ Case [finished] lot 1176 |
| BB082427 | 2 | Granola SS Chocolate Chip 12x10 OZ Case [finished] lot 1327 ; Granola SS Cranberry 12x10 OZ Case [finished] lot 1328 |
| FEB-04-2026 | 2 | Sweetened Flake Coconut [ingredient] lot 73 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 76 |
| FN-0219 | 2 | Granola Fruit Nut Batch [batch] lot 213 ; Granola Fruit Nut 25 LB [finished] lot 214 |
| JAN 20 2026 | 2 | Batch Granola Vanilla Almond 380 lb [batch] lot 276 ; Granola Vanilla Almond 25 LB [finished] lot 277 |
| JUL 14 2026 | 2 | Batch Classic Granola #9 [batch] lot 992 ; CQ Granola 10 LB [finished] lot 1001 |
| JUL 15 2026 | 2 | CQ Granola 10 LB [finished] lot 1002 ; Batch Classic Granola #9 [batch] lot 1003 |
| JUL 16 2026 | 2 | Batch Classic Granola #9 [batch] lot 1004 ; CQ Granola 10 LB [finished] lot 1007 |
| JUL 19 2026 | 2 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 1012 ; Batch Coconut Toasted Sweetened Flake [batch] lot 1014 |
| JUL 29 2026 | 2 | Batch Classic Granola #9 [batch] lot 1076 ; CQ Granola 10 LB [finished] lot 1087 |
| JUL 31 2026 | 2 | Batch Classic Granola #9 [batch] lot 1091 ; CQ Granola 10 LB [finished] lot 1093 |
| JUN 04 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 744 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 746 |
| JUN 05 2026 | 2 | Batch Coconut Toasted Sweetened Flake [batch] lot 749 ; Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 757 |
| JUN 11 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 806 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 808 |
| JUN 16 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 822 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 824 |
| JUN 21 2026 | 2 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 845 ; Batch Coconut Toasted Sweetened Flake [batch] lot 846 |
| LOT FEB 24 2026 | 2 | Coconut Sweetened Flake UNIPRO 10 LB [finished] lot 230 ; Coconut Sweetened Flake CNS 10 LB [finished] lot 231 |
| MAR 13 2026 | 2 | Batch Coconut Sweetened Fancy [batch] lot 313 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 315 |
| MAR 17 2026 | 2 | Batch BS Dark Chocolate Granola 350 [batch] lot 322 ; Batch Coconut Toasted Sweetened Flake [batch] lot 323 |
| MAR 20 2026 | 2 | Batch BS Dark Chocolate Granola 350 [batch] lot 345 ; BS Granola – Dark Chocolate – 6x7 OZ Case [finished] lot 452 |
| MAR 23 2026 | 2 | Batch Classic Granola #9 [batch] lot 351 ; Granola Classic 25 LB [finished] lot 355 |
| MAR 30 2026 | 2 | Batch Coconut Sweetened Fancy [batch] lot 382 ; Coconut Sweetened Fancy UNIPRO 10 LB [finished] lot 384 |
| MAY 04 2026 | 2 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 535 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 537 |
| MAY 25 2026 | 2 | Coconut Toasted Sweetened Flake CNS 25 LB [finished] lot 669 ; Batch Coconut Toasted Sweetened Flake [batch] lot 676 |
| MAY 31 2026 | 2 | Batch Coconut Sweetened Flake [batch] lot 708 ; CQ Coconut Sweetened Flake 10 LB [finished] lot 710 |
| SW2607708 | 2 | BS Granola – Peanut Butter Banana – 6x7 OZ Case [finished] lot 336 ; BS Granola – Peanut Butter Banana – 6x8 OZ Case [finished] lot 343 |

### 3i-ii. Every lot record coded "AUG 21 2026"

| lot id | product | SKU | type | entry_source | on-hand lb |
|---|---|---|---|---|---|
| 1318 | Batch Classic Granola #9 | 90002 | batch | production_output | 1588.0 |
| 1321 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | finished | pack_output | 40.0 |
| 1326 | Granola Classic 25 LB | 70050 | finished | pack_output | 350.0 |

### 3i-iii. Every transaction the backward trace would attach to code "AUG 21 2026" (all lots with that code)

| txn | type | business_date | product | SKU | qty lb | customer | status | lot |
|---|---|---|---|---|---|---|---|---|
| 2076 | make | 2026-08-21 | Batch Classic Granola #9 | 90002 | 1938.0000 | — | posted | lot 1318 |
| 2079 | pack | 2026-08-21 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | 200.0000 | — | posted | lot 1321 |
| 2087 | ship | 2026-08-24 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | -140.0000 | Dingman's Dairy | posted | lot 1321 |
| 2090 | ship | 2026-08-24 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | -20.0000 | Wards Ice Cream | posted | lot 1321 |
| 2093 | pack | 2026-08-24 | Granola Classic 25 LB | 70050 | 350.0000 | — | posted | lot 1326 |
| 2093 | pack | 2026-08-24 | Batch Classic Granola #9 | 90002 | -350.0000 | — | posted | lot 1318 |

### 3i-iv. Txns #2087 and #2090 as actually recorded

| txn | type | business_date | product | SKU | lot_code | qty lb | customer |
|---|---|---|---|---|---|---|---|
| 2087 | ship | 2026-08-24 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | JUL 27 2026 | -60.0000 | Dingman's Dairy |
| 2087 | ship | 2026-08-24 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | AUG 21 2026 | -140.0000 | Dingman's Dairy |
| 2090 | ship | 2026-08-24 | Coconut Toasted Sweetened Flake CNS 10 LB | 10010 | AUG 21 2026 | -20.0000 | Wards Ice Cream |

## 3j. On-hand drift — batch products (epoch vs today; made:packed over last 30 d; flag = on-hand rose >20% with both flows non-zero)

| batch product | on-hand @ epoch | on-hand today | made:packed 30 d | flag |
|---|---|---|---|---|
| Batch BS Dark Chocolate Granola 350 | 4,463.8 | 1,050.0 | 0:0 |  |
| Batch BS Peanut Butter Banana Granola | 2.4 | 2.4 | 3,616:3,614 (1.00) |  |
| Batch Classic Chocolate Chip Granola #9 | 17,748.0 | 0.0 | 3,828:0 |  |
| Batch Classic Granola #9 | 24,855.0 | 13,839.0 | 63,631:44,870 (1.42) |  |
| Batch Coconut Sweetened Fancy | 3,454.4 | 3,602.8 | 3,596:3,260 (1.10) |  |
| Batch Coconut Sweetened Flake | 24,061.6 | 26,478.8 | 66,334:61,135 (1.09) |  |
| Batch Coconut Sweetened Medium | 954.4 | 633.6 | 2,398:2,560 (0.94) |  |
| Batch Coconut Toasted Sweetened Flake | 600.0 | 500.0 | 0:2,000 (0.00) |  |
| Batch SS Chocolate Chip Granola #2 | 20,599.5 | 1,605.0 | 22,401:23,625 (0.95) |  |
| Batch SS Cranberry Granola #3 | 552.5 | 249.0 | 2,274:2,025 (1.12) |  |
| Batch SS Original Granola #1 | 3,120.0 | 3,650.0 | 6,300:2,400 (2.62) |  |
| Granola Fruit Nut Batch | 169.0 | 169.0 | 0:0 |  |

### 3j-ii. On-hand drift — top 20 finished SKUs by 30-d shipped lb (packed:shipped over 30 d)

| finished SKU | SKU code | on-hand @ epoch | on-hand today | packed:shipped 30 d | flag |
|---|---|---|---|---|---|
| CQ Granola 10 LB | 1614 | 4,200.0 | 1,400.0 | 35,000:33,600 (1.04) |  |
| CQ Coconut Sweetened Flake 10 LB | 893 | 8,540.0 | 23,290.0 | 47,590:28,000 (1.70) | **FLAG** |
| Granola SS Chocolate Chip 12x10 OZ Case | 70003 | 22,237.5 | 16,650.0 | 23,625:25,328 (0.93) |  |
| Coconut Toasted Sweetened Flake CNS 25 LB | 10029 | 1,600.0 | 1,050.0 | 1,900:12,750 (0.15) |  |
| Graham Cracker Crumbs – 10 LB | 31012 | 4,200.0 | 0.0 | 9,240:12,340 (0.75) |  |
| Granola SS Classic #9 Bulk per/lb | 70013 | 0.0 | 0.0 | 12,115:12,115 (1.00) |  |
| Kookies & Kreme – 25 LB | 10304 | 29,575.0 | 24,775.0 | 0:8,025 (0.00) |  |
| Sprinkles Rainbow 10 LB | 10302 | 17,960.0 | 10,680.0 | 0:7,680 (0.00) |  |
| BS Granola – Peanut Butter Banana – 6x7 OZ Case | 70073 | 0.0 | 0.0 | 3,614:6,241 (0.58) |  |
| Coconut Sweetened Flake UNIPRO 10 LB | 67476 | 6,370.0 | 3,170.0 | 8,770:5,600 (1.57) |  |
| Sprinkles Rainbow 25 LB | 10305 | 5,725.0 | 4,725.0 | 0:5,375 (0.00) |  |
| Coconut Sweetened Flake CNS 25 LB | 10020 | 3,450.0 | 0.0 | 3,625:3,625 (1.00) |  |
| Granola Classic 25 LB | 70050 | 1,375.0 | 2,300.0 | 5,175:3,325 (1.56) | **FLAG** |
| Granola SS Chocolate Chip Low Carb 12x10 OZ Case | 70070 | 1,050.0 | 0.0 | 0:2,618 (0.00) |  |
| Granola SS Original Low Carb 12x10 OZ Case | 70010 | 975.0 | 0.0 | 0:2,498 (0.00) |  |
| Coconut Sweetened Fancy UNIPRO 10 LB | 67470 | 1,740.0 | 260.0 | 3,260:2,400 (1.36) |  |
| Granola SS Cranberry 12x10 OZ Case | 70011 | 1,342.5 | 2,025.0 | 2,025:2,122 (0.95) | **FLAG** |
| Desiccated Flake 50 LB | 10047 | 104,100.0 | 102,500.0 | 0:2,000 (0.00) |  |
| Granola SS Original Bulk per/lb | 70004 | 0.0 | 0.0 | 1,875:1,875 (1.00) |  |
| Granola Crunchy CNS 10 LB Case | 10300 | 600.0 | 0.0 | 1,120:1,560 (0.72) |  |

## 3k. Customer entity hygiene

`customers` rows: 77. Normalized-name collisions (case/punctuation/suffix folded):

| normalized | customer records |
|---|---|
| GINSBERG S FOODS | #54 `Ginsberg's Foods`; #129 `Ginsberg’s Foods` |
| HIALEAH NEW URBAN FARMS | #33 `Hialeah / New Urban Farms`; #61 `Hialeah/New Urban Farms` |
| INTER COUNTY BAKERS | #9 `Inter-County Bakers`; #222 `Inter-County Bakers, Inc.` |
| RESTAURANT DEPOT JETRO HAINES CITY 407 | #10 `Restaurant Depot / Jetro Haines City #407`; #43 `RESTAURANT DEPOT/Jetro Haines City #407` |
| SPARTA FOODS DISTRIBUTORS | #39 `Sparta Foods Distributors`; #60 `Sparta Foods Distributors Inc.` |

### 3k-ii. Free-text `transactions.customer_name` collisions on ship txns (this is what the Sankey nodes come from)

| normalized | variants |
|---|---|
| GINSBERG S FOODS | `Ginsberg's Foods` (11×, last 2026-07-08); `Ginsberg’s Foods` (4×, last 2026-04-30) |
| HIALEAH NEW URBAN FARMS | `Hialeah / New Urban Farms` (6×, last 2026-03-19); `Hialeah/New Urban Farms` (1×, last 2026-04-14) |
| INTER COUNTY BAKERS | `Inter-County Bakers` (26×, last 2026-07-13); `Inter-County Bakers, Inc.` (7×, last 2026-08-12) |
| QUALI PACK | `Quali Pack` (1×, last 2026-02-09); `Quali-Pack` (1×, last 2026-02-09); `QUALI-PACK USA` (1×, last 2026-05-11) |
| RESTAURANT DEPOT JETRO HAINES CITY 407 | `Restaurant Depot / Jetro Haines City #407` (1×, last 2026-07-22); `RESTAURANT DEPOT/Jetro Haines City #407` (12×, last 2026-07-07) |
| SPARTA FOODS DISTRIBUTORS | `Sparta Foods Distributors` (3×, last 2026-03-03); `Sparta Foods Distributors Inc.` (14×, last 2026-08-04) |

### 3k-iii. Restaurant Depot / Jetro variants seen anywhere

| name | ship-txn count / source |
|---|---|
| Restaurant Depot | 3 |
| Restaurant Depot / Jetro Haines City #407 | 1 |
| Restaurant Depot LLC #499 Commerce | 2 |
| RESTAURANT DEPOT/Jetro Haines City #407 | 12 |
| Restaurant Depot/Jetro-RD #189 Chicago IL | 3 |
| Restaurant Depot/Jetro-RD #436 Streetsboro OH | 4 |
| Restaurant Depot | customers-table row #1 |
| Restaurant Depot / Jetro Haines City #407 | customers-table row #10 |
| Restaurant Depot LLC #499 Commerce | customers-table row #11 |
| RESTAURANT DEPOT/Jetro Haines City #407 | customers-table row #43 |
| Restaurant Depot/Jetro-RD #189 Chicago IL | customers-table row #73 |
| Restaurant Depot/Jetro-RD #436 Streetsboro OH | customers-table row #132 |
| Restaurant Depot #499 - Commerce Dist. Center GA | customers-table row #221 |

## 3l. Packs drawing from 3+ source lots, last 60 d

Count: **5**

| pack txn | business_date | source product | lots drawn | lot codes |
|---|---|---|---|---|
| 1599 | 2026-07-09 | Batch Coconut Sweetened Medium | 4 | JUL 08 2026, JUN 02 2026, JUN 08 2026, MAY 21 2026 |
| 1639 | 2026-07-13 | Batch SS Chocolate Chip Granola #2 | 3 | JUN 15 2026, JUN 17 2026, JUN 18 2026 |
| 1783 | 2026-07-29 | Batch Classic Granola #9 | 3 | JUN 26 2026, JUN 28 2026, JUN 29 2026 |
| 1843 | 2026-08-04 | Batch SS Chocolate Chip Granola #2 | 3 | JUN 18 2026, JUN 22 2026, JUN 23 2026 |
| 1921 | 2026-08-13 | Batch SS Chocolate Chip Granola #2 | 3 | JUL 08 2026, JUL 09 2026, JUL 20 2026 |

## Reading notes on the measured tables

- **3a (entry lag): occurred_at is NOT independently captured for floor entries.** Every write endpoint inserts `timestamp = get_plant_now()` and the 039 trigger derives `occurred_at` from it, so for live floor entries occurred ≡ entered (85–100% of live rows have |lag| < 2 min). Entry lag is therefore **not measurable** from this schema for normal floor work. The only >24 h lags post-epoch are the **72 txns of the 2026-08-17 inventory recon** (`operator_id = 'inv-recon-2026-08-17'`, backdated to business_date 2026-08-14, uniform ~75.8 h) — the one caller that ever set a true occurred-vs-entered gap, and it did so correctly.
- **3b:** the backfill boundary is sharp: 100% backfilled through the week of Aug 3, ~0% from Aug 11 on (epoch confirmed). The 68 live adjusts in the Aug-10 week are the recon.
- **3c:** 2026-08-17's 96-txn day with a 74-txn burst (77.1%) is the recon posting, not floor behavior; 2026-08-25's flag is a single-transaction day (trivially 100%). No true floor day exceeded 80%-in-one-burst in the window.
- **3d:** every weekday post-epoch has ≥1 make except 2026-08-25 (report generated that morning — first make is typically entered 13:25–16:35 ET, so this is expected timing, not necessarily a gap).
- **3e:** every posted ship txn in all windows has a customer_name and either an SO link or an order_reference. The free-text `order_reference` column itself is rarely used (0–13%); linkage lives in `sales_order_shipments`. By pounds, only 57.3% of post-epoch ship lb is SO-linked because the recon's large "forced" YTD ships carry only order_reference text.
- **3f:** supplier and BOL capture on receives is 100% in every window; expected-receipt matching is 0% because **zero `expected_receipts` rows have ever been created** (FR-2 deployed 2026-08-18, unused so far).
- **3g:** 0/49 batches fully resolve, but the gap is concentrated: 34 batches miss exactly 1 lot, 12 miss 2, 3 miss 3. The dominant unresolved lots are the found-inventory coconut lots `25120` (25 batches), `6013` (17), `26-02-03-FOUND-002` Flavor–Almond (9) — long-lived `found_inventory` lots with no supplier-bearing receive by design. Until those specific lots are either exhausted or retro-attributed to a supplier, batch-level trace completeness stays at 0%.
- **3i:** 159 lot codes resolve to >1 lot record; date-style codes routinely span 3–14 products in one day. For `AUG 21 2026` specifically: three lots share the code (Batch Classic Granola #9 lot 1318, Coconut Toasted Sweetened Flake CNS 10 LB lot 1321, Granola Classic 25 LB lot 1326). **Txns #2087 (−140 lb) and #2090 (−20 lb) on 2026-08-24 are confirmed ship lines of SKU 10010 Coconut Toasted Sweetened Flake CNS 10 LB from lot 1321 (Dingman's Dairy and Wards Ice Cream respectively) — not granola.** A code-only backward trace would nonetheless surface the granola make #2076 and pack #2093 under the same code (the API returns 409 ambiguous_lot_code unless product_id is passed).
- **3k:** beyond the normalized collisions shown, `Restaurant Depot LLC #499 Commerce` (customer #11) and `Restaurant Depot #499 - Commerce Dist. Center GA` (customer #221) are the same store under two names that do NOT collide under simple normalization — this pair (plus #10 vs #43 Haines City) is what splits the Sankey's Restaurant Depot/Jetro nodes.
- **3l (auto-excluded IDs):** `batch_formulas.exclude_from_inventory` marks formula ingredients that should not draw down inventory (e.g. untracked items). At /make preview+commit, `main.py:5373–5377` collects those ingredient product ids into `auto_excluded_ids`, unions them with any manually excluded ids, skips lot allocation for them, and stamps `"(auto-excluded IDs: [...])"` into the transaction notes (`main.py:5565`) so the exclusion is visible on the ledger record.

## 4. Code-only: why the forward trace from a batch lot dead-ends

**Function: `trace_ingredient` (`main.py:6761`, output-lot branch ~`main.py:6825`), driven by `traceForward` in `dashboard/traceability.html:700`.** The dashboard's forward trace calls `/trace/ingredient/{lot_code}`. For a batch lot (`entry_source='production_output'`) the endpoint takes the `is_output_lot` branch, which hard-codes `batches = []` with the comment "output lots aren't consumed as ingredients" — so `used_in_batches` comes back empty, the front-end renders "not consumed in any production batches", and `findSupplierForLot` (which only scans the last 100 *receive* txns) finds no receive for a produced lot, adding the "Supplier unknown" gap node.

The premise is false: **pack transactions DO consume batch lots and DO write `ingredient_lot_consumption` rows** (`main.py:6036`, `main.py:6083`).

**Fix (≤5 lines, not implemented):** in the `is_output_lot` branch of `trace_ingredient`, instead of `batches = []`, run the same downstream query used for raw ingredient lots (`SELECT ... FROM ingredient_lot_consumption ilc ... WHERE ilc.ingredient_lot_id = %s AND t.effective_status='posted'`), so a batch lot's `used_in_batches` returns the pack transactions' output lots; the existing front-end loop then follows each pack lot through `findShipmentsForLot` → ship → customer. (Cosmetic follow-on: for output lots, skip the supplier lookup and label the origin node "Produced in-house" instead of "Supplier unknown".)

## 5. Schema notes — what exists vs. what the metrics had to approximate

Fields that EXIST (verified in `tests/schema/schema.sql`, re-dumped from prod 2026-08-20):

| Concept | Actual field | Notes |
|---|---|---|
| occurred timestamp | `transactions.occurred_at` (NOT NULL) | derived by trigger from `"timestamp"`; never independently supplied by any endpoint |
| authoritative business day | `transactions.business_date` (NOT NULL) | ET date, trigger-derived |
| entered timestamp | `transactions.created_at` + `created_at_source` | `'database'` = live; `'migration_backfill_039'`/`'legacy_unverified'` = backfilled |
| backfilled flag | **none** — proxy is `created_at_source <> 'database'` | original insert time survives in legacy `"timestamp"` |
| txn source/actor | `transactions.operator_id` (default `'legacy-shared-key'`) | recon used `'inv-recon-2026-08-17'`; floor GPT entries are indistinguishable from each other |
| lot code / product / type / source | `lots.lot_code`, `lots.product_id`, `lots.lot_type`, `lots.entry_source`, `lots.supplier_lot_code`, `lots.received_at` | `entry_source`: received / found_inventory / adjusted / production_output / pack_output |
| product SKU | `products.odoo_code` | there is no column named `sku` |
| case size | `products.case_size_lb` (also `default_batch_lb`, `yield_multiplier`) | |
| product family | **no `family` column** — proxied here as parent batch product via `products.parent_batch_product_id`, else the product itself | `product_category` exists but is sparsely populated |
| ship customer | `transactions.customer_name` (free text) | **no customer_id FK on transactions** — entity linkage is by name string only |
| SO linkage of a ship | `sales_order_shipments.transaction_id` → `sales_order_lines` → `sales_orders` | `transactions.order_reference` free text is mostly blank |
| shipment record | `shipments` + `shipment_lines.transaction_id` | `shipments.transaction_id` itself is typically NULL; written mechanically per ship |
| receipt supplier/ref | `transactions.shipper_name`, `shipper_code`, `bol_reference` | free text; `suppliers` table exists (FR-2) but receives don't carry supplier_id |
| expected receipts | `expected_receipts` + `transactions.expected_receipt_id` | table has **0 rows ever**; `ledger_current_transactions` view does NOT expose `expected_receipt_id` (view predates migration 041) — join base `transactions` |
| batch ingredient consumption | `ingredient_lot_consumption` (txn, ingredient product, ingredient lot, lb) | written by /make AND /pack |
| formula | `batch_formulas` incl. `exclude_from_inventory` | source of "(auto-excluded IDs: ...)" |
| void/correction state | `ledger_corrections` + `ledger_current_transactions.effective_status` | always query `effective_status`, not raw `status` |

Fields that DO NOT exist (each absence is itself a finding):

- No `backfilled` boolean; no per-user actor identity on floor entries (shared key ⇒ `operator_id` is a source tag at best).
- No `customers.customer_id` FK anywhere in the transaction path — ship attribution is free-text `customer_name` (hence 3k collisions feed the Sankey directly).
- No `supplier_id` on receive transactions (FR-2's `suppliers` table is only reachable through `expected_receipts`, which is unused).
- No uniqueness on `lots.lot_code` (159 collisions) and no lot-code format constraint (7+ formats within one family, Spanish-month twins).
- No dispatch-proof columns (POD/photo/signature) on ships; `shipments.transaction_id` NULL by convention.
- No independent "occurred" capture: no endpoint accepts a caller-supplied occurred_at, so entry lag will remain unmeasurable until one does.
- `production_schedule` exists but is empty; `certifications` holds only two 1900-01-01 smoke rows.

<!-- Section 6 follows: generated by scripts/forms_crosscheck.py on 2026-08-25 after the form exports landed in data/forms/ (initially absent). -->
## 6. Forms vs FL cross-check (floor Google-Form exports as the independent record)

Forms read from `data/forms/` (not committed; gitignored). FL side: posted-only effective ledger, read-only via the 6543 pooler. 'FL entered' uses `created_at` for live rows and the legacy `"timestamp"` column for backfilled rows (their `created_at` is the 039 migration stamp). FL ledger coverage starts 2026-01-28 (first posted txn); first FL coconut make is 2026-02-05.

### 6a. Receiving form vs FL receipts (match: supplier from the CARRIER column + same business date, ±1 day fallback, then exact-BOL ±3 d; FL grouped into day×supplier delivery events since one truck spans several FL txns; names fuzzy-normalized — 'DUTC'-style lot prefixes never used for matching)

Form rows (Feb 2 → Aug 25 08:27): **112** · FL receives since Feb 1: **176** txns (148 real in **113** delivery events; 28 sentinel found/count rows excluded) · matched form-row↔event pairs: **75**

FL entered − form timestamp, by FL row source (backfilled rows' legacy `"timestamp"` is server-clock based, so a systematic ~+240 min there is a UTC-vs-ET labeling artifact, not real lag):
- `database` rows (n=4): median **-2 min**, range -28…-2; 4 pair(s) have FL entered BEFORE the form was submitted.
- `migration_backfill_039` rows (n=71): median **+235 min**, range -1997…+554; 1 pair(s) have FL entered BEFORE the form was submitted.

Matched pairs:

| form date | form carrier | FL shipper | FL txn(s) | FL business_date | FL entered − form (min) | match | FL row source |
|---|---|---|---|---|---|---|---|
| 2026-02-04 | Blue Stripes | Blue Stripes | 63 | 2026-02-04 | 294 | exact | migration_backfill_039 |
| 2026-02-05 | Star Snacks | Star Snacks | 79 | 2026-02-05 | 298 | exact | migration_backfill_039 |
| 2026-02-06 | Blue Stripes | Blue Stripes | 112 | 2026-02-06 | 297 | exact | migration_backfill_039 |
| 2026-02-09 | CBS Food | CBS Food | 129 | 2026-02-09 | 290 | exact | migration_backfill_039 |
| 2026-02-10 | Creative Foods | Creative Foods | 154, 155 | 2026-02-10 | 357 | exact | migration_backfill_039 |
| 2026-02-11 | Dutch Gold | Dutch Gold Honey | 175 | 2026-02-11 | 299 | contain | migration_backfill_039 |
| 2026-02-16 | Dutch Valley | Dutch Valley | 237, 238, 239, 240, 241, 242 | 2026-02-16 | 288 | exact | migration_backfill_039 |
| 2026-02-16 | EURO | Euro | 247 | 2026-02-16 | 297 | exact | migration_backfill_039 |
| 2026-02-17 | Jack’s Egg’s | Jack's Egg's | 255 | 2026-02-17 | 297 | exact | migration_backfill_039 |
| 2026-02-18 | Bender Warehouse | Blender Warehouse | 261 | 2026-02-18 | 305 | fuzzy 0.97 | migration_backfill_039 |
| 2026-02-24 | Moran Logistics | Phildesco c/o Moran Logistics | 305 | 2026-02-24 | 377 | contain | migration_backfill_039 |
| 2026-02-25 | Dutch Valley | Dutch Valley | 322, 323, 324 | 2026-02-25 | 293 | exact | migration_backfill_039 |
| 2026-02-25 | Star Snacks | Star Snacks | 325 | 2026-02-25 | 298 | exact | migration_backfill_039 |
| 2026-02-27 | Euro Good | Euro Good | 341 | 2026-02-27 | 321 | exact | migration_backfill_039 |
| 2026-03-03 | Blue Stripes | Blue Stripes | 352, 353 | 2026-03-03 | 291 | exact | migration_backfill_039 |
| 2026-03-04 | Jack’s Eggs | Jack’s Eggs | 365 | 2026-03-04 | 296 | exact | migration_backfill_039 |
| 2026-03-13 | CBS Food | CBS Food | 445 | 2026-03-13 | 236 | exact | migration_backfill_039 |
| 2026-03-25 | Blue Stripes | Blue Stripes | 538 | 2026-03-25 | 236 | exact | migration_backfill_039 |
| 2026-04-17 | Euro Good | Euro Goods | 642 | 2026-04-17 | 245 | exact | migration_backfill_039 |
| 2026-04-21 | Parker Flavor | Parker Flavors | 666 | 2026-04-21 | 237 | contain | migration_backfill_039 |
| 2026-04-21 | Essex Food | Essex Foods | 667, 668 | 2026-04-21 | 234 | exact | migration_backfill_039 |
| 2026-04-22 | Sam International | Sam International | 680 | 2026-04-22 | 200 | exact | migration_backfill_039 |
| 2026-04-24 | Tilley | Tilley | 699 | 2026-04-24 | 244 | exact | migration_backfill_039 |
| 2026-04-27 | Dutch Valley | Dutch Valley | 707, 708, 709, 710, 711, 712, 713 | 2026-04-27 | 216 | exact | migration_backfill_039 |
| 2026-04-28 | Dutch Gold | Dutch Gold | 738 | 2026-04-28 | 236 | exact | migration_backfill_039 |
| 2026-05-04 | CBS Food | CBS Food | 789 | 2026-05-04 | 228 | exact | migration_backfill_039 |
| 2026-05-04 | Dutch Valley | Dutch Valley | 790, 791, 792, 793, 798 | 2026-05-04 | 224 | exact | migration_backfill_039 |
| 2026-05-05 | Kadouri Food | Kadouri | 804 | 2026-05-05 | 350 | exact | migration_backfill_039 |
| 2026-05-05 | Jack’s Eggs | Jack's Eggs | 805 | 2026-05-05 | 232 | exact | migration_backfill_039 |
| 2026-05-07 | Sam International | Sam International | 836 | 2026-05-07 | 230 | exact | migration_backfill_039 |
| 2026-05-08 | Quali Pack | Quali Pack | 842 | 2026-05-08 | 188 | exact | migration_backfill_039 |
| 2026-05-11 | Blue Stripes | Blue Stripes | 846, 847 | 2026-05-11 | 230 | exact | migration_backfill_039 |
| 2026-05-13 | A-1 Bakery | A1 Bakery | 878 | 2026-05-13 | 233 | fuzzy 0.95 | migration_backfill_039 |
| 2026-05-26 | Jack’s Eggs | Jack's Eggs | 1059 | 2026-05-26 | 236 | exact | migration_backfill_039 |
| 2026-05-27 | Tri State | Tri State | 1066, 1067, 1068 | 2026-05-27 | 221 | exact | migration_backfill_039 |
| 2026-05-28 | CBS Food | CBS Food | 1082 | 2026-05-28 | 232 | exact | migration_backfill_039 |
| 2026-06-02 | Creative Food | Creative Foods | 1130, 1131 | 2026-06-02 | 229 | exact | migration_backfill_039 |
| 2026-06-05 | Quali Pack | Quali Pack | 1174 | 2026-06-05 | 207 | exact | migration_backfill_039 |
| 2026-06-05 | Euro Good | EURO Good | 1175 | 2026-06-05 | 233 | exact | migration_backfill_039 |
| 2026-06-08 | Dutch Valley | Dutch Valley | 1179 | 2026-06-08 | 235 | exact | migration_backfill_039 |
| 2026-06-09 | Creative Foods | Creative Foods | 1232 | 2026-06-09 | 234 | exact | migration_backfill_039 |
| 2026-06-09 | A1 Bakery | A1 Bakery | 1298 | 2026-06-09 | 213 | exact | migration_backfill_039 |
| 2026-06-10 | Tri State | Tri State | 1306, 1307 | 2026-06-10 | 233 | exact | migration_backfill_039 |
| 2026-06-10 | New England | NEW ENGLAND | 1312 | 2026-06-10 | 233 | exact | migration_backfill_039 |
| 2026-06-12 | Quali Pack | Quali Pack | 1352 | 2026-06-12 | 233 | exact | migration_backfill_039 |
| 2026-06-24 | Dutch Valley | Dutch Valley | 1419, 1421 | 2026-06-22 | -1997 | BOL, ±-2 d | migration_backfill_039 |
| 2026-06-24 | Star Snack | Star Snacks | 1440 | 2026-06-24 | 238 | contain | migration_backfill_039 |
| 2026-06-25 | CBS Food | CBS Food | 1455 | 2026-06-25 | 233 | exact | migration_backfill_039 |
| 2026-06-28 | Jack’s Eggs | Jack's Eggs | 1467 | 2026-06-28 | 233 | exact | migration_backfill_039 |
| 2026-06-29 | Dutch Valley | Dutch Valley | 1471, 1472 | 2026-06-29 | 224 | exact | migration_backfill_039 |
| 2026-06-30 | Quali Pack | Quali Pack | 1504 | 2026-06-30 | 228 | exact | migration_backfill_039 |
| 2026-06-30 | Blue Stripes | Blue Stripes | 1505 | 2026-06-30 | 231 | exact | migration_backfill_039 |
| 2026-07-01 | CBS Food | CBS Food | 1519 | 2026-07-01 | 234 | exact | migration_backfill_039 |
| 2026-07-06 | Jack’s Eggs | Jack's Eggs | 1550 | 2026-07-06 | 150 | exact | migration_backfill_039 |
| 2026-07-06 | Nation Harvest | National Harvest | 1551 | 2026-07-06 | 229 | fuzzy 0.93 | migration_backfill_039 |
| 2026-07-09 | Quali Pack | Quali Pack | 1594 | 2026-07-09 | 236 | exact | migration_backfill_039 |
| 2026-07-09 | Jack’s Eggs | Jack's Eggs | 1595 | 2026-07-09 | 238 | exact | migration_backfill_039 |
| 2026-07-10 | Dutch Gold | Dutch Gold | 1612 | 2026-07-10 | 236 | exact | migration_backfill_039 |
| 2026-07-13 | Essex Food | Essex Food | 1621, 1622 | 2026-07-13 | 234 | exact | migration_backfill_039 |
| 2026-07-15 | Quali Pack | Quali Pack | 1650 | 2026-07-15 | 236 | exact | migration_backfill_039 |
| 2026-07-21 | Parker Flavors | Parker Flavors | 1700, 1701, 1702 | 2026-07-21 | 536 | exact | migration_backfill_039 |
| 2026-07-24 | Refrieg-IT | Refrig-IT | 1734 | 2026-07-24 | 221 | fuzzy 0.95 | migration_backfill_039 |
| 2026-07-24 | CBS Food | CBS Food | 1746 | 2026-07-24 | 233 | exact | migration_backfill_039 |
| 2026-07-28 | Creative Foods | Creative Foods | 1760 | 2026-07-28 | 235 | exact | migration_backfill_039 |
| 2026-08-03 | Euro Good | Euro Good | 1805 | 2026-08-03 | 242 | exact | migration_backfill_039 |
| 2026-08-03 | Dutch Valley | Dutch Valley | 1806, 1807, 1808 | 2026-08-03 | 237 | exact | migration_backfill_039 |
| 2026-08-04 | Euro Good | Euro Good | 1827 | 2026-08-04 | 238 | exact | migration_backfill_039 |
| 2026-08-04 | A1 Bakery | A1 Bakery | 1825, 1826 | 2026-08-04 | 231 | exact | migration_backfill_039 |
| 2026-08-04 | Jack’s Eggs | Jack's Eggs | 1836 | 2026-08-04 | 207 | exact | migration_backfill_039 |
| 2026-08-05 | Essex Food Ingredients | Essex Food | 1852 | 2026-08-05 | 554 | contain | migration_backfill_039 |
| 2026-08-06 | Linking Logistics | Sweet New England | 1862 | 2026-08-06 | 237 | BOL | migration_backfill_039 |
| 2026-08-14 | Euro Good | Euro Good | 1930 | 2026-08-14 | -2 | exact | database |
| 2026-08-17 | Dutch Valley | Dutch Valley | 1941 | 2026-08-17 | -3 | exact | database |
| 2026-08-17 | A1 Bakery | A1 Bakery | 2028 | 2026-08-17 | -28 | exact | database |
| 2026-08-25 | CBS Food | CBS Food | 2100 | 2026-08-25 | -2 | exact | database |

Form rows with NO FL receipt:

| form date | carrier | BOL | products | carrier known to FL? |
|---|---|---|---|---|
| 2026-02-02 | Jack’s Eggs | 632054 | Sugar 6X-50lbs-225 | yes |
| 2026-02-03 | Jack’s Eggs | 632192 | Sugar brown-50lbs-100 | yes |
| 2026-02-09 | Moran Logistics |  | Dedicated flake-50lbs-196 | yes |
| 2026-02-10 | Prime Packaging | 84702007 | cardboard-2372-5075 | NO |
| 2026-02-16 | Prime packaging | 149982 | Cardboard 28F-2lbs-2000; Cardboard 1412-2lbs-1000 | NO |
| 2026-02-27 | Prime Packaging | 84721206 | Carboard9x-1875-4500 | NO |
| 2026-03-09 | Prime Packaging | 84736368 | Cardboard CNS1-2453lbs-5250 | NO |
| 2026-03-30 | Dutch Valley | 4491435 | Oats GF-50lbs-89 | yes |
| 2026-03-31 | Jack’s Eggs | 636617 | Sugar 6x-50lbs-157; Brown Sugar-50lbs-100 | yes |
| 2026-04-10 | Quali Pack | CV261767 | Oats-50lbs-875 | yes |
| 2026-04-10 | Dutch Gold | 432350026 | Honey-3000lbs-1 | yes |
| 2026-04-14 | Star Snacks | 41052 | Sunflowers-50lb-50 | yes |
| 2026-04-14 | Prime  Packing | 158613 | Cartons 28F-2000lbs-2000 | NO |
| 2026-04-14 | New England | 261136 | Sugar 6x-50lb-880 | yes |
| 2026-04-16 | Creative Food | 244661 | Graham Crumb-50lbs-810 | yes |
| 2026-04-23 | Euro Good | EG-002395 | Rainbow sprinkles-25lbs-98; Rainbow sprinkles-10lbs-99 | yes |
| 2026-04-28 | Pallets | 1421 | Pallets-1lbs-270 | NO |
| 2026-04-29 | Prime Packaging | 84794893 | Cardboard CNS1-2442lbs-5225 | NO |
| 2026-05-13 | Prime Packaging | 84819769 | Cardboard CNS1-2313lbs-4950 | NO |
| 2026-05-14 | Prime Packaging | 84821409 | Cardboard PL9X-1968.79lbs-4725 | NO |
| 2026-05-19 | Prime Packaging | 84824923 | Cardboard CN2-2000lbs-4500 | NO |
| 2026-05-19 | Prime Packaging | SS164799 | Cardboard 1412-1lbs-2000; Cardboard 28F-1lbs-2000 | NO |
| 2026-05-21 | Prime Packaging | 84825599 | Cardboard CNS1-2462lbs-5268; Cardboard CNS13-2442lbs-5250 | NO |
| 2026-06-08 | Prime packaging | 167589 | Clear Tape-1lbs-65 | NO |
| 2026-06-09 | Prime Packaging | 167935 | Cases 28F-1lbs-2000 | NO |
| 2026-06-19 | Prime Packaging | SS169732 | Cases 28F-1lbs-2000 | NO |
| 2026-06-25 | Prime Packaging | 170804 | Stretch wrap-20lbs-40 | NO |
| 2026-06-30 | A-1 Bakery | 67324 | Chocolate chips 4000ct-25lbs-80 | yes |
| 2026-07-10 | Prime Packaging | 84891728 | Cases-3748lbs-8020 | NO |
| 2026-07-14 | Prime Packaging | 173319 | Cardboard 28F-2000-1 | NO |
| 2026-07-14 | Prime Packaging | 173365 | Cardboard 1412-1lbs-2000 | NO |
| 2026-07-21 | Prime packaging | 1746665 | Bags clear-1lbs-45000 | NO |
| 2026-07-31 | Prime Packaging | 84915983 | Card board-1849lbs-3975 | NO |
| 2026-08-04 | Prime Packaging | 176583 | Cardboard 28F-1lbs-2000 | NO |
| 2026-08-18 |  | SS178866 | Cardboard 28f-1lbs-2000; Cardboard 14.12-1lbs-2000 | NO |
| 2026-08-20 | Prime Packaging | 84941502 | Cardboard CNS2-1998lbs-4275 | NO |
| 2026-08-21 | Prime Packaging | 84944528 | Cardboard CNS2-415lbs-889; Cardboard 13-2442lbs-5250 | NO |

FL delivery events (non-sentinel) with NO form row:

| FL txn(s) | business_date | shipper | BOL | source |
|---|---|---|---|---|
| 61 | 2026-02-03 | BLUE STRIPES | 0226 | migration_backfill_039 |
| 65 | 2026-02-04 | Acme Foods | 11111 | migration_backfill_039 |
| 64 | 2026-02-04 | Vinnapro | 98765 | migration_backfill_039 |
| 88, 90 | 2026-02-05 | SAEM | 3270, S679 | migration_backfill_039 |
| 113 | 2026-02-06 | Creative Foods | 12345 | migration_backfill_039 |
| 121 | 2026-02-06 | Jack's Egg's | 12345 | migration_backfill_039 |
| 130 | 2026-02-09 | Franklin Baker | Moran Logistics | migration_backfill_039 |
| 161 | 2026-02-10 | Barry Callebaut | 12345 | migration_backfill_039 |
| 230 | 2026-02-12 | Barry Callebaut | 12345 | migration_backfill_039 |
| 185 | 2026-02-12 | Blue Stripes | 1138 | migration_backfill_039 |
| 231 | 2026-02-12 | LaCrosse Milling | 12345 | migration_backfill_039 |
| 264 | 2026-02-18 | Dutch Valley | 12345 | migration_backfill_039 |
| 276 | 2026-02-19 | Grain Supply | GS-20260219 | migration_backfill_039 |
| 309 | 2026-02-24 | EURO | PHYSICAL-ADJUST-25132 | migration_backfill_039 |
| 330 | 2026-02-26 | Franklin Baker | 12345 | migration_backfill_039 |
| 349 | 2026-03-02 | JOEL | FEB-25-2026 | migration_backfill_039 |
| 398 | 2026-03-10 | Blue Stripes | 12345 | migration_backfill_039 |
| 418 | 2026-03-11 | David Rosen | 482885 | migration_backfill_039 |
| 434 | 2026-03-12 | Jack's Eggs | 635358 | migration_backfill_039 |
| 466 | 2026-03-18 | Star Snacks | 40934 | migration_backfill_039 |
| 492 | 2026-03-19 | Blue Stripes | 35932590 | migration_backfill_039 |
| 491 | 2026-03-19 | Dutch Gold | 22526 | migration_backfill_039 |
| 603 | 2026-04-14 | Dutch Valley Food Dist. | SO-260320-002 | migration_backfill_039 |
| 605 | 2026-04-14 | Linking Logistics | L7 Lot | migration_backfill_039 |
| 608, 611 | 2026-04-14 | Quali Pack | 250408, 260407 Lote | migration_backfill_039 |
| 685 | 2026-04-22 | Franklin Baker | FOUND-INVENTORY | migration_backfill_039 |
| 715 | 2026-04-27 | Quali Pack | 321654 | migration_backfill_039 |
| 753 | 2026-04-29 | CBS Food | 2026 | migration_backfill_039 |
| 754 | 2026-04-29 | Jack's Eggs | 2426 | migration_backfill_039 |
| 764 | 2026-04-30 | Star Snacks | 26-04-14-STAR-001 | migration_backfill_039 |
| 772 | 2026-05-01 | Euro Good | UNKNOWN | migration_backfill_039 |
| 783 | 2026-05-01 | Franklin Baker | 4062026 | migration_backfill_039 |
| 797 | 2026-05-04 | A1 Bakery Supply | 2604 | migration_backfill_039 |
| 1119 | 2026-06-01 | Blue Stripes | 85062 | migration_backfill_039 |
| 1420 | 2026-06-22 | DUTC Valley | 4523832 | migration_backfill_039 |
| 1824 | 2026-08-04 | Quali Pack | 9914 | migration_backfill_039 |
| 2029 | 2026-08-17 | A1 Baker | 69383 | database |
| 2082 | 2026-08-21 | Quali Pack | 10195 | database |

### 6b. Coconut form vs FL — pans (batches) per production day

Form rows: 177 (last submission 2026-08-24 16:47). Production day per row = DAILY BATCH LOT# embedded date vs submission date, arbitrated by the DAY PRODUCED weekday (heuristic fallback on 0 rows).

Days with coconut production on either side: **137** · exact pan agreement: **69** · pan-count mismatches: **31** · form-only days before FL coconut coverage (2026-02-05): **17** (expected — pre-go-live) · form-only days in FL's era: **11** (2026-02-11, 2026-02-25, 2026-03-09, 2026-03-18, 2026-03-25, 2026-04-10, 2026-04-13, 2026-04-24, 2026-05-25, 2026-05-27, 2026-07-20) · FL-only days: **9** (2026-02-26, 2026-03-17, 2026-05-01, 2026-05-08, 2026-05-15, 2026-06-05, 2026-06-26, 2026-07-09, 2026-07-17)

Per-day detail (every mismatch/one-sided day in FL's era, plus all days from Jul 1):

| day | form pans | FL pans | verdict |
|---|---|---|---|
| 2026-02-05 | 10.0 | 26.0 | Δ -16 |
| 2026-02-11 | 8.0 | — | FORM ONLY |
| 2026-02-12 | 12.0 | 28.0 | Δ -16 |
| 2026-02-16 | 12.0 | 13.0 | Δ -1 |
| 2026-02-25 | 10.0 | — | FORM ONLY |
| 2026-02-26 | — | 10.0 | FL ONLY |
| 2026-03-09 | 8.0 | — | FORM ONLY |
| 2026-03-10 | 10.0 | 18.0 | Δ -8 |
| 2026-03-17 | — | 3.0 | FL ONLY |
| 2026-03-18 | 3.0 | — | FORM ONLY |
| 2026-03-25 | 10.0 | — | FORM ONLY |
| 2026-03-26 | 6.0 | 21.0 | Δ -15 |
| 2026-04-10 | 4.0 | — | FORM ONLY |
| 2026-04-13 | 12.0 | — | FORM ONLY |
| 2026-04-14 | 12.0 | 17.0 | Δ -5 |
| 2026-04-16 | 12.0 | 24.0 | Δ -12 |
| 2026-04-17 | 2.0 | 8.0 | Δ -6 |
| 2026-04-20 | 11.0 | 5.0 | Δ +6 |
| 2026-04-24 | 2.0 | — | FORM ONLY |
| 2026-04-27 | 14.0 | 15.0 | Δ -1 |
| 2026-04-28 | 12.0 | 13.0 | Δ -1 |
| 2026-05-01 | — | 6.0 | FL ONLY |
| 2026-05-04 | 14.0 | 8.0 | Δ +6 |
| 2026-05-08 | — | 6.0 | FL ONLY |
| 2026-05-11 | 14.0 | 8.0 | Δ +6 |
| 2026-05-15 | — | 6.0 | FL ONLY |
| 2026-05-18 | 14.0 | 8.0 | Δ +6 |
| 2026-05-21 | 6.0 | 12.0 | Δ -6 |
| 2026-05-25 | 6.0 | — | FORM ONLY |
| 2026-05-27 | 10.0 | — | FORM ONLY |
| 2026-05-28 | 13.0 | 23.0 | Δ -10 |
| 2026-06-05 | — | 6.0 | FL ONLY |
| 2026-06-08 | 14.0 | 8.0 | Δ +6 |
| 2026-06-12 | 5.0 | 11.0 | Δ -6 |
| 2026-06-15 | 17.0 | 11.0 | Δ +6 |
| 2026-06-18 | 8.0 | 13.0 | Δ -5 |
| 2026-06-21 | 5.0 | 6.0 | Δ -1 |
| 2026-06-22 | 14.0 | 8.0 | Δ +6 |
| 2026-06-26 | — | 6.0 | FL ONLY |
| 2026-06-29 | 15.0 | 9.0 | Δ +6 |
| 2026-06-30 | 6.0 | 12.0 | Δ -6 |
| 2026-07-01 | 14.0 | 8.0 | Δ +6 |
| 2026-07-02 | 10.0 | 10.0 | ✓ |
| 2026-07-06 | 8.0 | 8.0 | ✓ |
| 2026-07-07 | 16.0 | 16.0 | ✓ |
| 2026-07-08 | 12.0 | 12.0 | ✓ |
| 2026-07-09 | — | 6.0 | FL ONLY |
| 2026-07-10 | 6.0 | 5.0 | Δ +1 |
| 2026-07-13 | 11.0 | 6.0 | Δ +5 |
| 2026-07-17 | — | 5.0 | FL ONLY |
| 2026-07-19 | 5.0 | 6.0 | Δ -1 |
| 2026-07-20 | 12.0 | — | FORM ONLY |
| 2026-07-21 | 12.0 | 18.0 | Δ -6 |
| 2026-07-22 | 3.0 | 9.0 | Δ -6 |
| 2026-07-23 | 6.0 | 6.0 | ✓ |
| 2026-07-24 | 6.0 | 6.0 | ✓ |
| 2026-07-26 | 6.0 | 6.0 | ✓ |
| 2026-07-27 | 12.0 | 6.0 | Δ +6 |
| 2026-07-28 | 12.0 | 12.0 | ✓ |
| 2026-07-30 | 10.0 | 10.0 | ✓ |
| 2026-08-03 | 11.0 | 11.0 | ✓ |
| 2026-08-04 | 10.0 | 10.0 | ✓ |
| 2026-08-05 | 12.0 | 12.0 | ✓ |
| 2026-08-06 | 12.0 | 12.0 | ✓ |
| 2026-08-07 | 8.0 | 8.0 | ✓ |
| 2026-08-10 | 12.0 | 12.0 | ✓ |
| 2026-08-11 | 12.0 | 12.0 | ✓ |
| 2026-08-12 | 12.0 | 12.0 | ✓ |
| 2026-08-13 | 12.0 | 12.0 | ✓ |
| 2026-08-14 | 8.0 | 8.0 | ✓ |
| 2026-08-17 | 12.0 | 12.0 | ✓ |
| 2026-08-18 | 8.0 | 8.0 | ✓ |
| 2026-08-20 | 12.0 | 12.0 | ✓ |
| 2026-08-24 | 12.0 | 12.0 | ✓ |

**Aug 20 check (6d):** form 12 pans vs FL 12 pans → **CONFIRMED equal.**

### 6b-ii. Coconut form vs FL — cases packed per day × product (FL's era; every non-agreeing pair plus all pairs from Jul 1)

| day | product | form cases | FL cases | verdict |
|---|---|---|---|---|
| 2026-02-05 | CQ Coconut Sweetened Flake 10 LB | 385.0 | — | FORM ONLY |
| 2026-02-06 | CQ Coconut Sweetened Flake 10 LB | 216.0 | — | FORM ONLY |
| 2026-02-09 | CQ Coconut Sweetened Flake 10 LB | 476.0 | — | FORM ONLY |
| 2026-02-10 | CQ Coconut Sweetened Flake 10 LB | 268.0 | — | FORM ONLY |
| 2026-02-11 | CQ Coconut Sweetened Flake 10 LB | 314.0 | — | FORM ONLY |
| 2026-02-12 | CQ Coconut Sweetened Flake 10 LB | 466.0 | — | FORM ONLY |
| 2026-02-13 | CQ Coconut Sweetened Flake 10 LB | 296.0 | — | FORM ONLY |
| 2026-02-18 | CQ Coconut Sweetened Flake 10 LB | 309.0 | — | FORM ONLY |
| 2026-02-19 | CQ Coconut Sweetened Flake 10 LB | 386.0 | 495.0 | Δ -109 |
| 2026-02-20 | CQ Coconut Sweetened Flake 10 LB | 307.0 | 311.0 | Δ -4 |
| 2026-02-25 | Coconut Sweetened Fancy UNIPRO 10 LB | 71.0 | — | FORM ONLY |
| 2026-02-25 | Coconut Sweetened Flake CNS 10 LB | 282.0 | — | FORM ONLY |
| 2026-02-25 | Coconut Sweetened Flake UNIPRO 10 LB | 20.0 | — | FORM ONLY |
| 2026-02-26 | Coconut Sweetened Fancy UNIPRO 10 LB | — | 71.0 | FL ONLY |
| 2026-02-26 | Coconut Sweetened Flake CNS 10 LB | — | 282.0 | FL ONLY |
| 2026-02-26 | Coconut Sweetened Flake UNIPRO 10 LB | — | 20.0 | FL ONLY |
| 2026-03-09 | CQ Coconut Sweetened Flake 10 LB | 307.0 | — | FORM ONLY |
| 2026-03-10 | CQ Coconut Sweetened Flake 10 LB | 383.0 | 690.0 | Δ -307 |
| 2026-03-12 | CQ Coconut Sweetened Flake 10 LB | 309.0 | — | FORM ONLY |
| 2026-03-13 | CQ Coconut Sweetened Flake 10 LB | — | 309.0 | FL ONLY |
| 2026-03-18 | Coconut Toasted Sweetened Flake CNS 25 LB | 19.0 | 1.0 | Δ +18 |
| 2026-03-25 | Coconut Sweetened Fancy UNIPRO 10 LB | 108.0 | — | FORM ONLY |
| 2026-03-25 | Coconut Sweetened Flake UNIPRO 10 LB | 253.0 | — | FORM ONLY |
| 2026-03-26 | Coconut Sweetened Fancy UNIPRO 10 LB | 72.0 | 180.0 | Δ -108 |
| 2026-03-26 | Coconut Sweetened Flake UNIPRO 10 LB | — | 253.0 | FL ONLY |
| 2026-03-30 | Coconut Sweetened Flake UNIPRO 10 LB | 219.0 | 218.0 | Δ +1 |
| 2026-04-10 | Coconut Sweetened Fancy UNIPRO 10 LB | 144.0 | — | FORM ONLY |
| 2026-04-13 | CQ Coconut Sweetened Flake 10 LB | 341.0 | — | FORM ONLY |
| 2026-04-13 | Coconut Sweetened Flake UNIPRO 10 LB | 101.0 | — | FORM ONLY |
| 2026-04-14 | CQ Coconut Sweetened Flake 10 LB | 439.0 | 341.0 | Δ +98 |
| 2026-04-14 | Coconut Sweetened Flake UNIPRO 10 LB | — | 101.0 | FL ONLY |
| 2026-04-15 | CQ Coconut Sweetened Flake 10 LB | 434.0 | 439.0 | Δ -5 |
| 2026-04-16 | CQ Coconut Sweetened Flake 10 LB | 438.0 | 872.0 | Δ -434 |
| 2026-04-17 | Coconut Sweetened Medium UNIPRO 10 LB | 72.0 | — | FORM ONLY |
| 2026-04-20 | Coconut Sweetened Flake UNIPRO 10 LB | — | 72.0 | FL ONLY |
| 2026-04-20 | Coconut Sweetened Medium UNIPRO 10 LB | — | 72.0 | FL ONLY |
| 2026-04-20 | Coconut Toasted Sweetened Flake CNS 25 LB | 72.0 | 60.0 | Δ +12 |
| 2026-04-22 | CQ Coconut Sweetened Flake 10 LB | 292.0 | 287.0 | Δ +5 |
| 2026-04-24 | CQ Coconut Sweetened Flake 10 LB | 74.0 | — | FORM ONLY |
| 2026-04-27 | CQ Coconut Sweetened Flake 10 LB | — | 74.0 | FL ONLY |
| 2026-04-27 | Coconut Sweetened Fancy UNIPRO 10 LB | 250.0 | — | FORM ONLY |
| 2026-04-27 | Coconut Sweetened Medium UNIPRO 10 LB | 35.0 | — | FORM ONLY |
| 2026-04-27 | Coconut Toasted Sweetened Flake CNS 25 LB | 76.0 | — | FORM ONLY |
| 2026-04-28 | Coconut Sweetened Fancy UNIPRO 10 LB | — | 250.0 | FL ONLY |
| 2026-04-28 | Coconut Sweetened Medium UNIPRO 10 LB | — | 35.0 | FL ONLY |
| 2026-05-01 | Coconut Sweetened Fancy CNS 10 LB | — | 40.0 | FL ONLY |
| 2026-05-01 | Coconut Sweetened Fancy UNIPRO 10 LB | — | 40.0 | FL ONLY |
| 2026-05-11 | CQ Coconut Sweetened Flake 10 LB | 285.0 | 292.0 | Δ -7 |
| 2026-05-18 | CQ Coconut Sweetened Flake 10 LB | 277.0 | 279.0 | Δ -2 |
| 2026-05-26 | CQ Coconut Sweetened Flake 10 LB | 127.0 | 218.0 | Δ -91 |
| 2026-05-26 | Coconut Sweetened Flake UNIPRO 10 LB | 91.0 | — | FORM ONLY |
| 2026-05-27 | CQ Coconut Sweetened Flake 10 LB | 365.0 | — | FORM ONLY |
| 2026-05-28 | CQ Coconut Sweetened Flake 10 LB | — | 338.0 | FL ONLY |
| 2026-06-01 | CQ Coconut Sweetened Flake 10 LB | 437.0 | 464.0 | Δ -27 |
| 2026-06-01 | Coconut Sweetened Fancy UNIPRO 10 LB | — | 5.0 | FL ONLY |
| 2026-06-04 | Coconut Toasted Sweetened Flake CNS 10 LB | — | 10.0 | FL ONLY |
| 2026-06-11 | Coconut Sweetened Medium UNIPRO 10 LB | — | 6.0 | FL ONLY |
| 2026-06-16 | CQ Coconut Sweetened Flake 10 LB | 219.0 | 218.0 | Δ +1 |
| 2026-06-17 | CQ Coconut Sweetened Flake 10 LB | 218.0 | 219.0 | Δ -1 |
| 2026-06-18 | CQ Coconut Sweetened Flake 10 LB | 292.0 | 291.0 | Δ +1 |
| 2026-06-23 | Coconut Sweetened Flake CNS 10 LB | — | 10.0 | FL ONLY |
| 2026-06-24 | Coconut Sweetened Flake CNS 10 LB | — | 5.0 | FL ONLY |
| 2026-07-01 | CQ Coconut Sweetened Flake 10 LB | 293.0 | 293.0 | ✓ |
| 2026-07-01 | Coconut Toasted Sweetened Flake CNS 10 LB | 40.0 | 40.0 | ✓ |
| 2026-07-01 | Coconut Toasted Sweetened Flake CNS 25 LB | 58.0 | 58.0 | ✓ |
| 2026-07-02 | CQ Coconut Sweetened Flake 10 LB | 366.0 | 366.0 | ✓ |
| 2026-07-02 | Coconut Toasted Sweetened Flake CNS 25 LB | — | 28.0 | FL ONLY |
| 2026-07-06 | CQ Coconut Sweetened Flake 10 LB | 291.0 | 291.0 | ✓ |
| 2026-07-07 | CQ Coconut Sweetened Flake 10 LB | 198.0 | 198.0 | ✓ |
| 2026-07-08 | CQ Coconut Sweetened Flake 10 LB | 280.0 | — | FORM ONLY |
| 2026-07-08 | Coconut Sweetened Flake UNIPRO 10 LB | — | 280.0 | FL ONLY |
| 2026-07-09 | Coconut Sweetened Flake CNS 10 LB | 40.0 | 40.0 | ✓ |
| 2026-07-09 | Coconut Sweetened Medium UNIPRO 10 LB | 6.0 | 42.0 | Δ -36 |
| 2026-07-10 | Coconut Sweetened Flake CNS 25 LB | — | 26.0 | FL ONLY |
| 2026-07-10 | Coconut Toasted Sweetened Flake CNS 25 LB | 75.0 | 75.0 | ✓ |
| 2026-07-13 | Coconut Sweetened Fancy UNIPRO 10 LB | 200.0 | 200.0 | ✓ |
| 2026-07-13 | Coconut Sweetened Flake CNS 25 LB | 40.0 | 40.0 | ✓ |
| 2026-07-13 | Coconut Sweetened Medium UNIPRO 10 LB | 20.0 | 20.0 | ✓ |
| 2026-07-13 | Coconut Toasted Sweetened Flake CNS 25 LB | 62.0 | 62.0 | ✓ |
| 2026-07-19 | Coconut Toasted Sweetened Flake CNS 25 LB | 63.0 | 63.0 | ✓ |
| 2026-07-20 | Coconut Sweetened Flake UNIPRO 10 LB | 219.0 | — | FORM ONLY |
| 2026-07-20 | Coconut Toasted Sweetened Flake CNS 25 LB | 75.0 | — | FORM ONLY |
| 2026-07-21 | Coconut Sweetened Flake UNIPRO 10 LB | 481.0 | 700.0 | Δ -219 |
| 2026-07-21 | Coconut Toasted Sweetened Flake CNS 10 LB | — | 5.0 | FL ONLY |
| 2026-07-21 | Coconut Toasted Sweetened Flake CNS 25 LB | — | 73.0 | FL ONLY |
| 2026-07-22 | Coconut Sweetened Fancy UNIPRO 10 LB | 140.0 | 140.0 | ✓ |
| 2026-07-22 | Coconut Sweetened Flake CNS 10 LB | 100.0 | 100.0 | ✓ |
| 2026-07-22 | Coconut Sweetened Medium UNIPRO 10 LB | 40.0 | 40.0 | ✓ |
| 2026-07-23 | Coconut Toasted Sweetened Flake CNS 25 LB | 74.0 | 74.0 | ✓ |
| 2026-07-24 | Coconut Toasted Sweetened Flake CNS 25 LB | 75.0 | 75.0 | ✓ |
| 2026-07-26 | Coconut Toasted Sweetened Flake CNS 10 LB | 55.0 | 55.0 | ✓ |
| 2026-07-26 | Coconut Toasted Sweetened Flake CNS 25 LB | 55.0 | 53.0 | Δ +2 |
| 2026-07-27 | CQ Coconut Sweetened Flake 10 LB | 133.0 | 133.0 | ✓ |
| 2026-07-27 | Coconut Sweetened Flake CNS 10 LB | 25.0 | 25.0 | ✓ |
| 2026-07-27 | Coconut Sweetened Flake CNS 25 LB | 25.0 | 25.0 | ✓ |
| 2026-07-27 | Coconut Toasted Sweetened Flake CNS 25 LB | 76.0 | 76.0 | ✓ |
| 2026-07-28 | CQ Coconut Sweetened Flake 10 LB | 439.0 | 439.0 | ✓ |
| 2026-07-30 | CQ Coconut Sweetened Flake 10 LB | 369.0 | 369.0 | ✓ |
| 2026-08-03 | Coconut Sweetened Flake UNIPRO 10 LB | 291.0 | 291.0 | ✓ |
| 2026-08-03 | Coconut Sweetened Medium UNIPRO 10 LB | 108.0 | 108.0 | ✓ |
| 2026-08-04 | Coconut Sweetened Fancy UNIPRO 10 LB | 181.0 | 181.0 | ✓ |
| 2026-08-04 | Coconut Sweetened Flake UNIPRO 10 LB | 147.0 | 147.0 | ✓ |
| 2026-08-04 | Coconut Sweetened Medium CNS 10 LB | 36.0 | 36.0 | ✓ |
| 2026-08-05 | Coconut Sweetened Flake CNS 10 LB | 80.0 | 80.0 | ✓ |
| 2026-08-05 | Coconut Sweetened Flake UNIPRO 10 LB | 359.0 | 359.0 | ✓ |
| 2026-08-06 | CQ Coconut Sweetened Flake 10 LB | 79.0 | 60.0 | Δ +19 |
| 2026-08-06 | Coconut Sweetened Flake CNS 25 LB | 120.0 | 120.0 | ✓ |
| 2026-08-06 | Coconut Sweetened Flake UNIPRO 10 LB | 80.0 | 80.0 | ✓ |
| 2026-08-07 | CQ Coconut Sweetened Flake 10 LB | 293.0 | 293.0 | ✓ |
| 2026-08-10 | CQ Coconut Sweetened Flake 10 LB | 440.0 | 440.0 | ✓ |
| 2026-08-11 | CQ Coconut Sweetened Flake 10 LB | 440.0 | 440.0 | ✓ |
| 2026-08-11 | Coconut Toasted Sweetened Flake CNS 10 LB | — | 10.0 | FL ONLY |
| 2026-08-12 | CQ Coconut Sweetened Flake 10 LB | 441.0 | 441.0 | ✓ |
| 2026-08-13 | CQ Coconut Sweetened Flake 10 LB | 220.0 | 220.0 | ✓ |
| 2026-08-13 | Coconut Sweetened Fancy UNIPRO 10 LB | 145.0 | 145.0 | ✓ |
| 2026-08-13 | Coconut Sweetened Medium UNIPRO 10 LB | 72.0 | 72.0 | ✓ |
| 2026-08-14 | CQ Coconut Sweetened Flake 10 LB | 295.0 | 295.0 | ✓ |
| 2026-08-17 | CQ Coconut Sweetened Flake 10 LB | 453.0 | 453.0 | ✓ |
| 2026-08-18 | CQ Coconut Sweetened Flake 10 LB | 296.0 | 296.0 | ✓ |
| 2026-08-19 | Coconut Sweetened Medium CNS 10 LB | — | 40.0 | FL ONLY |
| 2026-08-20 | CQ Coconut Sweetened Flake 10 LB | 429.0 | 439.0 | Δ -10 |
| 2026-08-20 | Coconut Sweetened Flake CNS 10 LB | 10.0 | 10.0 | ✓ |
| 2026-08-21 | Coconut Toasted Sweetened Flake CNS 10 LB | — | 20.0 | FL ONLY |
| 2026-08-24 | CQ Coconut Sweetened Flake 10 LB | 441.0 | 441.0 | ✓ |

### 6b-iii. Physical case lot# (form) vs FL pack output lot — the F4 error size

Matched form pack lines (same day+product packed in FL, form lot present): **30840 cases**. Attributed in FL to a lot with a DIFFERENT embedded date than the one physically on the cases: **1911 cases = 6.2%** (month-spelling, 'Lot' noise, and pure format differences — e.g. `Apr 20 2026` vs `2026-04-20` — normalized away first; what remains is genuine day-level misattribution).

| day | product | form cases | physical lot on cases | FL pack output lot(s) that day |
|---|---|---|---|---|
| 2026-03-30 | Coconut Sweetened Flake UNIPRO 10 LB | 219.0 | Mar 30 2026 | MAR 03 2026 |
| 2026-04-14 | CQ Coconut Sweetened Flake 10 LB | 360.0 | Apr 14 2026 | APR 13 2026 |
| 2026-04-15 | CQ Coconut Sweetened Flake 10 LB | 374.0 | Apr 15 2026 | 2026-04-13, 2026-04-14 |
| 2026-04-30 | CQ Coconut Sweetened Flake 10 LB | 93.0 | Apr 29 2026 | APR 30 2026 |
| 2026-05-18 | Coconut Toasted Sweetened Flake CNS 25 LB | 74.0 | May 18 2026 | MAY 15 2026 |
| 2026-06-08 | Coconut Toasted Sweetened Flake CNS 25 LB | 77.0 | Jun 08 2026 | JUN 05 2026 |
| 2026-06-09 | Coconut Sweetened Flake CNS 25 LB | 100.0 | Jan 09 2026 | JUN 09 2026 |
| 2026-06-09 | Coconut Sweetened Flake UNIPRO 10 LB | 192.0 | Jan 09 2026 | JUN 09 2026 |
| 2026-06-18 | CQ Coconut Sweetened Flake 10 LB | 1.0 | Jun 16 2026 | JUN 17 2026, JUN 18 2026 |
| 2026-07-30 | CQ Coconut Sweetened Flake 10 LB | 341.0 | Jul 30 3026 | JUL 28 2026, JUL 30 2026 |
| 2026-08-06 | Coconut Sweetened Flake UNIPRO 10 LB | 80.0 | Aug 06 3026 | AUG 06 2026 |

**8/7 79-case check:** the form (submitted 08/07 08:05, production day 2026-08-06) reports **79 cases of CQ Coconut Sweetened Flake 10 LB packed from lot `Jul 30 2026`**. FL records the Jul-30-lot CQ packs in August as: txn 1876 on 2026-08-06 60 cases; txn 1880 on 2026-08-07 19 cases — i.e. FL split the same 79 Jul-30-lot cases across two business days (60 + 19) instead of the form's single 79-case event, and a further form line (8/7 afternoon, 19 cases `Jul 30 2026`) overlaps the FL 8/7 19-case pack. Case totals reconcile at the lot level (79 form-morning vs 60+19 FL) but day attribution differs — flagged as requested.

### 6c. Lot-code sanity on form free-text lot fields

Issues found: **18** (same-day fields — receiving sticker lot#, DAILY BATCH LOT#, case lot# — checked for >7 d date drift; ingredient lot#s are legitimately older stock, so they are only checked for impossible years, month misspellings, and FUTURE dates).

| issue type | n |
|---|---|
| embedded date … | 13 |
| misspelled/Spanish month '…' | 2 |
| implausible year 3026 | 2 |
| implausible year 2027 | 1 |

Full list:

| form | row date | field | code | issue |
|---|---|---|---|---|
| receiving | 2026-04-21 | What lot# did you use? | 260521PARK001 | embedded date 2026-05-21 is 30 d from row date |
| receiving | 2026-08-17 | What lot# did you use? | 260827ABAK | embedded date 2026-08-27 is 10 d from row date |
| coconut | 2026-01-02 | case lot# | De 30 2025 | misspelled/Spanish month 'DE' |
| coconut | 2026-01-05 | DAILY BATCH LOT# | Jan 05 2025 | embedded date 2025-01-05 is 365 d from row date |
| coconut | 2026-01-05 | case lot# | Dec 24 2025 | embedded date 2025-12-24 is 12 d from row date |
| coconut | 2026-01-05 | case lot# | Jan 05 2025 | embedded date 2025-01-05 is 365 d from row date |
| coconut | 2026-01-21 | case lot# | Dec 15 2025 | embedded date 2025-12-15 is 37 d from row date |
| coconut | 2026-03-09 | case lot# | Feb 20 2026 | embedded date 2026-02-20 is 17 d from row date |
| coconut | 2026-03-26 | ingredient lot# (sugar, 6x) | 261230JA | embedded date 2026-12-30 is 279 d in the FUTURE |
| coconut | 2026-06-09 | DAILY BATCH LOT# | Jan 09 2026 | embedded date 2026-01-09 is 151 d from row date |
| coconut | 2026-06-09 | case lot# | Jan 09 2026 | embedded date 2026-01-09 is 151 d from row date |
| coconut | 2026-06-09 | case lot# | Jan 09 2026 | embedded date 2026-01-09 is 151 d from row date |
| coconut | 2026-07-27 | case lot# | Jul07 2026 | embedded date 2026-07-07 is 20 d from row date |
| coconut | 2026-07-30 | case lot# | Jul 30 3026 | implausible year 3026 |
| coconut | 2026-08-04 | DAILY BATCH LOT# | Aug 04 2027 | implausible year 2027 |
| coconut | 2026-08-06 | case lot# | Aug 06 3026 | implausible year 3026 |
| coconut | 2026-08-07 | case lot# | Jul 30 2026 | embedded date 2026-07-30 is 8 d from row date |
| coconut | 2026-08-24 | case lot# | Agu 24 2026 | misspelled/Spanish month 'AGU' |

### 6e. Shipping form vs FL ship transactions (compared at day × normalized-customer grain, ±1 day; one truck can span several FL txns/form rows)

**The shipping form export contains NO product, case-count, or lot-number columns** — its fields are timestamp, name, carrier, customer, BOL photo link, BOL#, and three condition checkboxes. The requested packing-slip-lot vs FL-allocated-lot comparison and quantity mismatches are therefore NOT computable from this export; '% of ship lines where FL's lot ≠ the slip's lot' would require the linked BOL photos or per-line form fields. Presence matching only, below.

Form rows (Jan 28 → Aug 24 12:06): **219** = **215** day×customer events · FL ship txns since Jan 28: **483** = **192** day×customer events · events matched: **162** · form-only events: **55** · FL-only events: **30**

By month:

| month | form events | FL events | form-only | FL-only |
|---|---|---|---|---|
| 2026-01 | 3 | 0 | 3 | 0 |
| 2026-02 | 27 | 8 | 20 | 1 |
| 2026-03 | 39 | 42 | 7 | 10 |
| 2026-04 | 25 | 17 | 14 | 6 |
| 2026-05 | 28 | 31 | 3 | 5 |
| 2026-06 | 34 | 37 | 3 | 5 |
| 2026-07 | 32 | 29 | 5 | 2 |
| 2026-08 | 27 | 28 | 0 | 1 |

Form shipment events with NO FL ship (from Jul 1; earlier months summarized above):

| date | customer | carrier | BOL# |
|---|---|---|---|
| 2026-07-02 | Grassland | Grassland | 27934 |
| 2026-07-08 | Clark | Clark | 8619496 |
| 2026-07-16 | DiCarlo Food | DiCarlo Food | 9337210 |
| 2026-07-17 | Roasted Coffee | Roasted Coffee | 28311 |
| 2026-07-20 | Bakers Depot | Bakers Depot | 21551 |

FL ship events with NO form row (from Jul 1):

| date | customer | FL txns | lb |
|---|---|---|---|
| 2026-07-13 | DiCarlo Food Service | 4 | 4150 |
| 2026-07-22 | Restaurant Depot / Jetro Haines City #407 | 1 | 36400 |
| 2026-08-14 | Sunshine Granola | 6 | 46668 |

### 6f. Form freshness

| form | last submission |
|---|---|
| coconut_batches.csv | 2026-08-24 16:47 |
| receiving.csv | 2026-08-25 08:27 |
| shipping.csv | 2026-08-24 12:06 |

### 6g. Reading notes on the Forms-vs-FL tables

- **Receiving (6a):** 75/112 form rows match an FL delivery event. Of the 37 unmatched form rows, **21 are Prime Packaging / Pallets / packaging-consumables deliveries whose supplier does not exist in FL at all** ("carrier known to FL? = NO") — packaging receipts are simply out of FL's scope, which is the single biggest reason the two records diverge. The remainder are ingredient deliveries FL genuinely lacks (e.g. Jack's Eggs sugar 2/2 and 2/3, Dutch Valley oats 3/30, Quali Pack oats 4/10, Euro Good sprinkles 4/23). In the other direction, **the Quali Pack 2026-08-04 receipt (FL txn 1824, BOL 9914) has no form row — confirming the expected gap** — alongside 41 other FL delivery events with no form (heavily: multi-truck Dutch Valley days and the Feb backfill era).
- **Receiving timing (6a):** for the only 4 live (post-epoch) matched pairs, FL was entered a median **2 min BEFORE** the form was submitted (range −28…−2 min) — FL entry precedes the paper form, as suspected; the sign matters. The backfilled rows' +235 min median is a UTC-vs-ET clock-labeling artifact of the legacy `"timestamp"` column, not real lag.
- **Coconut pans (6b):** post-epoch the two records agree perfectly — **every production day from Aug 3 onward matches pan-for-pan** (incl. **Aug 20: form 12 = FL 12, CONFIRMED**). All 31 pan-count mismatch days and nearly all one-sided days are pre-epoch, and the dominant pattern is a **±6-pan pair on adjacent days** (e.g. 2/25 form-only ↔ 2/26 FL-only; 3/12 ↔ 3/13; 3/25 ↔ 3/26): the backfilled FL entry landed on the day after the floor's production day. "Aug 21 in FL, not the form" holds for **packs**: FL pack txn 2079 (20 cases Coconut Toasted Sweetened Flake CNS 10 LB, lot `AUG 21 2026`) has no coconut-form row — the form recorded no coconut work that day (FL also has no coconut *make* on 8/21; the 8/21 make was Classic Granola).
- **F4 error size (6b-iii): 6.2% of matched cases (1,911 of 30,840)** carry a physical case lot# whose embedded date differs from the FL pack-output lot that day — genuine day-level lot misattribution after normalizing month spellings and formats. The list is dominated by day-shifted codes (Apr 14 vs APR 13, Jun 08 vs JUN 05) and typo'd months/years (`Jan 09 2026` written for JUN 09; `Jul 30 3026`).
- **The 8/7 79-case pack** reconciles at the lot level but not the day level: form says 79 CQ cases from lot `Jul 30 2026` on production day 8/6; FL recorded 60 cases on 8/6 (txn 1876) + 19 on 8/7 (txn 1880) from that lot.
- **Lot-code sanity (6c): 18 real defects** in the free-text lot fields, incl. two year-3026 codes (`Jul 30 3026`, `Aug 06 3026`), one 2027 (`Aug 04 2027`), misspelled months (`Agu 24 2026`, `De 30 2025`), a future-dated sugar lot (`261230JA` = Dec 30 used in March), the receiving sticker `260827ABAK` written on Aug 17, and a January/June transposition (`Jan 09 2026` on Jun 9 — the same defect that shows up in 6b-iii).
- **Shipping (6e):** the form and FL agree well from May onward (Aug: 0 form-only, 1 FL-only); February's 20 form-only events are the FL pre/early-go-live gap, mirroring coconut. The 3 FL-only events since Jul 1 are explained: DiCarlo 7/13 pairs with the form's 7/16 row (3-day offset, beyond the ±1 d window), Restaurant Depot/Jetro 7/22 (36,400 lb, single txn) has no form row, and Sunshine Granola 8/14 (6 txns, 46,668 lb) is the 8/17 inventory-recon YTD forced ships — not a real truck. **No product/case/lot columns exist in this form**, so slip-lot-vs-FL-lot error cannot be measured from it (schema finding: to get the shipping F4 equivalent, the form needs per-line product/cases/lot fields like the coconut form, or the BOL photos must be read).
### 7. Backfill re-measure — entry basis = legacy `"timestamp"` for migration_backfill_039 rows (their created_at is just the 039 stamp), created_at for live rows. Weekly, by business week from Jul 1:

| week | txns | % same-day | % next-day | % ≥2 d (true backfill / >24 h certain) | % unknowable |
|---|---|---|---|---|---|
| 2026-06-29 | 31 | 100.0 | 0.0 | 0.0 | 0.0 |
| 2026-07-06 | 68 | 100.0 | 0.0 | 0.0 | 0.0 |
| 2026-07-13 | 63 | 100.0 | 0.0 | 0.0 | 0.0 |
| 2026-07-20 | 70 | 97.1 | 1.4 | 1.4 | 0.0 |
| 2026-07-27 | 50 | 96.0 | 0.0 | 4.0 | 0.0 |
| 2026-08-03 | 81 | 100.0 | 0.0 | 0.0 | 0.0 |
| 2026-08-10 | 128 | 43.8 | 0.0 | 56.2 | 0.0 |
| 2026-08-17 | 67 | 100.0 | 0.0 | 0.0 | 0.0 |
| 2026-08-24 | 19 | 100.0 | 0.0 | 0.0 | 0.0 |

By product family (Jul 1 → today):

| family | txns | % same-day | % next-day | % ≥2 d true backfill | % unknowable | latest truly-backfilled business_date |
|---|---|---|---|---|---|---|
| Granola (all) | 228 | 72.4 | 0.4 | 27.2 | 0.0 | 2026-08-14 |
| Other/ingredient-level | 122 | 98.4 | 0.0 | 1.6 | 0.0 | 2026-08-14 |
| Coconut Flake | 106 | 95.3 | 0.0 | 4.7 | 0.0 | 2026-08-14 |
| Coconut Toasted Flake | 45 | 97.8 | 0.0 | 2.2 | 0.0 | 2026-08-14 |
| Receives (ingredients) | 32 | 100.0 | 0.0 | 0.0 | 0.0 | — |
| Coconut Medium | 25 | 88.0 | 0.0 | 12.0 | 0.0 | 2026-08-14 |
| Coconut Fancy | 19 | 89.5 | 0.0 | 10.5 | 0.0 | 2026-08-14 |

Weekly × family: truly-backfilled / total txns:

| week | Coconut Medium | Coconut Flake | Coconut Fancy | Coconut Toasted Flake | Granola (all) | Receives (ingredients) |
|---|---|---|---|---|---|---|
| 2026-06-29 | — | 0/6 | — | 0/5 | 0/17 | 0/1 |
| 2026-07-06 | 0/5 | 0/12 | 0/2 | 0/4 | 0/21 | 0/5 |
| 2026-07-13 | 0/3 | 0/7 | 0/4 | 0/6 | 0/23 | 0/3 |
| 2026-07-20 | 0/2 | 0/10 | 0/4 | 0/17 | 1/19 | 0/5 |
| 2026-07-27 | — | 0/14 | — | 0/5 | 2/24 | 0/1 |
| 2026-08-03 | 0/6 | 0/16 | 0/3 | 0/2 | 0/16 | 0/11 |
| 2026-08-10 | 3/6 | 5/23 | 2/4 | 1/3 | 59/80 | 0/1 |
| 2026-08-17 | 0/2 | 0/14 | 0/2 | 0/1 | 0/22 | 0/4 |
| 2026-08-24 | 0/1 | 0/4 | — | 0/2 | 0/6 | 0/1 |

Reconstruction sessions (entry day of truly-backfilled rows):

| entered on | family | txns | business_date range covered |
|---|---|---|---|
| 2026-08-17 | Coconut Fancy | 2 | 2026-08-14 → 2026-08-14 |
| 2026-08-17 | Coconut Flake | 5 | 2026-08-14 → 2026-08-14 |
| 2026-08-17 | Coconut Medium | 3 | 2026-08-14 → 2026-08-14 |
| 2026-08-17 | Coconut Toasted Flake | 1 | 2026-08-14 → 2026-08-14 |
| 2026-08-17 | Granola (all) | 62 | 2026-07-24 → 2026-08-14 |
| 2026-08-17 | Other/ingredient-level | 2 | 2026-08-14 → 2026-08-14 |

### 8. Aged on-hand by lot-age bucket (age = now − coalesce(received_at, lot created_at))

Batch products (positive on-hand only):

| product | SKU | ≤14 d | 15–30 d | 31–60 d | >60 d | oldest open lot | lb | age d |
|---|---|---|---|---|---|---|---|---|
| Batch Coconut Sweetened Flake | 90004 | 2517.2 | 2781.4 | 2987.2 | 18193.0 | 26-05-05-COCO-002 | 2074.0 | 112 |
| Batch Classic Granola #9 | 90002 | 13839.0 | 0 | 0 | 0 | AUG 13 2026 | 3853.0 | 12 |
| Batch SS Original Granola #1 | 90016 | 3650.0 | 0 | 0 | 0 | AUG 18 2026 | 150.0 | 7 |
| Batch Coconut Sweetened Fancy | 90003 | 148.4 | 188.0 | 596.0 | 2670.4 | 2026-04-20 | 1680.0 | 127 |
| Batch SS Chocolate Chip Granola #2 | 90011 | 1605.0 | 0 | 0 | 0 | AUG 18 2026 | 1605.0 | 7 |
| Batch BS Dark Chocolate Granola 350 | 95002 | 0 | 0 | 1050.0 | 0 | JUL 24 2026 | 1050.0 | 32 |
| Batch Coconut Sweetened Medium | 90005 | 79.2 | 158.4 | 396.0 | 0 | JUL 08 2026 | 396.0 | 48 |
| Batch Coconut Toasted Sweetened Flake | 90007 | 0 | 0 | 0 | 500.0 | MAY 01 2026 | 500.0 | 116 |
| Batch SS Cranberry Granola #3 | 90013 | 245.0 | 4.0 | 0 | 0 | AUG 06 2026 | 4.0 | 19 |
| Granola Fruit Nut Batch | 90008 | 0 | 0 | 169.0 | 0 | JUN 29 2026 | 169.0 | 57 |
| Batch BS Peanut Butter Banana Granola | 95005 | 0 | 2.4 | 0 | 0 | JUL 27 2026 | 2.4 | 27 |

Top 20 finished SKUs (by 30-d shipped lb):

| product | SKU | ≤14 d | 15–30 d | 31–60 d | >60 d | oldest open lot | lb | age d |
|---|---|---|---|---|---|---|---|---|
| Desiccated Flake 50 LB | 10047 | 0 | 0 | 0 | 102500.0 | 26-04-22-FRAN-001 | 102500.0 | 125 |
| Kookies & Kreme – 25 LB | 10304 | 0 | 0 | 0 | 24775.0 | 26-06-02-CREA-002 | 24775.0 | 84 |
| CQ Coconut Sweetened Flake 10 LB | 893 | 23290.0 | 0 | 0 | 0 | AUG 12 2026 | 2800.0 | 13 |
| Granola SS Chocolate Chip 12x10 OZ Case | 70003 | 16650.0 | 0 | 0 | 0 | BB081027 | 4500.0 | 13 |
| Sprinkles Rainbow 10 LB | 10302 | 0 | 0 | 0 | 10680.0 | 26-06-10-TRIS-001 | 10680.0 | 76 |
| Sprinkles Rainbow 25 LB | 10305 | 0 | 4725.0 | 0 | 0 | 26-08-03-EURO-001 | 2275.0 | 22 |
| Coconut Sweetened Flake UNIPRO 10 LB | 67476 | 0 | 3170.0 | 0 | 0 | AUG 05 2026 | 1400.0 | 20 |
| Granola Classic 25 LB | 70050 | 2300.0 | 0 | 0 | 0 | AUG 12 2026 | 1950.0 | 5 |
| Granola SS Cranberry 12x10 OZ Case | 70011 | 2025.0 | 0 | 0 | 0 | BB081727 | 1852.5 | 4 |
| CQ Granola 10 LB | 1614 | 1400.0 | 0 | 0 | 0 | AUG 12 2026 | 1400.0 | 4 |
| Coconut Toasted Sweetened Flake CNS 25 LB | 10029 | 0 | 1050.0 | 0 | 0 | JUL 27 2026 | 1050.0 | 29 |
| Coconut Sweetened Fancy UNIPRO 10 LB | 67470 | 260.0 | 0 | 0 | 0 | AUG 13 2026 | 260.0 | 12 |

Lots 616 / 620 / 624 (25120, 6013):

| lot | code | product | on-hand lb | first consumed | last consumed |
|---|---|---|---|---|---|
| 616 | 25120 | Coconut Macaroon Desiccated | 1013.0 | 2026-05-14 | 2026-08-24 |
| 620 | 25120 | Coconut Medium Desiccated | 16375.0 | None | None |
| 624 | 6013 | Coconut Flake Desiccated | 6750.0 | 2026-07-24 | 2026-08-24 |

Products consuming them since Aug 11:

| lot | consuming product (batch) | makes | lb drawn | first | last |
|---|---|---|---|---|---|
| 616 | Batch Classic Granola #9 | 6 | 1050.0 | 2026-08-11 | 2026-08-24 |
| 616 | Batch SS Chocolate Chip Granola #2 | 3 | 665.0 | 2026-08-14 | 2026-08-18 |
| 616 | Batch SS Original Granola #1 | 2 | 273.0 | 2026-08-18 | 2026-08-19 |
| 616 | Batch SS Cranberry Granola #3 | 1 | 105.0 | 2026-08-19 | 2026-08-19 |
| 624 | Batch Coconut Sweetened Flake | 8 | 16400.0 | 2026-08-11 | 2026-08-24 |
### 7b. Reading notes on the backfill re-measure (F1 refinement)

- **The July "100% backfilled" picture in §3b was an artifact of `created_at`.** Re-measured on the legacy `"timestamp"` column (the true insert time for pre-039 rows), every week from Jun 29 through Aug 8 is 96–100% same-day, for every family — coconut medium, flake, fancy, toasted, granola, and receives alike. The floor was entering live all along; only the `created_at` column was backfilled.
- **There is exactly one true reconstruction window post-Jul-1: the 2026-08-17 recon session.** All ≥2-day rows were entered on 8/17 and cover business dates 7/24 → 8/14 (62 granola txns — the QB YTD forced ships/packs and count adjustments — plus 11 coconut/other rows all backdated to the 8/14 count date). That is the entire 56.2% spike in the Aug-10 week.
- **The expected "Aug 11 10:32 coconut-medium rebuild" does not exist.** The only transactions whose legacy timestamps fall on 2026-08-11 are the nine live evening floor entries #1892–1900 (15:36–16:18 ET, all same-day, `created_at_source='database'`). Coconut-medium rows from June through August each carry a legacy timestamp on their own business date (typical 16:30–17:10 ET evening cadence, make then pack minutes apart). The 10:32 ET / 14:32 UTC timestamp associated with that memory is the **039 migration `created_at` stamp carried by every backfilled row** — the "rebuild" was of the `created_at` column, not of the data.
- **Per-family epoch (entry has been live since at least):** every family — coconut medium, flake, fancy, toasted, granola, receives — measures live from **Jul 1 onward** (earlier not re-measured here), with the single exception of the 8/17-entered recon patch covering 7/24–8/14 granola and the 8/14 coconut count rows. So for post-epoch analyses, the honest boundary is: **use 8/11 for `created_at`-based metrics, but the underlying entry behavior was already live well before it**; treat 8/14-dated recon rows as reconstructed regardless of family.
- Caveats: (1) a reconstruction that forged historical `timestamp` values would be undetectable by this method — the per-row minutes-apart make→pack cadence argues against that here; (2) legacy `"timestamp"` for backfilled rows is UTC-wall, so a floor entry after ~8 pm ET would misread as next-day — the measured next-day rate is 0–1.4%, so this barely binds; (3) 0 rows were unknowable (no NULL timestamps).

### 8b. Reading notes on aged on-hand

- **Batch Coconut Sweetened Flake is the aging problem:** 18,193 lb of its 26,479 lb on-hand is >60 d old, oldest open lot `26-05-05-COCO-002` at 2,074 lb / 112 d. Granola batch stock by contrast is entirely ≤14 d.
- **Dead >60 d finished stock:** Desiccated Flake 50 LB 102,500 lb (125 d, lot `26-04-22-FRAN-001`), Kookies & Kreme 25 LB 24,775 lb (84 d), Sprinkles Rainbow 10 LB 10,680 lb (76 d).
- **Lot 620 (`25120`, Coconut Medium Desiccated, 16,375 lb) has NEVER been consumed** — it is pure dead stock from the 5/14 count split. Lot 616 (`25120`, Macaroon, 1,013 lb) is actively consumed by granola batches (Classic #9, SS Choc Chip, SS Original, SS Cranberry — 2,093 lb drawn since 8/11), and lot 624 (`6013`, Flake, 6,750 lb) feeds Batch Coconut Sweetened Flake (16,400 lb drawn since 8/11). Supplier retro-attribution on 616/624 (docs/sql/found-lot-suppliers-2026-08-25.sql) is what flips ongoing production back to traceable.

### 9. Duplicate-branch check: fix/trace-batch-output-lots vs PR #17

**There is no divergent duplicate — they are the same commit.** PR #17's head (`headRefOid 9e78fbc`), `origin/fix/trace-batch-output-lots`, and the local branch in the Codex worktree (`~/Documents/Codex/2026-08-11/g/work/factory-ledger-current`) are all identical at `9e78fbc` (5 commits over main: 6f85045 → 9e78fbc; diff = main.py +58/−? , dashboard/traceability.html, tests/test_batch1_correctness_security.py +258, changelogs). The 13:10 change-log entry and PR #17 describe one and the same branch — nothing in "the branch" is missing from "#17", so there is nothing to cherry-pick and no second branch to close. Content sanity-checked: it implements exactly the §4 fix (output-lot downstream via `ingredient_lot_consumption` keyed by numeric lot id, `lot_id`/`product_id` added to trace payloads additively, dashboard identity by numeric id, collision 409 kept) and its regression test uses this report's lot 1318/1321/1326 case. Three genuinely old trace branches exist (`claude/blissful-bohr` v2.5.0-era, `claude/funny-pascal` / `claude/peaceful-mayer` 2026-03-25 "product_id disambiguation") — all superseded (the disambiguation is in main today); left untouched per instructions. **Recommendation: proceed with PR #17 as-is; close nothing, cherry-pick nothing.**

---
