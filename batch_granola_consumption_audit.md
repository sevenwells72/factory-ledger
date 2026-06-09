# Batch Granola Consumption Audit

Generated: 2026-06-08 12:04:49 local time. Database session was opened read-only; no data was modified.

## Summary

Current `/pack` commits generally do reduce batch inventory correctly: each modern pack transaction creates a positive finished-good line and matching negative source-batch lines, with `ingredient_lot_consumption` trace rows. For direct finished goods mapped to the four target batch components, I found 83 posted pack output rows across 83 pack transactions. 81 of those rows balance exactly between finished-good pounds created and expected target-batch pounds consumed. 2 direct pack rows do not: both used the wrong source item (`Batch BS Dark Chocolate Granola 350`) for Classic Granola finished goods.

I also found 23 positive finished-good inventory rows for mapped finished goods where no expected batch-granola consumption was recorded in the same transaction. These are manual `adjust`, `receive`, or wrong-source `pack` entries rather than normal balanced pack commits. They add finished goods without reducing the target batch lot in that transaction.

## Target Batch Components

| id | odoo_code | name | type | uom | default_batch_lb | active |
| --- | --- | --- | --- | --- | --- | --- |
| 107 | 90002 | Batch Classic Granola #9 | batch | lb | 323 | True |
| 108 | 90001 | Batch Classic Chocolate Chip Granola #9 | batch | lb | 348 | True |
| 114 | 90011 | Batch SS Chocolate Chip Granola #2 | batch | lb | 393 | True |
| 116 | 90016 | Batch SS Original Granola #1 | batch | lb | 350 | True |

## Finished Goods Mapped To These Components

| fg_id | fg_sku | finished_product | case_size_lb | active | batch_id | batch_sku | batch_component |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 143 | 70053 | Granola Chocolate Chip 25 LB | 25 | True | 108 | 90001 | Batch Classic Chocolate Chip Granola #9 |
| 137 | 10300 | Granola Crunchy CNS 10 LB Case | 10 | True | 107 | 90002 | Batch Classic Granola #9 |
| 144 | 1614 | CQ Granola 10 LB | 10 | True | 107 | 90002 | Batch Classic Granola #9 |
| 138 | 70012 | Granola Wheat Free 25 LB | 25 | True | 107 | 90002 | Batch Classic Granola #9 |
| 136 | 70050 | Granola Classic 25 LB | 25 | True | 107 | 90002 | Batch Classic Granola #9 |
| 129 | 70081 | Granola Setton Good Ol 25 LB | 25 | True | 107 | 90002 | Batch Classic Granola #9 |
| 145 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 7.5 | True | 114 | 90011 | Batch SS Chocolate Chip Granola #2 |
| 146 | 70002 | Granola SS Original 12x10 OZ Case | 7.5 | True | 116 | 90016 | Batch SS Original Granola #1 |
| 183 | 70004 | Granola SS Original Bulk per/lb |  | True | 116 | 90016 | Batch SS Original Granola #1 |
| 185 | 70006 | Granola SS Mini 100 |  | True | 116 | 90016 | Batch SS Original Granola #1 |
| 141 | 70060 | Granola Honey Nut 25 LB | 25 | True | 116 | 90016 | Batch SS Original Granola #1 |

## Consumption Summary By Transaction Type

| type | rows | transactions | consumed_lb | rows_with_matching_ilc |
| --- | --- | --- | --- | --- |
| adjust | 14 | 14 | 18469 | 0 |
| make | 4 | 4 | 3327 | 3 |
| pack | 123 | 85 | 132787.5 | 119 |

Note: `adjust` rows and the reversal-style `make` row do not have matching `ingredient_lot_consumption` rows. Modern `pack` and normal `make` commits usually do.

## Direct Finished-Good Pack Consumption

Each row is a consumed source lot for a posted `pack` transaction that created a mapped finished good and consumed the expected target batch component.

| production_date | finished_sku | finished_product | finished_lot | finished_created_lb | component_sku | batch_component | batch_lot | consumed_lb | transaction_table | transaction_record_id | consumption_table | consumption_record_id | ingredient_lot_consumption_id | output_transaction_line_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-16 | 1614 | CQ Granola 10 LB | FEB 06 2026 | 20 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 20 | transactions | 245 | transaction_lines | 1585 | 199 | 1584 |
| 2026-02-18 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB021627 | 4500 | 90011 | Batch SS Chocolate Chip Granola #2 | B26-0212-002 | 3930 | transactions | 260 | transaction_lines | 1639 | 238 | 1638 |
| 2026-02-18 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB021627 | 4500 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 13 2026 | 570 | transactions | 260 | transaction_lines | 1640 | 239 | 1638 |
| 2026-02-18 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 13 2026 | 1912.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 13 2026 | 1912.5 | transactions | 270 | transaction_lines | 1680 | 269 | 1679 |
| 2026-02-18 | 70002 | Granola SS Original 12x10 OZ Case | FEB 16 2026 | 1852.5 | 90016 | Batch SS Original Granola #1 | FEB 16 2026 | 1852.5 | transactions | 271 | transaction_lines | 1682 | 270 | 1681 |
| 2026-02-19 | 70050 | Granola Classic 25 LB | FEB 06 2026 | 600 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 600 | transactions | 281 | transaction_lines | 1714 | 292 | 1713 |
| 2026-02-20 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 13 2026 | 952.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 13 2026 | 661.5 | transactions | 298 | transaction_lines | 1748 | 307 | 1747 |
| 2026-02-20 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 13 2026 | 952.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 18 2026 | 291 | transactions | 298 | transaction_lines | 1749 | 308 | 1747 |
| 2026-02-24 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB022427 Lot | 562.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 18 2026 | 562.5 | transactions | 315 | transaction_lines | 1796 | 337 | 1795 |
| 2026-02-24 | 70002 | Granola SS Original 12x10 OZ Case | BB022427 | 900 | 90016 | Batch SS Original Granola #1 | FEB 16 2026 | 900 | transactions | 316 | transaction_lines | 1798 | 338 | 1797 |
| 2026-03-02 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 18 2026 | 2002.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 18 2026 | 718.5 | transactions | 345 | transaction_lines | 1894 | 400 | 1893 |
| 2026-03-02 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 18 2026 | 2002.5 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 24 2026 | 1179 | transactions | 345 | transaction_lines | 1895 | 401 | 1893 |
| 2026-03-02 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | FEB 18 2026 | 2002.5 | 90011 | Batch SS Chocolate Chip Granola #2 | 26-02-25 | 105 | transactions | 345 | transaction_lines | 1896 | 402 | 1893 |
| 2026-03-03 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 375 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 375 | transactions | 351 | transaction_lines | 1929 | 429 | 1928 |
| 2026-03-03 | 70002 | Granola SS Original 12x10 OZ Case | BB030327 | 2017.5 | 90016 | Batch SS Original Granola #1 | FEB 16 2026 | 47.5 | transactions | 363 | transaction_lines | 1952 | 438 | 1951 |
| 2026-03-03 | 70002 | Granola SS Original 12x10 OZ Case | BB030327 | 2017.5 | 90016 | Batch SS Original Granola #1 | B26-0302-001 | 1970 | transactions | 363 | transaction_lines | 1953 | 439 | 1951 |
| 2026-03-03 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB030327 | 247.5 | 90011 | Batch SS Chocolate Chip Granola #2 | 26-02-25 | 247.5 | transactions | 364 | transaction_lines | 1955 | 440 | 1954 |
| 2026-03-04 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 26-02-25 | 1125 | 90011 | Batch SS Chocolate Chip Granola #2 | 26-02-25 | 1125 | transactions | 368 | transaction_lines | 1978 | 459 | 1977 |
| 2026-03-04 | 70050 | Granola Classic 25 LB | B26-0304-001 | 1500 | 90002 | Batch Classic Granola #9 | B26-0304-001 | 1500 | transactions | 371 | transaction_lines | 1983 | 461 | 1982 |
| 2026-03-05 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 26-02-25 | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | 26-02-25 | 1273.5 | transactions | 384 | transaction_lines | 2025 | 476 | 2024 |
| 2026-03-05 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | 26-02-25 | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 26 2026 | 976.5 | transactions | 384 | transaction_lines | 2026 | 477 | 2024 |
| 2026-03-06 | 10300 | Granola Crunchy CNS 10 LB Case | FEB 09 2026 | 40 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 40 | transactions | 387 | transaction_lines | 2052 | 500 | 2051 |
| 2026-03-10 | 10300 | Granola Crunchy CNS 10 LB Case | FEB 06 2026 | 150 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 150 | transactions | 405 | transaction_lines | 2110 | 539 | 2109 |
| 2026-03-10 | 1614 | CQ Granola 10 LB | FEB 06 2026 | 2800 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 780 | transactions | 415 | transaction_lines | 2134 | 553 | 2133 |
| 2026-03-10 | 1614 | CQ Granola 10 LB | FEB 06 2026 | 2800 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 2020 | transactions | 415 | transaction_lines | 2135 | 554 | 2133 |
| 2026-03-11 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 1250 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 1250 | transactions | 422 | transaction_lines | 2153 | 565 | 2152 |
| 2026-03-11 | 1614 | CQ Granola 10 LB | FEB 09 2026 | 1400 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 1400 | transactions | 423 | transaction_lines | 2155 | 566 | 2154 |
| 2026-03-12 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 1250 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 107 | transactions | 427 | transaction_lines | 2168 | 575 | 2167 |
| 2026-03-12 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 1250 | 90002 | Batch Classic Granola #9 | FEB 10 2026 | 644 | transactions | 427 | transaction_lines | 2169 | 576 | 2167 |
| 2026-03-12 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 1250 | 90002 | Batch Classic Granola #9 | FEB 10 2026-02 | 499 | transactions | 427 | transaction_lines | 2170 | 577 | 2167 |
| 2026-03-12 | 70050 | Granola Classic 25 LB | FEB 09 2026 | 1250 | 90002 | Batch Classic Granola #9 | FEB 10 2026-02 | 1250 | transactions | 435 | transaction_lines | 2179 | 578 | 2178 |
| 2026-03-12 | 70060 | Granola Honey Nut 25 LB | MAR 12 2026 | 1000 | 90016 | Batch SS Original Granola #1 | B26-0302-001 | 130 | transactions | 438 | transaction_lines | 2198 | 594 | 2197 |
| 2026-03-12 | 70060 | Granola Honey Nut 25 LB | MAR 12 2026 | 1000 | 90016 | Batch SS Original Granola #1 | MAR 12 2026 | 870 | transactions | 438 | transaction_lines | 2199 | 595 | 2197 |
| 2026-03-12 | 1614 | CQ Granola 10 LB | MAR 03 2026 | 1400 | 90002 | Batch Classic Granola #9 | FEB 10 2026-02 | 183 | transactions | 440 | transaction_lines | 2208 | 602 | 2207 |
| 2026-03-12 | 1614 | CQ Granola 10 LB | MAR 03 2026 | 1400 | 90002 | Batch Classic Granola #9 | FEB 26 2026 | 969 | transactions | 440 | transaction_lines | 2209 | 603 | 2207 |
| 2026-03-12 | 1614 | CQ Granola 10 LB | MAR 03 2026 | 1400 | 90002 | Batch Classic Granola #9 | MAR 03 2026 | 248 | transactions | 440 | transaction_lines | 2210 | 604 | 2207 |
| 2026-03-12 | 1614 | CQ Granola 10 LB | MAR 10 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAR 03 2026 | 2800 | transactions | 441 | transaction_lines | 2212 | 605 | 2211 |
| 2026-03-13 | 1614 | CQ Granola 10 LB | MAR 11 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAR 03 2026 | 1474 | transactions | 447 | transaction_lines | 2226 | 613 | 2225 |
| 2026-03-13 | 1614 | CQ Granola 10 LB | MAR 11 2026 | 2800 | 90002 | Batch Classic Granola #9 | B26-0304-001 | 115 | transactions | 447 | transaction_lines | 2227 | 614 | 2225 |
| 2026-03-13 | 1614 | CQ Granola 10 LB | MAR 11 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAR 10 2026 | 1211 | transactions | 447 | transaction_lines | 2228 | 615 | 2225 |
| 2026-03-16 | 1614 | CQ Granola 10 LB | MAR 11 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAR 10 2026 | 1373 | transactions | 456 | transaction_lines | 2250 | 627 | 2249 |
| 2026-03-16 | 1614 | CQ Granola 10 LB | MAR 11 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAR 11 2026 | 1427 | transactions | 456 | transaction_lines | 2251 | 628 | 2249 |
| 2026-03-16 | 1614 | CQ Granola 10 LB | MAR 16 2026 | 1400 | 90002 | Batch Classic Granola #9 | MAR 11 2026 | 1400 | transactions | 457 | transaction_lines | 2253 | 629 | 2252 |
| 2026-03-24 | 70050 | Granola Classic 25 LB | MAR 23 2026 | 2500 | 90002 | Batch Classic Granola #9 | MAR 11 2026 | 2341 | transactions | 533 | transaction_lines | 2459 | 744 | 2458 |
| 2026-03-24 | 70050 | Granola Classic 25 LB | MAR 23 2026 | 2500 | 90002 | Batch Classic Granola #9 | MAR 12 2026 | 159 | transactions | 533 | transaction_lines | 2460 | 745 | 2458 |
| 2026-03-24 | 70050 | Granola Classic 25 LB | MAR 24 2026 | 1000 | 90002 | Batch Classic Granola #9 | MAR 12 2026 | 1000 | transactions | 534 | transaction_lines | 2462 | 746 | 2461 |
| 2026-03-24 | 10300 | Granola Crunchy CNS 10 LB Case | MAR 16 2026 | 500 | 90002 | Batch Classic Granola #9 | MAR 12 2026 | 133 | transactions | 535 | transaction_lines | 2464 | 747 | 2463 |
| 2026-03-24 | 10300 | Granola Crunchy CNS 10 LB Case | MAR 16 2026 | 500 | 90002 | Batch Classic Granola #9 | MAR 16 2026 | 367 | transactions | 535 | transaction_lines | 2465 | 748 | 2463 |
| 2026-03-24 | 10300 | Granola Crunchy CNS 10 LB Case | MAR 24 2026 | 350 | 90002 | Batch Classic Granola #9 | MAR 16 2026 | 350 | transactions | 536 | transaction_lines | 2467 | 749 | 2466 |
| 2026-03-27 | 70050 | Granola Classic 25 LB | MAR 24 2026 | 1500 | 90002 | Batch Classic Granola #9 | MAR 16 2026 | 1500 | transactions | 562 | transaction_lines | 2541 | 792 | 2540 |
| 2026-03-27 | 70050 | Granola Classic 25 LB | MAR 26 2026 | 1500 | 90002 | Batch Classic Granola #9 | MAR 16 2026 | 1013 | transactions | 563 | transaction_lines | 2543 | 793 | 2542 |
| 2026-03-27 | 70050 | Granola Classic 25 LB | MAR 26 2026 | 1500 | 90002 | Batch Classic Granola #9 | MAR 18 2026 | 487 | transactions | 563 | transaction_lines | 2544 | 794 | 2542 |
| 2026-03-30 | 70012 | Granola Wheat Free 25 LB | MAR 24 2026 | 250 | 90002 | Batch Classic Granola #9 | MAR 18 2026 | 250 | transactions | 580 | transaction_lines | 2579 | 810 | 2578 |
| 2026-04-14 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 Lot | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | FEB 26 2026 | 2167.5 | transactions | 616 | transaction_lines | 2705 | 899 | 2704 |
| 2026-04-14 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 Lot | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 10 2026 Lot | 82.5 | transactions | 616 | transaction_lines | 2706 | 900 | 2704 |
| 2026-04-15 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 | 3375 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 10 2026 Lot | 1882.5 | transactions | 625 | transaction_lines | 2749 | 934 | 2748 |
| 2026-04-15 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 | 3375 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 12 2026 Lote | 1492.5 | transactions | 625 | transaction_lines | 2750 | 935 | 2748 |
| 2026-04-16 | 70002 | Granola SS Original 12x10 OZ Case | BB041327 | 1980 | 90016 | Batch SS Original Granola #1 | MAR 12 2026 | 180 | transactions | 637 | transaction_lines | 2782 | 951 | 2781 |
| 2026-04-16 | 70002 | Granola SS Original 12x10 OZ Case | BB041327 | 1980 | 90016 | Batch SS Original Granola #1 | APR 13 2026 | 1400 | transactions | 637 | transaction_lines | 2783 | 952 | 2781 |
| 2026-04-16 | 70002 | Granola SS Original 12x10 OZ Case | BB041327 | 1980 | 90016 | Batch SS Original Granola #1 | APR 14 2026 | 400 | transactions | 637 | transaction_lines | 2784 | 953 | 2781 |
| 2026-04-16 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 | 1125 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 12 2026 Lote | 1125 | transactions | 638 | transaction_lines | 2786 | 954 | 2785 |
| 2026-04-17 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB041327 | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 12 2026 Lote | 2250 | transactions | 644 | transaction_lines | 2808 | 970 | 2807 |
| 2026-04-20 | 70012 | Granola Wheat Free 25 LB | MAR 27 2026 | 1000 | 90002 | Batch Classic Granola #9 | MAR 18 2026 | 232 | transactions | 660 | transaction_lines | 2851 | 997 | 2850 |
| 2026-04-20 | 70012 | Granola Wheat Free 25 LB | MAR 27 2026 | 1000 | 90002 | Batch Classic Granola #9 | MAR 23 2026 | 768 | transactions | 660 | transaction_lines | 2852 | 998 | 2850 |
| 2026-04-20 | 70050 | Granola Classic 25 LB | 2026-04-13 | 250 | 90002 | Batch Classic Granola #9 | MAR 23 2026 | 250 | transactions | 661 | transaction_lines | 2854 | 999 | 2853 |
| 2026-04-21 | 70050 | Granola Classic 25 LB | 2026-04-17 | 1500 | 90002 | Batch Classic Granola #9 | MAR 23 2026 | 1500 | transactions | 677 | transaction_lines | 2893 | 1022 | 2892 |
| 2026-04-27 | 70050 | Granola Classic 25 LB | ABR 15 2026 | 1200 | 90002 | Batch Classic Granola #9 | MAR 23 2026 | 389 | transactions | 702 | transaction_lines | 2960 | 1064 | 2959 |
| 2026-04-27 | 70050 | Granola Classic 25 LB | ABR 15 2026 | 1200 | 90002 | Batch Classic Granola #9 | MAR 24 2026 | 811 | transactions | 702 | transaction_lines | 2961 | 1065 | 2959 |
| 2026-04-27 | 70053 | Granola Chocolate Chip 25 LB | ABR 24 2026 | 600 | 90001 | Batch Classic Chocolate Chip Granola #9 | APR 24 2026 | 600 | transactions | 703 | transaction_lines | 2963 | 1066 | 2962 |
| 2026-04-28 | 70060 | Granola Honey Nut 25 LB | 2026-04-17 | 1000 | 90016 | Batch SS Original Granola #1 | 2026-04-17 | 1000 | transactions | 729 | transaction_lines | 3043 | 1120 | 3042 |
| 2026-04-28 | 10300 | Granola Crunchy CNS 10 LB Case | MAR 24 2026 | 100 | 90002 | Batch Classic Granola #9 | MAR 24 2026 | 100 | transactions | 735 | transaction_lines | 3052 | 1121 | 3051 |
| 2026-04-29 | 70060 | Granola Honey Nut 25 LB | APR 24 2026 | 600 | 90016 | Batch SS Original Granola #1 | APR 24 2026 | 600 | transactions | 747 | transaction_lines | 3097 | 1154 | 3096 |
| 2026-04-29 | 1614 | CQ Granola 10 LB | MAR 24 2026 | 9800 | 90002 | Batch Classic Granola #9 | MAR 24 2026 | 3288 | transactions | 756 | transaction_lines | 3117 | 1165 | 3116 |
| 2026-04-29 | 1614 | CQ Granola 10 LB | MAR 24 2026 | 9800 | 90002 | Batch Classic Granola #9 | APR 13 2026 | 1938 | transactions | 756 | transaction_lines | 3118 | 1166 | 3116 |
| 2026-04-29 | 1614 | CQ Granola 10 LB | MAR 24 2026 | 9800 | 90002 | Batch Classic Granola #9 | 2026-04-15 | 3230 | transactions | 756 | transaction_lines | 3119 | 1167 | 3116 |
| 2026-04-29 | 1614 | CQ Granola 10 LB | MAR 24 2026 | 9800 | 90002 | Batch Classic Granola #9 | 2026-04-21 | 969 | transactions | 756 | transaction_lines | 3120 | 1168 | 3116 |
| 2026-04-29 | 1614 | CQ Granola 10 LB | MAR 24 2026 | 9800 | 90002 | Batch Classic Granola #9 | ABR 23 2026 | 375 | transactions | 756 | transaction_lines | 3121 | 1169 | 3116 |
| 2026-05-01 | 1614 | CQ Granola 10 LB | ABR 23 2026 | 1400 | 90002 | Batch Classic Granola #9 | ABR 23 2026 | 1400 | transactions | 780 | transaction_lines | 3182 | 1205 | 3181 |
| 2026-05-01 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | MAY 01 2026 | 1125 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 12 2026 Lote | 1125 | transactions | 786 | transaction_lines | 3213 | 1220 | 3212 |
| 2026-05-01 | 70050 | Granola Classic 25 LB | APR 29 2026 | 1500 | 90002 | Batch Classic Granola #9 | ABR 23 2026 | 1500 | transactions | 787 | transaction_lines | 3215 | 1221 | 3214 |
| 2026-05-04 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 3375 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 12 2026 Lote | 295.5 | transactions | 800 | transaction_lines | 3263 | 1256 | 3262 |
| 2026-05-04 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 3375 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 13 2026 | 786 | transactions | 800 | transaction_lines | 3264 | 1257 | 3262 |
| 2026-05-04 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 3375 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 14 2026 | 2293.5 | transactions | 800 | transaction_lines | 3265 | 1258 | 3262 |
| 2026-05-05 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 4365 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 14 2026 | 457.5 | transactions | 813 | transaction_lines | 3318 | 1297 | 3317 |
| 2026-05-05 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 4365 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 28 2026 | 393 | transactions | 813 | transaction_lines | 3319 | 1298 | 3317 |
| 2026-05-05 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB050427 | 4365 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 30 2026 | 3514.5 | transactions | 813 | transaction_lines | 3320 | 1299 | 3317 |
| 2026-05-06 | 10300 | Granola Crunchy CNS 10 LB Case | APR 29 2026 | 150 | 90002 | Batch Classic Granola #9 | ABR 23 2026 | 150 | transactions | 818 | transaction_lines | 3337 | 1311 | 3336 |
| 2026-05-06 | 70050 | Granola Classic 25 LB | MAY 05 2026 | 1500 | 90002 | Batch Classic Granola #9 | ABR 23 2026 | 1097 | transactions | 823 | transaction_lines | 3358 | 1326 | 3357 |
| 2026-05-06 | 70050 | Granola Classic 25 LB | MAY 05 2026 | 1500 | 90002 | Batch Classic Granola #9 | APR 28 2026 | 403 | transactions | 823 | transaction_lines | 3359 | 1327 | 3357 |
| 2026-05-06 | 70050 | Granola Classic 25 LB | MAY 05 2026 | 625 | 90002 | Batch Classic Granola #9 | APR 28 2026 | 625 | transactions | 824 | transaction_lines | 3361 | 1328 | 3360 |
| 2026-05-06 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 05 2026 | 200 | 90002 | Batch Classic Granola #9 | APR 28 2026 | 200 | transactions | 825 | transaction_lines | 3363 | 1329 | 3362 |
| 2026-05-12 | 70053 | Granola Chocolate Chip 25 LB | MAY 11 2026 | 300 | 90001 | Batch Classic Chocolate Chip Granola #9 | MAY 11 2026 | 300 | transactions | 868 | transaction_lines | 3535 | 1447 | 3534 |
| 2026-05-12 | 70050 | Granola Classic 25 LB | MAY 12 2026 | 700 | 90002 | Batch Classic Granola #9 | MAY 12 2026 | 700 | transactions | 870 | transaction_lines | 3540 | 1450 | 3539 |
| 2026-05-13 | 70012 | Granola Wheat Free 25 LB | MAY 12 2026 | 2000 | 90002 | Batch Classic Granola #9 | APR 28 2026 | 1356 | transactions | 883 | transaction_lines | 3574 | 1471 | 3573 |
| 2026-05-13 | 70012 | Granola Wheat Free 25 LB | MAY 12 2026 | 2000 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 644 | transactions | 883 | transaction_lines | 3575 | 1472 | 3573 |
| 2026-05-13 | 70050 | Granola Classic 25 LB | MAY 08 2026 | 1500 | 90002 | Batch Classic Granola #9 | MAY 08 2026 | 1500 | transactions | 884 | transaction_lines | 3577 | 1473 | 3576 |
| 2026-05-13 | 70050 | Granola Classic 25 LB | MAY 13 2026 | 1500 | 90002 | Batch Classic Granola #9 | MAY 13 2026 | 1500 | transactions | 885 | transaction_lines | 3579 | 1474 | 3578 |
| 2026-05-14 | 70050 | Granola Classic 25 LB | MAY 13 2026 | 3000 | 90002 | Batch Classic Granola #9 | MAY 13 2026 | 3000 | transactions | 951 | transaction_lines | 3690 | 1516 | 3689 |
| 2026-05-20 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 19 2026 | 150 | 90002 | Batch Classic Granola #9 | MAY 19 2026 | 150 | transactions | 1011 | transaction_lines | 3820 | 1574 | 3819 |
| 2026-05-20 | 70050 | Granola Classic 25 LB | MAY 19 2026 | 575 | 90002 | Batch Classic Granola #9 | MAY 19 2026 | 575 | transactions | 1012 | transaction_lines | 3822 | 1575 | 3821 |
| 2026-05-21 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 19 2026 | 600 | 90002 | Batch Classic Granola #9 | MAY 19 2026 | 600 | transactions | 1044 | transaction_lines | 3880 | 1598 | 3879 |
| 2026-05-26 | 1614 | CQ Granola 10 LB | MAY 19 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAY 19 2026 | 1905 | transactions | 1057 | transaction_lines | 3921 | 1626 | 3920 |
| 2026-05-26 | 1614 | CQ Granola 10 LB | MAY 19 2026 | 2800 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 895 | transactions | 1057 | transaction_lines | 3922 | 1627 | 3920 |
| 2026-05-26 | 1614 | CQ Granola 10 LB | MAY 20 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAY 20 2026 | 2800 | transactions | 1058 | transaction_lines | 3924 | 1628 | 3923 |
| 2026-05-28 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB052627 | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | APR 30 2026 | 1201.5 | transactions | 1100 | transaction_lines | 4028 | 1684 | 4027 |
| 2026-05-28 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB052627 | 2250 | 90011 | Batch SS Chocolate Chip Granola #2 | MAY 01 2026 | 1048.5 | transactions | 1100 | transaction_lines | 4029 | 1685 | 4027 |
| 2026-06-01 | 1614 | CQ Granola 10 LB | MAY 20 2026 | 1400 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 1400 | transactions | 1113 | transaction_lines | 4062 | 1702 | 4061 |
| 2026-06-01 | 1614 | CQ Granola 10 LB | MAY 21 2026 | 1400 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 1400 | transactions | 1114 | transaction_lines | 4064 | 1703 | 4063 |
| 2026-06-01 | 70012 | Granola Wheat Free 25 LB | MAY 21 2026 | 1000 | 90002 | Batch Classic Granola #9 | MAY 21 2026 | 1000 | transactions | 1120 | transaction_lines | 4082 | 1706 | 4081 |
| 2026-06-01 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 21 2026 | 560 | 90002 | Batch Classic Granola #9 | MAY 21 2026 | 560 | transactions | 1121 | transaction_lines | 4084 | 1707 | 4083 |
| 2026-06-01 | 70003 | Granola SS Chocolate Chip 12x10 OZ Case | BB060127 | 1125 | 90011 | Batch SS Chocolate Chip Granola #2 | MAY 26 2026 | 1125 | transactions | 1126 | transaction_lines | 4093 | 1710 | 4092 |
| 2026-06-04 | 1614 | CQ Granola 10 LB | MAY 21 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAY 21 2026 | 2800 | transactions | 1169 | transaction_lines | 4212 | 1777 | 4211 |
| 2026-06-04 | 1614 | CQ Granola 10 LB | MAY 28 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAY 28 2026 | 2800 | transactions | 1170 | transaction_lines | 4214 | 1778 | 4213 |
| 2026-06-05 | 1614 | CQ Granola 10 LB | JUN 01 2026 | 1400 | 90002 | Batch Classic Granola #9 | JUN 01 2026 | 1400 | transactions | 1177 | transaction_lines | 4234 | 1791 | 4233 |
| 2026-06-05 | 1614 | CQ Granola 10 LB | MAY 28 2026 | 2800 | 90002 | Batch Classic Granola #9 | MAY 28 2026 | 430 | transactions | 1178 | transaction_lines | 4236 | 1792 | 4235 |
| 2026-06-05 | 1614 | CQ Granola 10 LB | MAY 28 2026 | 2800 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 1721 | transactions | 1178 | transaction_lines | 4237 | 1793 | 4235 |
| 2026-06-05 | 1614 | CQ Granola 10 LB | MAY 28 2026 | 2800 | 90002 | Batch Classic Granola #9 | 26-05-04-GRAN-001 | 649 | transactions | 1178 | transaction_lines | 4238 | 1794 | 4235 |
| 2026-06-08 | 70050 | Granola Classic 25 LB | JUN 01 2026 | 1500 | 90002 | Batch Classic Granola #9 | JUN 01 2026 | 1500 | transactions | 1180 | transaction_lines | 4241 | 1795 | 4240 |
| 2026-06-08 | 10300 | Granola Crunchy CNS 10 LB Case | JUN 02 2026 | 300 | 90002 | Batch Classic Granola #9 | JUN 02 2026 | 300 | transactions | 1181 | transaction_lines | 4243 | 1796 | 4242 |

## Indirect Finished-Good Use Via Granola Fruit Nut Batch

Classic Granola #9 is also an ingredient in intermediate `Granola Fruit Nut Batch` (`90008`). Those intermediate lots were later packed into finished SKU `70061` where available.

| production_date | finished_sku | finished_product | finished_lot | finished_created_lb | component_sku | batch_component | batch_lot | consumed_lb | intermediate_sku | intermediate_batch | intermediate_lot | intermediate_created_lb | make_transaction_id | make_consumption_tl_id | ingredient_lot_consumption_id | downstream_pack_transaction_id | downstream_output_tl_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-19 | 70061 | Granola Fruit Nut 25 LB | FN-0219 | 25 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 20 | 90008 | Granola Fruit Nut Batch | FN-0219 | 25 | 283 | 1727 | 303 | 284 | 1728 |
| 2026-05-14 | 70061 | Granola Fruit Nut 25 LB | MAY 12 2026 | 500 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 20 | 90008 | Granola Fruit Nut Batch | MAY 12 2026 | 25 | 891 | 3594 | 1483 | 893 | 3601 |
| 2026-05-14 | 70061 | Granola Fruit Nut 25 LB | MAY 12 2026 | 500 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 380 | 90008 | Granola Fruit Nut Batch | MAY 12 2026 | 475 | 892 | 3600 | 1488 | 893 | 3601 |

## Finished Goods Created Without Target Batch Consumption

These positive finished-good inventory rows are for products mapped to one of the target batch components, but the same transaction has no negative line for the expected batch component.

| production_date | type | transaction_id | output_transaction_line_id | finished_sku | finished_product | finished_lot | finished_created_lb | expected_component_sku | expected_batch_component | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | adjust | 105 | 1333 | 1614 | CQ Granola 10 LB | 26-02-05-FOUND-007 | 140 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-06 | adjust | 124 | 1370 | 1614 | CQ Granola 10 LB | 26-02-06-FOUND-009 | 140 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-09 | adjust | 139 | 1398 | 1614 | CQ Granola 10 LB | 26-02-09-FOUND-006 | 1400 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-09 | adjust | 140 | 1399 | 1614 | CQ Granola 10 LB | 26-02-09-FOUND-007 | 1400 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-09 | adjust | 143 | 1402 | 1614 | CQ Granola 10 LB | 26-02-09-FOUND-008 | 1400 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-09 | adjust | 144 | 1403 | 1614 | CQ Granola 10 LB | 26-02-09-FOUND-009 | 1400 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-10 | adjust | 166 | 1458 | 1614 | CQ Granola 10 LB | 26-02-10-FOUND-006 | 740 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-10 | adjust | 168 | 1460 | 1614 | CQ Granola 10 LB | 26-02-10-FOUND-007 | 1840 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-02-12 | adjust | 212 | 1514 | 1614 | CQ Granola 10 LB | 26-02-05-FOUND-007 | 2660 | 90002 | Batch Classic Granola #9 | Adjustment: 2660.0 lb |
| 2026-02-12 | adjust | 213 | 1515 | 1614 | CQ Granola 10 LB | 26-02-06-FOUND-009 | 1260 | 90002 | Batch Classic Granola #9 | Adjustment: 1260.0 lb |
| 2026-02-13 | adjust | 233 | 1548 | 1614 | CQ Granola 10 LB | 26-02-13-FOUND-001 | 1400 | 90002 | Batch Classic Granola #9 | Found inventory: found_during_count |
| 2026-03-20 | receive | 496 | 2376 | 10300 | Granola Crunchy CNS 10 LB Case | MAR 11 2026 | 200 | 90002 | Batch Classic Granola #9 |  |
| 2026-03-20 | pack | 512 | 2412 | 70050 | Granola Classic 25 LB | MAR 18 2026 | 625 | 90002 | Batch Classic Granola #9 | Pack 25 cases of Granola Classic 25 LB from Batch BS Dark Chocolate Granola 350 lots: MAR 18 2026 (625.0 lb) |
| 2026-03-20 | pack | 513 | 2414 | 70012 | Granola Wheat Free 25 LB | MAR 18 2026 | 1000 | 90002 | Batch Classic Granola #9 | Pack 40 cases of Granola Wheat Free 25 LB from Batch BS Dark Chocolate Granola 350 lots: MAR 18 2026 (4.12 lb), MAR 19 2026 (995.88 lb) |
| 2026-03-23 | receive | 516 | 2419 | 1614 | CQ Granola 10 LB | 26-03-23-PHYS-002 | 1400 | 90002 | Batch Classic Granola #9 |  |
| 2026-05-15 | receive | 965 | 3711 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 08 2026 | 400 | 90002 | Batch Classic Granola #9 |  |
| 2026-05-15 | receive | 967 | 3713 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 08 2026 | 100 | 90002 | Batch Classic Granola #9 |  |
| 2026-05-19 | adjust | 1005 | 3798 | 70050 | Granola Classic 25 LB | ABR 15 2026 | 625 | 90002 | Batch Classic Granola #9 | Adjustment: 625.0 lb |
| 2026-05-20 | adjust | 1013 | 3823 | 70050 | Granola Classic 25 LB | MAY 08 2026 | 1500 | 90002 | Batch Classic Granola #9 | Adjustment: 1500.0 lb |
| 2026-05-20 | adjust | 1014 | 3824 | 70050 | Granola Classic 25 LB | MAY 13 2026 | 4500 | 90002 | Batch Classic Granola #9 | Adjustment: 4500.0 lb |
| 2026-05-21 | receive | 1031 | 3857 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 14 2026 | 100 | 90002 | Batch Classic Granola #9 |  |
| 2026-05-21 | adjust | 1032 | 3858 | 10300 | Granola Crunchy CNS 10 LB Case | MAY 19 2026 | 110 | 90002 | Batch Classic Granola #9 | Adjustment: 110.0 lb |
| 2026-05-26 | receive | 1051 | 3906 | 70012 | Granola Wheat Free 25 LB | MAY 13 2026 | 1000 | 90002 | Batch Classic Granola #9 |  |

## Suspicious Or Exception Cases

### Wrong Source Item

| date | transaction_id | finished_sku | finished_product | finished_lot | finished_created_lb | expected_batch_component | actual_negative_lines | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-03-20 | 512 | 70050 | Granola Classic 25 LB | MAR 18 2026 | 625 | Batch Classic Granola #9 | Batch BS Dark Chocolate Granola 350 lot MAR 18 2026 625.0000 lb tl#2413 | Pack 25 cases of Granola Classic 25 LB from Batch BS Dark Chocolate Granola 350 lots: MAR 18 2026 (625.0 lb) |
| 2026-03-20 | 513 | 70012 | Granola Wheat Free 25 LB | MAR 18 2026 | 1000 | Batch Classic Granola #9 | Batch BS Dark Chocolate Granola 350 lot MAR 18 2026 4.1200 lb tl#2415; Batch BS Dark Chocolate Granola 350 lot MAR 19 2026 995.8800 lb tl#2416 | Pack 40 cases of Granola Wheat Free 25 LB from Batch BS Dark Chocolate Granola 350 lots: MAR 18 2026 (4.12 lb), MAR 19 2026 (995.88 lb) |

### Target Batch Reductions Not Tied To A Finished-Good Pack Output

These rows reduce a target batch lot but are not ordinary finished-good pack-outs. They are mostly legacy/manual adjustments plus one posted reversal transaction. Some old `pack` rows have no positive output line and no ILC row, so they are not reliable finished-good traceability records.

| date | type | transaction_id | transaction_line_id | component_sku | batch_component | batch_lot | consumed_lb | ingredient_lot_consumption_id | outputs | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2026-02-05 | adjust | 93 | 1314 | 90002 | Batch Classic Granola #9 | 26-02-05-FOUND-006 | 1500 |  |  | Adjustment: -1500.0 lb |
| 2026-02-05 | pack | 94 | 1315 | 90002 | Batch Classic Granola #9 | JAN 30 2026 | 1292 |  |  |  |
| 2026-02-05 | pack | 95 | 1316 | 90002 | Batch Classic Granola #9 | FEB-02-26 | 208 |  |  |  |
| 2026-02-05 | pack | 97 | 1325 | 90002 | Batch Classic Granola #9 | FEB-03-2026 | 3000 |  |  |  |
| 2026-02-05 | pack | 98 | 1326 | 90002 | Batch Classic Granola #9 | FEB-04-26 | 3000 |  |  |  |
| 2026-02-05 | adjust | 99 | 1327 | 90002 | Batch Classic Granola #9 | B26-0204-001 | 646 |  |  | Adjustment: -646.0 lb |
| 2026-02-05 | adjust | 100 | 1328 | 90002 | Batch Classic Granola #9 | FEB-02-26 | 2376 |  |  | Adjustment: -2376.0 lb |
| 2026-02-05 | adjust | 101 | 1329 | 90002 | Batch Classic Granola #9 | FEB-03-2026 | 2168 |  |  | Adjustment: -2168.0 lb |
| 2026-02-05 | adjust | 102 | 1330 | 90002 | Batch Classic Granola #9 | FEB-04-26 | 810 |  |  | Adjustment: -810.0 lb |
| 2026-02-05 | adjust | 103 | 1331 | 90002 | Batch Classic Granola #9 | FEB-04-26 | 1358 |  |  | Adjustment: -1358.0 lb |
| 2026-02-05 | adjust | 104 | 1332 | 90002 | Batch Classic Granola #9 | FEB-05-2026 | 42 |  |  | Adjustment: -42.0 lb |
| 2026-02-06 | adjust | 123 | 1369 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 1400 |  |  | Adjustment: -1400.0 lb |
| 2026-02-09 | adjust | 142 | 1401 | 90002 | Batch Classic Granola #9 | FEB 09 2026 | 2800 |  |  | Adjustment: -2800.0 lb |
| 2026-02-12 | adjust | 219 | 1521 | 90002 | Batch Classic Granola #9 | FEB-05-2026 | 5126 |  |  | Adjustment: -5126.0 lb |
| 2026-02-12 | adjust | 220 | 1522 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 220 |  |  | Adjustment: -220.0 lb |
| 2026-02-12 | adjust | 222 | 1524 | 90002 | Batch Classic Granola #9 | FEB 10 2026 | 2 |  |  | Adjustment: -2.0 lb |
| 2026-02-12 | adjust | 223 | 1525 | 90002 | Batch Classic Granola #9 | FEB 10 2026-02 | 6 |  |  | Adjustment: -6.0 lb |
| 2026-02-12 | adjust | 224 | 1526 | 90001 | Batch Classic Chocolate Chip Granola #9 | FEB 10 2026 | 15 |  |  | Adjustment: -15.0 lb |
| 2026-02-19 | make | 283 | 1727 | 90002 | Batch Classic Granola #9 | FEB 06 2026 | 20 | 303 | Granola Fruit Nut Batch [90008] lot FN-0219 +25.0000 lb tl#1722 | 1 batch(es) of Batch Granola Fruit Nut |
| 2026-05-12 | make | 864 | 3506 | 90002 | Batch Classic Granola #9 | MAY 12 2026 | 2907 |  | Coconut Macaroon Desiccated [11013] lot 26-02-03-FOUND-001 +112.5000 lb tl#3507; Flavor – Almond [11018] lot 26-02-03-FOUND-002 +16.2000 lb tl#3508; Honey [11030] lot 26-03-19-DUTC-001 +14.4000 lb tl#3509; Oats [11032] lot 26-05-08-QUAL-001 +2025.0000 lb tl#3510; Oil – Canola [11034] lot 26-04-29-CBSF-001 +270.0000 lb tl#3511; Salt [11039] lot 26-02-03-FOUND-016 +15.3000 lb tl#3512; Sugar – Light Brown [11044] lot 26-04-29-JACK-001 +450.0000 lb tl#3513 | Reversal of transaction #863 |
| 2026-05-14 | make | 891 | 3594 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 20 | 1483 | Granola Fruit Nut Batch [90008] lot MAY 12 2026 +25.0000 lb tl#3589 | 1 batch(es) of Batch Granola Fruit Nut |
| 2026-05-14 | make | 892 | 3600 | 90002 | Batch Classic Granola #9 | APR 29 2026 | 380 | 1488 | Granola Fruit Nut Batch [90008] lot MAY 12 2026 +475.0000 lb tl#3595 | 19 batch(es) of Batch Granola Fruit Nut |

### Missing Lot / Unit / Quantity Checks

- Missing lot: none found in the audited consumption rows; `transaction_lines.lot_id` is required and all audited rows resolved to a `lots.lot_code`.
- Wrong unit: none found for the four target components; all target batch products have `uom = lb`, and consumption quantities are stored in `transaction_lines.quantity_lb` / `ingredient_lot_consumption.quantity_lb`.
- Suspicious quantity: the two wrong-source pack rows are the main quantity/source problem. Direct balanced pack rows consume the same pounds of expected batch granola as finished goods created. Manual `adjust`/`receive` rows create finished goods without any same-transaction batch reduction, so they are suspicious for inventory traceability even when their pound amounts are plausible case multiples.

## Backend/Schema Notes

- `products.parent_batch_product_id` maps finished goods to the expected batch source.
- `/pack` commit inserts a positive finished-good `transaction_lines` row, negative source-batch `transaction_lines` rows, and matching `ingredient_lot_consumption` rows.
- `/make` commit does the same for batch production: positive output batch line plus negative ingredient lines and ILC rows. This is how Classic Granola #9 flows into `Granola Fruit Nut Batch`.
- Lot-level balances are derived from signed `transaction_lines.quantity_lb`; negative rows reduce inventory.
