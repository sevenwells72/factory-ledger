# Batch Granola Inventory Adjustment Backup

Generated before any database write. This report is read-only and captures current system balances for Arturo's June 8, 2026 physical review list.

## Adjustment Note

Physical inventory review performed with Arturo on June 8, 2026. Batch granola inventory was compared against physical stock on hand. Only Batch Classic Granola #9, lot JUN 02 2026, 4,868 lb, was confirmed physically present. This lot is not adjusted. The listed lots were determined to be previously consumed or otherwise not physically present and are adjusted to zero to align system inventory with physical inventory. This is an inventory correction only, not product disposal. Historical lot consumption logic remains under review.

## Summary

- Listed lot rows found: 32
- Lots to adjust to zero: 31
- Lots kept unchanged: 1
- Total negative adjustment prepared: -57110.5 lb
- Missing listed lots: 0

## All Affected Lots Before Adjustment

| sku | item | lot | current_system_on_hand_lb | adjustment_quantity_needed_lb | final_quantity_after_adjustment_lb | action | table_name | record_id | product_id | lot_id | related_transaction_ids | transaction_line_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90002 | Batch Classic Granola #9 | 26-05-04-GRAN-001 | 966 | -966 | 0 | ADJUST TO ZERO | lots | 529 | 107 | 529 | 795, 1178 | 2 |
| 90002 | Batch Classic Granola #9 | 26-05-05-GRAN-002 | 1938 | -1938 | 0 | ADJUST TO ZERO | lots | 542 | 107 | 542 | 811 | 1 |
| 90002 | Batch Classic Granola #9 | 26-05-06-GRAN-003 | 969 | -969 | 0 | ADJUST TO ZERO | lots | 550 | 107 | 550 | 821 | 1 |
| 90002 | Batch Classic Granola #9 | MAY 08 2026 | 1407 | -1407 | 0 | ADJUST TO ZERO | lots | 568 | 107 | 568 | 844, 884 | 2 |
| 90002 | Batch Classic Granola #9 | MAY 12 2026 | 1238 | -1238 | 0 | ADJUST TO ZERO | lots | 582 | 107 | 582 | 863, 864, 865, 870 | 4 |
| 90002 | Batch Classic Granola #9 | MAY 13 2026 | 22 | -22 | 0 | ADJUST TO ZERO | lots | 594 | 107 | 594 | 882, 885, 951 | 3 |
| 90002 | Batch Classic Granola #9 | MAY 14 2026 | 646 | -646 | 0 | ADJUST TO ZERO | lots | 630 | 107 | 630 | 950 | 1 |
| 90002 | Batch Classic Granola #9 | MAY 20 2026 | 3014 | -3014 | 0 | ADJUST TO ZERO | lots | 655 | 107 | 655 | 1025, 1058 | 2 |
| 90002 | Batch Classic Granola #9 | MAY 21 2026 | 808 | -808 | 0 | ADJUST TO ZERO | lots | 662 | 107 | 662 | 1043, 1120, 1121, 1169 | 4 |
| 90002 | Batch Classic Granola #9 | MAY 27 2026 | 2261 | -2261 | 0 | ADJUST TO ZERO | lots | 686 | 107 | 686 | 1077 | 1 |
| 90002 | Batch Classic Granola #9 | JUN 01 2026 | 2268 | -2268 | 0 | ADJUST TO ZERO | lots | 722 | 107 | 722 | 1128, 1177, 1180 | 3 |
| 90002 | Batch Classic Granola #9 | JUN 02 2026 | 4868 | 0 | 4868 | KEEP / no adjustment | lots | 726 | 107 | 726 | 1135, 1181 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | FEB 10 2026 | 1725 | -1725 | 0 | ADJUST TO ZERO | lots | 129 | 108 | 129 | 162, 224 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | FEB 26 2026 | 1044 | -1044 | 0 | ADJUST TO ZERO | lots | 245 | 108 | 245 | 338 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAR 09 2026 | 2088 | -2088 | 0 | ADJUST TO ZERO | lots | 280 | 108 | 280 | 400 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | 2026-04-20 | 3132 | -3132 | 0 | ADJUST TO ZERO | lots | 431 | 108 | 431 | 656 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | APR 24 2026 | 96 | -96 | 0 | ADJUST TO ZERO | lots | 462 | 108 | 462 | 700, 703 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | APR 28 2026 | 2436 | -2436 | 0 | ADJUST TO ZERO | lots | 491 | 108 | 491 | 740 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAY 11 2026 | 3528 | -3528 | 0 | ADJUST TO ZERO | lots | 572 | 108 | 572 | 852, 868 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAY 27 2026 | 3132 | -3132 | 0 | ADJUST TO ZERO | lots | 687 | 108 | 687 | 1078 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | JUN 03 2026 | 2088 | -2088 | 0 | ADJUST TO ZERO | lots | 733 | 108 | 733 | 1149 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 01 2026 | 130.5 | -130.5 | 0 | ADJUST TO ZERO | lots | 519 | 114 | 519 | 785, 1100 | 2 |
| 90011 | Batch SS Chocolate Chip Granola #2 | 26-05-04-SSCHOC-001 | 2751 | -2751 | 0 | ADJUST TO ZERO | lots | 530 | 114 | 530 | 796 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 15 2026 | 3144 | -3144 | 0 | ADJUST TO ZERO | lots | 639 | 114 | 639 | 975 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 18 2026 | 5502 | -5502 | 0 | ADJUST TO ZERO | lots | 642 | 114 | 642 | 986 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 26 2026 | 4377 | -4377 | 0 | ADJUST TO ZERO | lots | 671 | 114 | 671 | 1056, 1126 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 14 2026 | 300 | -300 | 0 | ADJUST TO ZERO | lots | 406 | 116 | 406 | 621, 637 | 2 |
| 90016 | Batch SS Original Granola #1 | 2026-04-17 | 750 | -750 | 0 | ADJUST TO ZERO | lots | 422 | 116 | 422 | 643, 729 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 24 2026 | 100 | -100 | 0 | ADJUST TO ZERO | lots | 463 | 116 | 463 | 701, 747 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 27 2026 | 700 | -700 | 0 | ADJUST TO ZERO | lots | 478 | 116 | 478 | 716 | 1 |
| 90016 | Batch SS Original Granola #1 | MAY 12 2026 | 350 | -350 | 0 | ADJUST TO ZERO | lots | 580 | 116 | 580 | 861 | 1 |
| 90016 | Batch SS Original Granola #1 | MAY 14 2026 | 4200 | -4200 | 0 | ADJUST TO ZERO | lots | 628 | 116 | 628 | 947 | 1 |

## Records To Be Adjusted

| sku | item | lot | current_system_on_hand_lb | adjustment_quantity_needed_lb | final_quantity_after_adjustment_lb | action | table_name | record_id | product_id | lot_id | related_transaction_ids | transaction_line_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90002 | Batch Classic Granola #9 | 26-05-04-GRAN-001 | 966 | -966 | 0 | ADJUST TO ZERO | lots | 529 | 107 | 529 | 795, 1178 | 2 |
| 90002 | Batch Classic Granola #9 | 26-05-05-GRAN-002 | 1938 | -1938 | 0 | ADJUST TO ZERO | lots | 542 | 107 | 542 | 811 | 1 |
| 90002 | Batch Classic Granola #9 | 26-05-06-GRAN-003 | 969 | -969 | 0 | ADJUST TO ZERO | lots | 550 | 107 | 550 | 821 | 1 |
| 90002 | Batch Classic Granola #9 | MAY 08 2026 | 1407 | -1407 | 0 | ADJUST TO ZERO | lots | 568 | 107 | 568 | 844, 884 | 2 |
| 90002 | Batch Classic Granola #9 | MAY 12 2026 | 1238 | -1238 | 0 | ADJUST TO ZERO | lots | 582 | 107 | 582 | 863, 864, 865, 870 | 4 |
| 90002 | Batch Classic Granola #9 | MAY 13 2026 | 22 | -22 | 0 | ADJUST TO ZERO | lots | 594 | 107 | 594 | 882, 885, 951 | 3 |
| 90002 | Batch Classic Granola #9 | MAY 14 2026 | 646 | -646 | 0 | ADJUST TO ZERO | lots | 630 | 107 | 630 | 950 | 1 |
| 90002 | Batch Classic Granola #9 | MAY 20 2026 | 3014 | -3014 | 0 | ADJUST TO ZERO | lots | 655 | 107 | 655 | 1025, 1058 | 2 |
| 90002 | Batch Classic Granola #9 | MAY 21 2026 | 808 | -808 | 0 | ADJUST TO ZERO | lots | 662 | 107 | 662 | 1043, 1120, 1121, 1169 | 4 |
| 90002 | Batch Classic Granola #9 | MAY 27 2026 | 2261 | -2261 | 0 | ADJUST TO ZERO | lots | 686 | 107 | 686 | 1077 | 1 |
| 90002 | Batch Classic Granola #9 | JUN 01 2026 | 2268 | -2268 | 0 | ADJUST TO ZERO | lots | 722 | 107 | 722 | 1128, 1177, 1180 | 3 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | FEB 10 2026 | 1725 | -1725 | 0 | ADJUST TO ZERO | lots | 129 | 108 | 129 | 162, 224 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | FEB 26 2026 | 1044 | -1044 | 0 | ADJUST TO ZERO | lots | 245 | 108 | 245 | 338 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAR 09 2026 | 2088 | -2088 | 0 | ADJUST TO ZERO | lots | 280 | 108 | 280 | 400 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | 2026-04-20 | 3132 | -3132 | 0 | ADJUST TO ZERO | lots | 431 | 108 | 431 | 656 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | APR 24 2026 | 96 | -96 | 0 | ADJUST TO ZERO | lots | 462 | 108 | 462 | 700, 703 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | APR 28 2026 | 2436 | -2436 | 0 | ADJUST TO ZERO | lots | 491 | 108 | 491 | 740 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAY 11 2026 | 3528 | -3528 | 0 | ADJUST TO ZERO | lots | 572 | 108 | 572 | 852, 868 | 2 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | MAY 27 2026 | 3132 | -3132 | 0 | ADJUST TO ZERO | lots | 687 | 108 | 687 | 1078 | 1 |
| 90001 | Batch Classic Chocolate Chip Granola #9 | JUN 03 2026 | 2088 | -2088 | 0 | ADJUST TO ZERO | lots | 733 | 108 | 733 | 1149 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 01 2026 | 130.5 | -130.5 | 0 | ADJUST TO ZERO | lots | 519 | 114 | 519 | 785, 1100 | 2 |
| 90011 | Batch SS Chocolate Chip Granola #2 | 26-05-04-SSCHOC-001 | 2751 | -2751 | 0 | ADJUST TO ZERO | lots | 530 | 114 | 530 | 796 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 15 2026 | 3144 | -3144 | 0 | ADJUST TO ZERO | lots | 639 | 114 | 639 | 975 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 18 2026 | 5502 | -5502 | 0 | ADJUST TO ZERO | lots | 642 | 114 | 642 | 986 | 1 |
| 90011 | Batch SS Chocolate Chip Granola #2 | MAY 26 2026 | 4377 | -4377 | 0 | ADJUST TO ZERO | lots | 671 | 114 | 671 | 1056, 1126 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 14 2026 | 300 | -300 | 0 | ADJUST TO ZERO | lots | 406 | 116 | 406 | 621, 637 | 2 |
| 90016 | Batch SS Original Granola #1 | 2026-04-17 | 750 | -750 | 0 | ADJUST TO ZERO | lots | 422 | 116 | 422 | 643, 729 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 24 2026 | 100 | -100 | 0 | ADJUST TO ZERO | lots | 463 | 116 | 463 | 701, 747 | 2 |
| 90016 | Batch SS Original Granola #1 | APR 27 2026 | 700 | -700 | 0 | ADJUST TO ZERO | lots | 478 | 116 | 478 | 716 | 1 |
| 90016 | Batch SS Original Granola #1 | MAY 12 2026 | 350 | -350 | 0 | ADJUST TO ZERO | lots | 580 | 116 | 580 | 861 | 1 |
| 90016 | Batch SS Original Granola #1 | MAY 14 2026 | 4200 | -4200 | 0 | ADJUST TO ZERO | lots | 628 | 116 | 628 | 947 | 1 |

## Records Kept Unchanged

| sku | item | lot | current_system_on_hand_lb | adjustment_quantity_needed_lb | final_quantity_after_adjustment_lb | action | table_name | record_id | product_id | lot_id | related_transaction_ids | transaction_line_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 90002 | Batch Classic Granola #9 | JUN 02 2026 | 4868 | 0 | 4868 | KEEP / no adjustment | lots | 726 | 107 | 726 | 1135, 1181 | 2 |

## Proposed Write Pattern - Not Executed

For each row in "Records To Be Adjusted", append one `transactions` row with `type = adjust`, then one negative `transaction_lines` row against the existing `lots.id`. No historical rows should be updated or deleted.

