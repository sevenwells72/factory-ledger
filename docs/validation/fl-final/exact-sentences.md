# Exact sentence acceptance

All 312 selected checks passed: 240 provider-backed, 72 deterministic. 679 distinct completed real-provider responses; model unchanged; store:false. Each row passed 3/3 in both modes. Historical failures are excluded and preserved separately.

Short/incomplete receives supply missing supplier, weight and BOL facts after the first reply; products and counts are never corrected by the runner. Physical make tags are supplied only by the explicit test worker.

| Suite | Sentence | Sessions off | Sessions on | Provider |
|---|---|---:|---:|---|
| main / bags_each_en | We received 10 bags of oats, 50 lb each, from Grain Supply, BOL 5521, supplier lot SL-4471. | 3/3 | 3/3 | Real model |
| main / bags_en | We received 10 bags of oats, 50 lb bags, from Grain Supply, BOL 5521, supplier lot SL-4471. | 3/3 | 3/3 | Real model |
| main / bags_es | Recibimos 10 bolsas de avena de 50 libras de Grain Supply, BOL 5521, lote del proveedor SL-4471. | 3/3 | 3/3 | Real model |
| main / cancel_en | Cancel order SO-261015-001 | 3/3 | 3/3 | No model needed |
| main / cancel_es | Cancela el pedido SO-261015-001 | 3/3 | 3/3 | No model needed |
| main / change_en | Change order SO-261015-001 to 3 cases | 3/3 | 3/3 | No model needed |
| main / change_es | Cambia el pedido SO-261015-001 a 3 cajas | 3/3 | 3/3 | No model needed |
| main / expected_en | Grain Supply is sending 20 cases of Oats next Tuesday, PO FINAL-1. Add it as an expected delivery. | 3/3 | 3/3 | No model needed |
| main / expected_es | Grain Supply enviará 20 cajas de avena el próximo martes, PO FINAL-1. Agrégalo como entrega esperada. | 3/3 | 3/3 | No model needed |
| main / found_en | We found 2 cases of Honey, 5 lb each, during the physical count. | 3/3 | 3/3 | Real model |
| main / found_es | Encontramos 2 cajas de miel, 5 lb cada una, en el conteo físico | 3/3 | 3/3 | Real model |
| main / found_reason_es | Encontramos 2 cajas de miel, 5 lb cada una. | 3/3 | 3/3 | Real model |
| main / kosher_en | We received 2 cases of product 283 from Grain Supply, 25 lb per case, BOL 5521. | 3/3 | 3/3 | Real model |
| main / kosher_es | Recibimos 2 cajas del producto 283 de Grain Supply, 25 lb por caja, BOL 5521. | 3/3 | 3/3 | Real model |
| main / make_en | We made 1 batch of Classic Granola #9 | 3/3 | 3/3 | Real model |
| main / make_es | Hicimos 1 lote de Classic Granola #9 | 3/3 | 3/3 | Real model |
| main / multi_en | We received 5 cases of Sunflower Seeds and 3 cases of Oats from Grain Supply, 50 lb per case, BOL MULTI-5521. | 3/3 | 3/3 | Real model |
| main / multi_es | Recibimos 5 cajas de semillas de girasol y 3 cajas de avena de Grain Supply, 50 lb por caja, BOL MULTI-5521. | 3/3 | 3/3 | Real model |
| main / receive_en | We received 6 cases of Sunflower Seeds from Grain Supply, 50 lb per case, BOL FS8996, supplier lot SL-64AED4 | 3/3 | 3/3 | Real model |
| main / receive_es | Recibimos 6 cajas de semillas de girasol de Grain Supply, 50 lb por caja, BOL FS8996, lote del proveedor SL-64AED4 | 3/3 | 3/3 | Real model |
| main / setton_en | Add Setton as a new customer | 3/3 | 3/3 | Real model |
| main / setton_es | Agrega Setton como cliente nuevo | 3/3 | 3/3 | Real model |
| main / shipped_en | Ship SO-261012-001 | 3/3 | 3/3 | No model needed |
| main / shipped_es | Envía el pedido SO-261012-001 | 3/3 | 3/3 | No model needed |
| main / shortage_en | Ship all remaining on order SO-261016-047, BOL FINAL-SHIP. | 3/3 | 3/3 | Real model |
| main / shortage_es | Envía todo lo pendiente del pedido SO-261016-047, BOL FINAL-SHIP. | 3/3 | 3/3 | Real model |
| main / sku_283 | how much two eighty three do we have | 3/3 | 3/3 | Real model |
| main / sku_70013 | how much seven zero zero one three do we have | 3/3 | 3/3 | Real model |
| main / supplier_es | Recibimos 4 cajas de avena de la cross milling, 50 lb por caja, BOL 5521. | 3/3 | 3/3 | Real model |
| main / supplier_voice | we received four cases of oats from la cross milling fifty pounds a case B O L five five two one | 3/3 | 3/3 | Real model |
| supplement / bags_each_exact_en | 10 bags of oats, 50 lb each | 3/3 | 3/3 | Real model |
| supplement / bags_exact_en | 10 bags of oats, 50 lb bags | 3/3 | 3/3 | Real model |
| supplement / bags_exact_es | 10 bolsas de avena de 50 libras | 3/3 | 3/3 | Real model |
| supplement / change_report_en | Change order SO-261015-001: Kookies & Kreme should be 6 cases, not 4 | 3/3 | 3/3 | No model needed |
| supplement / change_report_es | Cambia el pedido SO-261015-001: Kookies & Kreme debe ser 6 cajas, no 4 | 3/3 | 3/3 | No model needed |
| supplement / expected_short_en | Grain Supply is sending 20 cases of Oats | 3/3 | 3/3 | No model needed |
| supplement / expected_short_es | Grain Supply está enviando 20 cajas de avena | 3/3 | 3/3 | No model needed |
| supplement / found_reason_record_en | We found 2 cases of Honey, 5 lb each | 3/3 | 3/3 | Real model |
| supplement / found_reason_record_es | Encontramos 2 cajas de miel, 5 lb cada una | 3/3 | 3/3 | Real model |
| supplement / found_record_en | We found 2 cases of Honey, 5 lb each, during the physical count | 3/3 | 3/3 | Real model |
| supplement / found_record_es | Encontramos 2 cajas de miel, 5 lb cada una, en el conteo físico | 3/3 | 3/3 | Real model |
| supplement / make_missing_en | We made 1 batch of Setton Coffee Granola #12 | 3/3 | 3/3 | Real model |
| supplement / make_missing_es | Hicimos 1 lote de Setton Coffee Granola #12 | 3/3 | 3/3 | Real model |
| supplement / multi_short_en | 5 cases of Sunflower Seeds and 3 cases of Oats | 3/3 | 3/3 | Real model |
| supplement / multi_short_es | 5 cajas de semillas de girasol y 3 cajas de avena | 3/3 | 3/3 | Real model |
| supplement / setton_bare_en | Add Setton | 3/3 | 3/3 | Real model |
| supplement / setton_bare_es | Agrega Setton | 3/3 | 3/3 | Real model |
| supplement / ss_original_voice | we made one batch of S S original number one | 3/3 | 3/3 | Real model |
| trace / different_en | We made 1 batch of Classic #9 | 3/3 | 3/3 | Real model |
| trace / different_es | Hicimos 1 lote de Classic #9 | 3/3 | 3/3 | Real model |
| trace / receive_en | We received 6 cases of Sunflower Seeds from Grain Supply, 50 lb per case, BOL FS8996, supplier lot SL-64AED4 | 3/3 | 3/3 | Real model |
| trace / receive_es | Recibimos 6 cajas de semillas de girasol de Grain Supply, 50 lb por caja, BOL FS8996, lote del proveedor SL-64AED4 | 3/3 | 3/3 | Real model |
