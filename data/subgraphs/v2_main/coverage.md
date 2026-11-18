# v2 子圖覆蓋率

| DN | bodyId | side | 補救前 | 補救後 | 全腦輸入 | 子圖輸入 |
| --- | --- | --- | --- | --- | --- | --- |
| DNb05 | 10065 | 1 | 0.590233 | 0.590233 | 18613 | 10986 |
| DNb05 | 10118 | -1 | 0.539469 | 0.539469 | 17938 | 9677 |
| DNa02 | 10360 | 1 | 0.593234 | 0.593234 | 24830 | 14730 |
| DNa01 | 10442 | -1 | 0.564453 | 0.564453 | 13087 | 7387 |
| DNa01 | 10760 | 1 | 0.566647 | 0.566647 | 13804 | 7822 |
| DNp09 | 10783 | -1 | 0.498958 | 0.498958 | 8640 | 4311 |
| DNa03 | 10975 | 1 | 0.516671 | 0.516671 | 16796 | 8678 |
| DNp09 | 11177 | 1 | 0.394658 | 0.401507 | 7300 | 2931 |
| DNa03 | 519624 | -1 | 0.523459 | 0.523459 | 18053 | 9450 |
| DNa02 | 523769 | -1 | 0.594651 | 0.594651 | 24527 | 14585 |

補救新增 3 節點、74 邊。
號誌不明移除 122 顆。

## 必要路徑

- required_dn_bilateral: PASS
- lc9_to_each_dnp09: PASS
- lc10a_aotu_to_each_dna02: PASS
- lc11_to_required_dn: PASS
- lc12_to_required_dn: PASS
- no_kc_apl_dpm: PASS
- weakly_connected: PASS
- orn_pn_lh_dn: PASS
- wind_types_to_required_dn: PASS
- all_requested_seed_types_retained: PASS
- dn_coverage: PASS
- visual_branch_all_inputs_to_all_motors: PASS
- olfactory_branch_all_inputs_to_all_motors: PASS
- wind_branch_all_inputs_to_all_motors: PASS

## 保留的 seed 型別

| 型別 | 發布顆數 | 可用顆數 | 保留顆數 | 最短 DN hops |
| --- | --- | --- | --- | --- |
| JO-CA1 | 10 | 10 | 9 | — |
| JO-CL | 19 | 19 | 3 | — |
| JO-CM | 30 | 30 | 2 | — |
| JO-ED1 | 18 | 18 | 5 | — |
| JO-ED2_a | 38 | 38 | 11 | — |
| JO-ED2_c | 14 | 14 | 4 | — |
| JO-EV1 | 50 | 50 | 9 | — |
| JO-EV2 | 36 | 36 | 25 | — |
| JO-EV3 | 45 | 45 | 30 | — |
| JO-EV5 | 23 | 23 | 13 | — |
| JO-EV6 | 21 | 21 | 3 | — |
| LC10a | 275 | 275 | 275 | — |
| LC11 | 143 | 143 | 73 | — |
| LC12 | 498 | 498 | 6 | — |
| LC15 | 126 | 126 | 69 | — |
| LC9 | 219 | 219 | 200 | — |
| LPLC2 | 185 | 185 | 180 | — |
| ORN_DM1 | 74 | 74 | 74 | — |
| ORN_VA2 | 83 | 74 | 74 | — |

## 拿掉的 seed 型別

| 型別 | 發布顆數 | 可用顆數 | 保留顆數 | 最短 DN hops |
| --- | --- | --- | --- | --- |
| JO-CA2 | 9 | 9 | 0 | 3 |
| JO-ED2_b | 21 | 21 | 0 | 3 |
| JO-EV4 | 1 | 1 | 0 | 3 |
| LC17 | 353 | 353 | 0 | 3 |

## 風的二階型別候選

僅為連接性候選，並非已確認的 AMMC／WED／SAD 命名。

- SAD077: 2991 突觸
- AMMC012: 2128 突觸
- CB3746: 1411 突觸
- AMMC013: 1363 突觸
- pIP1: 1320 突觸
- AMMC008: 1181 突觸
- SAD110: 944 突觸
- SAD004: 914 突觸
- CB3673b: 720 突觸
- DNg29: 708 突觸
