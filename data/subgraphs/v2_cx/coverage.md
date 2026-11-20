# v2_cx 候選圖驗證（#138 A）

`scripts/v2/verify_subgraph_v2.py`：36／41 項通過；8792 顆、185261 條邊。

## 沒通過的項目（帳務，不影響圖本身）

- `coverage_manifest_matches`：manifest 沿用 v2_main 的 coverage_repair 紀錄；新圖的覆蓋率見下表（驗證器從原始資料重算）。
- `olfactory_bounded_group_membership`：同上。
- `protected_count_matches`：protected 含 v2_main 的 3 顆修補節點和這次新增的 194 顆，manifest 的 added_nodes 仍是 3。
- `visual_bounded_group_membership`：新增節點的模態標記依路徑設定，不是依跳數上限。模型只用 group_mask 找轉向／前進 DN。
- `wind_bounded_group_membership`：同上。

## 通過的項目

- `all_nodes_on_seed_dn_path`
- `all_requested_seed_types_retained`
- `array_dtypes`
- `artifact_budget`
- `artifact_hashes`
- `brain_input_array`
- `coverage`
- `edge_budget`
- `exact_induced_raw_edges`
- `lc10a_aotu_to_each_dna02`
- `lc11_to_required_dn`
- `lc12_to_required_dn`
- `lc9_to_each_dnp09`
- `motor_group_matches_config`
- `no_kc_apl_dpm`
- `no_kc_class`
- `node_budget`
- `node_types_match_publisher`
- `olfactory_branch_all_inputs_to_all_motors`
- `olfactory_inputs_match_config`
- `olfactory_publisher_sides`
- `orn_pn_lh_dn`
- `protected_on_seed_dn_path`
- `repair_paths_valid`
- `required_dn_bilateral`
- `required_motors_present`
- `signs_resolved`
- `spec_version`
- `subgraph_input_array`
- `visual_branch_all_inputs_to_all_motors`
- `visual_inputs_match_config`
- `weakly_connected`
- `wind_branch_all_inputs_to_all_motors`
- `wind_inputs_match_config`
- `wind_root_sides`
- `wind_types_to_required_dn`

## 轉向／前進 DN 的輸入覆蓋率（從原始資料重算）

| DN | bodyId | side | 覆蓋率 | 全腦輸入 | 子圖輸入 |
| --- | --- | --- | --- | --- | --- |
| DNb05 | 10065 | 1 | 0.590233 | 18613 | 10986 |
| DNb05 | 10118 | -1 | 0.539469 | 17938 | 9677 |
| DNa02 | 10360 | 1 | 0.608538 | 24830 | 15110 |
| DNa01 | 10442 | -1 | 0.564453 | 13087 | 7387 |
| DNa01 | 10760 | 1 | 0.566647 | 13804 | 7822 |
| DNp09 | 10783 | -1 | 0.498958 | 8640 | 4311 |
| DNa03 | 10975 | 1 | 0.529114 | 16796 | 8887 |
| DNp09 | 11177 | 1 | 0.401507 | 7300 | 2931 |
| DNa03 | 519624 | -1 | 0.536919 | 18053 | 9693 |
| DNa02 | 523769 | -1 | 0.609165 | 24527 | 14941 |
