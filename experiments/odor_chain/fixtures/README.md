這兩個固定輸入取自 seed 0 的原始診斷，供所有訓練 seed 共用，維持原量測的可比性。
它們不是訓練資料或權重，不依賴任何既有 `runs/` 目錄。

- 診斷 seed：1610000000；200 個 teacher-driven observations；normal、puff、curved。
- `observations.npz` SHA-256：`fb016dea946d0430dbea2809c2ea2a38475b4e907805ca1f8475289e35d927e8`
- `codes.npz` SHA-256：`4fb766b9a7851c6fa43795943d05750cd6e2409f8c31113ca2489b204a1c871e`
- graph SHA-256：`a3c70c2759b0c9af179bb40ef0ff6e46ff469c96df0155c8fbe82bc4715f2e71`
- frontend SHA-256：`47cccf33ad8a5ba708d557e500cf48b7f3946be1b0f9f6eecffec6349955ec69`

`measure.py` 使用原始 observations 呼叫 `scripts/diagnostics/modality_sensitivity.py`，
並使用凍結的 `original` sensory codes 量測 DN 工作點；沿用 seed 0 的量測方式。
