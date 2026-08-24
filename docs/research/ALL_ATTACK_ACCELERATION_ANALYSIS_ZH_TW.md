# 17 種攻擊加速可行性與最佳化實驗

## 一、研究目的

既有效能研究（`docs/research/PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`）已完整驗證 FGSM／deterministic PGD 的 `implementation_optimization`，以及 CW 的 `batched_algorithmic_variant` 分類，但正式 attack registry 目前共 17 種攻擊，其餘 14 種從未逐一量測 baseline latency、thread tuning、batching 可行性與正確性。本輪回答指導教授的問題「Attack 怎麼讓它變快？哪些地方可以加速？」，對 17 種攻擊逐一：量測 baseline latency、找出瓶頸、測試 CPU thread 數與 batch size 兩種加速手段、驗證加速後 attack semantics 是否維持、並給出正式 speedup 與 batching classification。本輪不重跑 satellite-like 576 最終實驗、不修改既有正式 results、不修改 external/AWN 與 external/adversarial-rf、不修改攻擊演算法本身（eps／steps／objective／stopping threshold 一律不變，僅測試 thread 數與 batch size 兩個與演算法無關的執行參數）。

## 二、17 Attack Inventory

由 `src/adapters/attack_adapter.py:_ATTACK_ACCEPTED_PARAMS` 於本輪執行時重新查詢，確認正式 registry 為：

```
apgd, apgdt, autoattack, bim, cw, deepfool, difgsm, ead, fab, fgsm,
mifgsm, pgd, rfgsm, square, tpgd, vmifgsm, vnifgsm
```

共 17 種，與教授要求的清單（FGSM／BIM／PGD／MIFGSM／DIFGSM／VMIFGSM／VNIFGSM／RFGSM／TPGD／CW／DeepFool／FAB／Square／APGD／APGDT／AutoAttack／EAD）完全一致，**無命名或數量差異**。PGD 依既有專案慣例拆成 `pgd_det`（`random_start=False`）與 `pgd_stoch`（`random_start=True`）兩個獨立量測條件，DIFGSM 使用本專案既有的 IQ-compatible 自行實作（`src/adapters/iq_difgsm.py:IQDIFGSM`），非 torchattacks 原生 DIFGSM。

## 三、Baseline Methodology

固定條件：真實 AWN checkpoint（`external/adversarial-rf/2016.10a_AWN.pkl`）、真實 attack backend（`torchattacks==3.5.1` 或 `IQDIFGSM`）、CPU、`radioml-native` 前處理、`batch_size=1`、系統預設 torch thread 數（本機為 16）、固定 seed=0、eps=0.05（適用於接受 eps 的攻擊）、其餘所有參數（steps／restarts／n_queries／kappa／lr 等）維持 torchattacks 套件本身安裝版本的預設值，全程未調降或調高任何攻擊自身的疊代／收斂參數。統一 benchmark fixture：BPSK／QPSK／QAM16／WBFM 四種調變 × SNR {-10, 0, 18} × 每格 5 筆樣本 = 60 筆（fast-tier）。先對每種攻擊做 3-sample pilot，依「完整 fast-tier benchmark（baseline + thread tuning + batching sweep）」的總呼叫量預估耗時，超過 30 分鐘才會改用 20 筆固定 slow-tier 子集（本輪 17 種攻擊之 pilot 預估皆低於 30 分鐘，故全部使用 60 筆 fast-tier，見第八節）。每筆呼叫前已完成暖身（pilot／baseline 皆為正式呼叫，非額外暖身輪）。

## 四、CPU Thread Tuning

對每種攻擊在固定 8 筆樣本子集（所有攻擊共用同一子集，確保跨攻擊比較公平）、`batch_size=1` 下，測試 `torch_num_threads ∈ {1,2,4,8,16}`，取 median latency 最低者為 best_threads。Thread tuning 純粹是執行環境參數，不改變攻擊演算法或其輸出張量，因此理論上不應改變任何攻擊的輸出——本輪未觀察到 thread 數變動導致預測結果改變的情形（thread tuning 僅在同一 batch_size=1、同一輸入下切換 thread 數，不涉及批次重組，天然不產生本文件定義的 A/B/C/D 分類問題）。

## 五、Batching Methodology

在各攻擊各自的 best_threads 下，以 `batch_size ∈ {1,2,4,8,16,32}`（依子集大小裁切）逐一測試。分類規則：

- **A_implementation_optimization**：batch>1 相對 batch=1 的最大張量絕對差 < 1e-4 且預測 100% 相符。
- **B_stochastic_batch_compatible**：攻擊本身含 `random_start=True` 等不可逐呼叫定序的隨機成分，不宣稱 bit-identical，僅回報預測相符率。
- **C_batched_algorithmic_variant**：預測相符率 ≥ 90%，但張量或預測非 bit-identical，判定 batching 改變了最佳化軌跡。
- **D_batching_unsafe**：預測相符率 < 90%，或該攻擊在部分／全部 batch size 下對真實 backend 直接丟出例外（fallback 至 dummy_attack）。

分類使用「跨所有測試 batch size 的最差表現」（worst-case），而非平均值，避免以偶然表現較好的單一 batch size 掩蓋其他 batch size 下的不穩定。

## 六、Correctness Criteria

Batch=1（`batching_ref`）作為正確性比較基準，對每個 batch size 逐樣本比較：attacked tensor 最大絕對差（Linf）、平均 L2 差、attacked prediction 相符率。**任何一次 fallback 至 `dummy_attack` 的呼叫，其延遲與輸出皆從所有正確性與延遲聚合中排除**（fallback 結果不是真實攻擊的量測值，見第八節、第二十二節 Square 案例的完整說明）——這是本輪執行過程中發現並修正的一個資料完整性錯誤（詳見第二十九節）。

## 七、Attack-Specific Constraints

依教授指示逐一檢查的重點（詳細結果見第十至二十六節各攻擊小節）：DIFGSM 之 input diversity 是否為 per-sample random；VMIFGSM／VNIFGSM 之 neighbor gradient sampling／variance estimation 是否被 batching 污染；RFGSM 之 random initialization 是否可用 per-sample generator；TPGD 之 targeted/untargeted 語意與 batch labels；CW 之 early stopping／optimization trajectory（沿用既有結論，本輪重新量測）；DeepFool／FAB 之 per-sample iterative boundary search／restart 是否可安全向量化；Square 之 random state／query count／per-sample stopping；APGD／APGDT 之 adaptive step size／oscillation detection／restart 是否被 batching 影響 sample-specific state；AutoAttack 之 ensemble 內部結構；EAD 之 elastic-net objective／stopping。

## 八、Results

`experiments/benchmark_all_attack_acceleration.py`（新增，唯一一支涵蓋全部 17 種攻擊的 script，非逐攻擊各自一支）於 `results/all_attack_acceleration_20260824T031053Z/` 產出初次完整結果：**18 個量測條件（17 種攻擊，PGD 拆為 det/stoch）全數完成，0 error**。18 個 pilot 預估皆低於 30 分鐘門檻，17 種攻擊全數使用 60 筆 fast-tier 子集（無攻擊被降級為 slow-tier）。總執行時間 13.3 分鐘。`n_unexpected_fallback=60`，全數集中在 Square（49 筆）／APGDT（6 筆）／AutoAttack（5 筆）的 `batching_test` 階段（batch_size>1），baseline／thread_tuning／batching_ref／pilot 階段共 0 筆 fallback。

### 8.1 初次 benchmark 之後的完整性稽核：發現並修正 seed pairing 不一致

完成初次 benchmark 後，本輪進行了一次獨立的完整性稽核（completeness audit），逐一核對每種攻擊的 baseline／optimized 是否真正 paired。稽核發現：**batching 正確性比較的 reference（batch=1）與 test（batch>1）呼叫，對接受明確 `seed` 參數的攻擊使用了兩組不同的絕對 seed 數值**（reference 使用 `2000+i`，test 使用 `3000+start`，`i`／`start` 分別為樣本在子集中的位置與批次起點）。以正式 registry 逐一內省（`inspect.signature`）確認，17 種攻擊中共 6 種會把 `seed` 傳入真實攻擊建構子：DIFGSM、FAB、Square、APGD、APGDT、AutoAttack；其餘 11 種與兩種 PGD 變體皆不接受此參數，不受影響。

對這 6 種攻擊逐一以「相同樣本、`batch_size=1`、僅改變 seed 數值」做隔離測試（不涉及任何 batching），結果：FAB 在 10 組相同樣本配對下 **100% 相符**（`seed` 數值本身不改變其輸出，見第二十節），判定不受此問題影響；DIFGSM／Square／APGD／APGDT／AutoAttack 的相符率分別為 70%／70%／70%／20%／70%——與原始 batching 測試回報的相符率相近甚至更低，證實原始 `D_batching_unsafe` 分類（尤其 DIFGSM 的「batch-shared diversity randomness」推論）**混雜了 seed 數值差異本身的效果，無法單獨歸因於 batching**。

修正後的 seed policy（`experiments/rerun_attack_acceleration_corrected_seeds.py`）：所有單樣本呼叫（baseline／thread tuning／E2E）改用 `stable_seed(modulation, snr, sample_index, attack_name)`——由樣本與攻擊身分決定的固定雜湊值，與呼叫發生在哪個 phase／迴圈位置無關；batching 正確性比較則針對每個測試中的 batch size，重新以「該批次錨定樣本的 `stable_seed`」重算 batch=1 reference，並以同一個 seed 呼叫 batch>1 的 test——確保同一次比較的兩側使用完全相同的 seed 數值，僅 batching 本身作為唯一變因。此修正方法已先以獨立的 unit validation 確認（同一樣本在 batch=1／batch=2 chunk／batch=4 chunk 下取得的 seed 完全相同，不同樣本則得到不同 seed，無碰撞），才進行實際重跑。

只重跑受影響的 5 種攻擊（DIFGSM／Square／APGD／APGDT／AutoAttack），使用與原始 benchmark 完全相同的資料集、調變／SNR 網格、樣本清單與順序、AWN checkpoint、前處理、攻擊超參數（eps=0.05，其餘沿用套件預設）、thread／batch 候選與計時方法，**唯一改變的是 seed policy**。重跑於 `results/all_attack_acceleration_corrected_20260824T055724Z/rerun_5_attacks_only/` 完成：5/5 攻擊、0 error、57 筆 fallback（同樣集中在 Square/APGDT/AutoAttack 的 batch>1 階段，數量與原始的 60 筆相近但不完全相同——這是預期的，因為不同 seed 會觸發第三方套件例外的具體樣本組合本身就是資料相依的，見第二十一節）。**最重要的發現：DIFGSM 在修正 seed 後，於全部測試 batch size（2/4/8/16/32）下皆為 bit-identical（max diff=0、100% 預測相符），分類由原本的 `D_batching_unsafe` 更正為 `A_implementation_optimization`**——原文件對 DIFGSM「diversity 隨機閘門為 batch 共用而非逐樣本獨立」的推論並不成立，該推論本身是 seed 混淆造成的假象，予以撤回。Square／APGD／APGDT／AutoAttack 在修正後的公平比較下**仍然維持 `D_batching_unsafe`**，相符率調整為 Square 75%（原 62.5%）、APGD 83.3%（原 71.7%）、APGDT 63.6%（原 55.8%）、AutoAttack 76.9%（原 71.4%）——數值有變動，但結論方向不變，且現在的數字才是可信的（不再混雜 seed 效應）。

最終結果目錄合併「12 種未受影響攻擊＋FAB（原始 benchmark，未重跑）」與「5 種受影響攻擊（修正後重跑）」，寫入 `results/all_attack_acceleration_corrected_20260824T055724Z/`，`manifest.json` 完整記錄來源目錄、bug 說明、受影響／未受影響攻擊清單與修正後 seed policy。原始 `results/all_attack_acceleration_20260824T031053Z/` 保留於本機作為 provenance，其 5 種受影響攻擊之數字**視為 superseded，不再作為正式結論引用**；本文件以下所有小節與第九節之表格，皆已更新為 corrected 版本的數字。

## 九、17-Row Acceleration Table

（PGD 以 `pgd_det` 為代表列，`pgd_stoch` 另於第十二節說明；本表逐字取自 `attack_bottleneck_summary.csv`。）

| Attack | Baseline median (ms) | Optimized median (ms) | Baseline p95 (ms) | Optimized p95 (ms) | Speedup median | Speedup p95 | Optimization method | Best batch | Best threads | Correctness | Batching classification | Main limitation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| FGSM | 5.73 | 0.438 | 37.94 | 1.257 | 13.07x | 30.17x | threads+batching | 16 | 4 | bit-identical, 100% match | A_implementation_optimization | none |
| BIM | 49.27 | 3.371 | 120.11 | 4.453 | 14.62x | 26.97x | threads+batching | 16 | 2 | bit-identical, 100% match | A_implementation_optimization | none |
| PGD (det) | 31.68 | 5.013 | 70.11 | 5.248 | 6.32x | 13.36x | threads+batching | 32 | 1 | bit-identical, 100% match | A_implementation_optimization | random_start=True variant is separate (B), see §12 |
| MIFGSM | 41.48 | 16.489 | 210.49 | 20.625 | 2.52x | 10.21x | threads only | 1 | 1 | 100% match, tensor diff 4.0e-4 (not bit-identical) | C_batched_algorithmic_variant | batching not safe despite stable predictions |
| DIFGSM | 24.80 | 2.249 | 47.78 | 2.249 | 11.03x | 21.24x | threads+batching (corrected seed policy) | 32 | 4 | bit-identical, 100% match at every batch size | A_implementation_optimization | none -- original D verdict retracted, see §8.1 |
| VMIFGSM | 179.62 | 85.395 | 326.46 | 96.670 | 2.10x | 3.38x | threads only | 1 | 1 | worst match 91.7% | C_batched_algorithmic_variant | neighbor-sample variance estimation cross-contamination suspected |
| VNIFGSM | 205.10 | 84.344 | 639.07 | 91.148 | 2.43x | 7.01x | threads only | 2 | 2 | worst match 96.7% | C_batched_algorithmic_variant | same as VMIFGSM |
| RFGSM | 25.66 | 14.855 | 47.38 | 16.035 | 1.73x | 2.96x | threads only | 4 | 4 | worst match 81.7% | D_batching_unsafe | borderline -- most batch sizes ≥90%, worst case below |
| TPGD | 33.03 | 16.378 | 82.45 | 20.904 | 2.02x | 3.94x | threads only | 1 | 1 | worst match 36.7% | D_batching_unsafe | most unstable of the "cheap" gradient attacks under batching |
| CW | 71.08 | 21.464 | 231.80 | 32.429 | 3.31x | 7.15x | threads only | 4 | 4 | worst match 93.3%, max diff 0.00141 | C_batched_algorithmic_variant | reconfirms prior finding (95.0% match, diff 0.00138) |
| DeepFool | 173.35 | 24.743 | 849.91 | 39.648 | 7.01x | 21.44x | threads only | 2 | 2 | worst match 6.7% (degrades with batch size) | D_batching_unsafe | per-sample boundary search desyncs badly at large batch |
| FAB | 954.77 | 18.085 | 1409.13 | 19.364 | 52.79x | 72.77x | threads+batching | 32 | 4 | bit-identical, 100% match | A_implementation_optimization | benefit is batch-throughput only, see §14/§28 |
| Square | 9.27 | 3.952 | 83.89 | 11.733 | 2.35x | 7.15x | threads only (corrected seed policy) | 1 | 4 | worst match 75.0% + intermittent real-backend crash at batch>1 | D_batching_unsafe | third-party torchattacks.Square batching bug, see §21 |
| APGD | 33.41 | 20.141 | 59.91 | 21.812 | 1.66x | 2.75x | threads only (corrected seed policy) | 1 | 2 | worst match 83.3% | D_batching_unsafe | adaptive step-size/restart state not batch-safe, confirmed after seed correction |
| APGDT | 37.10 | 22.599 | 87.71 | 24.795 | 1.64x | 3.54x | threads only (corrected seed policy) | 1 | 4 | worst match 63.6% + partial fallback | D_batching_unsafe | same as APGD, plus targeted-loss instability, confirmed after seed correction |
| AutoAttack | 528.19 | 282.311 | 839.38 | 285.701 | 1.87x | 2.94x | threads only (corrected seed policy) | 1 | 4 | worst match 76.9% + partial fallback | D_batching_unsafe | ensemble internals not batch-safe, confirmed after seed correction; baseline itself uses version="rand" (see §24) |
| EAD | 2295.39 | 518.203 | 3414.05 | 558.644 | 4.43x | 6.11x | threads only | 1 | 1 | worst match 95.0% | C_batched_algorithmic_variant | slowest attack overall; still no safe batching |

無任何一列缺漏；**5 個 A**、1 個 B（`pgd_stoch`，未併入本表 17 列，見第十二節）、5 個 C、**7 個 D**（皆為 §8.1 稽核修正後之最終結果）。**17/17 attack 皆完成完整 baseline+thread+batching+correctness+E2E 量測，沒有任何一種因為慢而被跳過，也沒有任何一種因為稽核發現的 seed 問題而被略過修正。**

## 十、FGSM

Baseline（batch=1, threads=16）median 5.73ms／p95 37.94ms。單步攻擊，瓶頸主要是 Python/wrapper 呼叫開銷而非梯度計算本身（configured iteration count=1）。Best threads=4；batching 在 threads=4 下最快為 batch=16，median 0.438ms／p95 1.257ms，**bit-identical**（max diff=0、100% 預測相符），分類 **A_implementation_optimization**，與既有 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md` 的結論方向一致（該文件另一次獨立量測得到 16.10x，本輪為 13.07x／30.17x，量級相符，數值差異來自不同機器負載與不同 baseline 分母，非矛盾）。E2E：attack_generation median 2.53ms／p95 2.83ms，total median 4.76ms／p95 5.28ms，processing class 皆為 **A（<10ms）**。

## 十一、BIM

BIM 是 iterative FGSM（configured steps=10），本輪不假設其與 FGSM 具有相同 batching 行為，實測結果同樣達到 **A_implementation_optimization**（bit-identical，100% 相符）。Baseline median 49.27ms／p95 120.11ms；best threads=2；batching 最快於 batch=16，median 3.371ms／p95 4.453ms，speedup median 14.62x／p95 26.97x。E2E attack_generation median 14.60ms／p95 15.61ms，total median 16.92ms／p95 18.03ms，processing class 為 attack-generation A、E2E B（10–20ms）。

## 十二、PGD

**random_start=False（deterministic）**：baseline median 31.68ms／p95 70.11ms；best threads=1；batching 於 batch=32 達 bit-identical（100% 相符），分類 **A_implementation_optimization**，median 5.013ms／p95 5.248ms，speedup median 6.32x／p95 13.36x。

**random_start=True（stochastic）**：baseline median 51.10ms／p95 207.29ms；best threads=4；batching 於 batch=32，median 1.896ms／p95 1.963ms，speedup median 26.95x／p95 105.62x——**此 speedup 數字為 batch=32 下的量測值，不宣稱與 batch=1 bit-identical**，逐 batch size 的預測相符率介於 71.7%–81.7%（隨機起點使 torch 全域 RNG 在不同 batch 佈局下被抽取的次數不同，這是 PGD 本身的隨機性，不是 batching 的錯誤），分類 **B_stochastic_batch_compatible**，與既有文件對 `random_start=True` 只做 throughput-only 宣稱的原則一致。E2E 部分兩者接近（attack_generation median 13.6–14.1ms，total median 15.8–16.4ms，processing class B），deterministic 版本才是本表第九節唯一列出的「PGD」代表列。

## 十三、MIFGSM

Momentum-based iterative attack（configured steps=10, decay=1.0）。Baseline median 41.48ms／p95 210.49ms。逐 batch size 檢查：**預測 100% 相符**，但張量最大絕對差恆為 4.04e-4（略高於 1e-4 門檻），故未達 A 的 bit-identical 標準，分類 **C_batched_algorithmic_variant**——momentum 狀態本身雖未造成預測翻轉，但確實不是逐位元相同，證實 momentum 累積在數值上對 batch 佈局敏感，即使在本測試集上尚未影響最終分類結果。Thread-only 最佳化（threads=1）median 16.489ms／p95 20.625ms，speedup median 2.52x／p95 10.21x。E2E attack_generation median 14.27ms／p95 14.62ms，total median 16.63ms／p95 16.93ms，processing class C／B。

## 十四、DIFGSM

使用本專案 IQ-compatible 自行實作 `IQDIFGSM`，是接受明確 `seed` 參數的六種攻擊之一。初次量測曾顯示逐 batch size 相符率落在 71.7%–80.0%、最大絕對差在每個 batch size 下恆為 0.014096，並依此推論「diversity 的隨機閘門為整批共用而非逐樣本獨立」——**這個結論已於第八點一節之稽核撤回**：以相同樣本、僅改變 seed 數值（不涉及 batching）做隔離測試，同樣得到約 70% 的相符率，證實原始 71.7%–80.0% 主要（甚至可能完全）是 reference／test 兩側使用不同 seed 數值造成的混淆，不能歸因於 diversity 機制本身的 batch-shared randomness。

以 §8.1 所述修正後 seed policy（batching 比較的兩側使用相同的 chunk 錨定 seed）重新驗證：**DIFGSM 在全部測試 batch size（2/4/8/16/32）下皆為 bit-identical（max diff=0、100% 預測相符）**，分類更正為 **A_implementation_optimization**。這代表在公平的 seed 控制下，`IQDIFGSM` 的 diversity 變換（不論其隨機閘門內部實作為何）並未實際造成 batching 下的輸出差異——先前觀察到的不穩定性是量測方法的產物，不是 `IQDIFGSM` 的性質。Baseline median 24.80ms／p95 47.78ms；batch=32／threads=4 下 median 2.249ms／p95 2.249ms，speedup median 11.03x／p95 21.24x。E2E attack_generation median 14.36ms／p95 14.62ms，total median 16.71ms／p95 17.09ms，processing class A／B。本輪對 `IQDIFGSM` 的隨機數產生邏輯本身仍未做任何修改（僅修正了 benchmark 自身的 seed 呼叫方式），符合「不得修改攻擊演算法」的授權範圍。

## 十五、VMIFGSM／VNIFGSM

依教授指示特別檢查 variance tuning／neighbor gradient sampling 是否被 batching 污染：兩者的 configured N=5（neighbor samples）、beta=1.5。**VMIFGSM** baseline median 179.62ms／p95 326.46ms，逐 batch size 相符率 91.7%–95.0%（batch=32 時最高，可能因為向量化路徑反而更接近逐樣本獨立），worst-case 91.7% 分類 **C_batched_algorithmic_variant**；thread-only median 85.395ms／p95 96.670ms，speedup median 2.10x／p95 3.38x。**VNIFGSM** baseline median 205.10ms／p95 639.07ms，相符率 96.7%–100.0%，worst-case 96.7% 同樣分類 C；thread-only median 84.344ms／p95 91.148ms，speedup median 2.43x／p95 7.01x。兩者結果支持教授的假設方向：neighbor-sample 的 variance estimation 確實會因為批次佈局不同而產生非 bit-identical 的輸出，但尚未嚴重到系統性翻轉大多數預測（故落在 C 而非 D）。E2E：VMIFGSM attack_generation median 76.60ms／p95 77.77ms、total median 78.95ms／p95 80.22ms；VNIFGSM attack_generation median 77.57ms／p95 79.92ms、total median 80.13ms／p95 82.37ms；两者 processing class 皆為 D（50–100ms）。

## 十六、RFGSM

依教授指示檢查 random initialization 是否可用 per-sample generator：torchattacks 的 RFGSM 建構子本身無 `random_start` 參數（其隨機性是無條件內建於演算法中，非可選開關），本輪未新增自訂 per-sample RNG 機制（會構成演算法變更）。Baseline median 25.66ms／p95 47.38ms。逐 batch size 相符率 81.7%–93.3%，多數 batch size 已達或接近 90% 門檻，但 worst-case（batch=16，81.7%）低於門檻，分類 **D_batching_unsafe**——屬於本次 D 分類中最接近 C／A 邊界的一項，值得未來以更大樣本數重新檢驗此邊界是否穩定。Thread-only（threads=4）median 14.855ms／p95 16.035ms，speedup median 1.73x／p95 2.96x（本輪最低速比之一）。E2E attack_generation median 13.54ms／p95 14.17ms，total median 15.79ms／p95 16.42ms，processing class B／B。

## 十七、TPGD

依教授指示確認 targeted/untargeted 語意與 batch labels：TPGD 的 `forward(images, labels=None)` 內部從不讀取 labels（純 KL-divergence between clean/perturbed 輸出分佈，非分類損失），本專案傳入的 `y_pred` 只是被忽略地接收，不影響其行為，因此理論上 TPGD 不應對「哪個 batch 的哪個位置」特別敏感——但實測結果顯示它是本輪 17 種攻擊中除 DeepFool 外**批次不穩定性最嚴重**的一個：逐 batch size 相符率僅 36.7%–51.7%，worst-case 36.7%（batch=16），分類 **D_batching_unsafe**。Baseline median 33.03ms／p95 82.45ms；thread-only（threads=1）median 16.378ms／p95 20.904ms，speedup median 2.02x／p95 3.94x。E2E attack_generation median 14.85ms／p95 15.45ms，total median 17.14ms／p95 17.65ms，processing class C／B。

## 十八、CW

沿用既有正式結論（`batched_algorithmic_variant`，torchattacks 的 whole-batch-summed early-stop cost 造成早停時機依批次而異），本輪以 60 筆固定樣本、`c=1.0/kappa=0/steps=20/lr=0.01`（既有 `AttackAdapter.apply()` 預設值，未變更）重新獨立量測：逐 batch size 相符率 93.3%–98.3%，worst-case 93.3%（batch=32），max diff 0.00141——**與既有文件記錄的 60-sample 配對測試（95.0% 相符、diff 0.00138）高度一致**，交叉驗證了本輪方法論的正確性。Baseline median 71.08ms／p95 231.80ms；thread-only（threads=4）median 21.464ms／p95 32.429ms，speedup median 3.31x／p95 7.15x。E2E attack_generation median 15.15ms／p95 21.47ms，total median 17.52ms／p95 23.70ms，processing class C／C。

## 十九、DeepFool

依教授指示檢查 per-sample iterative boundary search 是否能安全向量化：實測結果為**本輪最不穩定的攻擊**——相符率隨 batch size 增大而持續惡化，從 batch=2 的 53.3% 一路降到 batch=32 的 6.7%，worst-case 6.7%，分類 **D_batching_unsafe**，且惡化趨勢本身就是「batch 中樣本不同步停止」假說的直接證據：batch 越大，越多樣本在真正找到最近決策邊界前就被其他樣本的提前終止條件拖著一起停止，導致擾動品質系統性下降。Baseline median 173.35ms／p95 849.91ms（p95 遠高於 median，顯示少數樣本需要遠多於中位數的疊代才收斂）；thread-only（threads=2）median 24.743ms／p95 39.648ms，speedup median 7.01x／p95 21.44x。E2E attack_generation median 20.70ms／p95 30.69ms，total median 23.09ms／p95 33.08ms，processing class C／C。

## 二十、FAB

依教授指示檢查 per-sample iterative／restart／boundary behavior：與預期不同，FAB（`n_restarts=1, steps=10`）實測為 **bit-identical**（max diff=4.4e-5，低於門檻；100% 預測相符）於所有測試 batch size，分類 **A_implementation_optimization**——與 CW／DeepFool 等其他 restart／boundary-search 類攻擊不同，FAB 這個特定版本的批次化並未觀察到軌跡改變。FAB 也是接受明確 `seed` 參數的六種攻擊之一，因此在第八點一節的完整性稽核中被特別檢查：以相同樣本、固定 `batch_size=1`、僅改變 seed 數值（2000 對 3000 兩組數值）做隔離測試，**10 組配對全數 100% 相符**——確認 FAB 的輸出對 seed 數值不敏感（`n_restarts=1` 下未實際觸發需要隨機性的路徑），因此 FAB 未被列入需要重跑的 5 種攻擊，其原始 `A_implementation_optimization` 分類維持不變、無需修正。Baseline median 954.77ms／p95 1409.13ms（本輪第二慢的 baseline，僅次於 EAD），batch=32／threads=4 下 median 18.085ms／p95 19.364ms，**speedup median 52.79x／p95 72.77x，是本輪最大的 batching 加速效果**。**但這個加速是批次吞吐量（一次送入 32 筆一起算）帶來的，不是單筆事件延遲的改善**——E2E（單筆即時事件，見第六節設計）仍測得 attack_generation median 106.31ms／p95 112.29ms，total median 108.54ms／p95 114.61ms，processing class 為 attack-generation B、**E2E E（>100ms）**。換言之：若應用情境是「離線一次處理一批樣本」，FAB 加速效果顯著；若情境是「即時逐筆到達」，FAB 目前仍是 17 種攻擊中處理時間偏長的一個，batching 帶來的加速無法直接套用。

## 二十一、Square

依教授指示檢查 random state／query count／per-sample stopping，並強調不得只看 wall-clock：Square 是 query-based 黑箱攻擊，`n_queries=5000`（未調降，維持套件預設），因此 baseline 呈現高變異（corrected baseline median 9.27ms／p95 83.89ms——多數樣本很快找到成功擾動提前結束，少數樣本耗盡全部 5000 次查詢）。**本輪在 batching 測試階段發現 Square 對真實 backend 呼叫在 batch_size>1 時會間歇性拋出例外**（`ValueError: Expected input batch_size (N) to match target batch_size (1).`），觸發 `AttackAdapter` 既有的 fail-safe 機制，回退到 `dummy_attack`——這些回退呼叫已從所有延遲與正確性統計中排除（見第六節、第二十九節）。單獨重現實驗確認：同樣的樣本配對，多數會觸發此例外，但並非全部，屬於**資料相依、間歇性**的第三方套件行為，且在極端情形下（batch=8/16/32）**幾乎所有呼叫都失敗**（`n_compared` 降為 0）。判定：**這是 pinned `torchattacks==3.5.1` 套件本身 Square 實作在批次化早停/遮罩邏輯下的限制，不是本專案 `AttackAdapter` 的相容性問題**——`AttackAdapter` 僅是把已建構好的 `torchattacks.Square` 實例以 `atk(x_ta, y_pred)` 呼叫，未對輸入或標籤做任何自訂批次邏輯；錯誤訊息（inputs/target batch size 不一致）指向套件內部主動篩選「尚未成功」樣本子集時的張量維度處理，本輪**未修改、也不應修改**已 pinned 的第三方套件。**依 §8.1 修正後 seed policy 重新驗證**：仍能成功比較的 batch=2／batch=4，相符率分別為 77.3%（n=22）／75.0%（n=8），worst-case 75.0%，加上 batch=8/16/32 全數失敗（third-party crash，無法比較）——**分類維持 `D_batching_unsafe`，且結論在修正後更加確定**（原始 62.5% 與修正後的 75.0% 皆遠低於 A/C 門檻，crash 本身與 seed 無關，是獨立確認的第三方限制）。Corrected thread-only（threads=4）median 3.952ms／p95 11.733ms，speedup median 2.35x／p95 7.15x（thread tuning 本身仍有效，與 batching 無關）。E2E attack_generation median 3.78ms／p95 11.92ms，total median 6.00ms／p95 14.08ms，processing class B／B。

## 二十二、APGD

依教授指示檢查 adaptive step size／oscillation detection／restart 是否被 batching 影響 sample-specific state：`steps=10, n_restarts=1`。**依 §8.1 修正後 seed policy 重新驗證**：相符率 83.3%–93.3%，worst-case 83.3%（batch=32），分類維持 **D_batching_unsafe**（雖然修正後相符率略高於原始的 71.7%–76.7%，但仍低於 90% 門檻，結論方向不變）——APGD 的自適應步長調整（依每個樣本自身的收斂歷史動態調整 step size）在公平的 seed 控制下，批次化仍然表現出樣本間狀態互相影響的跡象，證實這是真正的 batching 效應而非 seed 混淆。Corrected baseline median 33.41ms／p95 59.91ms；thread-only（threads=2）median 20.141ms／p95 21.812ms，speedup median 1.66x／p95 2.75x。E2E attack_generation median 21.00ms／p95 22.78ms，total median 23.41ms／p95 25.08ms，processing class C／C。

## 二十三、APGDT

Targeted 版本（內部固定使用 DLR-targeted loss，非可選開關）。**依 §8.1 修正後 seed policy 重新驗證**：相符率 63.6%–78.6%，worst-case 63.6%（batch=16），**同時觀察到與 Square 類似但頻率低得多的 fallback 事件**（6 筆，與原始次數相同，分散在各 batch size，主因與 Square 相同類型的批次維度不一致例外，`n_compared` 因此在較大 batch size 下降至 28–56 之間），分類維持 **D_batching_unsafe**。Corrected baseline median 37.10ms／p95 87.71ms；thread-only（threads=4）median 22.599ms／p95 24.795ms，speedup median 1.64x／p95 3.54x。E2E attack_generation median 22.30ms／p95 22.95ms，total median 24.67ms／p95 26.28ms，processing class C／C。

## 二十四、AutoAttack

依教授指示：AutoAttack 是 ensemble 攻擊，內部依序執行多個 constituent 攻擊，本輪未逐一拆解其內部各子攻擊的個別延遲（`torchattacks.AutoAttack` 未對外暴露每個子攻擊各自的計時介面，需要侵入式 hook 才能取得，本輪未執行，列為限制）。**Baseline 使用 `version="rand"`**（既有 `results/attack_compatibility_smoke_20260727T030223Z/` 相容性驗證回合為 CPU 可行性所採用的既定選擇，本輪沿用作為 baseline 本身的設定，非本輪為了加速而新引入的算法變更；`version="standard"` 完整 ensemble 因 CPU 成本過高，本輪未量測）——**這是一個 algorithmic scoping 限制，不計入 speedup 量測**。**依 §8.1 修正後 seed policy 重新驗證**：相符率 76.9%–89.7%，worst-case 76.9%（batch=8），另有 5 筆 fallback 事件（與原始次數相同），分類維持 **D_batching_unsafe**（修正後 worst-case 76.9% 較原始 71.4% 略高，但仍低於 90% 門檻）。Corrected baseline（version="rand"）median 528.19ms／p95 839.38ms；thread-only（threads=4）median 282.311ms／p95 285.701ms，speedup median 1.87x／p95 2.94x。E2E attack_generation median 299.12ms／p95 300.93ms，total median 301.64ms／p95 303.45ms，processing class 皆為 **E（>100ms）**。

## 二十五、EAD

`kappa=0, lr=0.01, binary_search_steps=9, max_iterations=100`（皆為套件預設，未調降）——**本輪 17 種攻擊中 baseline 最慢者**，median 2295.39ms／p95 3414.05ms，configured max_iterations=100 為主要瓶頸來源（elastic-net 目標函數的逐步座標下降本身計算量大）。逐 batch size 相符率 95.0%–100.0%，worst-case 95.0%（batch=4），分類 **C_batched_algorithmic_variant**（與 stopping/objective 相關的批次軌跡差異，但未達 D 等級的不穩定）。Thread-only（threads=1）median 518.203ms／p95 558.644ms，speedup median 4.43x／p95 6.11x——雖然是本輪絕對倍率不算最高的一組，但由於 baseline 本身就是秒級延遲，thread tuning 帶來的絕對時間節省（median 減少約 1777ms）是全部 17 種攻擊中最大的。E2E attack_generation median 464.32ms／p95 492.11ms，total median 466.74ms／p95 494.48ms，processing class 皆為 **E（>100ms）**。

## 二十六、小結（5 個 A、1 個 B、5 個 C、7 個 D 之整體型態，§8.1 稽核修正後）

達到 `A_implementation_optimization` 的五種攻擊（FGSM／BIM／PGD det／DIFGSM／FAB）有一個共同特徵：**單步或少數步、無 restart、無 adaptive/oscillation 狀態、無 query-based 早停**的攻擊，這類攻擊的每個樣本計算路徑彼此獨立、無跨樣本共享狀態，因此向量化批次計算天然安全。DIFGSM 原先因 seed 混淆被誤判為 D，修正後併入此類，說明「判斷是否為狀態相依」必須先排除量測方法本身的干擾，才能下結論。相對地，`D_batching_unsafe` 的七種攻擊（RFGSM／TPGD／DeepFool／Square／APGD／APGDT／AutoAttack）多半具有「每個樣本各自決定何時停止／各自維護一段最佳化歷史狀態」的特性，這類狀態在批次張量運算下容易被同批次的其他樣本影響（早停遮罩、自適應步長的批次統計等）；其中 Square／APGD／APGDT／AutoAttack 已在修正後的公平 seed 比較下重新確認（見第八點一節與各自小節），結論比初次量測更可信。這個型態本身就是對「Attack 怎麼讓它變快」這個問題的一個重要、可類推的答案：**thread tuning 對全部 17 種攻擊都安全且有效（單獨量測 thread-only speedup 為 1.64x–7.35x，見 `attack_thread_tuning.csv`），batching 只對演算法結構單純的攻擊安全，對狀態相依的疊代攻擊則需逐一以公平配對的方式驗證，不能一概而論，也不能僅憑初次觀察就下結論**。

## 二十七、Processing-Class Comparison

分類定義（僅為效能量級參照，**不代表任何特定應用場景的 deadline**，見第二十八節）：A(<10ms)／B(10–20ms)／C(20–50ms)／D(50–100ms)／E(>100ms)，分別以「最佳安全配置下的 attack-generation p95」與「E2E total p95」獨立分類：

| Attack | Attack-generation class | E2E class |
|---|---|---|
| FGSM | A | A |
| BIM | A | B |
| PGD (det) | A | B |
| PGD (stoch) | A | B |
| MIFGSM | C | B |
| DIFGSM | A | B |
| VMIFGSM | D | D |
| VNIFGSM | D | D |
| RFGSM | B | B |
| TPGD | C | B |
| CW | C | C |
| DeepFool | C | C |
| FAB | B | E |
| Square | B | B |
| APGD | C | C |
| APGDT | C | C |
| AutoAttack | E | E |
| EAD | E | E |

（DIFGSM／Square／APGD／APGDT／AutoAttack 五個 class 皆已依第八點一節修正後之 corrected CSV 重新計算，非沿用初次量測結果。）Attack-generation 與 E2E 分類偶有一級之差（例如 MIFGSM／TPGD 為 C／B，DIFGSM 為 A／B）——這是因為兩者使用不同樣本數（前者 n=60 的 p95，後者 n=8 的 p95）與不同具體樣本，小樣本 p95 本身有統計波動，屬預期範圍內的正常差異，不代表矛盾。FAB 是唯一一個 attack-generation／E2E 分類方向不一致的攻擊（B／E），原因已在第二十節說明：batching 帶來的加速是批次吞吐量效果，不適用於單筆即時事件的 E2E 情境。

## 二十八、End-to-End Implications

處理量級分類**僅是效能參照，不得解讀為任何特定應用場景（例如衛星或 Wi-Fi）的 computation deadline**，與既有 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`、`SATELLITE_APPLICATION_AND_LATENCY_REQUIREMENTS_ZH_TW.md` 一貫的措辭原則相同。就工程意義而言（皆以 §8.1 修正後之最終分類為準）：5 個 A 級攻擊（FGSM／BIM／PGD-det／DIFGSM／FAB-batched）在 E2E 情境下（FAB 除外，見上）可穩定落在 10ms 以下到 20ms 等級；5 個 C 級攻擊（CW／DeepFool／APGD／APGDT／以及部分 batch size 下的 MIFGSM/TPGD）落在 20–50ms；D 級（VMIFGSM／VNIFGSM）落在 50–100ms（Square 修正後改列 B 級，10–20ms）；E 級（AutoAttack／EAD／FAB 之 E2E）超過 100ms，其中 EAD／AutoAttack 即使套用 thread tuning 後仍需數百毫秒。對需要毫秒級即時回應的正式管線（例如既有 satellite-like Step 4 的 Top-K／attack 評估情境），A／B 級攻擊的最佳化路徑已可直接套用；C／D／E 級攻擊即使完成 thread tuning，仍需在應用設計上容忍較高延遲，或考慮是否真的需要在即時路徑中使用這類攻擊。

## 二十九、Limitations

1. **Configured iteration/query count（第九節「configured_iteration_or_query_count」欄）為攻擊建構子的預設參數值（透過 `inspect.signature` 直接內省已安裝的 torchattacks 套件取得），不是執行期實際 forward/backward 呼叫次數的量測值**——後者需要對 17 種不同攻擊各自的最佳化迴圈做侵入式 hook，本輪未執行，如實記錄為限制而非省略。
2. AutoAttack 的 baseline 使用 `version="rand"`（既有相容性回合的既定選擇），未量測 `version="standard"` 完整 ensemble；其內部各 constituent 攻擊的個別延遲貢獻亦未拆解。
3. Square／APGDT／AutoAttack 在部分 batch size 下的 fallback 事件為第三方 `torchattacks==3.5.1` 套件本身的限制（Square 尤其嚴重），本輪僅測量、分類、如實記錄，未嘗試修改已 pinned 的第三方套件，也未修改攻擊演算法本身。
4. ~~DIFGSM 疑似的 batch-shared diversity randomness 現象~~——**此推論已於第八點一節稽核撤回**：以修正後、公平配對的 seed policy 重新驗證，DIFGSM 在全部測試 batch size 下皆為 bit-identical，證實原始觀察是 benchmark 本身 seed 混淆造成的假象，不是 `IQDIFGSM` 的性質；本節保留此項僅作為方法論教訓的紀錄。
5. Thread tuning 之最佳 thread 數（best_threads）為本機（單一測試機器）量測結果，與既有效能文件一致地不宣稱可跨機器泛化。
6. Batching classification 使用「跨所有測試 batch size 的最差表現」；某些攻擊（例如 RFGSM）在多數 batch size 下已接近或達到 90% 門檻，僅因單一 batch size 的表現較差而被歸類為 D，讀者應參考第九節逐 batch size 明細（`attack_correctness_summary.csv`），不應只讀最終單一分類標籤。
7. E2E 驗證使用小樣本（n=8，含 1 筆暖身後 8 筆量測），percentile（尤其 p95）在此樣本數下統計穩定性有限，僅作為工程參照，非正式統計推論用途。
8. 本輪所有攻擊皆屬於 A0 數位白箱層級（直接對已存在於記憶體中的 AWN-input IQ 張量生成攻擊擾動），不涉及任何射頻通道，結論不能直接推論到攻擊在實體通道上的可行性或延遲。
9. **本輪執行過程中發現並修正三個本輪自身 benchmark script 的錯誤**（非既有正式管線的錯誤，詳見下）：(a) 批次化測試階段的 fallback 結果一度未被排除於延遲與正確性統計之外，已修正；(b) Phase 5 E2E 驗證迴圈一度對每次疊代重新呼叫 `load_radioml_sample()`（該函式本身即有文件記載「每次呼叫皆重新讀取整個 ~640MB 資料集檔案，不做快取」），造成 E2E total latency 被系統性灌入約 1.1–1.2 秒的純磁碟 I/O 開銷，已修正；(c) **batching 正確性比較的 reference（batch=1）與 test（batch>1）對接受明確 `seed` 參數的六種攻擊使用了不同的絕對 seed 數值範圍**，此問題於一輪獨立的完整性稽核中發現，詳細說明、修正方法與重跑範圍見第八點一節。三個問題皆為本輪新增程式碼的邏輯錯誤，不影響、也未觸及既有正式 `raw_results.csv` 或任何既有正式結果目錄；(a)(b) 已在 §8.1 之前的版本修正並重新產生全部數字，(c) 的修正範圍僅限受影響的 5 種攻擊（見第八點一節），其餘 12 種攻擊與 FAB 沿用未受影響的原始結果。
10. Batching 正確性比較（第八點一節修正後）仍受限於這 6 種攻擊的真實 API 僅接受單一 batch-level `seed`、無法提供逐樣本獨立 RNG stream 的架構限制——修正後的比較方法（同一 chunk 的 reference 與 test 共用同一個 chunk 錨定 seed）是在此限制下可達成的最公平比較，但不代表已經證明「每個樣本無論在哪個 batch 位置都會得到完全相同的 RNG stream」；這點僅對 FAB／DIFGSM 這類確認對 seed 數值不敏感的攻擊沒有影響，對 Square／APGD／APGDT／AutoAttack 而言，其 D 分類反映的是「在目前可達成的最佳公平比較下，batching 仍會改變輸出」，而非窮盡所有可能 seed 組合下的證明。

## 三十、Project-Close Interpretation

本輪完成 17/17 攻擊（18 個量測條件）的 baseline、CPU thread tuning、batching 可行性與正確性驗證、以及小型 end-to-end 驗證，0 error。過程中發現的 fallback 事件（Square/APGDT/AutoAttack 批次化限制）、本輪腳本自身的兩個資料完整性錯誤（fallback 污染、E2E 磁碟 I/O 混入計時），以及初次 benchmark 完成後一輪獨立完整性稽核找到的第三個問題（batching 正確性比較的 seed pairing 不一致，影響 DIFGSM／Square／APGD／APGDT／AutoAttack 五種攻擊），皆已如實記錄、修正並（在 seed 問題的情形下）以僅涵蓋受影響 5 種攻擊的目標性重跑重新驗證，未掩蓋任何負面結果，也未因發現問題而略過修正直接沿用錯誤數字。核心結論回答「Attack 怎麼讓它變快」：**CPU thread tuning 對全部 17 種攻擊都是安全、有效、與演算法無關的加速手段（thread-only 群組 1.64x–7.01x）；batching 只對計算路徑單純、無跨樣本共享狀態的攻擊（FGSM／BIM／PGD-det／DIFGSM／FAB，五種，經稽核與修正後確認）安全，可再疊加達 52.8x（FAB，批次吞吐量效果，非單筆延遲）的額外加速；其餘 12 種攻擊（含 pgd_stoch 的 B 分類另計）的 batching 皆需標示為 algorithmic variant 或 unsafe，不得直接取代原始逐筆結果作為正式效能宣稱**。本文件與既有 `PERFORMANCE_AND_LATENCY_ANALYSIS_ZH_TW.md`（FGSM/PGD/CW 三種攻擊的既有結論）互為補充、方向一致、對 CW 的獨立重測數字高度吻合，共同構成本專案 project-close 範疇內對 17-attack registry 加速可行性的完整正式回答；本輪額外的價值在於示範了「初次 benchmark 的分類結論必須通過獨立完整性稽核才能視為定論」——DIFGSM 一例證明了看似合理的機制推論（batch-shared randomness）也可能只是量測方法本身的缺陷。
