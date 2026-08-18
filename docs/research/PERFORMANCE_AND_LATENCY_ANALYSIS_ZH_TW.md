# 能量觸發 AMC 系統之 CPU 效能與延遲分析

## 一、研究背景

系統目前的正式流程（偵測、分段、AWN 分類、攻擊、Top-K 防禦）皆已在 CPU 上以真實後端驗證過功能正確性，但先前的研究文件並未系統性回答各階段實際花費多少時間、哪一個環節是延遲的主要來源，也未回答對抗攻擊的計算成本是否可以在不改變攻擊語意的前提下降低。這些問題在評估系統是否能支援更大規模的正式實驗矩陣、或未來是否需要考慮即時處理時，是必須先有實測數據才能回答的問題，而不是憑印象判斷。

## 二、研究問題

**RQ1**：在目前的 CPU 環境下，spectrum sensing、分段、AWN 前處理與推論、攻擊生成、Top-K 防禦各自花費多少時間？哪一個階段是端到端延遲的主要來源？

**RQ2**：FGSM、PGD、CW 三種攻擊的計算瓶頸實際落在哪裡——是 PyTorch 本身的張量運算，還是 Python 層的迴圈與物件建構開銷？

**RQ3**：在不改變攻擊參數與結果語意的前提下，能否透過批次化、執行緒設定等純實作層級的調整，降低攻擊生成延遲？效果有多大？

**RQ4**：目前的 spectrum sensing 是否必須對每一段新進資料重新從頭計算，還是可以用分塊、滾動緩衝的方式處理而不改變偵測結果？

**RQ5**：現有的效能數據是否足以判斷是否需要將任何部分改寫為 C++／LibTorch？

## 三、方法

所有量測皆使用正式流程共用的真實元件：真實 AWN checkpoint（`external/adversarial-rf/2016.10a_AWN.pkl`）、真實的 `energy_detect`／`select_aligned_segments`／`apply_awn_preprocess`／`to_awn_input`（`src/sensing/*.py`）、真實的 `AttackAdapter`（`src/adapters/attack_adapter.py`，串接 torchattacks）與 `TopKAdapter`（`src/adapters/topk_adapter.py`）。沒有任何階段使用替代數值或 placeholder 推論。計時使用 `time.perf_counter_ns()`（Phase A／B 逐階段計時）或 `time.perf_counter()`（Phase D／E 整體耗時），皆為單調時鐘，不受系統時間調整影響。所有正式量測前皆執行暖身樣本（Phase A：50 筆；Phase B：每個攻擊 20 筆；Phase C：每個攻擊 10 次呼叫；Phase E：每個攻擊 20 筆），暖身樣本本身不計入統計。資料集初始化、checkpoint 載入、模組匯入、CSV 寫入與繪圖時間皆不計入任何階段的計時區間。

某些函式內部的子步驟耦合在同一次呼叫中，本輪未修改對應原始程式以硬性拆分：`energy_detect()`（`src/sensing/energy_detection.py`）內部的噪聲底估計與門檻比較是同一次呼叫中的向量化運算，沒有可獨立量測的呼叫邊界；`select_aligned_segments()`（`src/sensing/segmentation.py`）在 `max-energy` 策略下，視窗搜尋本身就是分段函式的內部迴圈。這兩項在逐筆結果中記錄為缺值，並附上原因說明，不做估計。

## 四、Benchmark Protocol

**Phase A（clean／sensing 基準）**：11 種調變 × 20 個訊噪比（-20 至 18 dB，間隔 2 dB）× 每格 10 筆，合計 2200 個基礎樣本，每筆記錄 `embedding_ms`、`energy_detection_ms`、`region_postprocess_ms`、`segmentation_ms`、`awn_preprocess_ms`、`awn_clean_inference_ms`、`clean_total_ms`，並在同一批樣本上額外測量 K=20 的 Top-K 防禦與受防禦推論（`topk_ms`、`defended_inference_ms`、`topk_total_ms`）。

**Phase B（攻擊延遲基準）**：FGSM、PGD、CW 三種攻擊各自使用 11 種調變 × 3 個訊噪比（-10、0、18 dB）× 每格 10 筆共 330 個基礎樣本，攻擊參數為目前正式流程的預設值（FGSM／PGD：eps=0.05；CW：既有驗證過的 c=1.0、kappa=0、steps=20、lr=0.01），逐筆記錄 `attack_generation_ms`、`awn_attacked_inference_ms`、`attack_total_ms`。

**Phase C（Profiler）**：對每種攻擊各執行 30 次 `AttackAdapter.apply()` 呼叫（10 次暖身另計），分別以 Python `cProfile`（依 self time／cumulative time 排序）與 `torch.profiler`（CPU activity）記錄。

**Phase D（加速候選試驗）**：固定 60 筆樣本的試驗集，測試批次大小（1／4／8／16／32）、CPU 執行緒數（1／2／4／8／16）、PGD／CW 迭代步數縮減，每個候選方案皆記錄延遲、conditional ASR、Linf／L2、決定性可重現性（同一輸入重複呼叫兩次比對輸出）與模型模式還原狀態。

**Phase E（正式 Before／After）**：從 Phase D 選出的候選方案，在與 Phase B 完全相同的 330 個基礎樣本（相同調變、訊噪比、樣本索引、種子、攻擊參數、AWN checkpoint、CPU 環境）上，各自重新執行一次未優化（batch_size=1，預設 16 執行緒）與優化後（batch_size=16，1 個執行緒）的完整流程。

**Phase G（Streaming Sensing 原型）**：對同一條含三個真實 RadioML burst 的合成長串流，比較一次性離線 sensing 與切分為固定大小 chunk（256／512／1024／2048 samples）搭配滾動緩衝（carry buffer=128 samples）的逐塊 sensing，在 ±4 samples 容許誤差內比對事件起訖位置。

## 五、Clean Pipeline Latency（Phase A，n=2200）

| stage | mean (ms) | median (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | % of clean_total |
|---|---|---|---|---|---|---|---|
| embedding_ms | 0.458 | 0.441 | 0.539 | 0.581 | 0.707 | 3.403 | 4.0% |
| energy_detection_ms | 0.315 | 0.308 | 0.359 | 0.380 | 0.504 | 1.784 | 2.7% |
| region_postprocess_ms | 0.046 | 0.044 | 0.055 | 0.061 | 0.093 | 0.358 | 0.4% |
| segmentation_ms | 0.681 | 0.690 | 0.805 | 0.848 | 1.057 | 4.128 | 5.9% |
| awn_preprocess_ms | 0.036 | 0.033 | 0.045 | 0.051 | 0.084 | 0.147 | 0.3% |
| awn_clean_inference_ms | 9.925 | 1.285 | 15.004 | 53.668 | 197.762 | 258.892 | 86.6% |
| **clean_total_ms** | **11.460** | **2.819** | **16.566** | **55.094** | **199.322** | **260.735** | **100%** |
| topk_ms | 0.354 | 0.303 | 0.419 | 0.480 | 0.913 | 11.045 | — |
| defended_inference_ms | 12.316 | 2.382 | 26.988 | 65.685 | 207.510 | 273.164 | — |
| **topk_total_ms**（= clean_total + topk_ms + defended_inference_ms） | **24.131** | **5.975** | **54.563** | **101.410** | **345.269** | **534.620** | — |

前四個純感測階段（embedding、energy detection、region postprocess、segmentation）與 awn preprocess 合計僅占 clean_total 平均值的 13.4%，數值本身也很小（中位數合計約 1.5 毫秒）。**以平均值衡量，AWN 乾淨推論是 clean-path 中最大的單一延遲來源**，佔平均延遲的 86.6%。

**這個結論必須依統計口徑分開陳述，不得只寫「AWN 占 clean total 86.6%，因此是 clean pipeline 唯一瓶頸」**——第十五節第五小節以 mean／median／p95 三種統計量分別重新計算各階段占比，三種口徑的結論並不相同：

- **average case（平均情況）**：mean 顯著受到 AWN 尾端延遲離群值（tail spike）拉高影響——AWN 推論平均值（9.925 ms）遠高於中位數（1.285 ms），占 clean_total 平均值的 86.6%。
- **typical case（典型情況）**：以中位數衡量，AWN 推論僅占 45.6%，與其餘感測／前處理階段合計的 53.8% 相當接近，**不能說 AWN 推論在典型情況下單方面主導**。
- **tail case（尾端情況）**：以 p95 衡量，AWN 推論占比進一步升高到 97.35%，此時 AWN 推論的尾端延遲離群值是主導因素。

三種口徑的結論確實不同，詳細數據見第十五節第五小節、`bottleneck_by_percentile.csv`。

這裡有一個必須明確指出的現象：`awn_clean_inference_ms` 的平均值（9.925 ms）遠高於中位數（1.285 ms），p99 與 max 更達到 197.8 ms 與 258.9 ms，即使在暖身之後仍然如此。這代表延遲分布有明顯的長尾，多數樣本推論很快，但少數樣本異常慢。這個現象在 Phase B 的攻擊延遲數據中同樣可見（見第六節）。**本輪驗證（第十五節第四小節）已透過受控的執行緒數微基準測試證實：本機測試環境下，CPU 執行緒數設定是這個延遲尾端（tail latency）的重要來源，且與樣本內容（調變類型、訊噪比）無關**：執行緒數為 1／2／4／8 時，500 次重複呼叫中沒有任何一次落入「超過中位數 5 倍」的離群範圍；執行緒數為預設的 16 時，離群比例高達 12.2–14.4%，且在所有測試過的樣本群組（原本被誤認為「AM-SSB 較慢」的群組、其他調變、AM-SSB 低訊噪比群組）中比例相近，而非集中在特定調變類型。本次測試中，執行緒數 2 在 mean／median／p95／p99／max／CV 每一項指標上皆為最佳，執行緒數 1 非常接近；**這是本機這台機器上的測試結果，不得推論所有 CPU／硬體都應固定使用 1 個執行緒**，其他環境的最佳執行緒數需要各自重新量測。

## 六、攻擊延遲基準（Phase B，n=330／攻擊）

| attack | mean (ms) | median (ms) | p90 (ms) | p95 (ms) | p99 (ms) | max (ms) | samples/sec |
|---|---|---|---|---|---|---|---|
| FGSM | 31.26 | 6.59 | 78.88 | 129.71 | 433.16 | 704.05 | 32.0 |
| PGD | 133.14 | 68.76 | 311.57 | 433.80 | 1026.92 | 1988.60 | 7.5 |
| CW | 121.19 | 85.85 | 245.17 | 327.45 | 695.22 | 959.75 | 8.3 |

（以上為 `attack_generation_ms`，即攻擊本身的計算時間，不含 AWN 推論。）三種攻擊的乾淨準確率皆為 64.2%（330 筆基礎樣本），FGSM 攻擊成功率 85.2%，PGD 97.3%，CW 91.8%，皆與既有低擾動攻擊實驗（`docs/research/DIGITAL_LOW_PERTURBATION_ATTACK_EXPERIMENT_ZH_TW.md`）的量級一致。與 clean-path 一樣，三種攻擊的平均延遲皆遠高於中位數，同樣的長尾現象。

## 七、Profiler 結果（Phase C）

| attack | cProfile 總耗時（30 次呼叫） | backward pass self-time | backward 占總時間比例 |
|---|---|---|---|
| FGSM | 0.796 s | 0.143 s | 18.0% |
| PGD | 2.878 s | 1.485 s | 51.6% |
| CW | 5.821 s | 3.912 s | 67.2% |

`torch.profiler` 的 CPU 時間分解（見 `{attack}_torch_profiler_top30.txt`）顯示，三種攻擊排名最前面的項目一致是 `aten::_slow_conv2d_forward`、`aten::_slow_conv2d_backward`、`aten::native_batch_norm`——也就是 AWN 模型本身的卷積層與批次正規化層的前向與反向計算，而不是 Python 迴圈、物件建構或資料轉換。以 FGSM 為例，`cProfile` 依 self time 排序的前五名（`torch.conv1d`、`run_backward`、`torch.conv2d`、`torch.batch_norm`、`torch._C._nn.pad`）合計占總時間的 81.2%，其餘包含 Python 層的模組呼叫、屬性檢查、正規化與反正規化轉換等，合計不到 19%。

回答 Phase C 的九個問題：**(1)** model forward（卷積＋批次正規化＋其他層）：FGSM 約占 60%（forward+backward 合計扣除 backward 後估算），隨迭代次數增加而在總時間中的絕對占比下降但絕對耗時上升；**(2)** backward：FGSM 18.0%，PGD 51.6%，CW 67.2%，隨攻擊迭代次數增加而提高；**(3)** Python 迴圈本身（`torch/nn/modules/module.py` 的 `_call_impl`／`__setattr__`／`named_modules` 等）：合計約數個百分點，不是主要瓶頸；**(4)** tensor copy／clone（`aten::copy_`）：torch.profiler 顯示自身時間占比約 1%，不是主要瓶頸；**(5)** preprocessing（正規化與 `_iq_to_ta_input_minmax`／`_ta_output_to_iq_minmax` 反轉換）：未出現在任一攻擊的 top-40 self-time 清單中，耗時可忽略；**(6)** attack object construction（`_build_torchattacks`）：同樣未出現在 top-40 self-time 清單中，耗時可忽略；**(7)** normalization／inverse transform：與 (5) 相同，可忽略；**(8)** 是否有重複 work：FGSM 30 次呼叫觀察到 `named_modules`／`__setattr__` 被呼叫數千次（4170～26550 次），這來自 `AttackAdapter.apply()` 的 `finally` 區塊每次呼叫都要走訪並還原模型每個子模組的 `training`／`requires_grad` 狀態，屬於真實存在但量測上非主要瓶頸的重複開銷，本輪未修改 `src/adapters/attack_adapter.py` 嘗試消除；**(9)** CPU 執行緒是否有效利用：**沒有**——見第八節，這是本輪最主要的發現之一。**不得假設 Python 是主要瓶頸的前提已被推翻**：三種攻擊超過八成的計算時間確實花在 PyTorch 自身的張量運算（含其 C++/ATen 後端），而不是 Python 解釋器本身的迴圈開銷。

## 八、加速候選試驗（Phase D，pilot n=60）

**批次化**：`AttackAdapter.apply()` 本身即已支援輸入形狀 `[N,2,128]` 中 N>1 的批次呼叫，不需要修改任何正式程式（`AttackAdapter.apply()` 的文件字串本輪已加入逐攻擊批次化安全性分類，見第二節第 2 部分的稽核結論）。本輪先以 4 筆樣本驗證批次呼叫（N=4）與逐筆迴圈呼叫（4 次 N=1）在 FGSM 下的輸出完全一致（最大差異為 0.0），確認**FGSM** 批次化屬於 `implementation_optimization`。批次大小從 1 提升到最佳值時，FGSM 每筆延遲從 21.5 ms 降到 2.6 ms，PGD 從 56.1 ms 降到 7.8 ms，CW 從 158.2 ms 降到 17.1 ms（CW 在批次 32 時反而回升到 77.6 ms，顯示批次大小並非單調改善，16 是本輪試驗範圍內的較佳值）；三種攻擊的 conditional ASR 在不同批次大小下維持穩定（FGSM 恆為 0.857，PGD 在 0.980–1.000 之間，CW 在 0.898–0.959 之間）。**PGD 與 CW 批次化的正式分類已在第十五節第一、二小節修正**：PGD 批次化在 `random_start=False` 下確認為 `implementation_optimization`，本節測得的 ASR 波動來自 PGD 套件預設 `random_start=True` 的隨機性，屬於 `stochastic_comparison`；CW 批次化確認為 `batched_algorithmic_variant`，不是純粹的實作加速。`acceleration_pilot_results.csv` 的 `optimization_type` 欄位在本輪驗證前產生，pgd／cw 兩者的 batching candidate 列仍記為 `implementation_optimization`，屬已知過時值，正確分類以本段文字與第十五節為準；`experiments/acceleration_pilot.py` 本輪已修正該欄位邏輯，下次重跑會自動產生正確標記。

**CPU 執行緒數**：以 PGD、batch_size=1 為代表測試，結果出乎預期——**執行緒數從預設的 16 降到 1，每筆延遲從 43.6 ms 降到 16.7 ms**（1 執行緒最快，16 執行緒最慢，2／4／8 執行緒介於中間但同樣快於 16）。這代表目前的預設執行緒設定**沒有被有效利用**，對單筆（或小批次）張量運算而言，執行緒管理與跨執行緒同步的開銷超過了平行運算帶來的效益，這在本機回報的「4 sockets × 4 cores × 1 thread」拓撲下尤其明顯（見 manifest 中的 `known_environment_characteristic` 說明）。

**PGD／CW 迭代步數縮減**：兩者皆明確標記為 `algorithmic_tradeoff`，不視為單純的實作加速。PGD steps 從 10 降到 5，60 筆總耗時從 2664 ms 降到 1058 ms，但 conditional ASR 從 1.000 降到 0.918；降到 3 步時耗時 964 ms，ASR 降到 0.837。CW steps 從 20 降到 5，60 筆總耗時從 5096 ms 降到 3836 ms，但 ASR 從 0.898 大幅降到 0.510。這組數據顯示步數縮減確實能降低延遲，但對 CW 的攻擊效果影響非常大，不建議作為預設加速手段。

**Attack object 重用與 torch.compile**：受限於本輪時間，未執行獨立試驗；Phase C 的 profiler 證據顯示物件建構本身耗時可忽略（未出現在 top-40 self-time），因此預期即使實作物件重用，效益也有限，這點在報告中明確列為未執行項目，而非以推測結果替代實測。

完整逐筆結果見 `acceleration_pilot_results.csv`。

## 九、正式 Before／After 比較（Phase E，n=330／攻擊）

依 Phase D 證據，選定的優化方案為**批次大小 16 搭配 `torch.set_num_threads(1)`**。**執行緒數設定本身**（單獨來看，不含批次化）純屬執行環境設定，不改變任何攻擊的演算法語意，是 `implementation_optimization`；但**批次化**對三種攻擊的語意影響並不相同，**不得將 FGSM／PGD／CW 三者的加速倍數並列陳述為同一種「純加速」**，必須依下方分類表分開解讀：

| Attack | Baseline (mean ms) | Optimized (mean ms) | Speedup (mean) | ASR before | ASR after | Linf before | Linf after | Prediction match | **正式分類** |
|---|---|---|---|---|---|---|---|---|---|
| FGSM | 13.83 | 0.86 | 16.10x | 0.783 | 0.783 | 0.001532 | 0.001532 | 100.0% | **A．confirmed implementation optimization** |
| PGD | 78.06 | 5.34 | 14.61x | 0.981 | 0.967 | 0.001532 | 0.001532 | 78.5% | **B．stochastic comparison**（批次化本身另以 random_start=False 證實為 A，見下） |
| CW | 99.99 | 9.32 | 10.73x | 0.887 | 0.939 | 0.000583 | 0.000620 | 95.8% | **C．batched algorithmic variant** |

**本表所在的 `attack_acceleration_comparison.csv`（Phase E，330 筆規模）之 `optimization_type` 欄位為本輪驗證前產生，三列皆記為 `implementation_optimization`，其中 pgd／cw 兩列已知過時，正確分類以上表與第十五節為準；`experiments/acceleration_before_after.py` 本輪已修正為輸出正確的 A／B／C 分類與 `classification_note` 欄位，下次重跑會自動產生正確標記，但既有的 `results/performance_latency_20260818T010552Z/` 目錄本身不予回溯修改。** 三者的 ASR／Linf 變化需要分別解讀，**其中 PGD 與 CW 的解讀在本輪驗證（第十五節第三、四小節）後已從推測修正為證據支持的結論**：

**FGSM**：攻擊後預測與擾動量完全相同（prediction match 100%），確認批次化對 FGSM 是純粹的 `implementation_optimization`，沒有任何語意變化。

**PGD**：這裡的 78.5% prediction match（在本節 330 筆規模下量測）本輪已用 60 筆樣本的**決定性等價測試**正式拆解成因（完整方法與數據見第十五節第一小節）：將 torchattacks 的 `random_start` 顯式設為 `False` 後，batch_size=1 與 batch_size=16 的輸出**逐筆完全一致**（60 筆樣本的 attacked tensor 最大差異、attacked logits 最大差異皆為 0.0）。這證實**批次化本身對 PGD 的計算沒有任何影響**，屬於 `implementation_optimization`。本節表格中觀察到的 prediction match 差異，來自目前正式流程呼叫 PGD 時沒有顯式關閉 torchattacks 套件預設開啟的 `random_start`，使每次呼叫都從獨立的隨機起點開始優化；由於 batch_size=1 與 batch_size=16 呼叫 PGD 的次數不同（60 筆樣本下分別呼叫 60 次與 4 次），兩者消耗隨機數的方式不同，導致起點不同、結果不同，這是**攻擊本身既有的隨機性表現，不是批次化造成的語意改變**，也不是實作錯誤。

**CW**：95.8% 的 prediction match（本節 330 筆量測）與 Linf 由 0.000583 上升到 0.000620，本輪已透過**逐行閱讀 `torchattacks.CW.forward()` 原始碼**（該函式屬於已釘選版本的第三方套件，未修改）與**60 筆樣本的決定性配對測試**（CW 無 `random_start`，天生決定性，見第十五節第二小節）確認成因：CW 內建的提前停止判斷式（`if step % max(self.steps // 10, 1) == 0: if cost.item() > prev_cost: return best_adv_images`）使用的是**整批加總後的純量 cost**，而非逐筆的個別 loss，代表提前停止是「整批一起停」而不是「每筆各自停」。本輪以 runtime monkeypatch（不修改任何檔案）直接觀測到：batch_size=1 時，60 筆樣本各自在第 2 到 16 步之間的不同步數觸發提前停止；batch_size=16 時，4 個批次分別在第 10、16、12、14 步「整批」觸發停止——代表某些原本應該提早停止的樣本被迫繼續疊代，另一些樣本可能被迫提早結束。60 筆決定性配對測試顯示 pred_match=95.0%（3/60 不一致），attacked tensor 最大差異達 0.00137，遠超浮點雜訊量級。**這是批次化改變了 CW 最佳化軌跡本身的真實證據，不是實作雜訊，因此 CW 批次化不得標記為 `implementation_optimization`，本文件正式改標為 `batched_algorithmic_variant`**（定義：批次大小改變了演算法的實際行為，但改變的幅度與方向因樣本而異，不是單純的參數調整如 steps 縮減）。本輪未發現可在本 repo 可編輯範圍內修正此問題的方法——早停邏輯是 torchattacks 套件內部硬編碼的行為，此套件為已釘選版本、不可修改，唯一能維持與逐筆執行完全等價的方式是不批次化（batch_size=1），這會放棄批次化帶來的加速。是否接受 CW 批次化的這個 tradeoff，需由後續使用者依實際場景（是否需要語意完全等價於逐筆執行）決定。完整比較欄位見 `attack_acceleration_comparison.csv`，60 筆等價測試原始資料見 `pgd_batch_equivalence_deterministic.csv`、`pgd_batch_stochastic_comparison.csv`、`cw_batch_equivalence.csv`。

## 十、是否建議 C/C++ 或 LibTorch 改寫

依第七節 Profiler 證據：三種攻擊超過八成的計算時間已經花在 PyTorch 自身的 C++/ATen 後端運算（卷積、批次正規化、反向傳播），Python 層開銷占比很小；且第九節顯示，單純透過批次化與執行緒設定調整（皆未離開現有 Python/PyTorch 架構），FGSM／PGD／CW 已分別取得 16.1／14.6／10.7 倍的加速。這符合本文件事先定義的判斷準則中的情境 **A**（主要時間在 PyTorch forward/backward）與 **C**（batch/vectorization 已能取得明顯 speedup）：**本輪不建議將攻擊生成或 AWN 推論整體改寫為 C++／LibTorch**。改寫成本高，而目前證據顯示同等或更大的效益已可透過調整既有 PyTorch 呼叫方式取得。

需要說明的例外情況：即使經過本輪優化，PGD／CW 優化後的平均延遲（5.34 ms／9.32 ms）仍略高於最嚴格的 5 毫秒處理預算（見第十一節），若未來出現這類極嚴格的延遲需求，可再評估是否值得投入 C++／LibTorch 改寫成本，但本輪僅在此提出這個可能性，不提出正式 migration plan，也不實作任何 C++ 程式碼。

## 十一、Streaming Sensing 原型（Phase G）

**正式狀態聲明（本輪驗證修正用詞）：chunk-based Spectrum Sensing prototype 已驗證可行，但目前仍缺完整 cross-chunk detector state。**

使用同一條含三個真實 RadioML burst（BPSK、QPSK、QAM16，訊噪比皆為 0 dB）的 8192-sample 合成串流，離線一次性 sensing 正確偵測到 3 個區域。切分為固定大小 chunk 並以 128-sample 的滾動緩衝（carry buffer）逐塊處理後：

| chunk size | 偵測到的區域數 | 與離線結果相符（±4 samples 內） |
|---|---|---|
| 256 | 0 | 0/3（完全未偵測到任何區域） |
| 512 | 3 | 1/3 精確相符，另外 2 個區域起訖誤差為 5–10 samples，超出容許範圍但相距不遠 |
| 1024 | 3 | 3/3，最大誤差 4 samples |
| 2048 | 3 | 3/3，最大誤差 2 samples |

這組結果回答 RQ4：**Spectrum sensing 不需要對每一段新進資料都從完整串流重新計算，chunk_size 夠大時（本例中達到視窗長度的 8 倍以上）偵測結果與離線一次性處理幾乎完全一致；但 chunk_size 較小時會系統性失敗，本輪已找出並以量化證據確認具體機制**（完整方法與逐 chunk 數據見 `streaming_failure_diagnosis.csv`，由 `experiments/diagnose_streaming_failure.py` 產生）。

失敗機制並非「事件被 chunk 邊界切開」本身，而是：`StreamingDetector._detect_mask()` 對每個 chunk 獨立以中位數（median）估計背景噪聲底，但用來平滑功率的移動平均視窗（window=128）會把每個 burst 樣本的影響向左右各擴散約半個視窗寬度，使得「有效受影響樣本數」接近 `burst 長度 + window`，而不是單純的 burst 長度本身。本輪逐 chunk 量測這個「有效受影響樣本數」占（chunk + carry buffer）總長度的比例，發現與偵測成敗高度對應：chunk_size=256 時此比例約 66.7%（超過一半，中位數統計的「背景為多數」前提被打破，導致噪聲底被高估、門檻過高、整個 chunk 完全偵測不到訊號）；chunk_size=512 時約 40%（前提大致仍成立，但已足以造成數個 samples 的邊界誤差）；chunk_size=1024／2048 時降到 22–25%／12–13%（前提穩固成立，偵測結果與離線幾乎一致）。這是一個關於「carry buffer／chunk 相對於感測視窗的比例是否足夠大」的統計前提問題，`src/sensing/streaming_detector.py` 模組文件字串中已提前記載這是已知限制之一（「每個 chunk 獨立估計噪聲底，不是全串流的持續估計」）。

**下一步最小修正方向（本輪僅提出設計，未實作，且不建議在未經獨立單元測試驗證前實作）**：
1. 將「噪聲底估計用的統計視窗」與「chunk 大小」解耦——維持一個遠大於感測視窗（例如視窗的 8–16 倍）的獨立滾動緩衝，僅用於統計噪聲底，而不是直接複用 chunk+carry 這個較小的緩衝。
2. 將噪聲底估計從「每個 chunk 各自重新計算中位數」改為跨 chunk 持續更新的估計（例如以 EMA 或滑動視窗中位數維護一個貫穿整個串流的背景噪聲估計），這是目前原型與離線版本最主要的行為差異來源。
3. 若要在未來支援任意 chunk_size（包含小於視窗 8 倍的情況），需要額外的 open-region 狀態（尚未結束、可能延伸到下個 chunk 的事件）與 refractory／guard period，避免同一事件因跨 chunk 邊界重複判斷造成誤判或重複計數；目前原型僅有簡單的尾端緩衝延續機制，未實作這些狀態。
4. 需要補上事件去重複（event deduplication，避免同一實體事件因跨 chunk 邊界或 carry buffer 重疊而被算成兩個獨立區域）與事件 ID／時間戳記（讓每個偵測到的區域有穩定的識別碼與絕對時間，供下游系統關聯，而不只是回傳一組 sample index 區間）；目前原型完全沒有這兩項。

以上四項總結為一句話：**continuous IQ stream 可以用 chunk-based stateful detector 處理，不必把每個 packet 當成一個完全獨立的 sensing job，但正式實作仍需要 carry-over moving-average context、rolling noise-floor state、open-region state、rolling IQ buffer、event deduplication、guard／refractory logic、以及 timestamps／event IDs 這幾項狀態，目前原型只具備其中最簡單的尾端緩衝延續機制。**

**明確聲明本原型的範圍限制**：這只是一個離線的分塊處理原型，用來比較演算法輸出是否一致，**不是**即時串流系統，沒有實際的資料輸入輸出、沒有時間保證，也不能直接串接真實 SDR 硬體。跨 chunk 事件的延續僅透過簡單的尾端緩衝處理，未實作事件去重複判斷、事件 ID、時間戳記或 refractory／guard period，這些在 `src/sensing/streaming_detector.py` 的模組文件字串中已明確列為未實作項目。本原型未修改 `src/utils/pipeline.py` 或任何正式流程程式碼，本輪診斷亦未修改 `src/sensing/streaming_detector.py`。

## 十二、Processing Budget 對照

以下數字回答「目前哪些 pipeline／攻擊在給定的處理時間預算內可以完成」，**budget 本身不代表任何特定應用場景的 deadline**，僅作為比較不同處理階段耗時量級的參照點。

| Budget | Clean pipeline 達成率 | Clean+TopK pipeline 達成率 | FGSM baseline 達成率 | PGD baseline 達成率 | CW baseline 達成率 | FGSM optimized（均值） | PGD optimized（均值） | CW optimized（均值） |
|---|---|---|---|---|---|---|---|---|
| 5 ms | 80.7% | 24.6% | 1.5% | 0.0% | 0.0% | 在預算內 | 超出預算 | 超出預算 |
| 10 ms | 87.4% | 73.0% | 58.8% | 0.0% | 0.0% | 在預算內 | 在預算內 | 在預算內 |
| 20 ms | 90.8% | 81.1% | 75.2% | 0.0% | 0.3% | 在預算內 | 在預算內 | 在預算內 |
| 35 ms | 93.4% | 86.0% | 81.2% | 25.5% | 8.5% | 在預算內 | 在預算內 | 在預算內 |
| 50 ms | 94.7% | 89.1% | 86.4% | 39.1% | 23.6% | 在預算內 | 在預算內 | 在預算內 |
| 100 ms | 97.0% | 94.8% | 90.9% | 65.5% | 59.7% | 在預算內 | 在預算內 | 在預算內 |
| 250 ms | 99.9% | 98.3% | 97.0% | 85.5% | 90.3% | 在預算內 | 在預算內 | 在預算內 |
| 1000 ms | 100% | 100% | 100% | 98.8% | 100% | 在預算內 | 在預算內 | 在預算內 |

「達成率」欄位是 Phase A／B 逐筆 `total_ms`／`attack_total_ms` 中小於等於該 budget 的比例；「optimized（均值）」欄位是 Phase E 優化後 330 筆的平均延遲是否小於等於該 budget（單一均值判斷，不是逐筆比例）。優化後的 FGSM 在所有列出的 budget 下皆達標；PGD／CW 優化後平均延遲分別為 5.34 ms／9.32 ms，在 10 ms 以上的 budget 下達標，但不滿足最嚴格的 5 ms budget。

## 十三、研究限制

本輪所有量測皆在單一機器、單一次執行中完成，沒有跨多次執行的重複測量與信賴區間估計。第五節與第六節提到的長尾延遲現象（平均值遠高於中位數），其中 **AWN 推論部分已在第十五節第四小節透過受控執行緒數微基準測試診斷出主要成因**（執行緒數設定，非樣本內容）；**攻擊生成本身（`attack_generation_ms`）的長尾現象本輪未進一步診斷**，僅确認 PGD／CW 的批次化語意差異（見第十五節第一、二小節），攻擊生成長尾是否同樣是執行緒數造成，本輪未驗證，留待後續。Phase D 的加速候選試驗僅使用 60 筆樣本的小規模試驗集，用於快速篩選候選方案，不作為正式結論的統計依據——正式結論以 Phase E 在 330 筆規模下的重新量測、以及第十五節在 60 筆樣本上的決定性等價測試為準。**CW 批次化後的數值差異已在第十五節第二小節確認根本原因（批次層級的提前停止判斷），並改標為 `batched_algorithmic_variant`**，不再是未知成因。Attack object 重用與 `torch.compile` 兩項候選加速方案本輪未執行，僅依 Profiler 證據推測效益有限，未經實測驗證。Streaming sensing 原型僅測試了一條含三個 burst 的合成串流，未涵蓋高密度多重疊事件、極低訊噪比或跨 chunk 邊界剛好切在 burst 中間等更嚴苛的情境；第十五節的失敗機制診斷本身也僅針對這條測試串流的三個 burst，未驗證是否適用於其他 burst 長度或密度組合。所有效能數據皆針對 CPU 執行，未包含任何 GPU 量測。第十五節的執行緒數微基準測試同樣僅在單一機器上執行一次，執行緒數建議（2 個執行緒最穩定）是否能推廣到其他硬體環境，本輪未驗證。

## 十四、後續工作

依本輪與驗證輪證據，下一步可優先考慮：在正式攻擊實驗（例如 `experiments/run_low_perturbation_attacks.py`）中導入 FGSM／PGD 的批次化執行（已確認為 `implementation_optimization`），作為降低大規模矩陣執行時間的直接手段；CW 是否批次化則需先評估 `batched_algorithmic_variant` 帶來的語意差異是否可接受；針對 PGD 顯式設定 `random_start=False` 或固定隨機種子，若需要逐筆可重現的攻擊結果；將正式流程與批次化攻擊實驗的 CPU 執行緒數從預設調整為第十五節建議的較低值（2 或 4），以降低 AWN 推論與（若成因相同）攻擊生成的尾端延遲；追查攻擊生成本身（相對於 AWN 推論）的長尾延遲是否同樣是執行緒數造成；以及在確認何種延遲預算是實際需求後，再評估是否需要以纜線或屏蔽環境驗證下的即時處理路徑取代目前的離線批次處理模式。Streaming sensing 原型若要進一步發展為可用於正式流程的 cross-chunk 偵測器，需要依第十一節提出的最小修正方向補上：與 chunk 大小解耦的噪聲底統計視窗、跨 chunk 持續更新的噪聲底估計、以及 open-region／refractory period 狀態，且應在每一項修正後以獨立單元測試驗證不影響離線路徑的既有正確性。

## 十五、效能正確性與穩定性驗證（Phase A–G 之後的追加驗證輪）

Phase A–G 的量測完成後，第九節（PGD／CW 批次化語意差異）與第五節（AWN 推論長尾延遲）留下兩類尚未驗證的推測。本節以額外的、目標明確的實驗回答這些問題，方法與結論分開陳述：**決定性等價測試（deterministic equivalence test）回答「批次化本身是否改變計算結果」，隨機性表現比較（stochastic performance comparison）回答「在攻擊本身具有隨機性的前提下，批次大小是否改變隨機性的表現方式」——兩者不得混為一談，隨機性造成的差異不等於實作錯誤。**

### 15.1　PGD 批次化等價性驗證

使用與 Phase D 相同的 60 筆試驗集生成方式（`experiments/acceleration_pilot.py:build_pilot_inputs()`），其餘參數（樣本、種子、eps=0.05、alpha、model、前處理、執行緒設定）與 Phase D／E 一致，僅批次大小（1 vs 16）與 `random_start` 兩者獨立變化：

| 測試 | random_start | pred_match_rate | attacked tensor 最大差異 | attacked logits 最大差異 | ASR（batch_size=1） | ASR（batch_size=16） |
|---|---|---|---|---|---|---|
| **決定性等價測試** | False | **100.0%（60/60）** | **0.0** | **0.0** | 1.000 | 1.000 |
| 隨機性表現比較 | True（套件預設，正式流程目前使用） | 86.7%（52/60） | 0.004287 | 21.54 | 1.000 | 0.980 |

`random_start=False` 時，batch_size=1 與 batch_size=16 的輸出在 60 筆樣本上**逐位元完全相同**（tensor 與 logits 最大差異皆為浮點 0.0，非近似 0）——這是批次化對 PGD 計算過程沒有任何影響的直接證據，確認 PGD 批次化屬於 `implementation_optimization`。`random_start=True` 時觀察到的 86.7% prediction match（與 Phase E 在 330 筆規模下觀察到的 78.5% 屬同一現象，數值不同因樣本集與規模不同）並非批次化造成的語意改變，而是 torchattacks 的 `random_start` 機制本身：batch_size=1 呼叫 PGD 60 次、batch_size=16 呼叫 4 次，兩者消耗隨機初始點的方式不同，起點不同導致收斂結果不同，這是攻擊算法既有的隨機性表現，**不得視為 bug**。原始逐筆資料見 `pgd_batch_equivalence_deterministic.csv`（決定性測試）與 `pgd_batch_stochastic_comparison.csv`（隨機性比較）。

### 15.2　CW 批次化語意稽核

CW 沒有 `random_start` 機制，天生決定性，因此 batch_size=1 與 batch_size=16 的任何輸出差異都不能歸因於隨機性，必須從程式邏輯本身找原因。本輪逐行閱讀 `torchattacks.CW.forward()`（`inspect.getsource()` 取得，該套件為已釘選版本、本輪未修改）並針對 12 項稽核問題逐一確認：

1. **Loss 是否逐筆計算**：是，`current_L2 = MSELoss(...).sum(dim=1)` 為逐筆 `[N]` 張量。
2. **Loss reduction 方式**：整批加總（`L2_loss = current_L2.sum()`、`f_loss = self.f(outputs, labels).sum()`），非 mean、非 none。
3. **批次大小是否改變 gradient 量級**：否——加總後獨立樣本的梯度分解不受批次中其他樣本影響（AWN 模型於 `.eval()` 模式下 BatchNorm 使用固定的 running statistics，本輪確認 `AttackAdapter.apply()` 只在呼叫前已為 train 模式時才會印出警告，本輪所有測試呼叫皆為 eval 模式，沒有跨樣本 BatchNorm 耦合）。
4. **optimizer learning rate 是否被批次大小影響**：否——Adam 對梯度張量的每個元素獨立維護動量狀態，逐樣本元素之間沒有交互。
5. **best adversarial example 是否逐筆儲存**：是，`best_adv_images` 透過逐筆 `mask` 更新。
6. **success mask 是否逐筆**：是，`condition = (pre != labels).float()` 為 `[N]` 張量。
7. **提前停止（early stop）是否逐筆判斷**：**否**——`if step % max(self.steps // 10, 1) == 0: if cost.item() > prev_cost: return best_adv_images` 使用整批加總後的純量 `cost`，是整批一起停止，不是逐筆各自停止。
8. **終止判斷是否為批次層級**：**是**（與第 7 項同一機制，本輪視為最關鍵的稽核發現）。
9. **c／confidence／steps／lr 是否一致**：是，本輪基準與優化兩次呼叫皆使用相同的空 `attack_params`，套件預設值（c=1.0, kappa=0, steps=20, lr=0.01）逐次呼叫相同。
10. **normalization／inverse transform 是否逐筆正確**：是，`_iq_to_ta_input_minmax` 以 `amin(dim=(1,2))` 逐筆計算自己的 min/max。
11. **clamp／project 是否逐筆正確**：是，`tanh_space`／`inverse_tanh_space` 為逐元素運算，無跨樣本耦合。
12. **梯度是否可能跨批次樣本聚合**：理論上除第 7／8 項的批次層級提前停止外，沒有其他跨樣本聚合點；本輪以 runtime monkeypatch（僅在記憶體中覆寫 `torchattacks.CW.forward`，不修改任何檔案）直接觀測提前停止是否實際觸發。

**實測結果**：batch_size=1 時，60 筆樣本**全部**（60/60）觸發提前停止，但觸發的步數因樣本而異（第 2 到 16 步不等，例如某些樣本在第 2 或第 4 步就已達到自己的最佳點）；batch_size=16 時，4 個批次分別在第 10、16、12、14 步「整批」觸發停止。這代表被批次在一起的樣本被迫共用同一個停止時機，而不是各自在自己的最佳步數停止。

60 筆決定性配對測試（`cw_batch_equivalence.csv`）結果：

| 項目 | 數值 |
|---|---|
| pred_match_rate | 95.0%（57/60，3 筆不一致：sample_id 5、26、37） |
| attacked tensor 最大差異 | 0.001377 |
| attacked logits 最大差異 | 12.42 |
| ASR（batch_size=1） | 0.898 |
| ASR（batch_size=16） | 0.959 |

由於 CW 沒有隨機性，這個差異只能來自批次層級提前停止對最佳化軌跡的實際影響（第 7／8 項稽核發現，並已由上述 monkeypatch 實測確認確實觸發）。**結論：CW 批次化不是純粹的 `implementation_optimization`，而是 `batched_algorithmic_variant`**——批次大小改變了 CW 內部提前停止的時機，進而改變部分樣本的最終擾動與預測結果，改變的方向與幅度因樣本而異，不是可預期、單方向的效果。本輪嘗試尋找修正方式：由於提前停止邏輯位於已釘選版本的 torchattacks 套件內部（非本 repo 原始碼，第三方函式庫，不可修改），在不修改該套件的前提下，唯一能維持與逐筆執行完全等價的方式是放棄批次化（batch_size=1）。因此**未進行修正重跑**，`batched_algorithmic_variant` 為本輪最終分類，是否接受此 tradeoff 換取批次化加速，留給後續使用場景決定。

### 15.3　AWN 推論延遲長尾——Top-50 離群值分析

從既有的 `pipeline_latency_raw.csv`（Phase A，2200 筆，本輪未重跑）萃取 `awn_clean_inference_ms` 最高的 50 筆（`awn_latency_outliers.csv`），連同執行順序、調變、訊噪比、前後鄰筆延遲、其他階段耗時一併記錄。發現：

- **高度集中於特定執行區段，而非隨機散布**：top-50 中有 19 筆集中在連續執行順序 552–594（AM-SSB，訊噪比 10–18 dB）這一段窄窗內，而非均勻散布在 2200 筆的整個執行過程中。
- **調變分布高度不均**：top-50 中 AM-SSB 占 24/50（48%），遠高於其在資料集中的比例（1/11 ≈ 9%）。
- 進一步檢視發現：AM-SSB 區塊（執行順序 400–599）內部，延遲隨訊噪比從 -20 dB 逐步升高到 16 dB 呈現明顯上升趨勢（訊噪比 16 dB 的中位數達 232 ms），但在該區塊結束、下一個調變（BPSK，執行順序 600 起）開始後，延遲**立即恢復正常**（中位數回到約 1–2 ms）。這個銳利的邊界排除了「隨執行時間單調惡化」（例如溫度累積）的假設，但當時尚不足以判斷是「AM-SSB 訊號內容特有」還是「剛好落在某個時間窗的環境爭用」。

### 15.4　AWN 推論延遲——受控執行緒數微基準測試

為了在排除訊號內容因素的前提下驗證上述假設，本輪建立一個新的受控微基準測試（`experiments/awn_thread_microbenchmark.py`）：固定 100 筆輸入（25 筆取自 15.3 節發現的「已知較慢」AM-SSB／高訊噪比組合、50 筆其他調變、25 筆「已知較快」AM-SSB／低訊噪比組合，三組樣本身份固定不變），在 6 種執行緒設定（目前預設、1、2、4、8、16）下，每種設定先執行 60 次暖身（捨棄），再對 100 筆輸入重複 5 輪、共 500 次計時呼叫（計時範圍不含資料載入與 CSV 寫入）。

整體統計（`awn_thread_microbenchmark_summary.csv`）：

| threads | mean (ms) | median (ms) | p95 (ms) | p99 (ms) | max (ms) | CV | outlier_rate（>5x 中位數，n=500） | throughput (samples/sec) |
|---|---|---|---|---|---|---|---|---|
| 目前預設（16） | 1.372 | 0.769 | 2.571 | 15.341 | 37.371 | 2.248 | 4.0% | 728.9 |
| 1 | 0.568 | 0.542 | 0.702 | 0.921 | 2.208 | 0.212 | 0.0% | 1759.7 |
| **2** | **0.551** | **0.534** | **0.649** | **0.769** | **0.876** | **0.101** | **0.0%** | **1813.4** |
| 4 | 0.611 | 0.571 | 0.895 | 1.093 | 1.285 | 0.218 | 0.0% | 1637.1 |
| 8 | 0.747 | 0.700 | 1.111 | 1.300 | 1.619 | 0.228 | 0.0% | 1338.7 |
| 16 | 10.062 | 0.831 | 39.444 | 203.309 | 254.872 | 3.728 | 12.2% | 99.4 |

按樣本群組分解（`awn_thread_microbenchmark_by_group.csv`）進一步顯示：**在執行緒數 1／2／4／8 下，「已知較慢」AM-SSB 組、其他調變組、「已知較快」AM-SSB 組三者的離群次數全部為 0**，沒有任何一組因內容不同而表現不同；只有在執行緒數 16（含「目前預設」）時才出現離群值，且三組的離群比例相近（11.2%、14.4%、8.8%），並非集中在 AM-SSB。**這排除了「AM-SSB 訊號內容導致 AWN 推論變慢」的假設**，確認 15.3 節觀察到的調變集中現象是執行到該區段時剛好處於高執行緒爭用狀態下的巧合，不是 AM-SSB 資料本身的性質；根本原因是**執行緒數設定**，與先前 Phase D（第八節）在 PGD 上觀察到的「執行緒數 16 反而比執行緒數 1 慢」現象一致且互相印證。

**最佳穩定性設定與最佳典型延遲設定並不衝突，且不存在真正的 throughput／tail latency tradeoff**：在本次測試範圍內，執行緒數 2 同時取得最低平均延遲、最低中位數、最低 p95／p99／max、最低變異係數（CV=0.101）與最高 throughput（1813.4 samples/sec），執行緒數 1 在每項指標上都非常接近（throughput 1759.7 samples/sec），執行緒數 4／8 表現稍遜但仍遠優於預設的 16；執行緒數 16 在每一項指標上都是最差的（throughput 僅 99.4 samples/sec）。也就是說，本次實驗沒有觀察到「犧牲穩定性換取更高吞吐量」的取捨關係——執行緒數 16 同時犧牲了吞吐量與穩定性，沒有任何補償性優勢。這與小張量（`[1,2,128]`）運算下執行緒協調開銷超過平行化效益的既有假說一致（見第八節）。

**本節建議（執行緒數 2，其次是 1）僅適用於本輪測試的這台機器與這個 `[1,2,128]` 小張量 AWN 推論工作負載，不得推論所有 CPU／硬體都應固定使用某個特定執行緒數**——不同核心數、不同 CPU 微架構、不同 PyTorch／oneDNN 版本、或不同輸入張量大小（例如批次推論而非單筆推論）都可能改變最佳執行緒數，任何要套用到其他環境的部署都應該重新執行本節的微基準測試方法（`experiments/awn_thread_microbenchmark.py`）驗證，而不是直接沿用本文件的數字。

### 15.5　Clean Pipeline 瓶頸依統計量重新計算

第五節的「AWN 推論占 86.6%」是以平均值計算，而平均值受少數尾端離群值嚴重影響（15.3／15.4 節已證實其成因與執行緒設定有關）。本節依 mean／median／p95 三種統計量分別重新計算各階段占比（`bottleneck_by_percentile.csv`）：

| stage | mean (ms) | median (ms) | p95 (ms) | 占 mean 比例 | 占 median 比例 | 占 p95 比例 |
|---|---|---|---|---|---|---|
| embedding_ms | 0.459 | 0.441 | 0.582 | 4.00% | 15.66% | 1.02% |
| energy_detection_ms | 0.315 | 0.308 | 0.380 | 2.75% | 10.94% | 0.66% |
| region_postprocess_ms | 0.046 | 0.044 | 0.061 | 0.40% | 1.54% | 0.11% |
| segmentation_ms | 0.681 | 0.690 | 0.848 | 5.94% | 24.50% | 1.48% |
| awn_preprocess_ms | 0.036 | 0.033 | 0.051 | 0.31% | 1.18% | 0.09% |
| **awn_clean_inference_ms** | **9.925** | **1.285** | **55.637** | **86.60%** | **45.60%** | **97.35%** |

**三種「瓶頸」結論並不相同，必須分開陳述**：
- **平均情況（mean）瓶頸**：AWN 推論占 86.60%，是絕對主導的瓶頸。
- **典型情況（median）瓶頸**：AWN 推論占 45.60%，仍是最大的單一階段，但感測與前處理各階段合計占 53.8%（略高於 AWN），兩者相差不大，**不能說 AWN 推論在典型情況下絕對主導**；segmentation（24.5%）與 embedding（15.66%）在典型情況下的占比並不可忽略。
- **尾端情況（p95）瓶頸**：AWN 推論占 97.35%，比平均情況更極端地主導，這與 15.3／15.4 節「AWN 推論的尾端延遲由執行緒設定造成」的發現一致——正是這些尾端事件把 AWN 推論的平均值從中位數的 1.285 ms 拉高到 9.925 ms。

因此，「AWN 推論是瓶頸」這句話在平均與尾端情況下成立，但在典型（中位數）情況下並非單方面成立；後續若要優化端到端延遲，需視優化目標是「降低典型延遲」（此時感測前端與 AWN 推論同等重要）還是「降低尾端延遲／提升穩定性」（此時應優先處理 15.4 節發現的執行緒設定問題）分別決定策略。

---

## 可重現性資訊

本文件記錄的全部數據對應目前主分支程式狀態，量測程式為 `experiments/benchmark_pipeline_latency.py`（Phase A／B）、`experiments/profile_attacks.py`（Phase C）、`experiments/acceleration_pilot.py`（Phase D）、`experiments/acceleration_before_after.py`（Phase E）、`experiments/test_streaming_sensing.py`（Phase G，搭配 `src/sensing/streaming_detector.py`），彙整程式為 `experiments/finalize_performance_results.py`。第十五節的追加驗證輪對應的量測程式為 `experiments/batch_equivalence_audit.py`（15.1／15.2，PGD／CW 批次化等價性與語意稽核，輸出 `pgd_batch_equivalence_deterministic.csv`、`pgd_batch_stochastic_comparison.csv`、`cw_batch_equivalence.csv`）、`experiments/analyze_awn_latency_outliers.py`（15.3，輸出 `awn_latency_outliers.csv`，讀取既有 `pipeline_latency_raw.csv`，未重跑 2200 筆基準）、`experiments/awn_thread_microbenchmark.py`（15.4，輸出 `awn_thread_microbenchmark_summary.csv`、`awn_thread_microbenchmark_by_group.csv`、`awn_thread_microbenchmark_raw.csv`）、一段即時計算 `bottleneck_by_percentile.csv` 的腳本（15.5）、`experiments/diagnose_streaming_failure.py`（第十一節失敗機制診斷，輸出 `streaming_failure_diagnosis.csv`）。逐筆原始資料、彙整統計、Profiler 輸出、圖表與完整環境資訊（CPU 型號、核心／執行緒數、記憶體、作業系統、Python／PyTorch／torchattacks 版本、checkpoint 雜湊、資料集路徑、種子、暖身與正式樣本數）保存於對應的本機結果目錄之 `manifest.json`，未納入版本控制。上游 AWN 與對抗式攻擊實作（包含 CW 提前停止邏輯所在的 torchattacks 套件）為固定版本的外部程式庫，本輪工作（含追加驗證輪）未對其進行修改；15.2 節提到的 runtime monkeypatch 僅在單一診斷腳本的記憶體中執行，未寫回任何檔案，且不影響同一腳本內用於產生 `cw_batch_equivalence.csv` 的實際等價測試呼叫路徑（等價測試呼叫的是未經 monkeypatch 的 `AttackAdapter.apply()`）。
