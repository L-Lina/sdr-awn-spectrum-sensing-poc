# AWN 模型於 Raspberry Pi 5 Hailo-8 NPU 部署之本週進度

日期：2026-09-15

## 1. 本週目標

本週目標為將既有 AWN modulation classification model 部署至 Raspberry Pi 5 搭配 Hailo AI HAT+（26 TOPS，Hailo-8）之硬體平台，建立 CPU AWN 與 NPU AWN 的後續比較基礎。

原始 AWN 模型規格如下：

- 資料集：RadioML2016.10a
- 類別數：11 種調變方式
- 輸入邏輯 shape：`[1, 2, 128]`
- 資料型別：float32
- 前處理：無 normalization

目標硬體：

- Raspberry Pi 5，16GB
- Raspberry Pi AI HAT+，26 TOPS
- Hailo-8

本週主要工作集中於下列部署流程：

PyTorch AWN → ONNX deployment graph → Hailo parse → quantization / optimization → compile → HEF

## 2. Raspberry Pi 與 CPU AWN baseline

Raspberry Pi 端 Hailo 環境已完成基本檢查：

- `hailortcli scan` 可偵測到裝置
- PCIe device：0001:01:00.0
- 晶片：Hailo-8
- firmware / HailoRT 版本：4.23.0

CPU AWN baseline 使用真實 RadioML2016.10a 資料集，於 SNR = 18 條件下，取 11 種調變方式各 20 筆樣本，共 220 筆樣本進行推論，結果為：

- accuracy：89.09%（196 / 220）
- WBFM：0%
- 其餘調變類別多數落在約 90% 至 100% 區間

CPU 端 model-forward latency 統計：

- mean：1.9513 ms
- median：1.9498 ms
- p95：1.9980 ms
- min：1.8944 ms
- max：2.1075 ms

此 CPU baseline 將作為後續 NPU accuracy 與 latency 比較之參考基準。

## 3. AWN deployment graph 之 Hailo compatibility adaptation

原始 PyTorch AWN 模型於轉換為 Hailo 可解析之 ONNX deployment graph 過程中，遭遇數項與 Hailo Dataflow Compiler（DFC）相容性相關之問題，逐一處理如下。

### 3.1 Conv stride export 問題

原始 ONNX export 中，第一層 Conv2D 之 stride 表示方式異常。處理方式為建立 deployment-only 之 deep copy，僅修正 `fixed.conv1[1].stride = (1, 1)`，未修改 external model 原始程式碼。

修正後之 ONNX 檔案為 `awn_2016_10a_exportfix.onnx`，驗證結果為：

- ONNX checker：PASS
- PyTorch 與 ONNX 預測結果一致
- QPSK@18 樣本預測類別為 class 9
- logits 最大絕對誤差約 9.54e-06，平均誤差約 3.17e-06

### 3.2 Hailo tensor rank 與 layout 相容性

原始 AWN 輸入 shape 為 `[1, 2, 128]`（rank-3）。Hailo parser 對 rank-3 tensor 之預設 layout 解讀與原模型之 NCW 語意不一致，導致後續 layer 之 shape 推論錯誤。

處理方式為將 deployment representation 改為 rank-4 `[1, 1, 2, 128]`，此為純粹的 representation-equivalent rewrite，未變更模型數學運算。以 22 組樣本進行語意一致性驗證，結果為 diff = 0。

### 3.3 Wavelet lifting block 相容性

AWN 模型中之 wavelet lifting 模組使用數種 Hailo 原生不支援之運算模式，包含 strided Slice（step=2）、reverse Slice（step=-1）以及部分 Gather / Shuffle pattern。

最終解法為將 wavelet 的 even/odd splitting 改寫為固定權重之 depthwise/grouped Conv1d selector：

- even selector kernel = [1, 0]
- odd selector kernel = [0, 1]
- groups = 64，kernel size = 2，stride = 2

驗證結果為 intermediate diff = 0，logits diff = 0，102 組樣本 prediction agreement 為 102/102。

Reflection padding 部分亦改寫為數學等價之 Gather + Concat representation。

## 4. Hailo parse / optimize 通過，但 compile 發生 native allocator crash

經過上述調整後，完整 AWN graph 已可通過 Hailo parse 與 optimize / quantize 階段，模型結構完整保留，共計 36 個 Hailo layer，最終 end node 為模型真正的輸出層，classifier 部分未遭截斷。

此階段之 final deployment ONNX 為 `awn_2016_10a_hailofull4d_nodropctrl.onnx`，語意一致性驗證結果為 102/102 logits allclose，102/102 predictions 完全一致。

然而，進入 compile 階段時，Hailo compiler 後端發生 native crash：

```
compiler: ../src/network_graph/racehorse.cpp:1682:
Assertion `pyramid_output->output_shapes().size() > index' failed.
[error] BackendAllocatorException: Compilation failed with unexpected crash
```

已測試之 compiler 配置包含 default single-context、forced multi-context、compiler optimization level max，三者結果皆為同一 assertion crash。

此結果不代表模型不支援 Hailo 部署。目前 evidence 指向 Hailo compiler 後端 allocator 對特定 graph topology 之處理存在問題。

## 5. 跨 DFC 版本 regression 驗證

為排除此問題為新版 DFC 之 regression，本週另外建立 Hailo AI Software Suite 2025-10 環境，實際版本為：

- DFC 3.33.0
- HailoRT 4.23.0
- Python 3.10.12

與原有環境 Hailo AI Software Suite 2026-07（DFC 3.34.0，HailoRT 4.24.0）並行比對。

使用完全相同之 ONNX、calibration set 與 parse → optimize → compile pipeline，結果為：

| DFC 版本 | parse | optimize | compile |
|---|---|---|---|
| 3.33.0 | PASS | PASS | FAIL |
| 3.34.0 | PASS | PASS | FAIL |

兩版本皆為同一 `racehorse.cpp:1682` assertion、同一 compile stage、皆未產生 HEF。由此結果可知，此問題並非單純之 DFC 3.34.0 regression。

## 6. Compiler crash 系統化 graph isolation

為進一步定位問題範圍，本週對完整 graph 進行系統化之子圖拆解測試。

粗粒度拆解結果為：

| 拆解邊界 | compile 結果 |
|---|---|
| conv2 end | PASS |
| wavelet end | PASS |
| pooling end | PASS |
| SE-attention end | FAIL |
| classifier-minus-last | FAIL |

由此結果可知，crash 邊界位於 SE-attention 模組內。

進一步對 SE-attention 內部進行細粒度拆解，範圍涵蓋 Reshape → fc1 Conv → ReLU → conv7 Conv → Sigmoid → Reshape → Mul 各節點，結果為每一個獨立節點（Reshape、fc1、ReLU、conv7、Sigmoid）皆可個別通過 compile，僅有最後之 element-wise Mul reconvergence 節點導致 compile 失敗。

由此結果，crash 範圍縮小至 SE-attention gating 之 graph reconvergence 結構。此處需說明，實驗 evidence 並未證明 Hailo 的 Mul 運算本身存在缺陷，而是將問題範圍縮小至 element-wise gating 及其 ancestor-descendant reconvergence topology。

## 7. Mathematically equivalent graph rewrite 實驗

基於前述定位結果，本週依序設計五組數學等價之 graph rewrite 方案，以 hypothesis-driven 方式逐一排除可能成因。每組實驗於通過語意一致性驗證後，方進入 Hailo compile 測試；語意驗證未通過者不予送入 compile。

| Candidate | 排除假設 | 實驗結果 | 解讀 |
|---|---|---|---|
| 1 | duplicated reshape / tensor reuse 是否為 allocator crash 主因 | semantic diff = 0，102/102 prediction agreement；compile FAIL，racehorse.cpp:1682 | duplicated reshape 並非充分條件 |
| 2 | Mul 兩側 input 是否需以獨立 static canonical 4D reshape 呈現方可通過 compile | semantic diff = 0，102/102；compile FAIL，同一 assertion | canonical reshape 形式並非充分條件 |
| 3a / 3b / 3c | element-wise Mul channel width（128 拆分為 2×64 / 4×32 / 8×16）是否為 allocator failure 之成因 | 三組拆分（Split → 多組 Mul → Concat）皆為 semantic equivalent，102/102；皆於相同 assertion 失敗 | 問題與單一 128-channel Mul 之 tensor 規模無直接對應 |
| 4A / 4B | Mul input 是否需以獨立 Hailo Conv layer materialize 方可通過 compile | 以 identity depthwise 1×1 Conv 分別對單側（4A）與雙側（4B）input 進行 materialize，identity Conv 於 Hailo graph 中保留，未被 optimizer fuse 消除；semantic diff = 0，102/102；compile FAIL，同一 assertion | 單純之 tensor materialization 並非充分條件 |
| 5 | ancestor-descendant reconvergence / shared graph provenance 是否為關鍵因素 | 複製 deterministic pooling branch 後，維持 exact semantic equivalence；DFC 3.33.0 與 3.34.0 均 compile SUCCESS | 實驗 evidence 強烈支持 reconvergence / graph-provenance-sensitive allocator behavior 為關鍵因素 |

## 8. Candidate 5：graph provenance 分離之 mathematically equivalent rewrite

目的：測試 SE-attention gating 中之 ancestor-descendant reconvergence 結構，是否為造成 allocator crash 之關鍵因素。

AWN 之 SE-attention 結構本質為：`x → SE MLP → sigmoid gate → gate × x`。在原始 graph 中，SE gate 分支與 passthrough x 分支共用同一個 pooling output tensor，因此 x 同時為 gate 分支之 ancestor，並於後續 Mul 節點處與 gate 分支重新匯合，形成 ancestor-descendant reconvergence 結構。

Candidate 5 之作法為複製此段 deterministic pooling block。原始結構為單一 pooling output 分別接往 SE gate 與 passthrough 兩路，於 Mul 節點匯合；改寫後之結構改為由共用之 wavelet output 分別接往兩組獨立之 pooling block（pooling_A、pooling_B），產生 x_A 與 x_B 兩個獨立 tensor，x_A 接往 SE gate，x_B 直接作為 Mul 之 passthrough 輸入。

pooling_A 與 pooling_B 使用相同輸入、相同之 deterministic 運算，不含 learned weights，不含 randomness，不含 state，因此 x_A 與 x_B 理論上應數值完全相同。實測結果為 max_abs_diff = 0.0，與預期一致。

Candidate 5 之語意一致性驗證結果為：

- 原始 deployment ONNX 與 Candidate 5 之 logits：exactly 0.0 diff
- Mul output：0.0 diff
- x_A 與 x_B：0.0 diff
- 102/102 prediction agreement
- 46 個 initializer 逐一 byte-for-byte 完全相同
- learned weights 無任何修改
- SE gate 保留，Mul 保留，classifier 結構完整，輸出仍為 11 類別

PyTorch AWN 與 Candidate 5 之比較結果為：logits 最大絕對誤差約 3.05e-5，中間張量最大絕對誤差約 8.9e-7，102/102 prediction agreement。此差異來自 PyTorch 之未融合 Conv 與 BatchNorm，與 ONNX 中已融合 Conv 之 FP32 累加順序不同，屬既有已知之浮點誤差範圍，非正確性問題。

解讀：實驗結果指出，僅在 graph provenance 上將 gate 分支與 passthrough 分支分離為兩組獨立節點，同時保持數值完全等價，即可解除 compile 階段之 native allocator crash。此結果與 Candidate 1 至 4 之排除結果一致支持，crash 與 reconvergence 結構本身相關，而非與 reshape 形式、tensor 規模或 tensor materialization 相關。

## 9. DFC 3.33.0 / 3.34.0 compile 結果

Candidate 5 於兩版本 DFC 之 compile 結果為：

DFC 3.33.0：

- parse：PASS
- optimize：PASS
- compile：SUCCESS
- HEF：`awn_2016_10a_c5.hef`，大小 1,544,987 bytes
- SHA256：6e54266f7e5931e432d4e5e67cd8e4a6e14863c16a94ff02a1946d21abc73dbb

DFC 3.34.0：

- parse：PASS
- optimize：PASS
- compile：SUCCESS
- HEF 大小：1,558,741 bytes

兩版本之 cluster allocation 統計一致：Control utilization 75%，Compute utilization 31.3%，Memory utilization 20.5%。

兩版本產生之 HEF 檔案並非 byte-identical，此為不同 compiler 版本之合理現象，不代表任一版本之編譯結果有誤。

## 10. HEF metadata

以 DFC 3.33.0 產生之 HEF 為例，透過 `hailortcli parse-hef` 讀取之實際 metadata 為：

- target architecture：HAILO8
- network：awn_2016_10a_c5
- single network，single context
- input vstream：`input_layer1`，UINT8，NHWC，shape (2, 128, 1)
- output vstream：`conv10`，UINT8，FCR，shape (1, 1, 11)

物理層 stream 因 Hailo 硬體對齊機制，輸出層存在 16-channel padding；邏輯層 vstream 已正確還原為 11 類別輸出，未見任何非預期之額外輸出。

## 11. 目前尚未完成的部分

目前已完成 Hailo-8 HEF 產生、跨 DFC 版本編譯驗證及模型語意一致性驗證；惟尚未完成 Raspberry Pi 5 + Hailo-8 實體裝置之 NPU inference validation。

因此目前尚未取得：

- NPU inference accuracy
- CPU 與 NPU prediction agreement
- NPU inference latency
- class-wise NPU classification performance
- end-to-end sensing / attack / Top-K defense pipeline 之 NPU 整合結果

上述項目列為下一階段實驗內容。

本階段結論應限定為：已完成 Hailo compiler-side deployment validation 與 HEF 產生。不得延伸宣稱 NPU accuracy、NPU latency、CPU/NPU consistency 或 full NPU pipeline 已完成。

## 12. 下一階段工作

1. Raspberry Pi 5 + Hailo-8 HEF compatibility validation
2. HailoRT input/output quantization metadata analysis
3. Single-sample functional inference validation
4. Fixed 220-sample RadioML2016.10a CPU / NPU comparison
5. NPU accuracy and prediction agreement evaluation
6. class-wise performance analysis
7. NPU inference latency measurement
8. integration with spectrum sensing / adversarial attack / Top-K defense pipeline

技術注意事項：HEF 之 input vstream 為 UINT8，並不代表原始 float32 IQ samples 可直接以 `astype(uint8)` 轉換。實機驗證時須依據 HailoRT 所提供之 quantization metadata 與 host-side transform 機制，正確處理輸入與輸出資料。

## 13. 本週結論

本週已完成 AWN 模型 Hailo-compatible deployment graph 之建立，並完成跨 DFC 3.33.0 與 3.34.0 的編譯驗證。針對原始完整模型在 Hailo allocator 階段發生的 racehorse.cpp:1682 native assertion，透過 coarse-to-fine graph isolation 將問題範圍縮小至 SE-attention gating reconvergence，並進一步設計多組 mathematically equivalent graph rewrites 進行假設排除。

Candidate 5 透過複製 deterministic pooling branch，使 SE gate branch 與 passthrough branch 在 graph provenance 上分離，同時保持與 baseline deployment ONNX 完全數值等價。Candidate 5 與 baseline 之 logits 差異為 0，102/102 predictions 一致，所有 learned initializers 未改動，並可於 DFC 3.33.0 與 3.34.0 成功產生 Hailo-8 HEF。上述 evidence 強烈支持原始 failure 與特定 reconvergence / graph-provenance allocator behavior 有關，惟目前尚不足以推定 Hailo compiler 內部之完整 root cause。

後續將進行 Raspberry Pi 5 + Hailo-8 實機推論、CPU/NPU 一致性與 latency evaluation，並於通過硬體驗證後整合至完整 spectrum sensing、adversarial attack 與 Top-K defense pipeline。
