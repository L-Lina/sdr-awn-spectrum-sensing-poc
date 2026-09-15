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

CPU AWN baseline 使用真實 RadioML2016.10a 資料集，於 SNR = 18 條件下，取 11 種調變方式各 20 筆樣本，共 220 筆樣本進行推論，結果如下：

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

## 3. AWN ONNX deployment graph 建立

原始 PyTorch AWN 模型於轉換為 Hailo 可解析之 ONNX deployment graph 過程中，遭遇數項與 Hailo Dataflow Compiler（DFC）相容性相關之問題，逐一處理如下。

### 3.1 Conv stride export 問題

原始 ONNX export 中，第一層 Conv2D 之 stride 表示方式異常。處理方式為建立 deployment-only 之 deep copy，僅修正 `fixed.conv1[1].stride = (1, 1)`，未修改 external model 原始程式碼。

修正後之 ONNX 檔案為 `awn_2016_10a_exportfix.onnx`，驗證結果如下：

- ONNX checker：PASS
- PyTorch 與 ONNX 預測結果一致
- QPSK@18 樣本預測類別為 class 9
- logits 最大絕對誤差約 9.54e-06，平均誤差約 3.17e-06

### 3.2 Hailo tensor rank 與 layout 相容性

原始 AWN 輸入 shape 為 `[1, 2, 128]`（rank-3）。Hailo parser 對 rank-3 tensor 之預設 layout 解讀與原模型之 NCW 語意不一致，導致後續 layer 之 shape 推論錯誤。

處理方式為將 deployment representation 改為 rank-4 `[1, 1, 2, 128]`，此為純粹的 representation-equivalent rewrite，未變更模型數學運算。以 22 組樣本進行 semantic validation，結果為 diff = 0。

### 3.3 Wavelet lifting block 相容性

AWN 模型中之 wavelet lifting 模組使用數種 Hailo 原生不支援之運算模式，包含 strided Slice（step=2）、reverse Slice（step=-1）以及部分 Gather / Shuffle pattern。

最終解法為將 wavelet 的 even/odd splitting 改寫為固定權重之 depthwise/grouped Conv1d selector：

- even selector kernel = [1, 0]
- odd selector kernel = [0, 1]
- groups = 64，kernel size = 2，stride = 2

驗證結果：intermediate diff = 0，logits diff = 0，102 組樣本 prediction agreement 為 102/102。

Reflection padding 部分亦改寫為數學等價之 Gather + Concat representation。

## 4. Hailo parse / optimize 成功，但 compile 發生 native allocator crash

經過上述調整後，完整 AWN graph 已可通過 Hailo parse 與 optimize / quantize 階段，網路結構完整保留，共計 36 個 Hailo layer，最終 end node 為模型真正的輸出層，classifier 部分未遭截斷。

此階段之 final deployment ONNX 為 `awn_2016_10a_hailofull4d_nodropctrl.onnx`，semantic validation 結果為 102/102 logits allclose，102/102 predictions 完全一致。

然而，進入 compile 階段時，Hailo native compiler 發生 crash：

```
compiler: ../src/network_graph/racehorse.cpp:1682:
Assertion `pyramid_output->output_shapes().size() > index' failed.
[error] BackendAllocatorException: Compilation failed with unexpected crash
```

已測試之 compiler 配置包含 default single-context、forced multi-context、compiler optimization level max，三者結果皆為同一 assertion crash。

此結果不代表模型不支援 Hailo 部署。目前 evidence 指向 Hailo compiler backend allocator 對特定 graph topology 之處理存在問題。

## 5. DFC 版本 regression

為排除此問題為新版 DFC 之 regression，本週另外下載並建立 Hailo AI Software Suite 2025-10 環境，實際版本為：

- DFC 3.33.0
- HailoRT 4.23.0
- Python 3.10.12

與原有環境 Hailo AI Software Suite 2026-07（DFC 3.34.0，HailoRT 4.24.0）並行比對。

使用完全相同之 ONNX、calibration set 與 parse → optimize → compile pipeline，結果如下：

| DFC 版本 | parse | optimize | compile |
|---|---|---|---|
| 3.33.0 | PASS | PASS | FAIL |
| 3.34.0 | PASS | PASS | FAIL |

兩版本皆為同一 `racehorse.cpp:1682` assertion、同一 compile stage、皆未產生 HEF。由此結果可知，此問題並非單純之 DFC 3.34.0 regression。

## 6. Compiler crash 系統化定位

為進一步定位問題範圍，本週對完整 graph 進行系統化之子圖拆解測試。

粗粒度拆解結果：

| 拆解邊界 | compile 結果 |
|---|---|
| conv2 end | PASS |
| wavelet end | PASS |
| pooling end | PASS |
| SE-attention end | FAIL |
| classifier-minus-last | FAIL |

由此結果可知，crash 邊界位於 SE-attention 模組內。

進一步對 SE-attention 內部進行細粒度拆解，範圍涵蓋 Reshape → fc1 Conv → ReLU → conv7 Conv → Sigmoid → Reshape → Mul 各節點，結果為每一個獨立節點（Reshape、fc1、ReLU、conv7、Sigmoid）皆可個別通過 compile，僅有最後之 element-wise Mul reconvergence 節點會導致 compile 失敗。

由此結果，將 crash 範圍縮小至 SE-attention gating 之 graph reconvergence 結構。此處需說明，實驗 evidence 並未證明 Hailo 的 Mul 運算本身存在缺陷，而是將問題範圍縮小至 element-wise gating 及其 ancestor-descendant reconvergence topology。

## 7. Equivalent rewrite 實驗

針對上述定位結果，依序測試五組數學等價之 graph rewrite 方案，結果整理如下表：

| Candidate | 方法 | Semantic validation | Compile | 結果 |
|---|---|---|---|---|
| 1 | 移除重複 Reshape，直接重用相同 4D tensor | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 2 | Mul 前兩側加入固定 static 4D canonical Reshape | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 3a | Split（2×64）→ 兩組 Mul → Concat | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 3b | Split（4×32）→ 四組 Mul → Concat | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 3c | Split（8×16）→ 八組 Mul → Concat | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 4A | Passthrough branch 加入 identity depthwise 1×1 Conv | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |
| 4B | 兩個 Mul input 均以 identity depthwise 1×1 Conv materialize | diff = 0，102/102 | FAIL | 同 racehorse.cpp:1682 |

由 Candidate 3 系列結果可知，Mul channel width（128 / 64 / 32 / 16）並非造成 crash 之主要因素。由 Candidate 4 系列結果可知，單純將 tensor materialize 為獨立 Hailo Conv layer 之輸出，並不足以解除此問題。所有 candidate 之 identity Conv 皆保留於 Hailo graph 中，未被 optimizer fuse 消除。

## 8. Candidate 5：成功之 workaround

AWN 之 SE-attention 結構本質為：`x → SE MLP → sigmoid gate → gate × x`。

在原始 graph 中，SE gate 分支與 passthrough x 分支共用同一個 pooling output tensor，因此 x 同時為 gate 分支之 ancestor，並於後續 Mul 節點處與 gate 分支重新匯合，形成 ancestor-descendant reconvergence 結構。

Candidate 5 之作法為複製此段 deterministic pooling block。原始結構為單一 pooling output 分別接往 SE gate 與 passthrough 兩路，最終於 Mul 匯合；改寫後之結構為由共用之 wavelet output 分別接往兩組獨立之 pooling block（pooling_A、pooling_B），產生 x_A 與 x_B 兩個獨立 tensor，x_A 接往 SE gate，x_B 直接作為 Mul 之 passthrough 輸入。

pooling_A 與 pooling_B 使用相同輸入、相同之 deterministic 運算，不含 learned weights，不含 randomness，不含 state，因此 x_A 與 x_B 理論上應數值完全相同。實測結果為 max_abs_diff = 0.0，驗證此一預期。

Candidate 5 之 semantic validation 結果如下：

- 原始 deployment ONNX 與 Candidate 5 之 logits：exactly 0.0 diff
- Mul output：0.0 diff
- x_A 與 x_B：0.0 diff
- 102/102 prediction agreement
- 46 個 initializer 逐一 byte-for-byte 完全相同
- learned weights 無任何修改
- SE gate 保留，Mul 保留，classifier 結構完整，輸出仍為 11 類別

PyTorch AWN 與 Candidate 5 之比較結果為：logits 最大絕對誤差約 3.05e-5，中間張量最大絕對誤差約 8.9e-7，102/102 prediction agreement。此差異來自 PyTorch 之未融合 Conv + BatchNorm 與 ONNX 中已融合 Conv 之 FP32 累加順序不同，屬既有已知之浮點誤差範圍，非正確性問題。

## 9. Hailo compile 最終成功

Candidate 5 於兩版本 DFC 之 compile 結果如下：

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

以 DFC 3.33.0 產生之 HEF 為例，透過 `hailortcli parse-hef` 讀取之實際 metadata 如下：

- target architecture：HAILO8
- network：awn_2016_10a_c5
- single network，single context
- input vstream：`input_layer1`，UINT8，NHWC，shape (2, 128, 1)
- output vstream：`conv10`，UINT8，FCR，shape (1, 1, 11)

物理層 stream 因 Hailo 硬體對齊機制，輸出層存在 16-channel padding；邏輯層 vstream 已正確還原為 11 類別輸出，未見任何非預期之額外輸出。

## 11. 目前尚未完成的部分

目前 HEF 已成功產生，但尚未在 Raspberry Pi Hailo-8 實機上完成 NPU inference validation。

原因為此 HEF 目前位於辦公室環境，尚未傳回 Raspberry Pi。

因此目前尚不能宣稱：

- NPU accuracy 已驗證
- CPU / NPU prediction agreement 已驗證
- NPU latency 已取得
- end-to-end sensing / attack / Top-K NPU pipeline 已完成

上述項目皆列為下一階段工作。

## 12. 下一步

1. 將 DFC 3.33.0 HEF 傳至 Raspberry Pi
2. 執行 `hailortcli scan` 與 `hailortcli parse-hef` 查核裝置與 HEF 狀態
3. 讀取實際 quantization metadata
4. 建立正確之 host-side IQ quantization / dequantization 流程
5. 先以單一已知樣本（QPSK@18）進行測試
6. 使用固定之 220 筆 RadioML 樣本，比較 CPU AWN accuracy、NPU AWN accuracy、prediction agreement 與 class-wise accuracy
7. 量測 NPU inference latency
8. 最後整合至 Spectrum Sensing → Hailo AWN AMC → adversarial attack → Top-K defense → defended inference 之完整流程

特別說明：HEF 之 input vstream 為 UINT8，不代表可直接將 float32 IQ 資料以 `astype(uint8)` 轉換。必須依據 HailoRT 所提供之 quantization metadata，進行正確之 host-side input transform，避免因量化方式錯誤而導致推論結果失真。

## 13. 本週結論

本週已完成完整 AWN deployment graph 之 Hailo compatibility adaptation，並將原本無法編譯之 native allocator crash，從完整模型逐步定位至 SE-attention gating 之 graph reconvergence 結構。

經多組數學等價之 graph rewrite 實驗，依序排除 reshape 重複、tensor materialization、Mul channel width 等可能因素後，透過複製 deterministic pooling branch，使兩條 gating path 在 graph provenance 上彼此獨立，同時維持數值完全一致，最終於 DFC 3.33.0 與 DFC 3.34.0 皆成功產生 Hailo-8 HEF。Candidate 5 已通過完整之 semantic validation，與原始 deployment ONNX 數值完全等價。

目前剩餘工作為 Raspberry Pi Hailo-8 實機之 accuracy 與 latency validation，以及後續整合至 sensing 與 adversarial attack defense pipeline。此處需說明，目前尚無法宣稱 Hailo compiler 之 root cause 已完全確定，實驗 evidence 強烈支持此為 graph-provenance / reconvergence-specific 之 allocator issue。
