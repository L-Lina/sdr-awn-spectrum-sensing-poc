# Threat Model Figures

這個目錄包含四張可用於 NDSS 論文與研究提案的威脅模型圖。圖中文字採英文，說明與使用方式採繁體中文。

## 檔案

| 圖 | SVG 向量檔 | PNG 預覽 | 建議用途 |
|---|---|---|---|
| Figure 1 | `fig1_pipeline_threat_model.svg` | `fig1_pipeline_threat_model.png` | System Model／Threat Model 主圖 |
| Figure 2 | `fig2_practical_rf_attacks.svg` | `fig2_practical_rf_attacks.png` | 實際頻譜感知攻擊機制 |
| Figure 3 | `fig3_attacker_tiers.svg` | `fig3_attacker_tiers.png` | 攻擊者能力與實驗證據邊界 |
| Figure 4 | `fig4_validation_ladder.svg` | `fig4_validation_ladder.png` | Evaluation Methodology／Deployment Validation |

`index.html` 是四張圖的本機預覽頁。`generate_figures.py` 是可重現的圖形產生器；請修改產生器後重新執行，不要只手動編輯產生的 SVG。

## 重新產生

```bash
cd docs/figures/threat_model
python3 generate_figures.py

# 選用：產生高解析 PNG 預覽
for f in fig*.svg; do
  convert -background white "$f" -resize 2400x -depth 8 "${f%.svg}.png"
done
```

## LaTeX 使用方式

SVG 是主要的向量版本。若投稿工具鏈支援 `svg` package：

```latex
\usepackage{svg}

\begin{figure*}[t]
  \centering
  \includesvg[width=\textwidth]{figures/fig1_pipeline_threat_model}
  \caption{...}
  \label{fig:threat-model}
\end{figure*}
```

正式投稿前應將字型嵌入並輸出裁切過的向量 PDF。不要用 PNG 版本作為最終論文圖，除非投稿工具鏈無法處理向量格式。

## 建議 Caption

### Figure 1 — End-to-end threat model

> Threat model for the energy-triggered AMC monitoring pipeline. An A1 attacker injects an additive RF waveform across the trust boundary into a defender-controlled receiver processing one pre-channelized complex-baseband channel. Numbered points show downstream attacks on noise estimation, event formation, window selection, AMC/defense decisions, evidence, and service availability. The attacker cannot modify the host or stored state and is not assumed to achieve phase-synchronized cancellation.

### Figure 2 — Practical RF attack mechanisms

> Practical attacks on energy-triggered RF event formation. Noise-floor poisoning raises the activity threshold and may hide a weak burst without destructive interference; short transmissions generate phantom events; a bridge merges otherwise separate events; prefixes and suffixes extend inferred boundaries; and a localized high-energy pulse hijacks max-energy window selection. All mechanisms use additive RF transmission and do not assume perfect cancellation of the legitimate signal.

### Figure 3 — Attacker tiers

> Attacker tiers and capability boundaries. A0 directly modifies offline IQ tensors and provides only an algorithmic upper bound. A1, the primary practical attacker, transmits an additive waveform using an independent SDR under power, bandwidth, timing, and channel constraints. A2 uses the same physical interface with reduced knowledge and limited observable feedback. The receiver host, model checkpoint, metadata, and evidence store remain defender controlled.

### Figure 4 — Validation ladder

> Validation ladder and claim boundaries. Digital tensor injection and cfile replay validate optimization and end-to-end software behavior, respectively. Independent cabled SDRs provide the first physical-attack evidence, while shielded OTA adds antenna and wireless-channel effects. Advisory shadow deployment measures receive-only operational behavior but does not establish causal attack effectiveness or autonomous-control safety.

## 科學性界線

- 目前系統應描述為「預先通道化複數基頻串流上的能量觸發 AMC」。
- Figure 1 不代表已實作 wideband channelizer、多通道 occupancy map、demodulation、CRC 或 autonomous control plane。
- Figure 2 的攻擊均以加性 RF 能量為主，不假設一般附近攻擊者可以精確相消合法訊號。
- Figure 3 的 A0、A1、A2 是不同的存取與知識模型，不是單調增加的攻擊強度。
- Figure 4 中只有 cabled dual-SDR 與 shielded OTA 可以支持實體攻擊可行性；shadow deployment 是營運證據，不是更強的主動攻擊證據。
