# 由正式採用之 evidence 直接讀取數值並產生 meeting_20260928 的三張圖。
# 不硬編碼任何結果數值；每張圖同時輸出實際用於繪圖的精簡 CSV 至 tables/figure_sources/。
# 執行環境：Windows PowerShell 5.1（.NET System.Drawing）。
#   powershell -NoProfile -ExecutionPolicy Bypass -File reports/meeting_20260928/analysis/make_figures.ps1
#   只產生指定圖：... make_figures.ps1 -Figures fig1

param([string[]]$Figures = @('fig1', 'fig2', 'fig3'))

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$Inv = [Globalization.CultureInfo]::InvariantCulture
$ReportDir = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent (Split-Path -Parent $ReportDir)
$Ev = Join-Path $RepoRoot 'meeting_20260928_evidence\results'
$FigDir = Join-Path $ReportDir 'figures'
$SrcDir = Join-Path $ReportDir 'tables\figure_sources'
New-Item -ItemType Directory -Force $FigDir | Out-Null
New-Item -ItemType Directory -Force $SrcDir | Out-Null

function P([string]$s) { return [double]::Parse($s, $Inv) }
function F([double]$v, [string]$fmt) { return $v.ToString($fmt, $Inv) }
function ReadJson([string]$path) { return (Get-Content -Raw -Encoding UTF8 $path | ConvertFrom-Json) }

$FontName = 'Microsoft JhengHei UI'
function NewFont([float]$size, [bool]$bold = $false) {
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    return New-Object System.Drawing.Font($FontName, $size, $style, [System.Drawing.GraphicsUnit]::Pixel)
}
function Col([string]$hex) { return [System.Drawing.ColorTranslator]::FromHtml($hex) }
function NewCanvas([int]$w, [int]$h) {
    $bmp = New-Object System.Drawing.Bitmap($w, $h)
    $bmp.SetResolution(150, 150)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $g.Clear([System.Drawing.Color]::White)
    return @($bmp, $g)
}
function Txt($g, [string]$s, $font, $color, [float]$x, [float]$y, [string]$align = 'left', [string]$valign = 'top') {
    $sf = New-Object System.Drawing.StringFormat
    $sf.Alignment = switch ($align) { 'center' { 'Center' } 'right' { 'Far' } default { 'Near' } }
    $sf.LineAlignment = switch ($valign) { 'middle' { 'Center' } 'bottom' { 'Far' } default { 'Near' } }
    $b = New-Object System.Drawing.SolidBrush($color)
    $g.DrawString($s, $font, $b, $x, $y, $sf)
    $b.Dispose(); $sf.Dispose()
}
function Line($g, $color, [float]$w, [float]$x1, [float]$y1, [float]$x2, [float]$y2, [bool]$dash = $false) {
    $p = New-Object System.Drawing.Pen($color, $w)
    if ($dash) { $p.DashStyle = [System.Drawing.Drawing2D.DashStyle]::Dash }
    $g.DrawLine($p, $x1, $y1, $x2, $y2); $p.Dispose()
}
function Rect($g, $color, [float]$x, [float]$y, [float]$w, [float]$h) {
    $b = New-Object System.Drawing.SolidBrush($color)
    if ($h -lt 0) { $y = $y + $h; $h = -$h }
    $g.FillRectangle($b, $x, $y, $w, $h); $b.Dispose()
}
function Dot($g, $color, [float]$x, [float]$y, [float]$r) {
    $b = New-Object System.Drawing.SolidBrush($color)
    $g.FillEllipse($b, $x - $r, $y - $r, 2 * $r, 2 * $r); $b.Dispose()
}
function NiceStep([double]$range) {
    $raw = $range / 6.0
    $mag = [math]::Pow(10, [math]::Floor([math]::Log10($raw)))
    foreach ($m in 1, 2, 2.5, 5, 10) { if ($raw -le $m * $mag) { return $m * $mag } }
    return 10 * $mag
}

$Ink = Col '#222222'; $Grid = Col '#DDDDDD'; $Axis = Col '#555555'; $Muted = Col '#666666'

# ------------------------------------------------------------------
# Figure 1：Phase 1d3 MKLDNN ON／OFF，B=1 攻擊延遲（cell 中位數之配對）
# ------------------------------------------------------------------
if ($Figures -contains 'fig1') {
$d3 = Join-Path $Ev 'accel_phase1d3_mkldnn_b1_20260927_135420'
$pairs = Import-Csv -Encoding UTF8 (Join-Path $d3 'paired_comparisons.csv')
$val1 = ReadJson (Join-Path $d3 'validation.json')

$rows1 = @()
foreach ($p in $pairs) {
    $rows1 += [pscustomobject]@{
        record = 'cell_pair'; attack = $p.attack.ToUpper(); pair_id = $p.pair_id; pair_valid = $p.pair_valid
        mkldnn_on_ms = F (P $p.on_median_ms) 'F4'; mkldnn_off_ms = F (P $p.off_median_ms) 'F4'
        delta_off_minus_on_ms = F (P $p.delta_ms_off_minus_on) 'F4'
        n = ''; sign_test_p = ''
        source = 'accel_phase1d3_mkldnn_b1_20260927_135420/paired_comparisons.csv'
    }
}
foreach ($atk in 'pgd', 'bim') {
    $on = $val1.per_condition."$atk/ON"; $off = $val1.per_condition."$atk/OFF"; $dec = $val1.decisions.$atk
    $rows1 += [pscustomobject]@{
        record = 'pooled_median'; attack = $atk.ToUpper(); pair_id = ''; pair_valid = ''
        mkldnn_on_ms = F ([double]$on.b1_latency_ms_median) 'F4'; mkldnn_off_ms = F ([double]$off.b1_latency_ms_median) 'F4'
        delta_off_minus_on_ms = F ([double]$dec.paired_median_delta_ms) 'F4'
        n = "$($on.b1_latency_ms_n)/$($off.b1_latency_ms_n) calls; $($dec.n_valid_pairs) pairs"; sign_test_p = F ([double]$dec.sign_test_two_sided_p) 'G4'
        source = 'accel_phase1d3_mkldnn_b1_20260927_135420/validation.json'
    }
}
$rows1 | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $SrcDir 'fig1_attack_acceleration.csv')

function Diamond($g, $fill, $edge, [float]$x, [float]$y, [float]$r) {
    $pts = [System.Drawing.PointF[]]@(
        (New-Object System.Drawing.PointF($x, ($y - $r))), (New-Object System.Drawing.PointF(($x + $r), $y)),
        (New-Object System.Drawing.PointF($x, ($y + $r))), (New-Object System.Drawing.PointF(($x - $r), $y)))
    $b = New-Object System.Drawing.SolidBrush($fill); $g.FillPolygon($b, $pts); $b.Dispose()
    $p = New-Object System.Drawing.Pen($edge, 2); $g.DrawPolygon($p, $pts); $p.Dispose()
}

$W = 1800; $H = 1100
$c = NewCanvas $W $H; $bmp = $c[0]; $g = $c[1]
$nPairs = @{}; foreach ($atk in 'pgd', 'bim') { $nPairs[$atk] = @($pairs | Where-Object { $_.attack -eq $atk }).Count }
Txt $g '圖 1　B=1 攻擊生成延遲：MKLDNN ON 與 OFF（僅作用於 attack call）' (NewFont 34 $true) $Ink 60 30
Txt $g ('Raspberry Pi 5，PGD／BIM ε = 0.03，88 樣本子集，熱控制（全部 cell 為 THERMAL_CLEAN）；PGD ' + $nPairs['pgd'] + ' 對、BIM ' + $nPairs['bim'] + ' 對相鄰 ON／OFF cell') (NewFont 22) $Muted 60 82

$allv = @(); foreach ($p in $pairs) { $allv += (P $p.on_median_ms); $allv += (P $p.off_median_ms) }
$ymin = [math]::Floor(([double]($allv | Measure-Object -Minimum).Minimum) - 1.5)
$ymax = [math]::Ceiling(([double]($allv | Measure-Object -Maximum).Maximum) + 1.5)
$ColOn = Col '#4C72B0'; $ColOff = Col '#DD8452'; $ColPair = Col '#9A9A9A'; $ColPool = Col '#111111'
$panels = @(@{atk = 'pgd'; x0 = 150 }, @{atk = 'bim'; x0 = 990 })
$top = 170; $bot = 860; $pw = 680
$jit = 9.0   # 水平 jitter（像素）：僅用於視覺分離，不改變任何 y 值，也不寫回任何 CSV
foreach ($pn in $panels) {
    $x0 = $pn.x0; $atk = $pn.atk
    $ymap = { param($v) $bot - ($v - $ymin) / ($ymax - $ymin) * ($bot - $top) }
    for ($t = [math]::Ceiling($ymin); $t -le $ymax; $t += 1.0) {
        $yy = & $ymap $t
        Line $g $Grid 1 $x0 $yy ($x0 + $pw) $yy
        Txt $g (F $t 'F0') (NewFont 20) $Axis ($x0 - 12) $yy 'right' 'middle'
    }
    Line $g $Axis 2 $x0 $top $x0 $bot; Line $g $Axis 2 $x0 $bot ($x0 + $pw) $bot
    $xa = $x0 + $pw * 0.36; $xb = $x0 + $pw * 0.64
    Txt $g 'MKLDNN ON' (NewFont 24 $true) $ColOn $xa ($bot + 14) 'center'
    Txt $g 'MKLDNN OFF' (NewFont 24 $true) $ColOff $xb ($bot + 14) 'center'
    Txt $g $atk.ToUpper() (NewFont 30 $true) $Ink ($x0 + $pw / 2) ($top - 50) 'center'
    # 個別 paired cell：依 pair_id 排序後給定固定的水平位移（deterministic），細線與小點
    $ap = @($pairs | Where-Object { $_.attack -eq $atk } | Sort-Object pair_id)
    for ($i = 0; $i -lt $ap.Count; $i++) {
        $dx = ($i - ($ap.Count - 1) / 2.0) * $jit
        $y1 = & $ymap (P $ap[$i].on_median_ms); $y2 = & $ymap (P $ap[$i].off_median_ms)
        Line $g $ColPair 1.2 ($xa + $dx) $y1 ($xb + $dx) $y2
        Dot $g $ColOn ($xa + $dx) $y1 4
        Dot $g $ColOff ($xb + $dx) $y2 4
    }
    # 合併中位：以菱形標記置於個別 cell 群組外側，與個別觀測分開
    $on = [double]$val1.per_condition."$atk/ON".b1_latency_ms_median
    $off = [double]$val1.per_condition."$atk/OFF".b1_latency_ms_median
    $dec = $val1.decisions.$atk
    $xpa = $xa - 80; $xpb = $xb + 80
    Diamond $g $ColPool (Col '#FFFFFF') $xpa (& $ymap $on) 11
    Diamond $g $ColPool (Col '#FFFFFF') $xpb (& $ymap $off) 11
    Txt $g ((F $on 'F2') + ' ms') (NewFont 21 $true) $Ink $xpa ((& $ymap $on) - 16) 'center' 'bottom'
    Txt $g ((F $off 'F2') + ' ms') (NewFont 21 $true) $Ink ($xpb + 18) (& $ymap $off) 'left' 'middle'
    $ann = '配對差中位 ' + (F ([double]$dec.paired_median_delta_ms) 'F2') + ' ms；OFF 較快 ' + $dec.n_pairs_off_faster + '/' + $dec.n_valid_pairs + ' 對；sign test p = ' + (F ([double]$dec.sign_test_two_sided_p) 'G4')
    Txt $g $ann (NewFont 21) $Ink ($x0 + $pw / 2) ($top - 8) 'center'
}
Txt $g '延遲 (ms)' (NewFont 22) $Axis 40 ($top - 52)
# 圖例
$ly = 930
Line $g $ColPair 1.2 150 $ly 200 $ly; Dot $g $ColOn 150 $ly 4; Dot $g $ColOff 200 $ly 4
Txt $g 'paired cell：每一對相鄰 ON／OFF cell 的中位延遲（paired_comparisons.csv）' (NewFont 21) $Ink 218 $ly 'left' 'middle'
Diamond $g $ColPool (Col '#FFFFFF') 175 ($ly + 36) 11
Txt $g 'pooled median：所有 accepted cell 的合併中位（validation.json）' (NewFont 21) $Ink 218 ($ly + 36) 'left' 'middle'
Txt $g '個別 paired cell 之水平位移（jitter）僅供視覺辨識，不改變任何量測值；pooled median 標記置於群組外側以便區分，x 方向位置不具數值意義。' (NewFont 20) $Muted 60 1000
Txt $g 'MKLDNN OFF 只作用於 B=1 的 PGD／BIM attack call；此圖為 benchmark 路徑結果，不代表正式 CPU full matrix 已重跑。來源：accel_phase1d3_mkldnn_b1_20260927_135420。' (NewFont 20) $Muted 60 1035
$bmp.Save((Join-Path $FigDir 'fig1_attack_acceleration.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
}

# ------------------------------------------------------------------
# Figure 2：CPU 與 Hailo-8 之 defense trade-off（24 profile 合併）
# ------------------------------------------------------------------
if ($Figures -contains 'fig2') {
$cpuCsv = Join-Path $Ev 'adaptive_k_effectiveness_20260927_164906\defense_summary.csv'
$hlCsv = Join-Path $Ev 'adaptive_k_hailo_paired_20260928\defense_summary_paired.csv'
$cpu = Import-Csv -Encoding UTF8 $cpuCsv | Where-Object { $_.scope -eq 'overall' }
$hl = Import-Csv -Encoding UTF8 $hlCsv | Where-Object { $_.scope -eq 'overall' }
$defs = @('topk10', 'topk20', 'topk30', 'topk40', 'adaptive_k_v2')
$defLabel = @{ topk10 = '固定 K=10'; topk20 = '固定 K=20'; topk30 = '固定 K=30'; topk40 = '固定 K=40'; adaptive_k_v2 = 'Adaptive-K v2' }

$rows2 = @()
foreach ($d in $defs) {
    $r = $cpu | Where-Object { $_.defense -eq $d }
    $rows2 += [pscustomobject]@{ backend = 'CPU'; defense = $d; clean_degradation_pct = F (P $r.clean_degradation_pct) 'F2'; recovery_pct = F (P $r.recovery_pct) 'F2'; retention_pct = F (P $r.retention_pct) 'F2'; retention_nodef_pct = F (P $r.retention_nodef_pct) 'F2'; retention_gain_pp = F (P $r.retention_gain_pp) 'F2'; source = 'adaptive_k_effectiveness_20260927_164906/defense_summary.csv (scope=overall)' }
}
foreach ($d in $defs) {
    $r = $hl | Where-Object { $_.defense -eq $d }
    $rows2 += [pscustomobject]@{ backend = 'Hailo-8'; defense = $d; clean_degradation_pct = F (P $r.hailo_clean_degradation_pct) 'F2'; recovery_pct = F (P $r.hailo_recovery_pct) 'F2'; retention_pct = F (P $r.hailo_retention_pct) 'F2'; retention_nodef_pct = F (P $r.hailo_retention_nodef_pct) 'F2'; retention_gain_pp = F (P $r.hailo_retention_gain_pp) 'F2'; source = 'adaptive_k_hailo_paired_20260928/defense_summary_paired.csv (scope=overall, hailo_*)' }
}
$rows2 | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $SrcDir 'fig2_adaptive_k_tradeoff.csv')

$W = 2000; $H = 1150
$c = NewCanvas $W $H; $bmp = $c[0]; $g = $c[1]
Txt $g '圖 2　固定 K 與 Adaptive-K v2 之防禦 trade-off（24 個攻擊 profile 合併）' (NewFont 34 $true) $Ink 60 30
Txt $g '左：CPU（PyTorch AWN，白箱攻擊）；右：Hailo-8（相同對抗 IQ，surrogate transfer）。兩個 backend 的數值不可合併比較。' (NewFont 22) $Muted 60 82
$metrics = @(
    @{ key = 'clean_degradation_pct'; label = 'Clean degradation (%)'; color = Col '#8C8C8C' },
    @{ key = 'recovery_pct'; label = 'Recovery (%)'; color = Col '#4C72B0' },
    @{ key = 'retention_gain_pp'; label = 'Retention 淨增益 (pp)'; color = Col '#55A868' }
)
$vals = @(); foreach ($r in $rows2) { foreach ($m in $metrics) { $vals += (P $r.($m.key)) } }
$vmin = [math]::Min(0, [double]($vals | Measure-Object -Minimum).Minimum); $vmax = [double]($vals | Measure-Object -Maximum).Maximum
$step = NiceStep ($vmax - $vmin)
$vmin = [math]::Floor($vmin / $step) * $step; $vmax = [math]::Ceiling($vmax / $step) * $step
$top = 200; $bot = 900; $pw = 860
foreach ($pn in @(@{b = 'CPU'; x0 = 140; t = 'CPU（PyTorch AWN）' }, @{b = 'Hailo-8'; x0 = 1090; t = 'Hailo-8（surrogate transfer）' })) {
    $x0 = $pn.x0
    $ymap = { param($v) $bot - ($v - $vmin) / ($vmax - $vmin) * ($bot - $top) }
    for ($t = $vmin; $t -le $vmax + 1e-9; $t += $step) {
        $yy = & $ymap $t
        Line $g $Grid 1 $x0 $yy ($x0 + $pw) $yy
        Txt $g (F $t 'F0') (NewFont 20) $Axis ($x0 - 12) $yy 'right' 'middle'
    }
    Line $g $Axis 2 $x0 $top $x0 $bot
    Line $g $Ink 2 $x0 (& $ymap 0) ($x0 + $pw) (& $ymap 0)
    Txt $g $pn.t (NewFont 28 $true) $Ink ($x0 + $pw / 2) ($top - 60) 'center'
    $gw = $pw / $defs.Count; $bw = $gw * 0.24
    for ($i = 0; $i -lt $defs.Count; $i++) {
        $r = $rows2 | Where-Object { $_.backend -eq $pn.b -and $_.defense -eq $defs[$i] }
        $gx = $x0 + $i * $gw + ($gw - 3 * $bw) / 2
        for ($j = 0; $j -lt 3; $j++) {
            $v = P $r.($metrics[$j].key)
            $bx = $gx + $j * $bw
            $y0 = & $ymap 0; $y1 = & $ymap $v
            Rect $g $metrics[$j].color $bx ([math]::Min($y0, $y1)) ($bw - 4) ([math]::Abs($y1 - $y0))
            $ly = if ($v -ge 0) { $y1 - 4 } else { $y1 + 4 }
            $va = if ($v -ge 0) { 'bottom' } else { 'top' }
            Txt $g (F $v 'F1') (NewFont 16) $Ink ($bx + ($bw - 4) / 2) $ly 'center' $va
        }
        Txt $g $defLabel[$defs[$i]] (NewFont 21) $Ink ($x0 + $i * $gw + $gw / 2) ($bot + 12) 'center'
    }
}
$lx = 140
foreach ($m in $metrics) { Rect $g $m.color $lx 965 26 26; Txt $g $m.label (NewFont 22) $Ink ($lx + 36) 978 'left' 'middle'; $lx += 380 }
Txt $g 'Retention 淨增益 = 含防禦之 retention − 無防禦 retention。各方法之相對排序在 CPU 與 Hailo-8 上不同；Adaptive-K 並非在所有指標或 backend 上皆優於固定 K。' (NewFont 20) $Muted 60 1020
Txt $g '來源：adaptive_k_effectiveness_20260927_164906/defense_summary.csv；adaptive_k_hailo_paired_20260928/defense_summary_paired.csv（scope = overall）。A0 digital 攻擊，不延伸至 OTA／RF。' (NewFont 20) $Muted 60 1055
$bmp.Save((Join-Path $FigDir 'fig2_adaptive_k_tradeoff.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
}

# ------------------------------------------------------------------
# Figure 3：formal_latency_20260928d 直接量測之延遲（measured only）
# ------------------------------------------------------------------
if ($Figures -contains 'fig3') {
$fl = Join-Path $Ev 'formal_latency_20260928d'
$pm = Import-Csv -Encoding UTF8 (Join-Path $fl 'primary_metrics_measured.csv')
$val3 = ReadJson (Join-Path $fl 'validation.json')
$sel = @(
    @{ comp = 'G_clean_pipeline_evaluation_path_measured_ms'; backend = 'cpu'; defense = 'none'; label = 'Clean pipeline：CPU，無防禦' },
    @{ comp = 'G_clean_pipeline_evaluation_path_measured_ms'; backend = 'cpu'; defense = 'adaptive_k_v2'; label = 'Clean pipeline：CPU，Adaptive-K' },
    @{ comp = 'G_clean_pipeline_evaluation_path_measured_ms'; backend = 'hailo'; defense = 'none'; label = 'Clean pipeline：Hailo-8，無防禦' },
    @{ comp = 'G_clean_pipeline_evaluation_path_measured_ms'; backend = 'hailo'; defense = 'adaptive_k_v2'; label = 'Clean pipeline：Hailo-8，Adaptive-K' },
    @{ comp = 'E_cpu_awn_adapter_ms'; backend = 'cpu'; defense = 'none'; label = '分類器 adapter：CPU' },
    @{ comp = 'F_hailo_adapter_roundtrip_ms'; backend = 'hailo'; defense = 'none'; label = '分類器 adapter：Hailo-8' }
)
$rows3 = @()
foreach ($s in $sel) {
    $r = $pm | Where-Object { $_.component -eq $s.comp -and $_.backend -eq $s.backend -and $_.defense -eq $s.defense }
    if (@($r).Count -ne 1) { throw "primary_metrics_measured.csv row not unique: $($s.comp) $($s.backend) $($s.defense)" }
    if ($r.metric_class -ne 'measured') { throw "non-measured row selected: $($s.comp)" }
    $rows3 += [pscustomobject]@{ label = $s.label; component = $s.comp; metric_class = $r.metric_class; backend = $s.backend; defense = $s.defense; configuration_role = $r.configuration_role; n = $r.n; median_ms = F (P $r.median_ms) 'F4'; p05_ms = F (P $r.p05_ms) 'F4'; p95_ms = F (P $r.p95_ms) 'F4'; source = 'formal_latency_20260928d/primary_metrics_measured.csv' }
}
$rows3 | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $SrcDir 'fig3_formal_latency.csv')

$W = 1900; $H = 1000
$c = NewCanvas $W $H; $bmp = $c[0]; $g = $c[1]
Txt $g '圖 3　CPU 與 Hailo-8 之直接量測延遲（PRIMARY_FORMAL，evaluation path）' (NewFont 34 $true) $Ink 60 30
$acc = "$($val3.info.run.accepted)/$($val3.info.run.planned)"
Txt $g ("formal_latency_20260928d：accepted cells " + $acc + "，overall = " + $val3.overall + "；B=1；不含攻擊生成時間。長條為中位數，橫線為 p05–p95 範圍（非信賴區間）。") (NewFont 22) $Muted 60 82
$xmax = [math]::Ceiling([double](($rows3 | ForEach-Object { P $_.p95_ms }) | Measure-Object -Maximum).Maximum + 0.5)
$left = 620; $right = 1780; $top = 170; $rowh = 105
$xmap = { param($v) $left + $v / $xmax * ($right - $left) }
for ($t = 0; $t -le $xmax; $t += 1) {
    $xx = & $xmap $t
    Line $g $Grid 1 $xx $top $xx ($top + $rowh * 6 + 30)
    Txt $g (F $t 'F0') (NewFont 20) $Axis $xx ($top + $rowh * 6 + 38) 'center'
}
Txt $g '延遲 (ms)' (NewFont 22) $Axis (($left + $right) / 2) ($top + $rowh * 6 + 72) 'center'
$ColCpu = Col '#4C72B0'; $ColHl = Col '#DD8452'
for ($i = 0; $i -lt $rows3.Count; $i++) {
    $r = $rows3[$i]
    $yc = $top + $i * $rowh + $rowh / 2 + $(if ($i -ge 4) { 30 } else { 0 })
    $colr = if ($r.backend -eq 'cpu') { $ColCpu } else { $ColHl }
    $med = P $r.median_ms
    Rect $g $colr $left ($yc - 28) ((& $xmap $med) - $left) 56
    Line $g $Ink 3 (& $xmap (P $r.p05_ms)) $yc (& $xmap (P $r.p95_ms)) $yc
    Line $g $Ink 3 (& $xmap (P $r.p05_ms)) ($yc - 12) (& $xmap (P $r.p05_ms)) ($yc + 12)
    Line $g $Ink 3 (& $xmap (P $r.p95_ms)) ($yc - 12) (& $xmap (P $r.p95_ms)) ($yc + 12)
    Txt $g $r.label (NewFont 23) $Ink ($left - 16) $yc 'right' 'middle'
    $mfmt = if ($med -lt 1) { 'F3' } else { 'F2' }
    Txt $g ((F $med $mfmt) + ' ms') (NewFont 21 $true) $Ink ((& $xmap (P $r.p95_ms)) + 12) $yc 'left' 'middle'
}
Line $g $Axis 2 $left $top $left ($top + $rowh * 6 + 30)
Txt $g '所有數值皆為 metric_class = measured；evaluation-path 延遲包含 evaluation 專用計算，不等同 deployment latency。推導之 deployment estimate 與 HailoRT 輔助量測未納入本圖。' (NewFont 20) $Muted 60 900
Txt $g '來源：formal_latency_20260928d/primary_metrics_measured.csv；accepted 數與 overall 取自 validation.json（全部 accepted cell 為 THERMAL_CLEAN）。' (NewFont 20) $Muted 60 935
$bmp.Save((Join-Path $FigDir 'fig3_formal_latency.png'), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
}

Write-Output ('figures written: ' + ($Figures -join ', '))
