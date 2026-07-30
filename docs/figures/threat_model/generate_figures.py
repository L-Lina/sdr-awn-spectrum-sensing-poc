#!/usr/bin/env python3
"""Generate publication-ready SVG threat-model figures.

The figures intentionally describe the current system as energy-triggered AMC on
one pre-channelized complex-baseband stream.  They do not depict an implemented
wideband channelizer or autonomous control plane.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Iterable, Sequence

OUT = Path(__file__).resolve().parent

INK = "#1f2937"
MUTED = "#667085"
LIGHT = "#f2f4f7"
LIGHTER = "#f8fafc"
GRID = "#d0d5dd"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#7E57C2"
RED = "#C62828"
WHITE = "#ffffff"


class SVG:
    def __init__(self, width: int, height: int, title: str, desc: str):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>",
            f"<desc id=\"desc\">{escape(desc)}</desc>",
            """<defs>
  <marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,4 L0,8 z" fill="#1f2937"/>
  </marker>
  <marker id="arrow-red" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L10,4 L0,8 z" fill="#C62828"/>
  </marker>
  <pattern id="hatch" width="8" height="8" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
    <rect width="8" height="8" fill="#fff5ef"/>
    <line x1="0" y1="0" x2="0" y2="8" stroke="#D55E00" stroke-width="2"/>
  </pattern>
  <style>
    text { font-family: Arial, Helvetica, sans-serif; fill: #1f2937; }
    .title { font-size: 26px; font-weight: 700; }
    .subtitle { font-size: 14px; fill: #667085; }
    .section { font-size: 14px; font-weight: 700; letter-spacing: .7px; }
    .head { font-size: 14px; font-weight: 700; }
    .body { font-size: 12px; }
    .small { font-size: 10.5px; }
    .tiny { font-size: 9px; }
    .italic { font-style: italic; }
  </style>
</defs>""",
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="white"/>',
        ]

    def add(self, raw: str) -> None:
        self.parts.append(raw)

    def rect(self, x, y, w, h, *, fill=WHITE, stroke=INK, sw=1.5, rx=8, dash=None) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.add(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash_attr}/>'
        )

    def line(self, x1, y1, x2, y2, *, stroke=INK, sw=1.5, dash=None, arrow=False, red_arrow=False) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow-red)"' if red_arrow else (' marker-end="url(#arrow)"' if arrow else "")
        self.add(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{sw}"{dash_attr}{marker}/>'
        )

    def path(self, d, *, fill="none", stroke=INK, sw=1.5, dash=None, arrow=False, red_arrow=False) -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        marker = ' marker-end="url(#arrow-red)"' if red_arrow else (' marker-end="url(#arrow)"' if arrow else "")
        self.add(
            f'<path d="{d}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round"{dash_attr}{marker}/>'
        )

    def circle(self, cx, cy, r, *, fill=WHITE, stroke=INK, sw=1.5) -> None:
        self.add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

    def text(self, x, y, lines: str | Sequence[str], *, cls="body", anchor="start", fill=None, weight=None,
             line_height=15, rotate=None) -> None:
        if isinstance(lines, str):
            lines = [lines]
        attrs = [f'x="{x}"', f'y="{y}"', f'class="{cls}"', f'text-anchor="{anchor}"']
        if fill:
            attrs.append(f'fill="{fill}"')
        if weight:
            attrs.append(f'font-weight="{weight}"')
        if rotate is not None:
            attrs.append(f'transform="rotate({rotate} {x} {y})"')
        tspans = []
        for i, line in enumerate(lines):
            dy = 0 if i == 0 else line_height
            tspans.append(f'<tspan x="{x}" dy="{dy}">{escape(str(line))}</tspan>')
        self.add(f'<text {" ".join(attrs)}>{"".join(tspans)}</text>')

    def save(self, name: str) -> None:
        self.parts.append("</svg>")
        (OUT / name).write_text("\n".join(self.parts) + "\n", encoding="utf-8")


def arrow_label(svg: SVG, x1, y1, x2, y2, label: str, *, color=INK, red=False, label_y=-8) -> None:
    svg.line(x1, y1, x2, y2, stroke=color, sw=2.1 if red else 1.7, arrow=not red, red_arrow=red)
    svg.text((x1 + x2) / 2, (y1 + y2) / 2 + label_y, label, cls="small", anchor="middle", fill=color)


def stage_box(svg: SVG, x: int, y: int, w: int, h: int, number: str, heading: str, rows: Sequence[str]) -> None:
    svg.rect(x, y, w, h, fill=WHITE, stroke=INK, sw=1.5, rx=6)
    svg.add(f'<rect x="{x}" y="{y}" width="{w}" height="34" rx="6" fill="{LIGHT}"/>')
    svg.line(x, y + 34, x + w, y + 34, stroke=GRID, sw=1)
    svg.text(x + 12, y + 22, f"{number}  {heading}", cls="head")
    row_h = (h - 34) / len(rows)
    for i, row in enumerate(rows):
        top = y + 34 + i * row_h
        if i:
            svg.line(x + 8, top, x + w - 8, top, stroke=GRID, sw=.7)
        svg.text(x + 10, top + row_h / 2 + 4, row, cls="small")


def attack_marker(svg: SVG, cx: int, cy: int, n: int, tx: int, ty: int, lines: Sequence[str]) -> None:
    svg.circle(cx, cy, 13, fill=WHITE, stroke=RED, sw=2.3)
    svg.text(cx, cy + 4, str(n), cls="small", anchor="middle", fill=RED, weight="700")
    svg.line(cx, cy + 13, tx, ty - 12, stroke=RED, sw=1.2, dash="3 3")
    svg.text(tx, ty, lines, cls="small", fill=RED, weight="700", line_height=13)


def figure1_pipeline() -> None:
    s = SVG(1400, 760, "End-to-end RF threat model",
            "A practical additive RF attacker crosses the trust boundary into an energy-triggered AMC pipeline.")
    s.text(50, 44, "Threat Model: Energy-Triggered AMC Monitoring", cls="title")
    s.text(50, 68, "Current scope: one pre-channelized complex-baseband channel; no implemented wideband occupancy inference", cls="subtitle")

    # Trust regions.
    s.text(30, 108, ["UNTRUSTED RF", "ENVIRONMENT"], cls="section", line_height=18)
    s.rect(300, 90, 1070, 610, fill=WHITE, stroke=INK, sw=2, rx=14, dash="9 6")
    s.add(f'<rect x="355" y="79" width="390" height="24" fill="white"/>')
    s.text(365, 97, "DEFENDER-CONTROLLED MONITORING NODE", cls="section")

    # External transmitters.
    s.rect(30, 145, 230, 72, fill=LIGHTER)
    s.text(145, 172, "Legitimate transmitter", cls="head", anchor="middle")
    s.text(145, 195, "burst waveform s[n]", cls="small", anchor="middle")
    s.rect(30, 275, 230, 180, fill="#fff7f5", stroke=RED, sw=2)
    s.text(145, 304, "RF attacker (A1)", cls="head", anchor="middle", fill=RED)
    s.text(50, 333, ["Independent SDR", "Additive waveform a[n]", "Controls power, bandwidth,", "timing, waveform, duty cycle", "No host access", "No precise phase synchronization"], cls="small", line_height=19)

    s.circle(280, 235, 18, fill=WHITE, stroke=INK, sw=2)
    s.text(280, 241, "+", cls="head", anchor="middle")
    s.text(280, 270, "RF channel", cls="small", anchor="middle")
    s.path("M260 181 C272 181 270 205 280 216", stroke=INK, sw=1.7, arrow=True)
    s.path("M260 324 C275 310 276 275 280 254", stroke=RED, sw=2.2, red_arrow=True)
    s.text(145, 261, "RF superposition: y[n] = h_s*s + h_a*a + w", cls="tiny", anchor="middle")

    # Pipeline stages.
    y, h = 145, 310
    stages = [
        (330, 200, "1", "RF ACQUISITION", ["Antenna / RF front end", "ADC + downconversion", "Upstream channel selection", "complex64 IQ y[n]", "Capture-health checks"]),
        (570, 245, "2", "DETECT & SEGMENT", ["Smoothed energy |y[n]|²", "Median noise floor N̂", "Threshold γ = αN̂", "Active-region extraction", "min length / merge gap", "Window selection"]),
        (855, 220, "3", "INFER & DECIDE", ["AWN preprocessing", "AMC classifier", "Top-K / Adaptive-K", "Event-level aggregation", "Confidence / abstention", "Failure reporting"]),
        (1115, 220, "4", "RECORD & TRIAGE", ["Event ID / timeline", "Label + confidence", "Defense / abstain status", "Raw IQ + guard samples", "Capture / model metadata", "Analyst queue"]),
    ]
    for x, w, num, head, rows in stages:
        stage_box(s, x, y, w, h, num, head, rows)

    s.line(298, 235, 330, 235, arrow=True)
    s.line(530, 300, 570, 300, arrow=True)
    s.line(815, 300, 855, 300, arrow=True)
    s.line(1075, 300, 1115, 300, arrow=True)

    # Attack points and callouts.
    attack_marker(s, 315, 235, 1, 320, 507, ["AP1  RF waveform injection"])
    attack_marker(s, 595, 248, 2, 455, 545, ["AP2  Noise-floor poisoning (G1)"])
    attack_marker(s, 790, 345, 3, 700, 505, ["AP3  Phantom events, bridging,", "and boundary extension (G2–G4)"])
    attack_marker(s, 790, 425, 4, 650, 585, ["AP4  Max-energy window hijacking (G5)"])
    attack_marker(s, 1048, 300, 5, 930, 535, ["AP5  AMC evasion and", "defense-router manipulation (G6–G7)"])
    attack_marker(s, 1305, 420, 6, 1135, 585, ["AP6  Evidence ambiguity and", "event / queue flooding (G2, G8)"])

    # Protected assets.
    s.rect(330, 620, 1005, 60, fill=LIGHT, stroke=INK, sw=1.2, rx=6)
    s.text(350, 644, "PROTECTED ASSETS", cls="head")
    s.text(350, 665, "Occupancy integrity  •  event-timeline integrity  •  AMC/abstention integrity  •  availability  •  evidence integrity  •  failure visibility", cls="small")

    # Legend and exclusions.
    s.line(35, 520, 95, 520, arrow=True)
    s.text(105, 524, "data flow", cls="small")
    s.line(35, 550, 95, 550, stroke=RED, sw=2.2, red_arrow=True)
    s.text(105, 554, "attacker-controlled RF", cls="small")
    s.circle(48, 582, 10, fill=WHITE, stroke=RED, sw=2)
    s.text(48, 586, "#", cls="tiny", anchor="middle", fill=RED)
    s.text(105, 586, "downstream attack point", cls="small")
    s.text(30, 642, ["Excluded: host compromise, checkpoint replacement,", "database tampering, metadata tampering, and", "perfect phase-synchronized cancellation."], cls="tiny", line_height=14)
    s.save("fig1_pipeline_threat_model.svg")


def panel_axes(s: SVG, x: int, y: int, w: int, h: int, label: str, title: str, left_axis: bool) -> tuple[int, int, int, int]:
    s.rect(x, y, w, h, fill=WHITE, stroke=GRID, sw=1.2, rx=4)
    s.text(x + 12, y + 23, label, cls="head")
    s.text(x + 46, y + 23, title, cls="head")
    px, py, pw, ph = x + 48, y + 44, w - 65, h - 77
    s.line(px, py + ph, px + pw, py + ph, stroke=INK, sw=1.2)
    s.line(px, py, px, py + ph, stroke=INK, sw=1.2)
    s.text(px + pw, py + ph + 25, "Time", cls="tiny", anchor="end")
    if left_axis:
        s.text(x + 15, py + ph / 2, "Smoothed energy p̄[n]", cls="tiny", anchor="middle", rotate=-90)
    return px, py, pw, ph


def event_bracket(s: SVG, x1: int, x2: int, y: int, label: str, *, dash=None, color=INK) -> None:
    s.path(f"M{x1} {y+8} V{y} H{x2} V{y+8}", stroke=color, sw=1.4, dash=dash)
    s.text((x1 + x2) / 2, y - 5, label, cls="tiny", anchor="middle", fill=color)


def tx_bar(s: SVG, x: int, y: int, w: int, label: str, attacker=False) -> None:
    fill = "url(#hatch)" if attacker else BLUE
    stroke = ORANGE if attacker else BLUE
    s.add(f'<rect x="{x}" y="{y}" width="{w}" height="10" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
    s.text(x + w / 2, y + 24, label, cls="tiny", anchor="middle", fill=stroke)


def figure2_attacks() -> None:
    s = SVG(1400, 900, "Practical RF attack mechanisms",
            "Six energy-over-time panels show practical additive RF attacks without assuming cancellation.")
    s.text(45, 42, "Practical Attacks on RF Event Formation", cls="title")
    s.text(45, 66, "The black trace is received smoothed energy; transmitter powers are not plotted as independently additive components.", cls="subtitle")

    coords = [(45, 100), (715, 100), (45, 365), (715, 365), (45, 630), (715, 630)]
    titles = ["Reference", "Noise-floor poisoning", "Phantom occupancy", "Event bridging", "Boundary extension", "Max-energy window hijacking"]
    panels = [panel_axes(s, x, y, 640, 230, f"({chr(97+i)})", titles[i], i % 2 == 0) for i, (x, y) in enumerate(coords)]

    # (a) reference.
    x, y, w, h = panels[0]
    threshold = y + 72
    s.line(x, threshold, x + w, threshold, stroke=INK, sw=1.2, dash="7 5")
    s.text(x + w - 5, threshold - 6, "γ", cls="tiny", anchor="end")
    s.path(f"M{x} {y+h-20} L{x+120} {y+h-18} C{x+145} {y+h-18} {x+155} {threshold-70} {x+195} {threshold-75} L{x+270} {threshold-72} C{x+300} {threshold-65} {x+310} {y+h-18} {x+335} {y+h-18} L{x+w} {y+h-20}", sw=2)
    tx_bar(s, x + 145, y + h + 5, 170, "legitimate", False)
    event_bracket(s, x + 140, x + 318, y + 10, "one legitimate event")
    s.rect(x + 175, y + 38, 105, 105, fill=LIGHT, stroke=INK, sw=1.2, rx=0)
    s.text(x + 227, y + 57, "selected", cls="tiny", anchor="middle")

    # (b) poisoning.
    x, y, w, h = panels[1]
    gamma0, gamma1 = y + 120, y + 65
    s.line(x, gamma0, x + w, gamma0, stroke=GRID, sw=1.2, dash="3 4")
    s.line(x, gamma1, x + w, gamma1, stroke=INK, sw=1.2, dash="7 5")
    s.text(x + w - 4, gamma0 - 5, "γ0 = αN0", cls="tiny", anchor="end", fill=MUTED)
    s.text(x + w - 4, gamma1 - 5, "γ1 = αN1", cls="tiny", anchor="end")
    s.path(f"M{x} {y+h-60} L{x+110} {y+h-58} C{x+145} {y+h-58} {x+158} {gamma0-30} {x+195} {gamma0-34} L{x+270} {gamma0-32} C{x+300} {gamma0-28} {x+315} {y+h-58} {x+345} {y+h-58} L{x+w} {y+h-60}", sw=2)
    tx_bar(s, x + 15, y + h + 5, w - 30, "", True)
    tx_bar(s, x + 150, y + h + 20, 165, "", False)
    s.text(x + 20, y + h + 1, "attacker: prolonged moderate power", cls="tiny", fill=ORANGE)
    s.text(x + 232, y + h + 44, "weak legitimate burst", cls="tiny", anchor="middle", fill=BLUE)
    s.text(x + 215, gamma0 - 42, "weak event missed", cls="small", anchor="middle", fill=RED, weight="700")
    s.text(x + 55, y + h - 68, "poisoned estimate", cls="tiny", fill=ORANGE)

    # (c) phantom.
    x, y, w, h = panels[2]
    threshold = y + 92
    s.line(x, threshold, x + w, threshold, stroke=INK, sw=1.2, dash="7 5")
    points = [(35, 80), (150, 70), (285, 90), (420, 65)]
    d = f"M{x} {y+h-18} "
    for off, height in points:
        d += f"L{x+off} {y+h-18} L{x+off+8} {threshold-height} L{x+off+42} {threshold-height+3} L{x+off+52} {y+h-18} "
        tx_bar(s, x + off + 7, y + h + 5, 45, "", True)
        event_bracket(s, x + off + 4, x + off + 55, y + 25, "false event")
    d += f"L{x+w} {y+h-18}"
    s.path(d, sw=2)
    s.text(x + w - 20, y + h - 35, "AMC → storage → analyst load", cls="small", anchor="end", fill=RED, weight="700")

    # (d) bridging.
    x, y, w, h = panels[3]
    threshold = y + 95
    s.line(x, threshold, x + w, threshold, stroke=INK, sw=1.2, dash="7 5")
    s.path(f"M{x} {y+h-18} L{x+70} {y+h-18} L{x+85} {threshold-55} L{x+205} {threshold-58} L{x+220} {threshold-10} L{x+280} {threshold-12} L{x+295} {threshold-60} L{x+420} {threshold-55} L{x+440} {y+h-18} L{x+w} {y+h-18}", sw=2)
    tx_bar(s, x + 85, y + h + 5, 125, "L1", False)
    tx_bar(s, x + 220, y + h + 5, 70, "bridge", True)
    tx_bar(s, x + 295, y + h + 5, 125, "L2", False)
    event_bracket(s, x + 78, x + 442, y + 18, "one merged event")
    event_bracket(s, x + 85, x + 210, y + 48, "without attack", dash="3 3", color=MUTED)
    event_bracket(s, x + 295, x + 420, y + 48, "", dash="3 3", color=MUTED)

    # (e) boundary extension.
    x, y, w, h = panels[4]
    threshold = y + 98
    s.line(x, threshold, x + w, threshold, stroke=INK, sw=1.2, dash="7 5")
    ts, te = x + 205, x + 365
    s.line(ts, y + 20, ts, y + h, stroke=MUTED, sw=1, dash="3 3")
    s.line(te, y + 20, te, y + h, stroke=MUTED, sw=1, dash="3 3")
    s.text(ts, y + 15, "t_s", cls="tiny", anchor="middle")
    s.text(te, y + 15, "t_e", cls="tiny", anchor="middle")
    s.path(f"M{x} {y+h-18} L{x+130} {y+h-18} L{x+140} {threshold-12} L{x+195} {threshold-16} L{x+205} {threshold-65} L{x+365} {threshold-62} L{x+375} {threshold-16} L{x+435} {threshold-12} L{x+445} {y+h-18} L{x+w} {y+h-18}", sw=2)
    tx_bar(s, x + 140, y + h + 5, 55, "prefix", True)
    tx_bar(s, x + 205, y + h + 5, 160, "legitimate", False)
    tx_bar(s, x + 375, y + h + 5, 60, "suffix", True)
    event_bracket(s, x + 137, x + 438, y + 27, "extended event t'_s ... t'_e")
    s.rect(x + 140, y + 55, 120, 80, fill=LIGHT, stroke=INK, sw=1.2, rx=0)
    s.text(x + 200, y + 72, "shifted window", cls="tiny", anchor="middle")
    s.rect(ts, y + 62, 120, 73, fill="none", stroke=MUTED, sw=1.1, rx=0, dash="4 3")
    s.text(ts + 60, y + 126, "reference", cls="tiny", anchor="middle", fill=MUTED)

    # (f) max-energy hijack.
    x, y, w, h = panels[5]
    threshold = y + 105
    s.line(x, threshold, x + w, threshold, stroke=INK, sw=1.2, dash="7 5")
    s.path(f"M{x} {y+h-18} L{x+85} {y+h-18} L{x+100} {threshold-50} L{x+290} {threshold-48} L{x+300} {threshold-90} L{x+335} {threshold-92} L{x+345} {threshold-45} L{x+420} {threshold-45} L{x+440} {y+h-18} L{x+w} {y+h-18}", sw=2)
    tx_bar(s, x + 100, y + h + 5, 320, "legitimate event", False)
    tx_bar(s, x + 300, y + h + 20, 35, "pulse", True)
    event_bracket(s, x + 95, x + 440, y + 20, "event boundary unchanged")
    s.rect(x + 120, y + 62, 120, 80, fill="none", stroke=MUTED, sw=1.2, rx=0, dash="4 3")
    s.text(x + 180, y + 80, "legitimate candidate", cls="tiny", anchor="middle", fill=MUTED)
    s.rect(x + 265, y + 52, 120, 90, fill=LIGHT, stroke=INK, sw=1.4, rx=0)
    s.text(x + 325, y + 70, "selected: max energy", cls="tiny", anchor="middle")
    s.text(x + 310, y + 42, "mean(W_A) > mean(W_L)", cls="small", anchor="middle", fill=RED, weight="700")

    s.save("fig2_practical_rf_attacks.svg")


def figure3_tiers() -> None:
    s = SVG(1400, 850, "Attacker tiers and capability boundaries",
            "A0 digital, A1 controlled RF, and A2 query-based RF attackers have different access and evidence roles.")
    s.text(45, 43, "Attacker Tiers and Capability Boundaries", cls="title")
    s.text(45, 67, "Tiers distinguish access modality and knowledge—not a monotonic strength ranking.", cls="subtitle")

    # Access path.
    y = 115
    labels = ["Untrusted RF", "Antenna / RF", "Trusted receiver", "Energy detection", "Segmentation", "Window selection", "AMC / defense", "Event record"]
    xs = [40, 205, 370, 535, 700, 865, 1030, 1195]
    widths = [125, 125, 125, 125, 125, 125, 125, 155]
    s.rect(345, 92, 1020, 120, fill=WHITE, stroke=INK, sw=1.7, rx=10, dash="7 5")
    s.add('<rect x="370" y="80" width="245" height="24" fill="white"/>')
    s.text(380, 98, "DEFENDER-CONTROLLED BOUNDARY", cls="section")
    for x, w, label in zip(xs, widths, labels):
        s.rect(x, y, w, 50, fill=LIGHTER, stroke=INK, sw=1.2, rx=5)
        s.text(x + w / 2, y + 30, label, cls="small", anchor="middle", weight="700")
    for i in range(len(xs) - 1):
        s.line(xs[i] + widths[i], y + 25, xs[i + 1], y + 25, arrow=True)

    s.rect(65, 235, 210, 54, fill="#eef8fc", stroke=BLUE, sw=2)
    s.text(170, 257, "A1  CONTROLLED RF", cls="head", anchor="middle", fill=BLUE)
    s.text(170, 276, "primary practical model", cls="tiny", anchor="middle")
    s.path("M170 235 C170 205 115 205 105 166", stroke=BLUE, sw=2, arrow=True)
    s.rect(285, 235, 210, 54, fill="#fff5ef", stroke=ORANGE, sw=2, dash="7 4")
    s.text(390, 257, "A2  QUERY-BASED RF", cls="head", anchor="middle", fill=ORANGE)
    s.text(390, 276, "black-box deployment model", cls="tiny", anchor="middle")
    s.path("M390 235 C390 205 180 205 165 166", stroke=ORANGE, sw=2, arrow=True)
    s.path("M1270 166 C1270 315 500 315 500 262", stroke=ORANGE, sw=1.3, dash="5 4", arrow=True)
    s.text(875, 309, "limited observable feedback", cls="tiny", anchor="middle", fill=ORANGE)
    s.rect(780, 235, 240, 54, fill=LIGHT, stroke=INK, sw=1.5)
    s.text(900, 257, "A0  DIGITAL INJECTION", cls="head", anchor="middle")
    s.text(900, 276, "algorithmic upper bound", cls="tiny", anchor="middle")
    s.path("M900 235 V193 H598 V166", stroke=INK, sw=1.6, arrow=True)
    s.text(750, 224, "offline benchmark access only", cls="tiny", anchor="middle")

    # Capability matrix.
    table_x, table_y, table_w = 40, 350, 1320
    col = [260, 353, 353, 354]
    row_h = 56
    rows = [
        ("Injection interface", "Direct offline IQ edit", "Independent SDR through RF", "Independent SDR through RF"),
        ("Knowledge", "Full pipeline + weights", "Known design; same/surrogate model", "Approximate architecture only"),
        ("Optimization signal", "Exact gradients + logits", "Channel-aware white/gray box", "Transfer or zeroth-order queries"),
        ("Timing / channel", "Exact; no RF channel", "Finite delay; own channel estimate", "Coarse timing; uncertain channel"),
        ("Resource limits", "Digital perturbation budget", "Power, BW, duty, CFO, duration", "RF limits + query/airtime budget"),
        ("Evidence role", "Debugging / worst-case bound", "Primary practical RF claims", "Information-limited deployment risk"),
    ]
    headers = ["ATTRIBUTE", "A0 — DIGITAL INJECTION", "A1 — CONTROLLED RF", "A2 — QUERY-BASED RF"]
    s.rect(table_x, table_y, table_w, 48 + row_h * len(rows), fill=WHITE, stroke=INK, sw=1.5, rx=4)
    cx = table_x
    for i, (w, head) in enumerate(zip(col, headers)):
        fill = LIGHT if i == 0 else ("#f7f7f7" if i == 1 else ("#eef8fc" if i == 2 else "#fff5ef"))
        s.add(f'<rect x="{cx}" y="{table_y}" width="{w}" height="48" fill="{fill}" stroke="{GRID}" stroke-width="1"/>')
        s.text(cx + w / 2, table_y + 29, head, cls="small", anchor="middle", weight="700")
        cx += w
    for r, values in enumerate(rows):
        y0 = table_y + 48 + r * row_h
        cx = table_x
        if r % 2:
            s.add(f'<rect x="{table_x}" y="{y0}" width="{table_w}" height="{row_h}" fill="{LIGHTER}"/>')
        for c, (w, value) in enumerate(zip(col, values)):
            s.line(cx, y0, cx, y0 + row_h, stroke=GRID, sw=.8)
            cls = "small"
            weight = "700" if c == 0 else None
            s.text(cx + 10, y0 + 32, value, cls=cls, weight=weight)
            cx += w
        s.line(table_x, y0, table_x + table_w, y0, stroke=GRID, sw=.8)

    bottom = table_y + 48 + row_h * len(rows)
    s.rect(40, bottom + 18, 1320, 44, fill=LIGHTER, stroke=INK, sw=1.2, rx=4)
    s.text(700, bottom + 45, "Excluded for all tiers: host compromise, checkpoint replacement, evidence-store modification, and metadata tampering.", cls="small", anchor="middle", weight="700")
    s.text(700, bottom + 82, "A1/A2 are additive asynchronous transmitters; they are not assumed to cancel legitimate RF energy on demand.", cls="small italic", anchor="middle", fill=MUTED)
    s.save("fig3_attacker_tiers.svg")


def ladder_card(s: SVG, x: int, y: int, w: int, h: int, num: str, title: str, setup: Sequence[str], supports: Sequence[str], does_not: Sequence[str], *, dashed=False, highlight=False) -> None:
    s.rect(x, y, w, h, fill=WHITE, stroke=INK, sw=2.8 if highlight else 1.4, rx=7, dash="8 5" if dashed else None)
    s.add(f'<path d="M{x+7} {y} H{x+w-7} Q{x+w} {y} {x+w} {y+7} V{y+52} H{x} V{y+7} Q{x} {y} {x+7} {y}" fill="{INK}"/>')
    s.text(x + 15, y + 32, num, cls="title", fill=WHITE)
    s.text(x + 54, y + 29, title, cls="head", fill=WHITE)
    for i, line in enumerate(setup):
        s.text(x + w / 2, y + 80 + i * 16, line, cls="small", anchor="middle", weight="700")
    split = y + 128
    support_h = 150
    s.add(f'<rect x="{x+1}" y="{split}" width="{w-2}" height="{support_h}" fill="{LIGHT}"/>')
    s.text(x + 12, split + 22, "SUPPORTS", cls="small", fill=GREEN, weight="700")
    for i, line in enumerate(supports):
        s.text(x + 14, split + 45 + i * 22, f"+  {line}", cls="small")
    sep = split + support_h
    s.line(x + 8, sep, x + w - 8, sep, stroke=INK, sw=1, dash="5 4")
    s.text(x + 12, sep + 23, "DOES NOT SUPPORT", cls="small", fill=RED, weight="700")
    for i, line in enumerate(does_not):
        s.text(x + 14, sep + 46 + i * 22, f"-  {line}", cls="small")


def figure4_ladder() -> None:
    s = SVG(1600, 820, "Validation ladder and claim boundaries",
            "Five stages distinguish algorithmic, software, physical, OTA, and operational evidence.")
    s.text(45, 43, "Validation Ladder and Claim Boundaries", cls="title")
    s.text(45, 67, "Evidence becomes more physical through shielded OTA; advisory shadow deployment changes the evidence type rather than proving a stronger attack.", cls="subtitle")
    s.line(55, 100, 1545, 100, stroke=INK, sw=1.5, arrow=True)
    s.text(55, 91, "CONTROL / REPRODUCIBILITY", cls="small", weight="700")
    s.text(1545, 91, "PHYSICAL + OPERATIONAL REALISM", cls="small", anchor="end", weight="700")

    y, w, h, gap = 135, 280, 565, 28
    cards = [
        ("1", "DIGITAL TENSOR", ["IQ tensor + delta", "=> full pipeline"], ["Algorithmic upper bound", "Adaptive-objective correctness", "White-box comparisons"], ["RF realizability", "Causality / synchronization", "Channel / ADC effects", "OTA feasibility"]),
        ("2", "CFILE REPLAY", ["complex64 file", "=> event pipeline"], ["End-to-end software", "Detection / event semantics", "Chunk + metadata handling", "Artifact replayability"], ["Live RF injection", "Receiver front-end effects", "Independent-source timing", "Physical transfer"]),
        ("3", "CABLED DUAL-SDR", ["Legit TX + Attack TX", "atten. + combiner => RX"], ["Independent additive RF", "Controlled SNR / SJR / PSR", "CFO, delay, gain, ADC", "Cabled feasibility"], ["Antenna propagation", "Multipath / mobility", "Radiated compliance", "Field generalization"]),
        ("4", "SHIELDED OTA", ["Legit + attack antennas", "=> shielded RX"], ["Radiated feasibility", "Antenna + channel effects", "Channel variability", "Controlled OTA repeatability"], ["Public operation", "Arbitrary range / environment", "Universal channel claims", "Perfect cancellation"]),
        ("5", "ADVISORY SHADOW*", ["Live RX => monitor", "=> log / analyst only"], ["Receive-only operation", "Natural workload exposure", "Latency / drops / alerts", "Analyst-facing utility"], ["Causal attack effectiveness", "Autonomous-control safety", "Mitigation efficacy", "All-adversary coverage"]),
    ]
    for i, card in enumerate(cards):
        x = 35 + i * (w + gap)
        ladder_card(s, x, y, w, h, *card, dashed=i == 4, highlight=i == 2)
        if i < 4:
            s.line(x + w + 4, y + h / 2, x + w + gap - 5, y + h / 2, arrow=True)
    s.text(35 + 2 * (w + gap) + w / 2, y - 14, "FIRST PHYSICAL-ATTACK EVIDENCE", cls="small", anchor="middle", fill=BLUE, weight="700")
    s.text(800, 738, "Claims accumulate only when the same end-to-end outcome, power constraints, and uncertainty model are evaluated at each stage.", cls="small", anchor="middle", weight="700")
    s.text(800, 765, "* Shadow deployment provides operational evidence, not a stronger active-attack tier. No automated action is taken.", cls="small italic", anchor="middle", fill=MUTED)
    s.save("fig4_validation_ladder.svg")


def gallery() -> None:
    figures = [
        ("Figure 1 — End-to-end threat model", "fig1_pipeline_threat_model.svg"),
        ("Figure 2 — Practical RF attack mechanisms", "fig2_practical_rf_attacks.svg"),
        ("Figure 3 — Attacker tiers", "fig3_attacker_tiers.svg"),
        ("Figure 4 — Validation ladder", "fig4_validation_ladder.svg"),
    ]
    cards = "\n".join(
        f'<section><h2>{escape(title)}</h2><img src="{escape(name)}" alt="{escape(title)}"></section>'
        for title, name in figures
    )
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Threat-model figures</title>
<style>
body{{margin:0;background:#eef2f6;color:#1f2937;font-family:Arial,Helvetica,sans-serif}}
main{{max-width:1500px;margin:auto;padding:28px}}h1{{margin:0 0 8px}}p{{color:#667085}}
section{{background:white;border:1px solid #d0d5dd;border-radius:10px;padding:18px;margin:24px 0;box-shadow:0 2px 8px #00000012}}
h2{{font-size:18px;margin:0 0 12px}}img{{display:block;width:100%;height:auto;border:1px solid #eaecf0}}
</style></head><body><main><h1>Practical Spectrum-Sensing Threat Model</h1>
<p>Print-friendly, paper-oriented SVG figures. Open each SVG separately for vector export.</p>{cards}</main></body></html>
"""
    (OUT / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    figure1_pipeline()
    figure2_attacks()
    figure3_tiers()
    figure4_ladder()
    gallery()
    print("Generated 4 SVG figures and index.html")


if __name__ == "__main__":
    main()
