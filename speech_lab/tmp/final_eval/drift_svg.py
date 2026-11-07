import json, time, statistics
rows = [json.loads(l) for l in open("tmp/final_eval/drift_s0v2.jsonl") if l.startswith("{")]
def tmin(r):
    t = time.localtime(r["mtime"]); return t.tm_hour * 60 + t.tm_min + t.tm_sec / 60
W, H, L, R, T, B = 720, 230, 58, 18, 18, 40
X0, X1, Y0, Y1 = 75, 390, -1.15, -0.95
x = lambda t: L + (t - X0) / (X1 - X0) * (W - L - R)
y = lambda v: T + (Y1 - v) / (Y1 - Y0) * (H - T - B)
boot1 = [r for r in rows if r["file"] in ("ambient.wav", "s2_ambient.wav")]
after = [r for r in rows if r["file"] not in ("ambient.wav", "s2_ambient.wav")]
med = statistics.median(r["tau_samples"] for r in after if r["file"].startswith("drift_"))
o = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Skew estimates through the night: about -1.12 samples in the first boot, then -0.98 to -1.04 after the resets">']
# target band: median +- 5 us (0.08 sample)
top = min(med + 0.08, Y1)
o.append(f'<rect class="band1" x="{x(115):.1f}" y="{y(top):.1f}" width="{x(X1)-x(115):.1f}" height="{y(med-0.08)-y(top):.1f}"/>')
for v in (-1.15, -1.10, -1.05, -1.00, -0.95):
    lab = f"{v:.2f}".replace("-", "−")
    o.append(f'<path class="grid" d="M{L} {y(v):.1f} H{W-R}"/><text x="{L-8}" y="{y(v)+4:.1f}" text-anchor="end">{lab}</text>')
for hh in (2, 3, 4, 5, 6):
    t = hh * 60
    o.append(f'<path class="axis" d="M{x(t):.1f} {H-B} v5"/><text x="{x(t):.1f}" y="{H-B+18}" text-anchor="middle">0{hh}:00</text>')
o.append(f'<path class="axis" d="M{L} {H-B} H{W-R}"/>')
# reset marker
o.append(f'<path class="lgrey" d="M{x(111):.1f} {T} V{H-B}"/><text x="{x(111)+5:.1f}" y="{T+10}">8 resets</text>')
for r in boot1:
    o.append(f'<circle class="copper" cx="{x(tmin(r)):.1f}" cy="{y(r["tau_samples"]):.1f}" r="4.5"/>')
for r in after:
    cls = "mask" if r["file"].startswith("drift_") else "ink"
    o.append(f'<circle class="{cls}" cx="{x(tmin(r)):.1f}" cy="{y(r["tau_samples"]):.1f}" r="{3.5 if cls=="mask" else 3}"/>')
o.append(f'<text x="{x(78):.1f}" y="{y(-1.12)-10:.1f}" class="t-copper">first boot</text>')
o.append(f'<text x="{x(386):.1f}" y="{y(med-0.08)-6:.1f}" class="t-mask" text-anchor="end">band: ±5 µs residual target around the median</text>')
o.append(f'<text x="14" y="{T+ (H-T-B)/2:.1f}" transform="rotate(-90 14 {T+(H-T-B)/2:.1f})" text-anchor="middle">skew (samples)</text>')
o.append("</svg>")
svg = "\n".join(o)
open("tmp/final_eval/drift.svg", "w").write(svg)
dr = [r["tau_samples"] for r in after if r["file"].startswith("drift_")]
print(len(dr), min(dr), max(dr), med, len(boot1), [r["tau_samples"] for r in boot1])
