"""Extract the CC inclusive sigma/E curves (NUANCE: TOTAL, QE, RES, DIS) and the data points (central values and
vertical error bars) from the vector figures cc_inclusive_nu.pdf / cc_inclusive_nubar.pdf of the arXiv source of
Formaggio & Zeller, Rev. Mod. Phys. 84, 1307 (2012) [arXiv:1305.7513], Fig. 15.  Requires pymupdf.
Axes from the major tick marks of each panel: x = log10(E/GeV) with 0.1 GeV at 92.8 (nu) / 84.2 pt (nubar) and
112.23 / 114.53 pt per decade; y linear with 0 at 322.5 pt and 0.2 per 40.5 pt (nu) / 0.05 per 36.16 pt (nubar).  Data points are the centres of the marker
shapes; the error is the larger extent of the vertical error-bar segments through the marker (symmetric errors
assumed).  The QE curve and QE data are per nucleon of an isoscalar target as in the paper (divided by two)."""
import sys, csv, numpy as np, pymupdf
src = sys.argv[1] if len(sys.argv) > 1 else "."
STYLE = {(): "TOTAL", (3, 3): "QE", (1, 2): "DIS", (3, 4, 1, 4): "RES"}
Y0 = 322.5
CAL = {"nu": (92.8, (429.5 - 92.8) / 3, (322.5 - 39.0) / 7, 0.2), "nubar": (84.2, (427.8 - 84.2) / 3, (322.5 - 33.2) / 8, 0.05)}
for name in ("nu", "nubar"):
    X0, DEC, MAJ, step = CAL[name]
    pg = pymupdf.open(f"{src}/cc_inclusive_{name}.pdf")[0]
    E = lambda x: 10 ** (-1 + (x - X0) / DEC)
    S = lambda y: (Y0 - y) / MAJ * step
    curves, marks, vsegs = {}, set(), []
    draws = pg.get_drawings()
    draws = draws[:len(draws) // 2]                          # each figure is drawn twice in the PDF: keep one copy
    for p in draws:
        w = p.get("width") or 0
        dashes = tuple(int(float(v)) for v in str(p.get("dashes") or "[] 0").split("]")[0].strip("[ ").split()) if p.get("dashes") else ()
        if abs(w - 3.0) < 0.01 and dashes in STYLE and len(p["items"]) > 20:
            xy = sorted({(round(q.x, 3), round(q.y, 3)) for it in p["items"] for q in it[1:] if hasattr(q, "x")})
            curves[STYLE[dashes]] = [(E(x), S(y)) for x, y in xy]
            continue
        r = p["rect"]
        inside = 70 < r.x0 and r.x1 < 509 and 20 < r.y0 and r.y1 < 322
        if inside and 2.0 <= r.width <= 9.0 and 2.0 <= r.height <= 9.0:
            # a marker: filled or outlined shape, or a cross/star made of short lines
            marks.add((round((r.x0 + r.x1) / 2, 1), round((r.y0 + r.y1) / 2, 1)))
        for it in p["items"]:
            if it[0] == "l":
                a_, b_ = it[1], it[2]
                if abs(a_.x - b_.x) < 0.05 and abs(a_.y - b_.y) > 0.5:
                    vsegs.append((a_.x, min(a_.y, b_.y), max(a_.y, b_.y)))
    # merge markers whose centres coincide within 1 pt (a marker drawn as several paths)
    pts = []
    for m in sorted(marks):
        if not any(abs(m[0] - q[0]) < 1.0 and abs(m[1] - q[1]) < 1.0 for q in pts):
            pts.append(m)
    bars = []
    for (x, y) in pts:
        col = [v for v in vsegs if abs(v[0] - x) < 0.4 and v[1] <= y + 6 and v[2] >= y - 6]
        lo = min([v[1] for v in col] + [y]); hi = max([v[2] for v in col] + [y])
        bars.append((x, y, max(y - lo, hi - y)))
    with open(f"fz_{name}_curves.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["curve", "E_GeV", "sigma_over_E_1e-38cm2_per_GeV"])
        for lab, xy in curves.items():
            for e, s in xy: w.writerow([lab, e, s])
    # error bars: drop the duplicated second copy and the short tick-like segments
    with open(f"fz_{name}_data.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["E_GeV", "sigma_over_E", "err"])
        for x, y, e in sorted(bars):
            w.writerow([E(x), S(y), e / MAJ * step])
    print(name, {k: len(v) for k, v in curves.items()}, "data points", len(bars))
