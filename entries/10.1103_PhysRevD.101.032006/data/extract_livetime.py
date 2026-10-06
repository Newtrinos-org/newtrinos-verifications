"""Extract the NMO-sensitivity-vs-livetime curves of IceCube-Gen2 & JUNO members, Phys. Rev. D 101, 032006 (2020)
[arXiv:1911.06745], Fig. 'livetimes' (images/nmo_sensitivity_vs_livetime.pgf of the arXiv source), for the JUNO
(8 cores) + IceCube Upgrade panels.  Axes from the tick marks: years = 1 + (x - x1)/0.186408 in, with x1 = 0.663279 in
(true NO) and 3.352992 in (true IO); significance = (y - 0.530914 in)/0.1439886 in.  Colours: blue = IceCube Upgrade,
black = JUNO (8 cores), dashed orange = simple (quadratic) sum, red = combined."""
import re, csv, sys
s = open((sys.argv[1] if len(sys.argv) > 1 else ".") + "/nmo_sensitivity_vs_livetime.pgf").read()
COL = {"0.254902,0.411765,0.882353": "upgrade", "0.000000,0.000000,0.000000": "juno_8cores",
       "1.000000,0.270588,0.000000": "combined", "1.000000,0.549020,0.000000": "simple_sum"}
PANELS = {"NO": 0.663279, "IO": 3.352992}
rows = []
for sc in s.split(r"\begin{pgfscope}"):
    pts = re.findall(r"\\pgfpath(?:moveto|lineto)\{\\pgfqpoint\{([-\d.]+)in\}\{([-\d.]+)in\}\}", sc)
    col = re.findall(r"\\definecolor\{currentstroke\}\{rgb\}\{([^}]*)\}", sc)
    if len(pts) < 5 or not col or col[-1] not in COL: continue
    x0 = float(pts[0][0])
    for order, x1 in PANELS.items():
        if x1 - 0.2 < x0 < x1 + 1.2:
            for x, y in pts:
                rows.append((order, COL[col[-1]], 1 + (float(x) - x1) / 0.186408, (float(y) - 0.530914) / 0.1439886))
with open("livetime_nmo_sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["true_ordering", "curve", "livetime_years", "significance_sigma"]); w.writerows(rows)
for o in PANELS:
    for c in COL.values():
        v = [r for r in rows if r[0] == o and r[1] == c]
        six = min(v, key=lambda r: abs(r[2] - 6))
        print(o, c, len(v), "points; at %.2f yr: %.2f sigma" % (six[2], six[3]))
