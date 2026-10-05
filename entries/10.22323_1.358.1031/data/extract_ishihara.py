"""Extract the official curves from the vector figures of A. Ishihara et al. (IceCube), PoS(ICRC2019)1031
[arXiv:1908.09441], the proceedings cited by the IceCube Upgrade MC data release (doi:10.21234/qfz1-yh02).
  - June26_Upgrade_NuMu_Disappearance_Sensitivity.pdf: 90% contours in (sin²θ23, Δm²32); the IceCube Upgrade 3 yr
    sensitivity (red, width 6) and DeepCore 3 yr 2018 (orange).  Axes from the labelled grid lines:
    sin²θ23 = 0.30 + (x - 104.8)/54.51*0.05,  Δm²32 = 0.0020 + (344.8 - y)/50.2*0.0002.
  - June26_Upgrade_NuTau_Appearance_Sensitivity.pdf: 1σ intervals of N_ντ (bar end points), N = (x - 30.0)/189.8.
Requires pymupdf."""
import sys, csv, pymupdf
src = sys.argv[1] if len(sys.argv) > 1 else "."
pg = pymupdf.open(f"{src}/June26_Upgrade_NuMu_Disappearance_Sensitivity.pdf")[0]
s23 = lambda x: 0.30 + (x - 104.8) / ((540.9 - 104.8) / 8) * 0.05
dm2 = lambda y: 0.0020 + (344.8 - y) / ((344.8 - 43.6) / 6) * 0.0002
cols = {(1.0, 0.0, 0.0): "upgrade_3yr", (1.0, 0.6470588445663452, 0.0): "deepcore_3yr_2018"}
with open("ishihara_contours_90cl.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["contour", "sin2_theta23", "dm2_32"])
    for p in pg.get_drawings():
        c = tuple(p.get("color") or ())
        if c in cols and len(p["items"]) > 20:
            pts = [p["items"][0][1]] + [it[-1] for it in p["items"]]
            for q in pts:
                w.writerow([cols[c], s23(q.x), dm2(q.y)])
pg = pymupdf.open(f"{src}/June26_Upgrade_NuTau_Appearance_Sensitivity.pdf")[0]
N = lambda x: (x - 30.0) / ((219.8 - 30.0) / 1.0)
bars = {(1.0, 0.0, 0.0): "upgrade_1yr", (1.0, 0.6470588445663452, 0.0): "deepcore_3yr", (0.0, 0.5019607543945312, 0.0): "opera",
        (0.0, 0.0, 1.0): "superk"}
with open("ishihara_nutau_1sigma.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["measurement", "low", "high"])
    for p in pg.get_drawings():
        c = tuple(p.get("color") or ())
        if c in bars and abs((p.get("width") or 0) - 6.0) < 0.01:
            w.writerow([bars[c], N(p["rect"].x0), N(p["rect"].x1)])
print(open("ishihara_nutau_1sigma.csv").read())
