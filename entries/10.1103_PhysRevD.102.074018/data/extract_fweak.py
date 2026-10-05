# Extract the shell-model weak form factors from the vector figures of Hoferichter, Menendez, Schwenk,
# PRD 102, 074018 (2020) [arXiv:2007.08529]: Fweak_CsI.pdf (Fig. 5) and Fweak_Ar.pdf (Fig. 6) of the arXiv source.
# Axis calibration from the tick marks: x 43.74 -> 279.39 pt <=> |q| = 0 -> 0.2 GeV;
# y 8.63 / 106.68 / 204.71 pt <=> F_w = 1 / 0.1 / 0.01 (log axis, PDF y downwards).  Requires pymupdf.
import sys, csv, pymupdf
def curves(fn, picks):
    pg = pymupdf.open(fn)[0]
    dr = pg.get_drawings()
    out = {}
    for name, idx in picks.items():
        pts = []
        for i in idx:
            for it in dr[i]["items"]:
                if it[0] == "l":
                    pts += [it[1], it[2]]
                elif it[0] == "c":
                    pts += [it[1], it[4]]
        seen, xy = set(), []
        for p in sorted(pts, key=lambda p: p.x):
            k = (round(p.x, 3), round(p.y, 3))
            if k in seen: continue
            seen.add(k)
            q = (p.x - 43.74) / (279.39 - 43.74) * 0.2
            logF = -(p.y - 8.63) / ((204.71 - 8.63) / 2)
            if p.y < 204.6:                       # drop points clipped at the lower frame edge
                xy.append((q, 10 ** logF))
        out[name] = xy
    return out
src = sys.argv[1] if len(sys.argv) > 1 else "."
data = curves(f"{src}/Fweak_CsI.pdf", {"I127": [3, 4], "Cs133": [5, 6]})
data |= curves(f"{src}/Fweak_Ar.pdf", {"Ar40": [7]})
for name, xy in data.items():
    with open(f"fweak_{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["q_GeV", "F_weak"]); w.writerows(xy)
    print(name, len(xy), "points, F(0) =", round(xy[0][1], 4))
