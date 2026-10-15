"""Inputs of this entry, extracted from the vector figures in the arXiv source of 2303.03222 (requires pymupdf, numpy):
  fig27_postpred.csv   Fig. 27: per sample, binned data and the mean posterior prediction (E_rec, 0.05 GeV bins)
  fig28_dcp.csv        Fig. 28/30: official Δχ²(δCP) from the data, NO and IO (relative to the global best fit)
Run in the arXiv source directory:  python extract.py <output dir>"""
import pymupdf, numpy as np, sys, os
out = sys.argv[1]
num = lambda s: float(s) if s.replace(".", "").isdigit() else None

def fig27(k):
    pg = pymupdf.open(f"Figures/OA/mach3/postPred_{k}_woRC_run1-10.pdf")[0]; dr = pg.get_drawings(); W = pg.get_text("words")
    # page rotated: E_rec along page-y (labels at page-x ≈ 333), events along page-x (labels at page-y ≈ 513)
    ex = [((w[1] + w[3]) / 2, num(w[4])) for w in W if 330 < w[0] < 340 and w[1] > 40 and num(w[4]) is not None]
    ey = [((w[0] + w[2]) / 2, num(w[4])) for w in W if 505 < w[1] < 520 and num(w[4]) is not None]
    A = np.polyfit(*zip(*ex), 1); B = np.polyfit(*zip(*ey), 1)
    red = max((d for d in dr if d["type"] == "s" and d.get("color") and tuple(round(v, 2) for v in d["color"]) == (1.0, 0.0, 0.0)), key=lambda d: len(d["items"]))
    pts = [red["items"][0][1]] + [it[-1] for it in red["items"]]
    E = np.polyval(A, [p.y for p in pts]); Y = np.polyval(B, [p.x for p in pts])
    steps = [(min(E[i], E[i + 1]), max(E[i], E[i + 1]), Y[i]) for i in range(len(E) - 1) if abs(Y[i] - Y[i + 1]) < 1e-3 and abs(E[i + 1] - E[i]) > 1e-3]
    w = np.median([h - l for l, h, _ in steps]); scale = 0.05 / w          # bins are 0.05 GeV (one panel has a different axis transform)
    e0 = min(l for l, _, _ in steps)
    nb = int(round((max(h for _, h, _ in steps) - e0) * scale / 0.05))
    pred = np.zeros(nb)
    for l, h, y in steps: pred[int(round((l - e0) * scale / 0.05))] = y
    # data markers (orange fill); the legend marker sits next to the word "Data"
    lab = [w for w in W if w[4] == "Data"][0]
    data = np.zeros(nb)
    for d in dr:
        if d.get("fill") and tuple(round(v, 2) for v in d["fill"]) == (1.0, 0.6, 0.0):
            r = d["rect"]; cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
            if abs(cx - (lab[0] + lab[2]) / 2) < 25 and abs(cy - lab[3]) < 40: continue
            e = (np.polyval(A, cy) - e0) * scale; data[int(e / 0.05)] = round(np.polyval(B, cx))
    return pred, data

with open(os.path.join(out, "fig27_postpred.csv"), "w") as f:
    f.write("sample,erec_lo,erec_hi,data,posterior_mean\n")
    for k in ["FHC1Rmu", "RHC1Rmu", "FHC1Re", "RHC1Re", "FHC1Re1de"]:
        pred, data = fig27(k)
        print(k, "bins", len(pred), "data", int(data.sum()), "posterior mean %.1f" % pred.sum())
        for i, (p, n) in enumerate(zip(pred, data)): f.write(f"{k},{0.05 * i:.2f},{0.05 * (i + 1):.2f},{int(n)},{p:.4f}\n")

with open(os.path.join(out, "fig28_dcp.csv"), "w") as f:
    f.write("ordering,dcp,dchi2\n")
    base = "Figures/OA/ptheta/brazil_20200525-03-muETh-run1to10-data_FC_20210226_FC_muETh_50ktoys_01_50kthrows_NH_m0500pi/"
    for mo, fn in (("NO", "brazil_dCP.pdf"), ("IO", "brazil_dCP_IH.pdf")):
        pg = pymupdf.open(base + fn)[0]; dr = pg.get_drawings(); W = pg.get_text("words")
        # δCP labels (−3 … 3, minus signs are separate words) along page-y, Δχ² labels along page-x at page-y ≈ 501
        xs = sorted([((w[1] + w[3]) / 2, num(w[4])) for w in W if 325 < w[0] < 335 and num(w[4]) is not None], reverse=True)
        dl = [(y, v * (-1 if any(m[4] == "−" and abs((m[1] + m[3]) / 2 - y - 9) < 6 for m in W) else 1)) for y, v in xs]
        cl = [((w[0] + w[2]) / 2, num(w[4])) for w in W if 495 < w[1] < 505 and num(w[4]) is not None]
        A = np.polyfit(*zip(*dl), 1); B = np.polyfit(*zip(*cl), 1)
        d = [x for x in dr if x["type"] == "s" and x.get("color") == (0.0, 0.0, 0.0) and len(x["items"]) > 50 and not (x.get("dashes") and "3" in x["dashes"])][0]
        pts = [d["items"][0][1]] + [it[-1] for it in d["items"]]
        x = np.polyval(A, [p.y for p in pts]); y = np.polyval(B, [p.x for p in pts]); o = np.argsort(x)
        print(mo, "δCP labels", [v for _, v in dl], "min %.2f max %.2f" % (y.min(), y.max()))
        for a, b in zip(x[o], y[o]): f.write(f"{mo},{a:.4f},{b:.4f}\n")
