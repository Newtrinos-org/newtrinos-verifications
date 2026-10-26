#!/usr/bin/env python3
"""
Reduce the T2K 2025 MCMC chain release (zenodo 17312780, T2KNuMCMCRelease.2506.05889.root, NuMCMCTools format)
to marginal posterior histograms, without and with the reactor constraint sin^2(theta13) = 0.0220 +- 0.0007
(applied by reweighting; the released chains have a flat prior in sin^2(theta13)).

usage: reduce_chains.py <chain .root file> <output dir>
Writes hist1d.csv (variable, ordering, weighting, bin_lo, bin_hi, density) and hist2d.csv, plus summary.csv
(posterior probability of the normal ordering, posterior means).
"""
import sys, os, math
import numpy as np
import uproot

path, outdir = sys.argv[1:3]
os.makedirs(outdir, exist_ok=True)
REACTOR = (0.0220, 0.0007)

edges1 = {
    "dcp": np.linspace(-math.pi, math.pi, 101),
    "sin2th23": np.linspace(0.30, 0.75, 91),
    "sin2th13": np.linspace(0.0, 0.06, 121),
    "dm2_32": np.linspace(-3.0e-3, 3.0e-3, 601),
}
edges2 = {
    ("sin2th23", "dm2_32"): (np.linspace(0.35, 0.70, 71), np.linspace(-2.9e-3, 2.9e-3, 581)),
    ("dcp", "sin2th13"): (np.linspace(-math.pi, math.pi, 61), np.linspace(0.0, 0.05, 101)),
    ("dcp", "sin2th23"): (np.linspace(-math.pi, math.pi, 61), np.linspace(0.35, 0.70, 71)),
}
orderings = ("NO", "IO", "both")
weightings = ("flat", "reactor")
h1 = {(v, o, w): np.zeros(len(e) - 1) for v, e in edges1.items() for o in orderings for w in weightings}
h2 = {(k, o, w): np.zeros((len(e[0]) - 1, len(e[1]) - 1)) for k, e in edges2.items() for o in orderings for w in weightings}
tot = {(o, w): 0.0 for o in orderings for w in weightings}
mean = {(v, o, w): 0.0 for v in edges1 for o in orderings for w in weightings}

tree = uproot.open(path)["mcmc"]
n = 0
for a in tree.iterate(["DeltaCP", "Deltam2_32", "Theta13", "Theta23"], step_size="400 MB", library="np"):
    x = {"dcp": np.mod(a["DeltaCP"] + math.pi, 2 * math.pi) - math.pi,
         "sin2th23": np.sin(a["Theta23"]) ** 2,
         "sin2th13": np.sin(a["Theta13"]) ** 2,
         "dm2_32": a["Deltam2_32"]}
    wts = {"flat": np.ones_like(x["dcp"]),
           "reactor": np.exp(-0.5 * ((x["sin2th13"] - REACTOR[0]) / REACTOR[1]) ** 2)}
    masks = {"NO": x["dm2_32"] > 0, "IO": x["dm2_32"] < 0, "both": np.ones_like(x["dcp"], dtype=bool)}
    for o, m in masks.items():
        for wn, wv in wts.items():
            ww = wv[m]
            tot[(o, wn)] += ww.sum()
            for v, e in edges1.items():
                h1[(v, o, wn)] += np.histogram(x[v][m], bins=e, weights=ww)[0]
                mean[(v, o, wn)] += (x[v][m] * ww).sum()
            for k, e in edges2.items():
                h2[(k, o, wn)] += np.histogram2d(x[k[0]][m], x[k[1]][m], bins=e, weights=ww)[0]
    n += len(x["dcp"])
    print(n, flush=True)

with open(os.path.join(outdir, "hist1d.csv"), "w") as f:
    f.write("variable,ordering,weighting,bin_lo,bin_hi,density\n")
    for (v, o, w), h in h1.items():
        e = edges1[v]
        norm = h.sum() * np.diff(e)
        for i in range(len(h)):
            f.write("%s,%s,%s,%r,%r,%r\n" % (v, o, w, e[i], e[i + 1], h[i] / norm[i] if h.sum() > 0 else 0.0))
with open(os.path.join(outdir, "hist2d.csv"), "w") as f:
    f.write("x,y,ordering,weighting,x_lo,x_hi,y_lo,y_hi,weight\n")
    for (k, o, w), h in h2.items():
        ex, ey = edges2[k]
        s = h.sum()
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                if h[i, j] > 0:
                    f.write("%s,%s,%s,%s,%r,%r,%r,%r,%r\n" % (k[0], k[1], o, w, ex[i], ex[i + 1], ey[j], ey[j + 1], h[i, j] / s))
with open(os.path.join(outdir, "summary.csv"), "w") as f:
    f.write("weighting,n_samples,P_NO,mean_sin2th23,mean_sin2th13,mean_dm2_32_NO\n")
    for w in weightings:
        f.write("%s,%d,%r,%r,%r,%r\n" % (w, n, tot[("NO", w)] / tot[("both", w)], mean[("sin2th23", "both", w)] / tot[("both", w)],
                                         mean[("sin2th13", "both", w)] / tot[("both", w)], mean[("dm2_32", "NO", w)] / tot[("NO", w)]))
print("done")
