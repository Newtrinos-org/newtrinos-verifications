#!/usr/bin/env python3
"""
Reference values from daemonflux (J. P. Yanez, A. Fedynitch, PRD 107, 123037 (2023); pip install daemonflux==0.9.0)
for the Newtrinos.jl entry: conventional fluxes and their relative 1-sigma errors (calibration covariance) for the
quantities of the paper's Fig. 12, at the "generic" location, for vertical down-going directions (zenith 0). (daemonflux's
"average" zenith is not used: for the ratio quantities it averages the ratio over zenith, not the fluxes.)

usage: python daemonflux_reference.py <output .csv>
"""
import sys
import numpy as np
from daemonflux import Flux

f = Flux(location="generic")
E = np.geomspace(5.0, 1.0e4, 60)
quantities = {"numu_sum": "numuflux", "nue_sum": "nueflux", "numu_ratio": "numuratio", "nue_ratio": "nueratio", "flavour_ratio": "flavorratio"}
# relative errors of the sums and ratios propagated linearly from daemonflux's own per-flavour parameter shifts (R = relative
# change of a flavour for +1 sigma; ratio a/b: R_a - R_b; sum a+b: (a R_a + b R_b)/(a + b)). daemonflux's ratio quantities
# carry separately tabulated Jacobians, which differ from this above ~200 GeV.
names = f.params.known_parameters
corr = np.asarray(f.params.cov) / np.outer(f.params.errors, f.params.errors)
fl = {q: f.flux(E, "0.0000", q) for q in ("numu", "antinumu", "nue", "antinue")}
R = [{q: f.flux(E, "0.0000", q, params={k: 1.0}) / fl[q] - 1 for q in fl} for k in names]
def rsum(r, a, b): return (fl[a] * r[a] + fl[b] * r[b]) / (fl[a] + fl[b])
lin = {"numu_sum": lambda r: rsum(r, "numu", "antinumu"), "nue_sum": lambda r: rsum(r, "nue", "antinue"),
       "numu_ratio": lambda r: r["numu"] - r["antinumu"], "nue_ratio": lambda r: r["nue"] - r["antinue"],
       "flavour_ratio": lambda r: rsum(r, "numu", "antinumu") - rsum(r, "nue", "antinue")}
def from_flavours(key):
    J = np.array([lin[key](r) for r in R])
    return np.sqrt(np.einsum("ie,ij,je->e", J, corr, J))
with open(sys.argv[1], "w") as out:
    out.write("quantity,E_GeV,value,relative_error,relative_error_from_flavours\n")
    for key, q in quantities.items():
        v = f.flux(E, "0.0000", q)
        e = f.error(E, "0.0000", q)
        ef = from_flavours(key)
        for i in range(len(E)):
            out.write("%s,%r,%r,%r,%r\n" % (key, E[i], v[i], e[i] / v[i], ef[i]))
