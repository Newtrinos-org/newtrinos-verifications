#!/usr/bin/env python3
"""
Extract the 2D credible-region contours (TGraphs) of the NOvA 2024 data release (zenodo 17822358, data_release_2024.zip,
files contours_{NOvAOnly,RCDB1D,RCDB2D}_{cond,noncond}.root) to one CSV: constraint, marginalisation, variables,
ordering, credible level, component, point index, x, y.

usage: extract_contours.py <directory with the contours_*.root files> <output .csv>   (pip install uproot)
"""
import sys, os, re, glob
import uproot

src, out = sys.argv[1:3]
pat = re.compile(r"contour_2D_\w+?_(conditional|nonconditional)_(dcpssth23|ssth23dm32)_(NO|IO|BO)_cred_int_(\d+)_(\d+)")
with open(out, "w") as f:
    f.write("constraint,marginalisation,variables,ordering,level,component,i,x,y\n")
    for path in sorted(glob.glob(os.path.join(src, "contours_*.root"))):
        constraint = os.path.basename(path).split("_")[1]
        r = uproot.open(path)
        for key in sorted(set(k.split(";")[0] for k in r.keys())):
            m = pat.fullmatch(key)
            if not m:
                continue
            marg, var, mo, lvl, comp = m.groups()
            level = float("0." + lvl[1:]) if lvl.startswith("0") else float(lvl)
            x, y = r[key].values()
            for i in range(len(x)):
                f.write("%s,%s,%s,%s,%r,%s,%d,%r,%r\n" % (constraint, marg, var, mo, level, comp, i, float(x[i]), float(y[i])))
