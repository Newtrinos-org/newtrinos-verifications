"""Official frequentist results (analysis "Frequentist2", profile likelihood, reactor constraint) of the
T2K+SK joint analysis data release (zenodo 12702685, CC-BY 4.0) → official_frequentist2.h5

    python extract_official.py FrequentistAnalysis2.root
"""
import sys, h5py, uproot
f = uproot.open(sys.argv[1])
with h5py.File("official_frequentist2.h5", "w") as out:
    out.attrs["source"] = "zenodo 12702685, FrequentistAnalysis2.root (CC-BY 4.0)"
    for mo in ("NO", "IO"):
        for v in ("dCP", "th23", "dm2"):
            h = f[f"h1D_{v}_chi2_wRC_{mo}"]
            out[f"{mo}/{v}/edges"] = h.axis().edges(); out[f"{mo}/{v}/dchi2"] = h.values()
        for pair in ("dCP_th23", "th23_dm2"):
            for k in f.keys():
                k = k.split(";")[0]
                if k.startswith(f"gr2D_{pair}_wRC_{mo}_conf") and k not in out:
                    x, y = f[k].values(); out[f"{k}/x"] = x; out[f"{k}/y"] = y
