"""KamLAND 2011 (PRD 83, 052002), Fig. 5 (fig5.eps -> ps2pdf -dEPSCrop; mutool trace): data points (Data - BG - Geo, ratio
to no oscillation) vs L0/E and the 3-nu (blue) and 2-nu (dotted) best-fit histograms. Axes from the tick marks:
L0/E = 20 km/MeV at x = 103.8 pt, 5.8 pt per km/MeV; P = 0 at y = 397.5 pt, 361 pt per unit (y increases downwards)."""
import sys, os
sys.path.insert(0, sys.argv[2])
from trace_parse import parse
paths, texts, H = parse(sys.argv[1])
X = lambda x: 20 + (x - 103.8) / 5.8
P = lambda y: (397.5 - y) / 361.25
blk = [s for p in paths if p['color'] == '0' and p['lw'] == 7.5 for s in p['segs'] if len(s) == 2]
pts = {}
for (a, b) in blk:
    if 77 < a[0] < 628 and a[1] < 385 and not (abs(a[1] - 336.2) < 0.1):       # points start at their centre
        c = (round(a[0], 1), round(a[1], 1))
        pts.setdefault(c, []).append(b)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig5_points.csv')
with open(out, 'w') as f:
    f.write('# KamLAND 2011, Fig. 5: data points from the vector graphics (extract_fig5.py)\nLE_lo,LE_hi,LE,P,P_lo,P_hi\n')
    for (cx, cy), ends in sorted(pts.items()):
        xs = [e[0] for e in ends]; ys = [e[1] for e in ends]
        if max(xs) - min(xs) < 1: continue                                       # tick marks
        f.write('%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n' % (X(min(xs)), X(max(xs)), X(cx), P(cy), P(max(ys)), P(min(ys))))
def hist(sel):
    v = []
    for p in paths:
        if not sel(p): continue
        for s in p['segs']:
            for a, b in zip(s[:-1], s[1:]):
                if abs(a[1] - b[1]) < 0.01 and abs(a[0] - b[0]) > 1: v.append((X(min(a[0], b[0])), X(max(a[0], b[0])), P(a[1])))
    return sorted(v)
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig5_histograms.csv'), 'w') as f:
    f.write('# KamLAND 2011, Fig. 5: best-fit histograms (3nu blue, 2nu dotted) from the vector graphics\nmodel,LE_lo,LE_hi,P\n')
    for name, sel in (('3nu', lambda p: p['color'] == '0 0 1'), ('2nu', lambda p: p['color'] == '0' and p['dash'] == '30 30' and p['lw'] == 15.0)):
        for lo, hi, pv in hist(sel):
            if abs(lo - 21.97) > 0.1: f.write('%s,%.2f,%.2f,%.4f\n' % (name, lo, hi, pv))   # 21.97: legend sample
print(open(out).read())
