"""KamLAND 2013 (PRD 88, 033001), Figs. 5 and 7a from the vector graphics (figX.eps -> ps2pdf -dEPSCrop; mutool trace).
Fig. 5: L0/E = 20 km/MeV at x = 79.2 pt, 6.38 pt per km/MeV; P = 0 at y = 397.5 pt, 361.25 pt per unit (y downwards).
Fig. 7a: (N_U - N_Th)/(N_U + N_Th) = -1 at x = 83.0 pt, +1 at 520.2 pt; N_U + N_Th = 0 at y = 427.8 pt, 1.6784 pt per event.
usage: python3 extract_fig5_fig7a.py fig5_trace.xml fig7a_trace.xml <dir of trace_parse.py>"""
import sys, os
sys.path.insert(0, sys.argv[3])
from trace_parse import parse
here = os.path.dirname(os.path.abspath(__file__))
paths, _, _ = parse(sys.argv[1])
X = lambda x: 20 + (x - 79.2) / 6.38
P = lambda y: (397.5 - y) / 361.25
blk = [s for p in paths if p['color'] == '0' and p['lw'] == 7.5 for s in p['segs'] if len(s) == 2]
pts = {}
for (a, b) in blk:
    if 77 < a[0] < 628 and a[1] < 385:
        pts.setdefault((round(a[0], 1), round(a[1], 1)), []).append(b)
with open(os.path.join(here, 'fig5_points.csv'), 'w') as f:
    f.write('# KamLAND 2013, Fig. 5: data points (Data - BG - Geo) from the vector graphics (extract_fig5_fig7a.py)\nLE_lo,LE_hi,LE,P,P_lo,P_hi\n')
    for (cx, cy), ends in sorted(pts.items()):
        xs = [e[0] for e in ends]; ys = [e[1] for e in ends]
        if max(xs) - min(xs) < 1 or len(ends) < 3: continue                     # tick marks, legend
        f.write('%.2f,%.2f,%.2f,%.4f,%.4f,%.4f\n' % (X(min(xs)), X(max(xs)), X(cx), P(cy), P(max(ys)), P(min(ys))))
with open(os.path.join(here, 'fig5_histogram.csv'), 'w') as f:
    f.write('# KamLAND 2013, Fig. 5: 3-nu best-fit histogram (blue) from the vector graphics\nLE_lo,LE_hi,P\n')
    for p in paths:
        if p['color'] != '0 0 1': continue
        for s in p['segs']:
            for a, b in zip(s[:-1], s[1:]):
                if abs(a[1] - b[1]) < 0.01 and abs(a[0] - b[0]) > 1 and a[1] < 340:        # legend sample is at y ~ 355
                    f.write('%.2f,%.2f,%.4f\n' % (X(min(a[0], b[0])), X(max(a[0], b[0])), P(a[1])))
paths, _, _ = parse(sys.argv[2])
A = lambda x: -1 + 2 * (x - 83.0) / (520.2 - 83.0)
N = lambda y: (427.8 - y) / 1.6784
cols = {'.399902 .199951 1': 68.3, '.399902 1 .800781': 95.4, '1 .399902 .399902': 99.7}
with open(os.path.join(here, 'fig7a_contours.csv'), 'w') as f:
    f.write('# KamLAND 2013, Fig. 7a: C.L. contours of the geo-nu event numbers from the vector graphics\ncl,segment,asymmetry,N_total\n')
    k = 0
    for p in paths:
        if p['color'] not in cols or p['lw'] != 22.5: continue
        for s in p['segs']:
            if len(s) < 10: continue                                                 # legend samples
            k += 1
            for x, y in s: f.write('%.1f,%d,%.4f,%.2f\n' % (cols[p['color']], k, A(x), N(y)))
print(open(os.path.join(here, 'fig5_points.csv')).read())
