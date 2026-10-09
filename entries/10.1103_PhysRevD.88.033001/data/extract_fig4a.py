"""KamLAND 2013 (PRD 88, 033001), Fig. 4(a) (fig4a.eps from arXiv:1303.4667; ps2pdf -dEPSCrop; mutool trace): the
KamLAND-only contours (black: 95 % dotted, 99 % dashed, 99.73 % solid) and KamLAND-only Δχ² profiles (thick black
dashed). Axes from the tick marks and the 1σ–4σ guide lines: tan²θ₁₂ 0.1 at x = 74 pt, 361.1 pt per unit; Δm²₂₁
0.1×10⁻⁴ eV² at y = 432 pt (y downwards), 141.7 pt per 10⁻⁴ eV²; top panel Δχ² = (120.2 − y)/5.7; right panel
Δχ² = (x − 399)/5.92."""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from trace_parse import parse
paths, texts, H = parse(sys.argv[1] if len(sys.argv) > 1 else 'fig4a_trace.xml')
T = lambda x: 0.1 + (x - 74.0) / 361.1
DM = lambda y: (0.1 + (432.0 - y) / 141.7) * 1e-4
OUT = os.path.dirname(os.path.abspath(__file__))
levels = {None: '99.73', '30 30': '99', '10 20': '95'}
with open(os.path.join(OUT, 'fig4a_kamland_only_contours.csv'), 'w') as f:
    f.write('# KamLAND-only allowed regions, θ13 free (PRD 88, 033001, Fig. 4a), from the vector graphics; segment = drawn polyline\n')
    f.write('cl,segment,tan2_theta12,dm2_21\n'); k = 0
    for p in paths:
        if p['color'] != '0' or p['lw'] != 15.0 or p['dash'] not in levels: continue
        for s in p['segs']:
            if len(s) < 10: continue                                    # legend samples
            k += 1
            for x, y in s: f.write('%s,%d,%.4f,%.6e\n' % (levels[p['dash']], k, T(x), DM(y)))
with open(os.path.join(OUT, 'fig4a_kamland_only_profiles.csv'), 'w') as f:
    f.write('# KamLAND-only Δχ² profiles, θ13 free (PRD 88, 033001, Fig. 4a side panels), from the vector graphics\n')
    f.write('variable,x,dchi2\n')
    for p in paths:
        if p['color'] != '0' or p['lw'] != 22.5 or p['dash'] != '30 30': continue
        pts = [q for s in p['segs'] for q in s]
        if len(pts) < 10: continue
        if max(q[1] for q in pts) < 125:                                # top panel: Δχ²(tan²θ12)
            for x, y in pts: f.write('tan2_theta12,%.4f,%.4f\n' % (T(x), (120.2 - y) / 5.7))
        else:                                                           # right panel: Δχ²(Δm²21)
            for x, y in pts: f.write('dm2_21,%.6e,%.4f\n' % (DM(y), (x - 399.0) / 5.92))
