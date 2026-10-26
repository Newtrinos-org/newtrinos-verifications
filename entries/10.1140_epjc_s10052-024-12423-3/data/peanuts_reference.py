#!/usr/bin/env python3
"""
Reference values from PEANUTS (Gonzalo & Lucente, EPJC 84 (2024) 119; github.com/michelelucente/PEANUTS)
for the Newtrinos.jl solar-oscillation cross-check.

usage: peanuts_reference.py <path to PEANUTS checkout> <B23 structure csv> <newtrinos zone layers csv> <output dir>

Inputs shared with Newtrinos: the oscillation parameters (ref_parameters.csv, written here), the B23 solar model
(radial electron density and production distributions) and, for the numerics check of the Earth crossing, the
constant-density zones of the Newtrinos solar Earth model (radius [km], electron density [mol/cm^3]).
Outputs (CSV):
  day_mass_fractions.csv  P(nu_e -> nu_i) at the Sun's surface, per component and energy (adiabatic, production-averaged)
  day_pee.csv             P(nu_e -> nu_beta) in daytime, per component and energy
  earth_mass_to_e.csv     P(nu_i -> nu_e) after crossing the Earth, per energy and nadir angle, for
                          the Lisi-Montanino PREM parametrisation (PEANUTS default; analytical and numerical evolutor)
                          and for the Newtrinos zones (numerical evolutor)
  exposure_night.csv      night-averaged and day P(nu_e -> nu_e) of 8B neutrinos at Kamioka, exposure-weighted
"""
import sys, os, csv, math
import numpy as np

peanuts_dir, b23_file, zones_file, outdir = sys.argv[1:5]
sys.path.insert(0, peanuts_dir)
from peanuts.pmns import PMNS
from peanuts.solar import solar_flux_mass, Psolar
from peanuts.earth import EarthDensity, Pearth, Pearth_integrated

os.makedirs(outdir, exist_ok=True)

# NuFIT 6.0-like normal-ordering point (same values in the Newtrinos script via ref_parameters.csv)
s12, s13, s23, dcp, d21, d31 = 0.307, 0.02195, 0.561, math.radians(177.0), 7.49e-5, 2.534e-3
th12, th13, th23 = math.asin(math.sqrt(s12)), math.asin(math.sqrt(s13)), math.asin(math.sqrt(s23))
pmns = PMNS(th12, th13, th23, dcp)
with open(os.path.join(outdir, "ref_parameters.csv"), "w") as f:
    f.write("s12,s13,s23,dcp,d21,d31\n%r,%r,%r,%r,%r,%r\n" % (s12, s13, s23, dcp, d21, d31))

# ---- matter-potential constants. PEANUTS writes V/k in the Sun as 3.868e-7/2.533 * n_e E / Dm^2 (peanuts/matter_mixing.py,
# Vk) and the Earth potential as 3.868e-7 m^-1 per mol/cm^3 (peanuts/potentials.py). With G_F and N_A of Newtrinos
# (osc.jl) the exact values are larger/smaller by the factors below; the "matched" references rescale the electron
# densities passed to PEANUTS by them, so that both codes use the same potential, the "published" ones do not.
G_F, N_A, HBARC = 8.961877245622253e-38, 6.02214076e23, 1.973269804e-7   # eV cm^3, 1/mol, eV m
V_eV = math.sqrt(2) * G_F * N_A                                           # eV per mol/cm^3
scale_sun = (V_eV * 2e6) / (3.868e-7 / 2.533)                             # V/k per (n_e E[MeV]/Dm^2)
scale_earth = (V_eV / HBARC) / 3.868e-7
with open(os.path.join(outdir, "constants.csv"), "w") as f:
    f.write("scale_sun,scale_earth\n%r,%r\n" % (scale_sun, scale_earth))

# ---- B23 solar model: r [R_sun], n_e [mol/cm^3], production distributions per component
import pandas as pd
data = pd.read_csv(b23_file, comment="#")
r, ne0 = data["r"].to_numpy(), data["ne"].to_numpy()
components = ["pp", "pep", "hep", "be7", "b8", "n13", "o15", "f17"]
lines = {"pep": [1.442], "be7": [0.3843, 0.8613]}
Egrid = {c: np.geomspace(0.1, 18.8, 60) for c in components}
for c, Es in lines.items():
    Egrid[c] = np.array(Es)

with open(os.path.join(outdir, "day_mass_fractions.csv"), "w") as fm, open(os.path.join(outdir, "day_pee.csv"), "w") as fp:
    fm.write("constants,component,E_MeV,P1,P2,P3\n")
    fp.write("constants,component,E_MeV,Pe,Pmu,Ptau\n")
    for consts, ne in (("published", ne0), ("matched", ne0 * scale_sun)):
        for c in components:
            frac = np.ascontiguousarray(data[c].to_numpy())
            for E in Egrid[c]:
                T = solar_flux_mass(th12, th13, d21, d31, E, r, ne, frac)
                P = Psolar(pmns, d21, d31, E, r, ne, frac)
                fm.write("%s,%s,%r,%r,%r,%r\n" % (consts, c, E, T[0], T[1], T[2]))
                fp.write("%s,%s,%r,%r,%r,%r\n" % (consts, c, E, P[0], P[1], P[2]))
print("day done", flush=True)

# ---- Earth crossing: P(nu_i -> nu_e) for pure mass states at Super-K depth (1 km)
depth_m = 1000.0
def write_density(fname, comment, rows):
    with open(fname, "w") as f:
        f.write("# %s\n" % comment)
        for row in rows:
            f.write(", ".join("%r" % x for x in row) + "\n")
    return fname
# Lisi-Montanino PREM parametrisation (PEANUTS default, Data/Earth_Density.csv): radius, alpha, beta, gamma
lm_rows = [[float(x) for x in l.split(",")] for l in open(os.path.join(peanuts_dir, "Data", "Earth_Density.csv")) if l.strip() and not l.startswith("#")]
# Newtrinos solar Earth zones as a tabulated (piecewise-constant) density, radius / 6371 km, innermost first
zr, zn = np.loadtxt(zones_file, delimiter=",", skiprows=1, unpack=True)
zone_rows = sorted([[ri / 6371.0, ni] for ri, ni in zip(zr, zn)])
models = {}
for consts, sc in (("published", 1.0), ("matched", scale_earth)):
    models[("lisi-montanino", consts)] = EarthDensity(density_file=write_density(os.path.join(outdir, "lm_%s.csv" % consts),
        "Lisi-Montanino PREM, coefficients x %r" % sc, [[row[0]] + [x * sc for x in row[1:]] for row in lm_rows]))
    models[("newtrinos-zones", consts)] = EarthDensity(density_file=write_density(os.path.join(outdir, "zones_%s.csv" % consts),
        "Newtrinos solar Earth zones (radius / R_E, n_e [mol/cm^3] x %r)" % sc, [[a, b * sc] for a, b in zone_rows]), tabulated_density=True)

Es_earth = [3.0, 5.0, 7.0, 10.0, 13.0, 16.0]
etas = np.linspace(0.0, math.pi / 2, 91)[:-1]  # nadir angle; cos(zenith of the Sun) = -cos(eta)
spot = [(5.0, 0.0), (5.0, 0.6), (10.0, 0.0), (10.0, 1.2)]
with open(os.path.join(outdir, "earth_mass_to_e.csv"), "w") as f:
    f.write("constants,model,mode,E_MeV,eta,cz,P1e,P2e,P3e\n")
    def row(key, mode, E, eta):
        P = [float(np.real(Pearth(np.eye(3)[i], models[key], pmns, d21, d31, E, eta, depth_m, mode=mode, massbasis=True)[0])) for i in range(3)]
        f.write("%s,%s,%s,%r,%r,%r,%r,%r,%r\n" % (key[1], key[0], mode, E, eta, -math.cos(eta), *P)); f.flush()
    for key in models:
        for E in Es_earth:
            for eta in etas:
                row(key, "analytical", E, eta)
    print("earth analytical done", flush=True)
    for key in (("lisi-montanino", "matched"), ("newtrinos-zones", "matched")):
        for E, eta in spot:
            row(key, "numerical", E, eta)
print("earth numerical done", flush=True)

# ---- exposure-weighted day and night for 8B at Kamioka (latitude 36.43 deg N), Lisi-Montanino Earth
lam = math.radians(36.43)
frac = np.ascontiguousarray(data["b8"].to_numpy())
with open(os.path.join(outdir, "exposure_night.csv"), "w") as f:
    f.write("constants,E_MeV,Pee_day,Pee_night\n")
    for consts, ne, sc in (("published", ne0, 1.0), ("matched", ne0 * scale_sun, scale_earth)):
        for E in np.linspace(3.5, 18.0, 30):
            T = solar_flux_mass(th12, th13, d21, d31, E, r, ne, frac)
            night = Pearth_integrated(T, models[("lisi-montanino", consts)], pmns, d21, d31, E, depth_m, lam=lam, ns=1000, normalized=True, daynight="night")
            day = Psolar(pmns, d21, d31, E, r, ne, frac)[0]
            f.write("%s,%r,%r,%r\n" % (consts, E, day, float(np.real(night[0]))))

# ---- the paper's validation (Sec. 6.4, Fig. 8): 8B and hep at the solar surface, BS05(AGS,OP), SNO parameters
from peanuts.solar import SolarModel
from peanuts.time_average import NadirExposure
bs05 = SolarModel(solar_model_file=os.path.join(peanuts_dir, "Data", "bs2005agsopflux.csv"))
th12s = math.atan(math.sqrt(0.469)); th13s = math.asin(math.sqrt(0.01)); d21s, d31s = 7.9e-5, 2.46e-3
pmns_s = PMNS(th12s, th13s, 0.85521, 3.4034)
rb, neb = np.ascontiguousarray(bs05.radius()), np.ascontiguousarray(bs05.density())
with open(os.path.join(outdir, "bs05_model.csv"), "w") as f:
    f.write("r,ne,b8,hep\n")
    for i in range(len(rb)):
        f.write("%r,%r,%r,%r\n" % (rb[i], neb[i], bs05.fraction("8B")[i], bs05.fraction("hep")[i]))
with open(os.path.join(outdir, "sno_validation.csv"), "w") as f:
    f.write("constants,component,E_MeV,Pe\n")
    for consts, ne in (("published", neb), ("matched", neb * scale_sun)):
        for comp in ("8B", "hep"):
            frac = np.ascontiguousarray(bs05.fraction(comp))
            for E in np.linspace(0.5, 20.0, 79):
                f.write("%s,%s,%r,%r\n" % (consts, comp, E, Psolar(pmns_s, d21s, d31s, E, rb, ne, frac)[0]))
with open(os.path.join(outdir, "sno_parameters.csv"), "w") as f:
    f.write("th12,th13,th23,dcp,d21,d31\n%r,%r,%r,%r,%r,%r\n" % (th12s, th13s, 0.85521, 3.4034, d21s, d31s))

# ---- ideal yearly exposure in nadir angle (Fig. 5) for latitudes 0, 45, 89 deg and Kamioka
with open(os.path.join(outdir, "nadir_exposure.csv"), "w") as f:
    f.write("lat_deg,eta,W\n")
    for lat in (0.0, 45.0, 89.0, 36.43):
        W = NadirExposure(lam=math.radians(lat), ns=1000, normalized=True)
        for eta, w in W:
            f.write("%r,%r,%r\n" % (lat, eta, w))
print("sno and exposure done", flush=True)
print("done")
