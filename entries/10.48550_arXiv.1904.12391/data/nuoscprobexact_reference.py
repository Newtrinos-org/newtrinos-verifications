"""Reference probabilities from NuOscProbExact (M. Bustamante, arXiv:1904.12391; pip install nuoscprobexact,
version 1.14.0) for the Newtrinos.jl verification entry.  Writes data/ref_*.csv.

The physical constants are those of Newtrinos' osc.jl (G_F, N_A, km -> 1/eV), so that the comparison tests the
solvers and not the last digits of the constants.  The oscillation parameters are NuOscProbExact's defaults
(NuFit 4.0, normal ordering).  Run from the entry's data/ directory:  python3 nuoscprobexact_reference.py
"""
import csv
import numpy as np
import globaldefs as gd, hamiltonians3nu, hamiltonians4nu, oscprob3nu, oscprob4nu

GF_EV_CM3 = 8.961877245622253e-38        # osc.jl: G_F in eV cm^3
N_A = 6.022e23                           # osc.jl
KM = 5.067730716156395e9                 # 1 km in 1/eV (osc.jl umev * 1e9)

par = dict(s12=gd.S12_NO_BF, s23=gd.S23_NO_BF, s13=gd.S13_NO_BF, dcp=gd.DCP_NO_BF, d21=gd.D21_NO_BF, d31=gd.D31_NO_BF,
           s14=np.sqrt(0.1), s24=np.sqrt(0.1), s34=0.0, d41=1.0, eps=dict(ee=gd.EPS_EE, em=gd.EPS_EM, mm=gd.EPS_MM))
with open("ref_parameters.csv", "w", newline="") as f:
    keys = ["s12", "s23", "s13", "dcp", "d21", "d31", "s14", "s24", "s34", "d41"]
    w = csv.writer(f); w.writerow(keys); w.writerow([par[k] for k in keys])

def vcc(p_density):   # sqrt(2) G_F n_e [eV], p_density = Y_e rho [mol/cm^3]
    return np.sqrt(2) * GF_EV_CM3 * p_density * N_A
def vnc(n_density):   # -G_F n_n / sqrt(2) [eV]
    return -GF_EV_CM3 * n_density * N_A / np.sqrt(2)

# ---- (1) three flavours, L = 1300 km, constant density rho = 3 g/cm^3, Y_e = 0.5 (the paper's Fig. 2 setting)
h3 = hamiltonians3nu.hamiltonian_3nu_vacuum_energy_independent(par["s12"], par["s23"], par["s13"], par["dcp"],
                                                                  par["d21"], par["d31"])
E = np.logspace(np.log10(0.5), np.log10(30.0), 400)       # GeV
V = vcc(1.5)
eps = [gd.EPS_EE, gd.EPS_EM, gd.EPS_ET, gd.EPS_MM, gd.EPS_MT, gd.EPS_TT]   # (ee, em, et, mm, mt, tt), relative to n_e
names = ["Pee", "Pem", "Pet", "Pme", "Pmm", "Pmt", "Pte", "Ptm", "Ptt"]
with open("ref_3nu_1300km.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["E_GeV"] + [f"{c}_{n}" for c in ("vacuum", "matter", "nsi") for n in names])
    for e in E:
        row = [e]
        for h in (h3 / (e * 1e9), hamiltonians3nu.hamiltonian_3nu_matter(h3, e * 1e9, V),
                  hamiltonians3nu.hamiltonian_3nu_nsi(h3, e * 1e9, V, eps)):
            row += list(oscprob3nu.probabilities_3nu(h, 1300.0 * KM))
        w.writerow(row)

# ---- (2) 3+1, antineutrino nu_mu survival through the Earth: the Newtrinos four-zone Earth (newtrinos_layers.csv),
#          chords computed here independently, one exact evolution operator per slab, composed in order
lay = list(csv.DictReader(open("newtrinos_layers.csv")))
R = np.array([float(l["radius_km"]) for l in lay])         # outer radius of each shell (outermost first)
P = np.array([float(l["p_density_mol_cm3"]) for l in lay])
N = np.array([float(l["n_density_mol_cm3"]) for l in lay])
RD = 6369.0                                                # detector radius (Newtrinos default r_detector)

def shell_of(r):
    i = np.nonzero(R >= r)[0]
    return i[-1]                                           # innermost shell whose outer radius is >= r

def slabs(cz):
    # point at distance s before the detector: r(s)^2 = RD^2 + s^2 + 2 RD s cz (arriving from below for cz < 0)
    smax = -RD * cz + np.sqrt((RD * cz) ** 2 - RD ** 2 + R[0] ** 2)
    cuts = [0.0, smax]
    for Rk in R:
        disc = (RD * cz) ** 2 - RD ** 2 + Rk ** 2
        if disc > 0:
            for s in (-RD * cz - np.sqrt(disc), -RD * cz + np.sqrt(disc)):
                if 0 < s < smax:
                    cuts.append(s)
    cuts = np.sort(cuts)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):                  # ordered from the detector backwards
        m = (a + b) / 2
        r = np.sqrt(RD ** 2 + m ** 2 + 2 * RD * m * cz)
        k = shell_of(r)
        out.append((b - a, P[k], N[k]))
    return out[::-1]                                       # in the order the neutrino crosses them

h4 = hamiltonians4nu.hamiltonian_4nu_vacuum_energy_independent(par["s12"], par["s23"], par["s13"], par["s14"],
                                                                  par["s24"], par["s34"], par["dcp"], par["d21"],
                                                                  par["d31"], par["d41"])
E4 = np.logspace(np.log10(300.0), np.log10(30000.0), 120)   # GeV
CZ = np.linspace(-1.0, -0.05, 96)
with open("ref_3p1_earth_numubar.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cz", "E_GeV", "Pmm_bar"])
    for cz in CZ:
        sl = slabs(cz)
        for e in E4:
            U = np.eye(4, dtype=complex)
            for (L, p, n) in sl:                           # antineutrinos: conjugate vacuum term, flip potentials
                h = hamiltonians4nu.hamiltonian_4nu_matter(np.conj(h4), e * 1e9, -vcc(p), -vnc(n))
                U = oscprob4nu.evolution_operator_4nu(h, L * KM) @ U
            w.writerow([cz, e, abs(U[1, 1]) ** 2])
print("done")
