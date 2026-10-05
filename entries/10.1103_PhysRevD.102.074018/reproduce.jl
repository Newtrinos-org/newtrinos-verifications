# Nuclear form factors of Newtrinos.jl (module `cevns_xsec`: Helm, Klein-Nystrand, symmetrised Fermi) compared with
# the shell-model weak form factors of Hoferichter, Menéndez, Schwenk, Phys. Rev. D 102, 074018 (2020), Figs. 5 and 6.
# The shell-model curves (data/fweak_*.csv) are extracted from the vector figures of the arXiv source (data/extract_fweak.py).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.cevns_xsec

# nuclei: mass [MeV]; weak radius R_w of the paper (Table IV); Newtrinos default radius Rn_nom of the COHERENT modules
NUC = (Cs133 = (label = "¹³³Cs", m = 123.8e3, Rw = 5.08, Rnom = 5.0),
       I127 = (label = "¹²⁷I", m = 118.21e3, Rw = 5.00, Rnom = 5.0),
       Ar40 = (label = "⁴⁰Ar", m = 37.3e3, Rw = 3.55, Rnom = 3.6))
MODELS = (:helm, :klein_nystrand, :sym_fermi)
MLABEL = (helm = "Helm", klein_nystrand = "Klein–Nystrand", sym_fermi = "symmetrised Fermi")
MCOL = (helm = :dodgerblue, klein_nystrand = :orangered, sym_fermi = :purple)

# |F(q)| from Newtrinos: ffsq(er, mn, rn) is a function of the recoil energy, q = sqrt(2 m E_r)
F(q_GeV, nuc, rn, model) = sqrt(X.ffsq((q_GeV * 1e3)^2 / (2nuc.m), nuc.m, rn; model = model))
shell(name) = CSV.read("data/fweak_$(name).csv", DataFrame)
q = collect(range(0, 0.2, length = 401))

function figure(fname, names, title)
    fig = Figure(size = (1100, 470))
    for (k, n) in enumerate(names)
        nuc = NUC[n]; sm = shell(n)
        ax = Axis(fig[1, k], yscale = log10, limits = (0, 0.2, 1e-2, 1.2), xlabel = "|q| (GeV)", ylabel = "|F(q²)|",
                  title = "$(nuc.label): $(title)")
        # break the line where the figure's path is clipped at the lower frame edge (around the diffraction zero)
        qs, fs = Float64[], Float64[]
        for i in eachindex(sm.q_GeV)
            i > 1 && sm.q_GeV[i] - sm.q_GeV[i-1] > 0.003 && (push!(qs, NaN); push!(fs, NaN))
            push!(qs, sm.q_GeV[i]); push!(fs, sm.F_weak[i])
        end
        lines!(ax, qs, fs, color = :black, linewidth = 3, label = "shell model (paper)")
        for m in MODELS
            lines!(ax, q, max.(F.(q, Ref(nuc), nuc.Rw, m), 1e-3), color = MCOL[m], linewidth = 1.8, label = "Newtrinos $(MLABEL[m]), R = R_w = $(nuc.Rw) fm")
        end
        lines!(ax, q, max.(F.(q, Ref(nuc), nuc.Rnom, :helm), 1e-3), color = :gray, linestyle = :dash, linewidth = 1.8,
               label = "Newtrinos default (Helm, R = $(nuc.Rnom) fm)")
        vspan!(ax, 0, 0.1, color = (:gray, 0.08))
        k == 1 && axislegend(ax, position = :lb, framevisible = false, labelsize = 10)
    end
    save(fname, fig)
end
figure("ours/fig5_fweak_csi.png", (:I127, :Cs133), "weak form factor")
figure("ours/fig6_fweak_ar.png", (:Ar40,), "weak form factor")

# ratio F²(Newtrinos) / F²(shell model) at the momentum transfers of COHERENT (shaded region in the figures)
interp(x, y, t) = (i = clamp(searchsortedlast(x, t), 1, length(x) - 1); y[i] + (y[i+1] - y[i]) * (t - x[i]) / (x[i+1] - x[i]))
rows = []
for (n, nuc) in pairs(NUC), qq in (0.02, 0.05, 0.08, 0.10)
    sm = shell(n); f2 = interp(sm.q_GeV, sm.F_weak, qq)^2
    push!(rows, (nucleus = String(n), q_GeV = qq, F2_shell = round(f2, digits = 4),
                 (Symbol("ratio_", m) => round(F(qq, nuc, nuc.Rw, m)^2 / f2, digits = 4) for m in MODELS)...,
                 ratio_default = round(F(qq, nuc, nuc.Rnom, :helm)^2 / f2, digits = 4)))
end
tab = DataFrame(rows)
CSV.write("results/form_factor_ratios.csv", tab)

# rms radius implied by each closed form: F(q) = 1 - q²⟨r²⟩/6 + O(q⁴) at small q must return the input radius
rms = [(nucleus = String(n), model = String(m), R_input = nuc.Rw,
        R_from_F = round(sqrt(6 * (1 - F(1e-3, nuc, nuc.Rw, m)) / (1.0 / X.hcut_c)^2), digits = 4)) for (n, nuc) in pairs(NUC) for m in MODELS]
CSV.write("results/rms_radius_check.csv", DataFrame(rms))
show(stdout, tab, allrows = true); println()
show(stdout, DataFrame(rms), allrows = true); println()
