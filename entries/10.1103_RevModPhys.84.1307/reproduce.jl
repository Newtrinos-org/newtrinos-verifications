# Charged-current νμ / ν̄μ cross sections of Newtrinos.jl (module `xsec`, model `H2O_PCA`: per-channel GENIE / NEUT
# curves for water) compared with the world data and the NUANCE prediction compiled by Formaggio & Zeller,
# Rev. Mod. Phys. 84, 1307 (2012), Fig. 15. The curves and data points are extracted from the vector figures of the
# arXiv source (data/extract_fz.py).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

E = 10 .^ range(-1, log10(500), length = 600)
TUNES = (:NEUT5_4_0, :G18_10a, :G21_11a, :G18_02a)
COL = (NEUT5_4_0 = :black, G18_10a = :dodgerblue, G21_11a = :orangered, G18_02a = :green3)
# total CC σ/E per nucleon of H₂O [10⁻³⁸ cm²/GeV] at nominal parameters, for each tune as nominal
σ = Dict((t, anti) => (x = X.configure(X.H2O_PCA(nominal = t, mc_nominal = t)); x.dσdE(E, :numu, :CC, anti, x.params))
         for t in TUNES, anti in (false, true))
# DIS channel of the default tune, for comparison with the NUANCE DIS curve
Eg, wester, allx = X._load_genie_data()
ilast = findlast(Eg .<= 26.0)
dis = Dict(anti => X.get_curve(wester, allx, "NEUT5_4_0", anti ? "numubar" : "numu", "CCDIS", ilast) for anti in (false, true))

fz(name) = (CSV.read("data/fz_$(name)_curves.csv", DataFrame), CSV.read("data/fz_$(name)_data.csv", DataFrame))
for (k, (anti, name, ymax, lab)) in enumerate(((false, "nu", 1.5, "ν"), (true, "nubar", 0.42, "ν̄")))
    cur, dat = fz(name)
    fig = Figure(size = (720, 500))
    ax = Axis(fig[1, 1], xscale = log10, xlabel = "E_ν (GeV)", ylabel = "$lab CC cross section / E (10⁻³⁸ cm²/GeV)",
              limits = (0.08, 500, 0, ymax), xticks = ([0.1, 1, 10, 100], ["0.1", "1", "10", "100"]))
    errorbars!(ax, dat.E_GeV, dat.sigma_over_E, dat.err, color = (:gray40, 0.7), whiskerwidth = 0)
    scatter!(ax, dat.E_GeV, dat.sigma_over_E, color = :gray40, markersize = 5, label = "world data (Formaggio & Zeller)")
    for (c, ls) in (("TOTAL", :solid), ("DIS", :dot))
        s = cur[cur.curve .== c, :]
        lines!(ax, s.E_GeV, s[!, 3], color = :gray20, linewidth = 4, linestyle = ls, alpha = 0.35,
               label = "NUANCE $(lowercase(c)), isoscalar")
    end
    for t in TUNES
        lines!(ax, E, σ[(t, anti)], color = COL[t], linewidth = 1.8, label = "Newtrinos $(t), H₂O")
    end
    lines!(ax, Eg, dis[anti], color = :black, linestyle = :dot, linewidth = 1.5, label = "Newtrinos NEUT5_4_0 DIS")
    axislegend(ax, position = k == 1 ? :rt : :rb, framevisible = false, labelsize = 10)
    save("ours/fig15_$(name).png", fig)
end

# numbers: σ/E at fixed energies vs NUANCE, and the 30–300 GeV average vs the data in that range
interp(x, y, t) = (i = clamp(searchsortedlast(x, t), 1, length(x) - 1); y[i] + (y[i+1] - y[i]) * (t - x[i]) / (x[i+1] - x[i]))
rows = []
for (anti, name) in ((false, "nu"), (true, "nubar"))
    cur, dat = fz(name)
    tot = cur[cur.curve .== "TOTAL", :]
    # inclusive measurements only: the figure also shows QE-only data, well below the total
    incl = dat.sigma_over_E .> 0.5 .* [interp(tot.E_GeV, tot[!, 3], e) for e in dat.E_GeV]
    hi = (dat.E_GeV .> 30) .& (dat.E_GeV .< 300) .& (dat.err .> 0) .& incl
    w = 1 ./ dat.err[hi] .^ 2
    data_hi = sum(w .* dat.sigma_over_E[hi]) / sum(w)
    for e in (0.5, 1.0, 3.0, 10.0, 100.0)
        push!(rows, (neutrino = name, E_GeV = e, nuance_isoscalar = round(interp(tot.E_GeV, tot[!, 3], e), digits = 3),
                     (Symbol(t) => round(interp(E, σ[(t, anti)], e), digits = 3) for t in TUNES)...))
    end
    sel = (E .> 30) .& (E .< 300)
    push!(rows, (neutrino = name, E_GeV = -1.0, nuance_isoscalar = round(data_hi, digits = 3),
                 (Symbol(t) => round(mean(σ[(t, anti)][sel]), digits = 3) for t in TUNES)...))
end
tab = DataFrame(rows)
CSV.write("results/cc_sigma_over_E.csv", tab)
println("(E_GeV = -1: 30-300 GeV average; first column = inverse-variance mean of the extracted inclusive data in that range)")
show(stdout, tab, allrows = true); println()
