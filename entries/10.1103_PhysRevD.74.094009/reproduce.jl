# Atmospheric neutrino flux uncertainties in Newtrinos.jl (module `atm_flux`, Barr systematics) compared with
# Barr, Robbins, Gaisser, Stanev, Phys. Rev. D 74, 094009 (2006), Figs. 7 and 9.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results")
const PHYS = joinpath(pkgdir(Newtrinos), "src/physics")

E = 10 .^ range(-1, 3, length = 81)
cz = collect(range(-0.9975, 0.9975, length = 400))                     # uniform in cos θ_z
cfg(sys) = Newtrinos.atm_flux.AtmFluxConfig(systematics_model = sys)
models = (Barr = Newtrinos.atm_flux.Barr(), BarrEnergyBands = Newtrinos.atm_flux.BarrEnergyBands())

# zenith-averaged fluxes (ν + ν̄ where the paper sums them); the paper's ranges: up cos θ_z < -0.6,
# horizontal |cos θ_z| < 0.3, down cos θ_z > 0.6 (Sec. V); flavour ratios averaged over all directions
avg(f, sel) = vec(mean(reshape(f, length(E), length(cz))[:, sel], dims = 2))   # fluxes are flattened (E, cos θ_z) grids
const UP, HOR, DOWN, ALL = cz .< -0.6, abs.(cz) .< 0.3, cz .> 0.6, trues(length(cz))
ratios(f) = (
    numu_numubar = avg(f.numu, ALL) ./ avg(f.numubar, ALL),
    nue_nuebar = avg(f.nue, ALL) ./ avg(f.nuebar, ALL),
    numu_nue = avg(f.numu .+ f.numubar, ALL) ./ avg(f.nue .+ f.nuebar, ALL),
    numu_updown = avg(f.numu .+ f.numubar, UP) ./ avg(f.numu .+ f.numubar, DOWN),
    nue_updown = avg(f.nue .+ f.nuebar, UP) ./ avg(f.nue .+ f.nuebar, DOWN),
    numu_uphor = avg(f.numu .+ f.numubar, UP) ./ avg(f.numu .+ f.numubar, HOR),
    nue_uphor = avg(f.nue .+ f.nuebar, UP) ./ avg(f.nue .+ f.nuebar, HOR),
)
# which Newtrinos parameter is meant to carry which of the paper's uncertainties
pars(m) = m == :Barr ?
    (numu_numubar = (:atm_flux_numunumubar_sigma,), nue_nuebar = (:atm_flux_nuenuebar_sigma,),
     numu_nue = (:atm_flux_nuenumu_sigma,), numu_updown = (:atm_flux_updown_sigma,), nue_updown = (:atm_flux_updown_sigma,),
     numu_uphor = (:atm_flux_uphorizonzal_sigma,), nue_uphor = (:atm_flux_uphorizonzal_sigma,)) :
    (numu_numubar = Symbol.("atm_flux_numunumubar_sigma_" .* ("lo", "mid", "hi")),
     nue_nuebar = Symbol.("atm_flux_nuenuebar_sigma_" .* ("lo", "mid", "hi")),
     numu_nue = Symbol.("atm_flux_nuenumu_sigma_" .* ("lo", "mid", "hi")),
     numu_updown = (:atm_flux_updown_sigma,), nue_updown = (:atm_flux_updown_sigma,),
     numu_uphor = (:atm_flux_uphorizontal_sigma,), nue_uphor = (:atm_flux_uphorizontal_sigma,))

# fractional change [%] of each ratio when its parameter(s) are set to +1σ (all others nominal)
shift = Dict()
for (m, sys) in pairs(models)
    atm = Newtrinos.atm_flux.configure(cfg(sys))
    nominal = atm.nominal_flux(E, cz)
    R0 = ratios(atm.sys_flux(nominal, atm.params))
    for (r, ps) in pairs(pars(m))
        p1 = merge(atm.params, NamedTuple{ps}(ntuple(_ -> 1.0, length(ps))))
        shift[(m, r)] = 100 .* abs.(getproperty(ratios(atm.sys_flux(nominal, p1)), r) ./ getproperty(R0, r) .- 1)
    end
end

# the paper's curves, as digitised for the Newtrinos parametrisation (src/physics/*.csv)
paper(f) = (d = CSV.read(joinpath(PHYS, f), DataFrame, normalizenames = true); (d[:, 1], d[:, 2]))
PAPER = (numu_numubar = "numunumubar.csv", nue_nuebar = "nunuebar.csv", numu_nue = "numunue.csv",
         numu_updown = "numuupdown.csv", nue_updown = "nueupdown.csv", numu_uphor = "numuuphorizontal.csv",
         nue_uphor = "nueuphorizontal.csv")
LABEL = (numu_numubar = "νμ/ν̄μ", nue_nuebar = "νe/ν̄e", numu_nue = "νμ/νe", numu_updown = "νμ up/down",
         nue_updown = "νe up/down", numu_uphor = "νμ up/horizontal", nue_uphor = "νe up/horizontal")
COLOR = (numu_numubar = :black, nue_nuebar = :red, numu_nue = :green3, numu_updown = :black, nue_updown = :red,
         numu_uphor = :green3, nue_uphor = :blue)

function figure(fname, keys_, title)
    fig = Figure(size = (1180, 520))
    for (k, m) in enumerate((:Barr, :BarrEnergyBands))
        ax = Axis(fig[1, k], xscale = log10, yscale = log10, limits = (0.1, 1000, 0.1, 100),
                  xlabel = "E_ν (GeV)", ylabel = "uncertainty (%)", title = "$title — Newtrinos $(m)()",
                  xminorticksvisible = true, yminorticksvisible = true)
        for r in keys_
            x, y = paper(PAPER[r])
            scatterlines!(ax, x, y, color = COLOR[r], marker = :circle, markersize = 6, linewidth = 1, label = "$(LABEL[r]) (paper)")
            s = shift[(m, r)]
            ok = s .> 0.1
            lines!(ax, E[ok], s[ok], color = COLOR[r], linewidth = 2.5, linestyle = :dash, label = "$(LABEL[r]) Newtrinos +1σ")
        end
        k == 1 && axislegend(ax, position = :lt, framevisible = false, labelsize = 11, nbanks = 2)
    end
    save(fname, fig)
end
figure("ours/fig7_flavour_ratios.png", (:numu_numubar, :nue_nuebar, :numu_nue), "flavour ratios")
figure("ours/fig9_directional_ratios.png", (:numu_updown, :nue_updown, :numu_uphor, :nue_uphor), "directional ratios")

# table: paper value vs Newtrinos at a few energies (log-interpolated paper curves)
interp(x, y, e) = (i = clamp(searchsortedlast(x, e), 1, length(x) - 1);
                   exp(log(y[i]) + (log(e) - log(x[i])) / (log(x[i+1]) - log(x[i])) * (log(y[i+1]) - log(y[i]))))
rows = []
for r in keys(PAPER), e in (0.3, 1.0, 3.0, 10.0, 100.0)
    x, y = paper(PAPER[r])
    e > maximum(x) && continue
    j = argmin(abs.(log.(E) .- log(e)))
    push!(rows, (ratio = LABEL[r], E_GeV = e, paper_pct = round(interp(x, y, e), digits = 2),
                 Barr_pct = round(shift[(:Barr, r)][j], digits = 2), BarrEnergyBands_pct = round(shift[(:BarrEnergyBands, r)][j], digits = 2)))
end
tab = DataFrame(rows)
CSV.write("results/barr_uncertainties.csv", tab)
show(stdout, tab, allrows = true); println()
