# T2K INGRID on-axis νμ CC inclusive cross sections per nucleon on iron and hydrocarbon (CH) and their ratio, flux
# averaged at ⟨E⟩ = 1.51 GeV, full phase space (PRD 90, 052010 (2014), in data/), compared with the cross sections of
# Newtrinos.jl (module `xsec`): the GENIE tunes (`generator_sigma`, ⁵⁶Fe and CH per nucleon) and the reference model of
# `CorrelatedXsec` with its prior 1σ band (Fe and CH sampled jointly). The models are averaged with the official T2K
# νμ flux at the INGRID Proton Module (data release ingriddata-numu-ccqe-xs-2015, which the release states was used for
# this measurement; data/flux_ProtonModule.csv); the iron modules see the same flux shape to 1–2 %.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, Random, Distributions

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

readc(f) = CSV.read(f, DataFrame; comment = "#", normalizenames = true)
A = readc("data/flux_averaged.csv")
Rt = readc("data/ratio.csv")
F = readc("data/flux_ProtonModule.csv")
dFe, dCH = A[A.target .== "Fe", :][1, :], A[A.target .== "CH", :][1, :]

# fine grid with the flux histogram (constant within its bins)
Ef = collect(range(0.01, F.E_hi[end], length = 6000))
Φ = [F.flux_per100MeV[clamp(searchsortedlast(F.E_lo, e), 1, nrow(F))] for e in Ef]
fluxavg(σ) = sum(Φ .* σ) / sum(Φ)
@printf("⟨E⟩ of the flux: %.3f GeV (paper 1.51)\n", fluxavg(Ef))

tab = X.load_table()
at(v, e) = X.at_energies(tab.E, v, e)
TUNES = Symbol.(sort(tab.tunes))
σ_tune(t, target, e) = at(sum(X.generator_sigma(t, target, :numu, false, :CC)), e) .* e

xs = X.configure(X.CorrelatedXsec())
ref = xs.cfg.reference
sf = Dict(tg => xs.sigma(Ef; flavour = :numu, anti = false, current = :CC, target = tg) for tg in (:Fe, :CH))
avg(p, tg) = fluxavg(sf[tg](p) .* Ef)
rng = MersenneTwister(1)
draw() = map(p -> p isa Distribution ? rand(rng, p) : p, xs.priors)
samples = [draw() for _ in 1:500]
p0 = xs.params
sFe = [avg(p, :Fe) for p in samples]; sCH = [avg(p, :CH) for p in samples]; sR = sFe ./ sCH

rows = [(model = "$(ref) (reference)", Fe = avg(p0, :Fe), CH = avg(p0, :CH), Fe_over_CH = avg(p0, :Fe) / avg(p0, :CH))]
push!(rows, (model = "prior 1σ of the reference", Fe = std(sFe), CH = std(sCH), Fe_over_CH = std(sR)))
for t in TUNES
    a, b = fluxavg(σ_tune(t, :Fe, Ef)), fluxavg(σ_tune(t, :CH, Ef))
    push!(rows, (model = string(t), Fe = a, CH = b, Fe_over_CH = a / b))
end
push!(rows, (model = "paper: NEUT", Fe = dFe.NEUT_pred, CH = dCH.NEUT_pred, Fe_over_CH = Rt.NEUT_pred[1]))
push!(rows, (model = "paper: GENIE", Fe = dFe.GENIE_pred, CH = dCH.GENIE_pred, Fe_over_CH = Rt.GENIE_pred[1]))
push!(rows, (model = "T2K data", Fe = dFe.sigma_1e_38cm2_per_nucleon, CH = dCH.sigma_1e_38cm2_per_nucleon, Fe_over_CH = Rt.value[1]))
out = DataFrame(rows)
out.Fe = round.(out.Fe, digits = 4); out.CH = round.(out.CH, digits = 4); out.Fe_over_CH = round.(out.Fe_over_CH, digits = 4)
CSV.write("results/flux_averaged.csv", out)
show(stdout, out, allrows = true); println()

# figures: σ(E) per nucleon for Fe and CH with the flux-averaged values, and the ratio Fe/CH
E = collect(range(0.3, 6.0, length = 300))
sig = Dict(tg => xs.sigma(E; flavour = :numu, anti = false, current = :CC, target = tg) for tg in (:Fe, :CH))
q(M, a) = [quantile(r, a) for r in eachrow(M)]
for (tg, d, name) in ((:Fe, dFe, "xsec_Fe"), (:CH, dCH, "xsec_CH"))
    M = reduce(hcat, [sig[tg](p) .* E for p in samples])
    fig = Figure(size = (820, 470))
    ax = Axis(fig[1, 1], xlabel = "E_ν (GeV)", ylabel = "νμ CC σ (10⁻³⁸ cm² per nucleon)", title = "T2K INGRID on-axis, $(tg)",
              limits = (0, 6, 0, 5))
    axf = Axis(fig[1, 1], yaxisposition = :right, ylabel = "flux (arb. units)", limits = (0, 6, 0, nothing))
    hidexdecorations!(axf); hideydecorations!(axf, label = false)
    stairs!(axf, vcat(F.E_lo, F.E_hi[end]), vcat(F.flux_per100MeV, F.flux_per100MeV[end]), step = :post, color = (:gray, 0.5))
    band!(ax, E, q(M, 0.16), q(M, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ")
    for (i, t) in enumerate(TUNES)
        lines!(ax, E, σ_tune(t, tg, E), color = (Makie.wong_colors()[mod1(i, 7)], 0.8), linewidth = 1, label = "GENIE $(t)")
    end
    lines!(ax, E, sig[tg](p0) .* E, color = :dodgerblue, linewidth = 2.5, label = "reference $(ref)")
    errorbars!(ax, [dCH.mean_E], [d.sigma_1e_38cm2_per_nucleon], [d.syst_down], [d.syst_up], color = :black, whiskerwidth = 6)
    scatter!(ax, [dCH.mean_E], [d.sigma_1e_38cm2_per_nucleon], color = :black, markersize = 12, label = "T2K (flux-averaged)")
    scatter!(ax, [dCH.mean_E], [avg(p0, tg)], color = :dodgerblue, marker = :diamond, markersize = 12, label = "reference, flux-averaged")
    Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
    save("ours/$(name).png", fig)
end
MR = reduce(hcat, [sig[:Fe](p) ./ sig[:CH](p) for p in samples])
fig = Figure(size = (820, 470))
ax = Axis(fig[1, 1], xlabel = "E_ν (GeV)", ylabel = "σ(Fe) / σ(CH) per nucleon", title = "T2K INGRID on-axis: Fe / CH", limits = (0, 6, 0.85, 1.2))
band!(ax, E, q(MR, 0.16), q(MR, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ")
for (i, t) in enumerate(TUNES)
    lines!(ax, E, σ_tune(t, :Fe, E) ./ σ_tune(t, :CH, E), color = (Makie.wong_colors()[mod1(i, 7)], 0.8), linewidth = 1, label = "GENIE $(t)")
end
lines!(ax, E, sig[:Fe](p0) ./ sig[:CH](p0), color = :dodgerblue, linewidth = 2.5, label = "reference $(ref)")
e = sqrt(Rt.stat[1]^2 + Rt.syst[1]^2)
errorbars!(ax, [Rt.mean_E[1]], [Rt.value[1]], [e], color = :black, whiskerwidth = 6)
scatter!(ax, [Rt.mean_E[1]], [Rt.value[1]], color = :black, markersize = 12, label = "T2K (flux-averaged)")
scatter!(ax, [Rt.mean_E[1]], [avg(p0, :Fe) / avg(p0, :CH)], color = :dodgerblue, marker = :diamond, markersize = 12, label = "reference, flux-averaged")
Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
save("ours/ratio.png", fig)
