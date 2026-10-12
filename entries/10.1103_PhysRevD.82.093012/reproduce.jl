# ντ charged-current cross sections: the models compared by Conrad, de Gouvêa, Shalgar & Spitz, PRD 82, 093012 (2010)
# (Kretzer–Reno, Paschos–Yu, Hagiwara–Mawatari–Yokoya on an isoscalar target, NUANCE on argon; vector-digitised from the
# arXiv source figure), the NLO DIS calculations of Kretzer & Reno (2002) and Jeong & Reno (2010) for σ(ντ)/σ(νμ), and
# the measured ντ normalisations (DONuT, OPERA, Super-K, IceCube DeepCore), compared with the ντ cross sections of
# Newtrinos.jl (module `xsec`): the GENIE tunes (`generator_sigma`, carbon = isoscalar, and argon) and the nominal model of
# `CorrelatedXsec` (the mean of the tunes) with its prior 1σ band, which includes the ντ normalisation (20 %).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, Random, Distributions

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

S = CSV.read("data/sigma_nutau.csv", DataFrame; normalizenames = true)
R = CSV.read("data/ratio_tau_mu.csv", DataFrame)
M = CSV.read("data/measurements.csv", DataFrame)
P = CSV.read("data/conrad2010_models_param_fluxavg.csv", DataFrame)

tab = X.load_table()
at(v, e) = X.at_energies(tab.E, v, e)
TUNES = Symbol.(sort(tab.tunes))
σE(t, target, flavour, anti, e) = at(sum(X.generator_sigma(t, target, flavour, anti, :CC)), e)

xs = X.configure(X.CorrelatedXsec())
ref = xs.cfg.reference
rng = MersenneTwister(1)
draw() = map(p -> p isa Distribution ? rand(rng, p) : p, xs.priors)
samples = [draw() for _ in 1:500]
p0 = xs.params
q(Mx, a) = [quantile(r, a) for r in eachrow(Mx)]
tunecol(i) = (Makie.wong_colors()[mod1(i, 7)], 0.6)

# --- σ/E for ντ, 0–30 GeV (Conrad et al. figure)
E = collect(range(3.0, 30.0, length = 300))
fτ = Dict(tg => xs.sigma(E; flavour = :nutau, anti = false, current = :CC, target = tg) for tg in (:C, :Ar))
fig = Figure(size = (860, 500))
ax = Axis(fig[1, 1], xlabel = "E_ν (GeV)", ylabel = "ντ CC σ / E (10⁻³⁸ cm²/GeV per nucleon)", limits = (0, 30, 0, 0.45),
          title = "ντ charged current: models of Conrad et al. (2010) and Newtrinos")
Mb = reduce(hcat, [fτ[:C](p) for p in samples])
band!(ax, E, q(Mb, 0.16), q(Mb, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ (C)")
for (i, t) in enumerate(TUNES)
    lines!(ax, E, σE(t, :C, :nutau, false, E), color = tunecol(i), linewidth = 1, label = "GENIE $(t), C")
end
lines!(ax, E, fτ[:C](p0), color = :dodgerblue, linewidth = 2.5, label = "nominal ($(ref)), C")
lines!(ax, E, fτ[:Ar](p0), color = :dodgerblue, linewidth = 1.5, linestyle = :dash, label = "nominal ($(ref)), Ar")
cs = S[S.source .== "Conrad_deGouvea_Shalgar_Spitz_2010_Fig", :]
for (m, col, ls) in zip(unique(cs.model), (:red, :green4, :purple, :black), (:solid, :dash, :dashdot, :dot))
    d = cs[cs.model .== m, :]
    lines!(ax, d.E_GeV, d.sigma_over_E_1e_38cm2_per_GeV, color = col, linestyle = ls, linewidth = 2, label = "Conrad et al.: " * replace(m, " (as plotted by Conrad et al.)" => ""))
end
kr = S[(S.source .== "Kretzer_Reno_2002_Fig6") .& (S.neutrino .== "nu_tau") .& occursin.("inclusive", S.process), :]
lines!(ax, kr.E_GeV, kr.sigma_over_E_1e_38cm2_per_GeV, color = :orange, linewidth = 2, label = "Kretzer & Reno 2002 (own figure, no cuts)")
Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
save("ours/sigma_over_E.png", fig)

# --- σ(ντ)/σ(νμ), 3–1000 GeV
E2 = 10 .^ range(log10(3.5), 3, length = 300)
fμ = xs.sigma(E2; flavour = :numu, anti = false, current = :CC, target = :C)
fτ2 = xs.sigma(E2; flavour = :nutau, anti = false, current = :CC, target = :C)
Mr = reduce(hcat, [fτ2(p) ./ fμ(p) for p in samples])
fig = Figure(size = (860, 480))
ax = Axis(fig[1, 1], xscale = log10, xlabel = "E_ν (GeV)", ylabel = "σ(ντ) / σ(νμ)", limits = (3.5, 1000, 0, 1.05),
          xticks = ([5, 10, 30, 100, 300, 1000], ["5", "10", "30", "100", "300", "1000"]), title = "ντ / νμ CC cross-section ratio (isoscalar)")
band!(ax, E2, q(Mr, 0.16), q(Mr, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ")
for (i, t) in enumerate(TUNES)
    lines!(ax, E2, σE(t, :C, :nutau, false, E2) ./ σE(t, :C, :numu, false, E2), color = tunecol(i), linewidth = 1, label = "GENIE $(t)")
end
lines!(ax, E2, fτ2(p0) ./ fμ(p0), color = :dodgerblue, linewidth = 2.5, label = "nominal ($(ref))")
for (src, col, lab) in (("Kretzer_Reno_2002_Fig6_lower", :orange, "Kretzer & Reno 2002"), ("Jeong_Reno_2010_Fig2", :red, "Jeong & Reno 2010 (W > 1.4 GeV)"))
    d = R[(R.source .== src) .& startswith.(R.neutrino, "nu ("), :]
    lines!(ax, d.E_GeV, d.ratio_sigma_tau_over_sigma_mu, color = col, linewidth = 2, linestyle = :dash, label = lab)
end
Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
save("ours/ratio_tau_mu.png", fig)

# --- the threshold parametrisation of Conrad et al., σ/E = (1 − exp(−a √(E − b))) × 10⁻³⁸ cm²/GeV, fitted on 4–30 GeV
Efit = collect(range(4.0, 30.0, length = 100))
function fitab(y)
    best = (Inf, 0.0, 0.0)
    for a in 0.05:0.0005:0.15, b in 2.5:0.02:7.0
        m = 1 .- exp.(-a .* sqrt.(max.(Efit .- b, 0)))
        c = sum((y .- m) .^ 2)
        c < best[1] && (best = (c, a, b))
    end
    best[2], best[3]
end
a, b = fitab(xs.sigma(Efit; flavour = :nutau, anti = false, current = :CC, target = :C)(p0))
rows = [(model = "nominal ($(ref)), C", a = a, b = b)]
for t in TUNES
    a, b = fitab(σE(t, :C, :nutau, false, Efit)); push!(rows, (model = "GENIE $(t), C", a = a, b = b))
end
a, b = fitab(xs.sigma(Efit; flavour = :nutau, anti = false, current = :CC, target = :Ar)(p0)); push!(rows, (model = "nominal ($(ref)), Ar", a = a, b = b))
for r in eachrow(P)
    push!(rows, (model = "Conrad et al.: " * r.model, a = r.a, b = r.b_GeV))
end
ab = DataFrame(rows)
CSV.write("results/threshold_parametrisation.csv", ab)
show(stdout, ab, allrows = true); println()

# --- measured ντ normalisations vs the prior of the nominal model's ντ normalisation (xsec_nutau_cc_norm)
sel = [r for r in eachrow(M) if occursin("normalisation", r.quantity) || occursin("ratio <sigma>", r.quantity) || startswith(r.quantity, "sigma_const") && !occursin("superseded", r.experiment)]
short(q) = startswith(q, "sigma_const") ? "σ_const / SM (0.50)" : q == "ratio <sigma>_meas/<sigma>_GENIE" ? "σ / σ_GENIE" :
           replace(q, "nu_tau normalisation" => "norm.", "tau normalisation" => "norm.")
labs = ["$(replace(r.experiment, " (Analysis A)" => " A", " (Analysis B; confirmatory)" => " B")): $(short(r.quantity))" for r in sel]
vals = [r.quantity |> s -> startswith(s, "sigma_const") ? r.value / 0.50 : r.value for r in sel]   # DONuT: σ_const / SM value 0.50
eu = [startswith(r.quantity, "sigma_const") ? r.err_up / 0.50 : r.err_up for r in sel]
ed = [startswith(r.quantity, "sigma_const") ? r.err_down / 0.50 : r.err_down for r in sel]
fig = Figure(size = (900, 420))
ax = Axis(fig[1, 1], xlabel = "measured / each experiment's model expectation", yticks = (1:length(sel), labs), limits = (0, 2.5, 0.3, length(sel) + 0.7),
          title = "Measured ντ normalisations vs the prior of xsec_nutau_cc_norm (band)")
nτ = xs.priors.xsec_nutau_cc_norm
vspan!(ax, [quantile(nτ, 0.16)], [quantile(nτ, 0.84)], color = (:dodgerblue, 0.25))
vlines!(ax, [1.0], color = :dodgerblue)
errorbars!(ax, vals, 1:length(sel), ed, eu, direction = :x, color = :black, whiskerwidth = 6)
scatter!(ax, vals, 1:length(sel), color = :black)
save("ours/measurements.png", fig)
