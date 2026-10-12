# MINERvA ν̄μ/νμ charged-current inclusive cross-section ratio and the νμ and ν̄μ σ/E per nucleon of carbon (isoscalar
# corrected), 2–22 GeV (PRD 95, 072009 (2017); Table IV and the supplemental covariances, from the arXiv:1701.04857v2
# source, in data/), compared with the cross sections of Newtrinos.jl (module `xsec`): the GENIE tunes (`generator_sigma`,
# carbon) and the reference model of `CorrelatedXsec` with its prior 1σ band (ν, ν̄ and their ratio sampled jointly).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, Random, Distributions, LinearAlgebra

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

readc(f) = CSV.read(f, DataFrame; comment = "#")
R = readc("data/ratio_nubar_nu.csv")
Sn, Sb = readc("data/sigma_nu_1701.csv"), readc("data/sigma_nubar_1701.csv")
covmat(f) = Matrix{Float64}(readc(f)[:, 2:end]) ./ 1000
CR, CS = covmat("data/cov_ratio_1701.csv"), covmat("data/cov_sigma_1701.csv")
@assert isapprox(sqrt.(diag(CR)), R.total; rtol = 0.05)
@assert isapprox(sqrt.(diag(CS)), vcat(Sn.total, Sb.total); rtol = 0.05)

tab = X.load_table()
E = 10 .^ range(log10(0.2), log10(25), length = 300)   # down to the threshold region
at(v, e = E) = X.at_energies(tab.E, v, e)
TUNES = Symbol.(sort(tab.tunes))
σ_tune(t, anti, e = E) = at(sum(X.generator_sigma(t, :C, :numu, anti, :CC)), e)
binE(lo, hi) = collect(range(lo, hi, length = 21))
bins = collect(zip(R.E_lo_GeV, R.E_hi_GeV))

xs = X.configure(X.CorrelatedXsec())
ref = xs.cfg.reference
sig = Dict(anti => xs.sigma(E; flavour = :numu, anti, current = :CC, target = :C) for anti in (false, true))
sigb = Dict(anti => [xs.sigma(binE(lo, hi); flavour = :numu, anti, current = :CC, target = :C) for (lo, hi) in bins] for anti in (false, true))
# model per bin: σ/E averaged flat in E within the bin; ratio of the bin-averaged σ (σ/E × E averaged)
mbin(p, anti) = [mean(f(p)) for f in sigb[anti]]
mratio(p) = [mean(fb(p) .* binE(b...)) / mean(fn(p) .* binE(b...)) for (fn, fb, b) in zip(sigb[false], sigb[true], bins)]

rng = MersenneTwister(1)
draw() = map(p -> p isa Distribution ? rand(rng, p) : p, xs.priors)
samples = [draw() for _ in 1:500]
p0 = xs.params
q(M, a) = [quantile(filter(!isnan, row), a) for row in eachrow(M)]   # ratios are undefined where a draw has σ = 0
band = Dict(anti => (M = reduce(hcat, [sig[anti](p) for p in samples]); (lo = q(M, 0.16), hi = q(M, 0.84))) for anti in (false, true))
Rs = reduce(hcat, [sig[true](p) ./ sig[false](p) for p in samples])
rband = (lo = q(Rs, 0.16), hi = q(Rs, 0.84))

ticks = ([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"])
tunecol(i) = (Makie.wong_colors()[mod1(i, 7)], 0.8)
function panel_ratio(pos)
    ax = Axis(pos, xscale = log10, xlabel = "E_ν (GeV)", ylabel = "σ(ν̄μ) / σ(νμ)", title = "MINERvA ratio, carbon (isoscalar)",
              limits = (0.2, 25, 0.2, 0.6), xticks = ticks)
    band!(ax, E, rband.lo, rband.hi, color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ")
    for (i, t) in enumerate(TUNES)
        lines!(ax, E, σ_tune(t, true) ./ σ_tune(t, false), color = tunecol(i), linewidth = 1, label = "GENIE $(t)")
    end
    lines!(ax, E, sig[true](p0) ./ sig[false](p0), color = :dodgerblue, linewidth = 2.5, label = "nominal ($(ref))")
    errorbars!(ax, R.E_center_GeV, R.ratio, R.total, color = :black, whiskerwidth = 4)
    errorbars!(ax, R.E_center_GeV, R.ratio, R.E_center_GeV .- R.E_lo_GeV, R.E_hi_GeV .- R.E_center_GeV, direction = :x, color = :black)
    scatter!(ax, R.E_center_GeV, R.ratio, color = :black, label = "MINERvA (total)")
    ax
end
function panel_sigma(pos, anti, S, lab, lims)
    ax = Axis(pos, xscale = log10, xlabel = "E_ν (GeV)", ylabel = "$lab CC σ / E (10⁻³⁸ cm²/GeV per nucleon)",
              title = "MINERvA $lab, carbon (isoscalar)", limits = (0.2, 25, lims...), xticks = ticks)
    band!(ax, E, band[anti].lo, band[anti].hi, color = (:dodgerblue, 0.25), label = "CorrelatedXsec prior 1σ")
    for (i, t) in enumerate(TUNES)
        lines!(ax, E, σ_tune(t, anti), color = tunecol(i), linewidth = 1, label = "GENIE $(t)")
    end
    lines!(ax, E, sig[anti](p0), color = :dodgerblue, linewidth = 2.5, label = "nominal ($(ref))")
    y = S[!, 4]
    errorbars!(ax, S.E_center_GeV, y, S.total, color = :black, whiskerwidth = 4)
    errorbars!(ax, S.E_center_GeV, y, S.E_center_GeV .- S.E_lo_GeV, S.E_hi_GeV .- S.E_center_GeV, direction = :x, color = :black)
    scatter!(ax, S.E_center_GeV, y, color = :black, label = "MINERvA (total)")
    ax
end
for (name, mk) in (("ratio", panel_ratio), ("nu", pos -> panel_sigma(pos, false, Sn, "νμ", (0.0, 1.05))),
                   ("nubar", pos -> panel_sigma(pos, true, Sb, "ν̄μ", (0.0, 0.42))))
    fig = Figure(size = (800, 470))
    ax = mk(fig[1, 1])
    Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
    save("ours/$(name).png", fig)
end

# χ² with the published covariances; for the reference also with the model's prior covariance (from the samples) added
chi2(d, m, C) = (r = d .- m; dot(r, C \ r))
dS = vcat(Sn[!, 4], Sb[!, 4])
mS(p) = vcat(mbin(p, false), mbin(p, true))
MSs = reduce(hcat, [mS(p) for p in samples]); MRs = reduce(hcat, [mratio(p) for p in samples])
rows = [(model = "$(ref) (nominal)", chi2_ratio = chi2(R.ratio, mratio(p0), CR), chi2_sigma = chi2(dS, mS(p0), CS),
         chi2_ratio_with_prior = chi2(R.ratio, mratio(p0), CR + cov(MRs, dims = 2)),
         chi2_sigma_with_prior = chi2(dS, mS(p0), CS + cov(MSs, dims = 2)))]
tbin(t, anti) = [mean(σ_tune(t, anti, binE(b...))) for b in bins]
tratio(t) = [mean(σ_tune(t, true, binE(b...)) .* binE(b...)) / mean(σ_tune(t, false, binE(b...)) .* binE(b...)) for b in bins]
for t in TUNES
    push!(rows, (model = string(t), chi2_ratio = chi2(R.ratio, tratio(t), CR), chi2_sigma = chi2(dS, vcat(tbin(t, false), tbin(t, true)), CS),
                 chi2_ratio_with_prior = NaN, chi2_sigma_with_prior = NaN))
end
out = DataFrame(rows)
out.ndf_ratio .= nrow(R); out.ndf_sigma .= length(dS)
CSV.write("results/chi2.csv", out)
CSV.write("results/ratio.csv", DataFrame(E_lo = R.E_lo_GeV, E_hi = R.E_hi_GeV, data = R.ratio, err = R.total,
                                         reference = round.(mratio(p0), digits = 4), prior_1sigma = round.(vec(std(MRs, dims = 2)), digits = 4)))
show(stdout, out, allrows = true, allcols = true); println()
