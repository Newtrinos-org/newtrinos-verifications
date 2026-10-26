# Bayesian T2K 2025 (module `t2k`) compared with the official T2K MCMC chain release (zenodo 17312780,
# arXiv:2506.05889). The official chains are reduced to marginal histograms by data/reduce_chains.py.
# Newtrinos posterior: adaptive importance sampling (MoleWhacker) of the T2K likelihood, separately per mass
# ordering (equal prior mass), reweighted to the priors of the release: flat in sin²θ₁₃, sin²θ₂₃, Δm²₃₂ and δCP,
# Gaussian sin²θ₁₂ = 0.307 ± 0.013 and Δm²₂₁ = (7.53 ± 0.18)·10⁻⁵ eV²; with and without the reactor constraint
# sin²θ₁₃ = 0.0220 ± 0.0007 (applied by reweighting, as for the official chains).
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, StatsBase, BAT, DensityInterface, CSV, DataFrames, CairoMakie, Statistics, Printf, Random, FileIO

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
Random.seed!(20251026)

experiments = (t2k = Newtrinos.t2k.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
priors = Newtrinos.get_priors(experiments)

s2(θ) = sin(θ)^2
θ₁₂c = asin(sqrt(0.307))
DM21 = (7.53e-5, 0.18e-5)
REACTOR = (0.0220, 0.0007)
# sampling priors: oscillation angles flat in θ over ranges well beyond the posterior (reweighted below)
base = merge(priors, (θ₁₂ = Normal(θ₁₂c, 0.013 / sin(2θ₁₂c)), Δm²₂₁ = Normal(DM21...),
                      θ₁₃ = Uniform(asin(sqrt(0.002)), asin(sqrt(0.07))), θ₂₃ = Uniform(asin(sqrt(0.30)), asin(sqrt(0.75))),
                      δCP = Uniform(-π, π)))
# Δm²₃₂ ∈ ±[2.1, 3.0]·10⁻³ eV² (Δm²₃₁ = Δm²₃₂ + Δm²₂₁ at the central Δm²₂₁; equal widths, i.e. equal prior mass)
orderings = (NO = (2.1e-3, 3.0e-3) .+ DM21[1], IO = (-3.0e-3, -2.1e-3) .+ DM21[1])

# ratio of the release's prior to the sampling prior, for one sample
function prior_ratio(x)
    r = sin(2x.θ₁₃) * sin(2x.θ₂₃)                                               # flat in sin²θ → flat in θ: Jacobian
    r *= pdf(Normal(0.307, 0.013), s2(x.θ₁₂)) * sin(2x.θ₁₂) / pdf(base.θ₁₂, x.θ₁₂)  # Gaussian in sin²θ₁₂
    r
end
reactor(x) = pdf(Normal(REACTOR...), s2(x.θ₁₃))

samples = map(keys(orderings), values(orderings)) do mo, range
    file = "cache/samples_$(mo).jld2"
    if isfile(file)
        return mo => FileIO.load(file, "samples")
    end
    posterior = PosteriorMeasure(likelihood, distprod(; merge(base, (Δm²₃₁ = Uniform(range...),))...))
    init = Newtrinos.make_init_samples(posterior, 16, 20_000)
    res = Newtrinos.whack_many_moles(posterior, init; target_samplesize = 250_000, maxiter = 120)
    # importance weights w.r.t. the final proposal mixture (unnormalised: their mean estimates the evidence)
    logw = res.samples_p.logd .- logdensityof.(Ref(res.approx_dist), res.samples_p.v)
    v = res.samples_user.v
    df = DataFrame(dcp = [mod(x.δCP + π, 2π) - π for x in v], sin2th23 = s2.(getproperty.(v, :θ₂₃)),
                   sin2th13 = s2.(getproperty.(v, :θ₁₃)), dm2_32 = [x.Δm²₃₁ - x.Δm²₂₁ for x in v],
                   logw = logw, prior_ratio = prior_ratio.(v), reactor = reactor.(v))
    FileIO.save(file, Dict("samples" => df))
    mo => df
end |> NamedTuple

# combine the orderings: weight ∝ evidence share, i.e. raw importance weight / number of samples of its ordering
m = maximum(maximum(df.logw) for df in samples)
S = vcat((begin
               d = copy(df)
               d.w0 = exp.(d.logw .- m) ./ nrow(d)
               d.ordering .= String(mo)
               d
           end for (mo, df) in pairs(samples))...)
weights = (flat = S.w0 .* S.prior_ratio, reactor = S.w0 .* S.prior_ratio .* S.reactor)
ess(w) = sum(w)^2 / sum(abs2, w)

# official histograms
h1 = CSV.read("data/hist1d.csv", DataFrame)
official = CSV.read("data/summary.csv", DataFrame)
summary = NamedTuple[]
fig = Figure(size = (1500, 780))
vars = (("dcp", "δCP", (-π, π), "both"), ("sin2th23", "sin²θ₂₃", (0.35, 0.70), "both"), ("sin2th13", "sin²θ₁₃", (0.0, 0.05), "both"),
        ("dm2_32", "Δm²₃₂ (NO) [10⁻³ eV²]", (2.25e-3, 2.75e-3), "NO"), ("dm2_32", "Δm²₃₂ (IO) [10⁻³ eV²]", (-2.75e-3, -2.25e-3), "IO"))
for (row, wn) in enumerate((:flat, :reactor))
    w = weights[wn]
    pno = sum(w[S.ordering .== "NO"]) / sum(w)
    off = official[official.weighting .== String(wn), :][1, :]
    push!(summary, (weighting = String(wn), P_NO_newtrinos = pno, P_NO_official = off.P_NO, ess = ess(w),
                    mean_sin2th23_newtrinos = sum(w .* S.sin2th23) / sum(w), mean_sin2th23_official = off.mean_sin2th23,
                    mean_sin2th13_newtrinos = sum(w .* S.sin2th13) / sum(w), mean_sin2th13_official = off.mean_sin2th13))
    for (col, (v, lab, lims, ord)) in enumerate(vars)
        sel = h1[(h1.variable .== v) .& (h1.ordering .== ord) .& (h1.weighting .== String(wn)), :]
        edges = vcat(sel.bin_lo, sel.bin_hi[end])
        mask = ord == "both" ? trues(nrow(S)) : S.ordering .== ord
        h = fit(Histogram, S[mask, v], Weights(w[mask]), edges).weights
        h = h ./ (sum(h) .* diff(edges))
        scale = v == "dm2_32" ? 1e3 : 1.0
        ax = Axis(fig[row, col], xlabel = lab, ylabel = col == 1 ? "posterior density ($(wn == :flat ? "no reactor constraint" : "reactor constraint"))" : "",
                  limits = (lims[1] * scale, lims[2] * scale, 0, nothing), title = col == 1 ? @sprintf("P(NO): Newtrinos %.2f, T2K %.2f", pno, off.P_NO) : "")
        stairs!(ax, edges[1:end-1] .* scale, sel.density ./ scale, step = :post, color = :black, label = "T2K MCMC release")
        stairs!(ax, edges[1:end-1] .* scale, h ./ scale, step = :post, color = :steelblue, linewidth = 2, label = "Newtrinos")
        row == 1 && col == 1 && axislegend(ax, position = :lb, framevisible = false, labelsize = 11)
    end
end
save("ours/fig_marginals.png", fig)
CSV.write("results/summary.csv", DataFrame(summary))
show(stdout, DataFrame(summary), allrows = true); println()
