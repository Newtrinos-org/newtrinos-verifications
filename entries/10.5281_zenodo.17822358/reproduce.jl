# Bayesian NOvA 2024 (module `nova`) compared with the official Bayesian results of the NOvA 2024 data release
# (zenodo 17822358; Phys. Rev. Lett. 136, 011802): the 2D credible regions released as TGraphs (data/extract_contours.py)
# and the Bayes factors of the paper's supplemental material.
# Newtrinos posterior: adaptive importance sampling (MoleWhacker) of the NOvA likelihood, separately per mass ordering
# (equal prior mass), reweighted to NOvA's priors (uniform in sin²θ₂₃, δCP, Δm²₃₂ and sin²2θ₁₃; θ₁₂, Δm²₂₁ fixed), without
# and with the 1D Daya Bay constraint sin²2θ₁₃ = 0.0851 ± 0.0024 (applied by reweighting).
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, StatsBase, BAT, DensityInterface, CSV, DataFrames, CairoMakie, Statistics, Printf, Random, FileIO

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
Random.seed!(20251026)

experiments = (nova = Newtrinos.nova.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
ref = Newtrinos.nova.ref_params()

s2(θ) = sin(θ)^2
ss2(θ) = sin(2θ)^2
DB = (0.0851, 0.0024)
# sampling priors: flat in the angles over ranges well beyond the posterior (reweighted below)
base = merge(priors, (θ₁₂ = ref.θ₁₂, Δm²₂₁ = ref.Δm²₂₁, θ₁₃ = Uniform(asin(sqrt(0.005)), asin(sqrt(0.06))),
                      θ₂₃ = Uniform(asin(sqrt(0.30)), asin(sqrt(0.72))), δCP = Uniform(0, 2π)))
# Δm²₃₂ ∈ ±[2.0, 3.0]·10⁻³ eV² (Δm²₃₁ = Δm²₃₂ + Δm²₂₁; equal widths, i.e. equal prior mass)
orderings = (NO = (2.0e-3, 3.0e-3) .+ ref.Δm²₂₁, IO = (-3.0e-3, -2.0e-3) .+ ref.Δm²₂₁)
# NOvA's priors relative to the sampling prior: flat in sin²θ₂₃ and sin²2θ₁₃ (Jacobians of flat-in-θ sampling)
prior_ratio(x) = sin(2x.θ₂₃) * abs(2sin(4x.θ₁₃))
daya_bay(x) = pdf(Normal(DB...), ss2(x.θ₁₃))

samples = map(keys(orderings), values(orderings)) do mo, range
    file = "cache/samples_$(mo).jld2"
    isfile(file) && return mo => FileIO.load(file, "samples")
    posterior = PosteriorMeasure(likelihood, distprod(; merge(base, (Δm²₃₁ = Uniform(range...),))...))
    init = Newtrinos.make_init_samples(posterior, 16, 20_000)
    res = Newtrinos.whack_many_moles(posterior, init; target_samplesize = 250_000, maxiter = 120)
    logw = res.samples_p.logd .- logdensityof.(Ref(res.approx_dist), res.samples_p.v)
    v = res.samples_user.v
    df = DataFrame(dcp = [mod(x.δCP, 2π) for x in v], sin2th23 = s2.(getproperty.(v, :θ₂₃)), ss2th13 = ss2.(getproperty.(v, :θ₁₃)),
                   dm2_32 = [x.Δm²₃₁ - x.Δm²₂₁ for x in v], logw = logw, prior_ratio = prior_ratio.(v), daya_bay = daya_bay.(v))
    FileIO.save(file, Dict("samples" => df))
    mo => df
end |> NamedTuple

# combine the orderings: weight ∝ evidence share, i.e. raw importance weight / number of samples of its ordering
m = maximum(maximum(df.logw) for df in samples)
S = vcat((begin
             d = copy(df); d.w0 = exp.(d.logw .- m) ./ nrow(d); d.ordering .= String(mo); d
         end for (mo, df) in pairs(samples))...)
weights = (NOvAOnly = S.w0 .* S.prior_ratio, RCDB1D = S.w0 .* S.prior_ratio .* S.daya_bay)
ess(w) = sum(w)^2 / sum(abs2, w)

summary = NamedTuple[]
official = (NOvAOnly = (P_NO = 0.70, P_UO = 0.56), RCDB1D = (P_NO = 0.77, P_UO = 0.68))   # supplemental Table I
for (k, w) in pairs(weights)
    push!(summary, (constraint = String(k), P_NO_newtrinos = sum(w[S.ordering .== "NO"]) / sum(w), P_NO_official = official[k].P_NO,
                    P_upper_octant_newtrinos = sum(w[S.sin2th23 .> 0.5]) / sum(w), P_upper_octant_official = official[k].P_UO, ess = ess(w)))
end
CSV.write("results/summary.csv", DataFrame(summary))

# highest-posterior-density regions from a weighted, lightly smoothed 2D histogram
function hpd(x, y, w, ex, ey; levels = (0.6827, 0.9), smooth = 1.0)
    h = fit(Histogram, (x, y), Weights(w), (ex, ey)).weights
    if smooth > 0     # Gaussian smoothing with σ = `smooth` bins
        k = [exp(-(i^2 + j^2) / (2smooth^2)) for i in -3:3, j in -3:3]; k ./= sum(k)
        hs = zeros(size(h))
        for i in axes(h, 1), j in axes(h, 2), a in -3:3, b in -3:3
            (1 <= i + a <= size(h, 1) && 1 <= j + b <= size(h, 2)) && (hs[i, j] += k[a+4, b+4] * h[i+a, j+b])
        end
        h = hs
    end
    h ./= sum(h)
    sorted = sort(vec(h), rev = true); c = cumsum(sorted)
    thr = [sorted[searchsortedfirst(c, l)] for l in levels]
    (; xc = (ex[1:end-1] .+ ex[2:end]) ./ 2, yc = (ey[1:end-1] .+ ey[2:end]) ./ 2, h, thr)
end

contours = CSV.read("data/nova2024_credible_contours.csv", DataFrame)
function official!(ax, var, mo, constraint)
    g = contours[(contours.constraint .== constraint) .& (contours.marginalisation .== "conditional") .& (contours.variables .== var) .& (contours.ordering .== mo), :]
    for (lvl, ls) in ((0.6827, :solid), (0.9, :dash))
        for c in groupby(g[g.level .== lvl, :], :component)
            # NOvA's IO files have Δm²₃₂ < 0; plotted as |Δm²₃₂| like ours
            lines!(ax, c.x, var == "ssth23dm32" ? abs.(c.y) : c.y, color = :black, linestyle = ls, linewidth = 1.5)
        end
    end
end
for constraint in (:RCDB1D, :NOvAOnly)
    fig = Figure(size = (1200, 980))
    for (row, mo) in enumerate(("NO", "IO"))
        sel = S.ordering .== mo
        w = weights[constraint][sel]
        sgn = mo == "NO" ? 1 : -1
        # (sin²θ₂₃, |Δm²₃₂|) in 10⁻³ eV², as NOvA's files
        r = hpd(S.sin2th23[sel], sgn .* S.dm2_32[sel] .* 1e3, w, range(0.35, 0.68, length = 67), range(2.2, 2.75, length = 56))
        ax = Axis(fig[row, 1], xlabel = "sin²θ₂₃", ylabel = "|Δm²₃₂| (10⁻³ eV²)", title = "$mo, $(constraint == :RCDB1D ? "1D Daya Bay constraint" : "NOvA only")")
        official!(ax, "ssth23dm32", mo, String(constraint)); ylims!(ax, 2.25, 2.7)
        contour!(ax, r.xc, r.yc, r.h, levels = [r.thr[1]], color = :steelblue, linewidth = 2.5)
        contour!(ax, r.xc, r.yc, r.h, levels = [r.thr[2]], color = :steelblue, linewidth = 2.5, linestyle = :dash)
        # (δCP/π, sin²θ₂₃)
        r2 = hpd(S.dcp[sel] ./ π, S.sin2th23[sel], w, range(0, 2, length = 61), range(0.35, 0.68, length = 56))
        ax2 = Axis(fig[row, 2], xlabel = "δCP / π", ylabel = "sin²θ₂₃", title = "$mo, $(constraint == :RCDB1D ? "1D Daya Bay constraint" : "NOvA only")")
        official!(ax2, "dcpssth23", mo, String(constraint))
        contour!(ax2, r2.xc, r2.yc, r2.h, levels = [r2.thr[1]], color = :steelblue, linewidth = 2.5)
        contour!(ax2, r2.xc, r2.yc, r2.h, levels = [r2.thr[2]], color = :steelblue, linewidth = 2.5, linestyle = :dash)
    end
    Legend(fig[3, 1:2], [LineElement(color = :black), LineElement(color = :black, linestyle = :dash),
                         LineElement(color = :steelblue, linewidth = 2.5), LineElement(color = :steelblue, linewidth = 2.5, linestyle = :dash)],
           ["NOvA 68.27 %", "NOvA 90 %", "Newtrinos 68.27 %", "Newtrinos 90 %"], orientation = :horizontal, framevisible = false)
    save("ours/fig_credible_regions_$(constraint).png", fig)
end
show(stdout, DataFrame(summary), allrows = true); println()
