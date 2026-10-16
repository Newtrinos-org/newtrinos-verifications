# Daya Bay full data release (Zenodo 10.5281/zenodo.17587229, analysis dataset, hdf5) with the full Newtrinos model of
# the Daya Bay analysis (module `dayabay_rewrite`: 8 ADs × 3 periods, IAV, LSNL, energy resolution, backgrounds, reactor
# and detector nuisance parameters), compared with the official reproduction of the analysis
# (https://github.com/dagflow-team/dayabay-analysis, dayabay_model 1.8.0; outputs in data/, see data/official_extract.py).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 32 --project=<path to Newtrinos.jl> reproduce.jl
# Needs curl, sha256sum and unzip to fetch the data release (1.3 MB, CC BY 4.0) on the first run.
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, Printf, Statistics
using BAT: distprod

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const N = Newtrinos
const D = N.dayabay_rewrite

# ---------------------------------------------------------------------------------------------------------------------
# data release: downloaded once and linked where the module expects it
const MODDIR = joinpath(pkgdir(Newtrinos), "src", "experiments", "daya_bay", "daya_bay_3158days")
const ZIP_URL = "https://zenodo.org/records/17587229/files/dayabay_analysis_dataset_hdf5_1-0-0.zip?download=1"
const ZIP_SHA256 = "9b5939e3dbdf1bedf83d5a2f24b292b4d7c6382a367e4f248889612cada7a1bc"
if !isdir(joinpath(MODDIR, "dayabay_data"))
    zipfile = abspath("cache/dayabay_analysis_dataset_hdf5_1-0-0.zip")
    isfile(zipfile) || run(`curl -sSfL -o $zipfile $ZIP_URL`)
    @assert split(read(`sha256sum $zipfile`, String))[1] == ZIP_SHA256 "checksum mismatch of $zipfile"
    run(`unzip -q -o $zipfile -d $(abspath("cache/dataset"))`)
    symlink(abspath("cache/dataset/hdf5"), joinpath(MODDIR, "dayabay_data"))
end

experiments = (dayabay = D.configure(),)
likelihood = N.generate_likelihood(experiments)
p = N.get_params(experiments)
priors = N.get_priors(experiments)
# solar parameters, θ₂₃ and δCP fixed (the official fit constrains the solar parameters, negligible here); normal ordering
priors = merge(priors, (θ₁₂ = p.θ₁₂, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₂₁ = p.Δm²₂₁,
                        θ₁₃ = Uniform(0.139, 0.161), Δm²₃₁ = Uniform(2.38e-3, 2.70e-3)))
s22(θ) = sin(2θ)^2
dm32(m31) = m31 - p.Δm²₂₁
# official real-data pull fit with the Poisson statistic (data/official_fits.csv)
offrow = first(eachrow(filter(r -> r.statistic == "full.pull.chi2poisson", CSV.read("data/official_fits.csv", DataFrame))))
offfits = Dict("sin2_2theta13" => offrow.sin2_2theta13, "sin2_2theta13_minos" => [offrow.sin2_2theta13_minos_lo, offrow.sin2_2theta13_minos_hi],
               "dm2_32" => offrow.dm2_32, "dm2_32_minos" => [offrow.dm2_32_minos_lo, offrow.dm2_32_minos_hi], "chi2_min" => offrow.chi2_min)

# ---------------------------------------------------------------------------------------------------------------------
# 1. prediction at the nominal parameters of the official model, bin by bin
Δm21 = 7.53e-5
p_off = merge(p, (θ₁₃ = asin(sqrt(0.0856)) / 2, θ₁₂ = asin(sqrt(0.851)) / 2, Δm²₂₁ = Δm21, Δm²₃₁ = 2.453e-3 + Δm21))
labels = [("AD$(D.retrieve_AD(k))", "$(D.retrieve_period(k))AD") for k in experiments.dayabay.assets.detector_list]
function prediction_table(pp, official)
    λ = [d.λ for d in experiments.dayabay.forward_model(pp).v]
    obs = experiments.dayabay.assets.observed
    off = CSV.read(official, DataFrame)
    rows = DataFrame(detector = String[], period = String[], bin = Int[], newtrinos = Float64[], official = Float64[], data = Float64[])
    for (i, (det, per)) in enumerate(labels)
        o = off[(off.detector .== det) .& (off.period .== per), :]
        r = (i - 1) * 26 .+ (1:26)
        @assert o.data == obs[r]
        append!(rows, DataFrame(detector = det, period = per, bin = o.bin, newtrinos = λ[r], official = o.pred, data = obs[r]))
    end
    rows
end
nominal = prediction_table(p_off, "data/official_nominal.csv")
CSV.write("results/prediction_nominal.csv", nominal)
ratio = nominal.newtrinos ./ nominal.official

fig = Figure(size = (900, 420))
ax = Axis(fig[1, 1], xlabel = "bin (21 AD-periods × 26 Erec bins)", ylabel = "Newtrinos / official",
          title = "Prediction at the nominal parameters of the official model")
for (i, _) in enumerate(labels)
    r = (i - 1) * 26 .+ (1:26)
    scatterlines!(ax, collect(r), ratio[r], markersize = 4, color = isodd(i) ? :steelblue : :darkorange)
end
hlines!(ax, [1.0], color = :black, linewidth = 0.5)
ylims!(ax, 1 - 5e-4, 1 + 5e-4)
text!(ax, 0.01, 0.97, space = :relative, align = (:left, :top),
      text = @sprintf("all 546 bins within [%.6f, %.6f]; total %.0f vs %.0f", minimum(ratio), maximum(ratio), sum(nominal.newtrinos), sum(nominal.official)))
save("ours/prediction_ratio.png", fig)

# ---------------------------------------------------------------------------------------------------------------------
# 2. profile likelihood scans (all nuisance parameters profiled; −2Δlog L of the Poisson likelihood with Gaussian priors,
#    the same statistic as the official `full.pull.chi2poisson`)
# global best fit first; the scans start from it (much faster convergence of the ~180 nuisance parameters)
mkpath("cache/bestfit")
bf_llh, bf_lpost, bf = N.find_mle_cached(likelihood, distprod(; priors...), p, "cache/bestfit"; adsel = N.ADTypes.AutoForwardDiff())
th13 = N.profile(likelihood, priors, OrderedDict(:θ₁₃ => 23), bf, cache_dir = "cache/th13")
dm31 = N.profile(likelihood, priors, OrderedDict(:Δm²₃₁ => 23), bf, cache_dir = "cache/dm31")
grid2d = N.profile(likelihood, priors, OrderedDict(:θ₁₃ => 15, :Δm²₃₁ => 15), bf, cache_dir = "cache/th13dm31")
FileIO.save("results/dayabay_full.jld2", Dict("th13" => th13, "dm31" => dm31, "th13dm31" => grid2d, "bestfit" => bf))

# a scan point fixes the scanned parameters, so its log posterior lacks their (flat) prior log density, which the global
# best fit includes: add it back before comparing (−2Δlog posterior = Δχ² incl. the Gaussian pull terms)
logpost(r) = r.values.log_posterior .+ sum(k -> logpdf.(priors[k], r.values[k]), keys(r.axes))
lmax = max(bf_lpost, maximum(logpost(th13)), maximum(logpost(dm31)), maximum(logpost(grid2d)))
Δχ²(r) = 2 .* (lmax .- logpost(r))

# official Δχ² table of the data release (profiled over everything but sin²2θ₁₃, Δm²₃₂)
off = CSV.read(joinpath(MODDIR, "DayaBay_DeltaChiSq_NO_3158days.txt"), DataFrame, skipto = 10, delim = " ",
               header = ["ss2th13", "dm232", "dchi2"], ignorerepeated = true)
o_s = reshape(off.ss2th13, 100, 100)[1, :]; o_m = reshape(off.dm232, 100, 100)[:, 1]; o_χ² = Array(reshape(off.dchi2, 100, 100))'

function onesigma(x, y)
    y = y .- minimum(y); i = argmin(y)
    j = findlast(>(1), y[1:i]); lo = j === nothing ? NaN : x[j] + (1 - y[j]) * (x[j+1] - x[j]) / (y[j+1] - y[j])
    j = findfirst(>(1), y[i:end]); hi = j === nothing ? NaN : (j += i - 1; x[j-1] + (1 - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1]))
    lo, hi
end

s_ours = s22.(th13.axes.θ₁₃); χ_s = Δχ²(th13)
m_ours = dm32.(dm31.axes.Δm²₃₁) .* 1e3; χ_m = Δχ²(dm31)
s_off = vec(minimum(o_χ², dims = 2)); m_off = vec(minimum(o_χ², dims = 1))
CSV.write("results/profile_sin2_2theta13.csv", DataFrame(sin2_2theta13 = s_ours, dchi2 = χ_s))
CSV.write("results/profile_dm2_32.csv", DataFrame(dm2_32 = m_ours .* 1e-3, dchi2 = χ_m))

for (file, x, χ, xo, χo, fit, minos, xlabel, scale) in (
        ("ours/profile_sin2_2theta13.png", s_ours, χ_s, o_s, s_off, offfits["sin2_2theta13"], offfits["sin2_2theta13_minos"], "sin²2θ₁₃", 1.0),
        ("ours/profile_dm2_32.png", m_ours, χ_m, o_m .* 1e3, m_off, offfits["dm2_32"] * 1e3, offfits["dm2_32_minos"] .* 1e3, "Δm²₃₂ (10⁻³ eV²)", 1.0))
    fig = Figure(size = (650, 480))
    ax = Axis(fig[1, 1], xlabel = xlabel, ylabel = "Δχ²", title = "Daya Bay full data release, profile likelihood")
    lines!(ax, xo, χo, color = :red, linewidth = 2, label = "Daya Bay Δχ² table (data release)")
    errorbars!(ax, [fit], [1.0], [-minos[1]], [minos[2]], direction = :x, color = :black, whiskerwidth = 8,
               label = "official model (dayabay-analysis), MINOS 1σ")
    scatterlines!(ax, x, χ, color = :blue, markersize = 6, label = "Newtrinos (dayabay_rewrite)")
    hlines!(ax, [1, 4], color = :gray, linestyle = :dot)
    xlims!(ax, extrema(x)...); ylims!(ax, 0, 12)
    axislegend(ax, position = :ct, backgroundcolor = (:white, 0.85))
    save(file, fig)
end

# 2D contours, 1σ/2σ/3σ (2 dof)
fig = Figure(size = (700, 560))
ax = Axis(fig[1, 1], title = "Daya Bay full data release: 1σ, 2σ, 3σ", xlabel = "sin²2θ₁₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
levels = quantile.(Chisq(2), 1 .- 2 .* ccdf.(Normal(), 1:3))
for (lv, ls) in zip(levels, (:solid, :dash, :dot))
    contour!(ax, o_s, o_m .* 1e3, o_χ², levels = [lv], color = :red, linestyle = ls, linewidth = 2)
    contour!(ax, s22.(grid2d.axes.θ₁₃), dm32.(grid2d.axes.Δm²₃₁) .* 1e3, Δχ²(grid2d), levels = [lv], color = :blue, linestyle = ls, linewidth = 2)
end
scatter!(ax, [s22(bf.θ₁₃)], [dm32(bf.Δm²₃₁) * 1e3], color = :blue, marker = :cross, markersize = 14)
scatter!(ax, [offfits["sin2_2theta13"]], [offfits["dm2_32"] * 1e3], color = :red, marker = :xcross, markersize = 14)
lines!(ax, [NaN], [NaN], color = :red, label = "Daya Bay Δχ² table (data release)")
lines!(ax, [NaN], [NaN], color = :blue, label = "Newtrinos (dayabay_rewrite)")
xlims!(ax, extrema(s22.(grid2d.axes.θ₁₃))...); ylims!(ax, extrema(dm32.(grid2d.axes.Δm²₃₁) .* 1e3)...)
axislegend(ax, position = :rt)
save("ours/contour_sin2_2theta13_dm2_32.png", fig)

# ---------------------------------------------------------------------------------------------------------------------
# 3. spectra at the best fit, summed per experimental hall, vs data and the official best fit
# (column `official`: the official model at its own best fit, matched by detector, period and bin)
bestfit = prediction_table(bf, "data/official_bestfit.csv")
CSV.write("results/prediction_bestfit.csv", bestfit)
edges = experiments.dayabay.assets.coarse_binning
hall(det) = "EH" * det[3:3]
fig = Figure(size = (1100, 650))
for (j, eh) in enumerate(("EH1", "EH2", "EH3"))
    sel = hall.(bestfit.detector) .== eh
    g = combine(groupby(bestfit[sel, :], :bin), :data => sum => :data, :newtrinos => sum => :newtrinos, :official => sum => :official)
    w = diff(edges)
    ax = Axis(fig[1, j], title = eh, ylabel = j == 1 ? "events / MeV" : "")
    stairs!(ax, edges, vcat(g.newtrinos ./ w, g.newtrinos[end] / w[end]), step = :post, color = :blue, label = "Newtrinos best fit")
    stairs!(ax, edges, vcat(g.official ./ w, g.official[end] / w[end]), step = :post, color = :red, linestyle = :dash, label = "official best fit")
    c = (edges[1:end-1] .+ edges[2:end]) ./ 2
    errorbars!(ax, c, g.data ./ w, sqrt.(g.data) ./ w, color = :black); scatter!(ax, c, g.data ./ w, color = :black, markersize = 5, label = "data")
    xlims!(ax, 0.7, 12); j == 3 && axislegend(ax, position = :rt)
    ax2 = Axis(fig[2, j], xlabel = "Erec (MeV)", ylabel = j == 1 ? "data / Newtrinos" : "")
    errorbars!(ax2, c, g.data ./ g.newtrinos, sqrt.(g.data) ./ g.newtrinos, color = :black); scatter!(ax2, c, g.data ./ g.newtrinos, color = :black, markersize = 5)
    lines!(ax2, c, g.official ./ g.newtrinos, color = :red, linestyle = :dash)
    hlines!(ax2, [1.0], color = :blue); xlims!(ax2, 0.7, 12); ylims!(ax2, 0.96, 1.04)
    linkxaxes!(ax, ax2)
end
rowsize!(fig.layout, 2, Relative(0.3))
save("ours/spectra_bestfit.png", fig)

# ---------------------------------------------------------------------------------------------------------------------
lo_s, hi_s = onesigma(s_ours, χ_s); lo_m, hi_m = onesigma(m_ours, χ_m)
olo_s, ohi_s = onesigma(o_s, s_off); olo_m, ohi_m = onesigma(o_m .* 1e3, m_off)
deviance = 2 * sum(r -> r.newtrinos - r.data + (r.data > 0 ? r.data * log(r.data / r.newtrinos) : 0.0), eachrow(bestfit))
# pull terms of the Gaussian priors (Normal / MvNormal), as in the official χ² with pull terms
gaussian(d) = d isa Normal || d isa MvNormal || (d isa AbstractVector && eltype(d) <: Normal)
pull(d, x) = d isa AbstractVector ? sum(((xi - mean(di)) / std(di))^2 for (di, xi) in zip(d, x)) :
             d isa MvNormal ? (x .- mean(d))' * (cov(d) \ (x .- mean(d))) : ((x - mean(d)) / std(d))^2
pulls = sum(pull(priors[k], bf[k]) for k in keys(priors) if gaussian(priors[k]))
open("results/summary.txt", "w") do io
    @printf(io, "Prediction at the official nominal parameters: Newtrinos/official per bin in [%.6f, %.6f], total %.1f vs %.1f\n",
            minimum(ratio), maximum(ratio), sum(nominal.newtrinos), sum(nominal.official))
    @printf(io, "Newtrinos best fit:  sin²2θ₁₃ = %.5f, Δχ² = 1 interval [%.5f, %.5f]\n", s22(bf.θ₁₃), lo_s, hi_s)
    @printf(io, "                     Δm²₃₂ = %.4f ×10⁻³ eV², Δχ² = 1 interval [%.4f, %.4f]\n", dm32(bf.Δm²₃₁) * 1e3, lo_m, hi_m)
    @printf(io, "                     χ²min = %.2f (Poisson deviance of the 546 bins %.2f + Gaussian pull terms %.2f)\n", deviance + pulls, deviance, pulls)
    @printf(io, "Official model (dayabay-analysis, full.pull.chi2poisson): sin²2θ₁₃ = %.5f −%.5f +%.5f, Δm²₃₂ = %.4f −%.4f +%.4f ×10⁻³ eV², χ²min = %.2f (incl. pull terms)\n",
            offfits["sin2_2theta13"], -offfits["sin2_2theta13_minos"][1], offfits["sin2_2theta13_minos"][2],
            offfits["dm2_32"] * 1e3, -offfits["dm2_32_minos"][1] * 1e3, offfits["dm2_32_minos"][2] * 1e3, offfits["chi2_min"])
    @printf(io, "Daya Bay Δχ² table (data release): sin²2θ₁₃ Δχ²=1 interval %.5f – %.5f, Δm²₃₂ %.4f – %.4f ×10⁻³ eV²\n", olo_s, ohi_s, olo_m, ohi_m)
end
print(read("results/summary.txt", String))
