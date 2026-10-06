# Tau-neutrino appearance with IceCube DeepCore: IceCube Collaboration, Phys. Rev. D 99, 032007 (2019), Analysis B
# ("DRAGON"), whose event sample is the public three-year high-statistics sample B (doi:10.21234/ac23-ra43) that the
# Newtrinos module `deepcore_3y` is built on. Reproduced: the CC-only ντ normalisation (the module scales ντ CC events;
# the paper's CC+NC normalisation also scales ντ NC events, which are merged with the other flavours in the release).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")

experiments = (deepcore_3y = Newtrinos.deepcore_3y.configure(),)
lik = Newtrinos.generate_likelihood(experiments)
p0 = Newtrinos.get_params(experiments)
# as in the paper: θ₂₃ and Δm²₃₂ free (NO), θ₁₃ = 8.5° ± 0.21°, solar parameters and δCP fixed
fixed = (Δm²₂₁ = 7.53e-5, θ₁₂ = asin(sqrt(0.304)), δCP = 0.0)
p0 = merge(p0, fixed, (θ₂₃ = deg2rad(46.0), Δm²₃₁ = 2.4e-3 + fixed.Δm²₂₁, θ₁₃ = deg2rad(8.5)))
priors = merge(Newtrinos.get_priors(experiments), fixed,
               (θ₂₃ = Uniform(deg2rad(35), deg2rad(55)), Δm²₃₁ = Uniform(1.8e-3, 3.2e-3),
                θ₁₃ = Truncated(Normal(deg2rad(8.5), deg2rad(0.21)), deg2rad(7), deg2rad(10)),
                nutau_cc_norm = Uniform(0.0, 2.0)))
res = Newtrinos.profile(lik, priors, OrderedDict(:nutau_cc_norm => 41), p0, cache_dir = "cache/nutau_cc")
FileIO.save("results/nutau_cc_profile.jld2", Dict("nutau_cc" => res))

x = collect(res.axes.nutau_cc_norm)
d = 2 .* (maximum(res.values.log_posterior) .- res.values.log_posterior)
# best fit by a parabola through the three lowest grid points; Wilks intervals by linear interpolation
i0 = argmin(d); i0 = clamp(i0, 2, length(x) - 1)
a, b, c = x[i0-1:i0+1], d[i0-1:i0+1], nothing
den = (a[1] - a[2]) * (a[1] - a[3]) * (a[2] - a[3])
A = (a[3] * (b[2] - b[1]) + a[2] * (b[1] - b[3]) + a[1] * (b[3] - b[2])) / den
B = (a[3]^2 * (b[1] - b[2]) + a[2]^2 * (b[3] - b[1]) + a[1]^2 * (b[2] - b[3])) / den
best = -B / (2A)
function crossing(level, dir)
    i = argmin(d)
    rng = dir < 0 ? (i:-1:2) : (i:length(x)-1)
    for k in rng
        k2 = k + dir
        if d[k2] >= level
            return x[k] + (level - d[k]) / (d[k2] - d[k]) * (x[k2] - x[k])
        end
    end
    dir < 0 ? x[1] : x[end]                                  # interval reaches the edge of the scan
end
ours = (best_fit = best, low68 = crossing(1.0, -1), high68 = crossing(1.0, 1), low90 = crossing(2.706, -1), high90 = crossing(2.706, 1),
        no_appearance_sigma = sqrt(d[1]))
off = CSV.read("data/official_nutau.csv", DataFrame)
o = off[(off.analysis .== "B (DRAGON)") .& (off.measurement .== "CC"), :][1, :]
CSV.write("results/nutau_cc_intervals.csv", DataFrame([
    (source = "IceCube (FC), Analysis B CC", best_fit = o.best_fit, low68 = o.low68, high68 = o.high68, low90 = o.low90, high90 = o.high90),
    (source = "Newtrinos (Wilks)", best_fit = ours.best_fit, low68 = ours.low68, high68 = ours.high68, low90 = ours.low90, high90 = ours.high90)]))

# ---- figure: Δχ² profile with the official and our intervals
fig = Figure(size = (760, 560))
ax = Axis(fig[1, 1], xlabel = "ντ CC normalisation", ylabel = "Δχ²", limits = (0, 2, 0, 12),
          title = "IceCube DeepCore 3 yr, sample B: ντ CC appearance")
lines!(ax, x, d, color = :dodgerblue, linewidth = 2.5, label = "Newtrinos profile")
hlines!(ax, [1.0, 2.706], color = :gray, linestyle = :dot)
for (y, lo, hi, lw, col, lab) in ((9.5, o.low90, o.high90, 3, (:purple, 0.6), "IceCube Analysis B (FC): 68 %, 90 %"),
                                  (9.5, o.low68, o.high68, 7, :purple, nothing),
                                  (8.0, ours.low90, ours.high90, 3, (:dodgerblue, 0.6), "Newtrinos (Wilks): 68 %, 90 %"),
                                  (8.0, ours.low68, ours.high68, 7, :dodgerblue, nothing))
    lines!(ax, [lo, hi], [y, y], color = col, linewidth = lw, label = lab)
end
scatter!(ax, [o.best_fit], [9.5], color = :purple, markersize = 14)
scatter!(ax, [ours.best_fit], [8.0], color = :dodgerblue, markersize = 14)
vlines!(ax, [1.0], color = :black, linestyle = :dash, linewidth = 1)
axislegend(ax, position = :rt, framevisible = false, labelsize = 11)
save("ours/nutau_cc_profile.png", fig)

# ---- best-fit nuisances vs. the paper's Table (Analysis B, CC fit)
bf = Newtrinos.bestfit(res)
cmp = DataFrame([
    (parameter = "θ₂₃ (°)", newtrinos = rad2deg(bf.θ₂₃), paper = 45.9),
    (parameter = "Δm²₃₂ (1e-3 eV²)", newtrinos = (bf.Δm²₃₁ - fixed.Δm²₂₁) * 1e3, paper = 2.34),
    (parameter = "θ₁₃ (°)", newtrinos = rad2deg(bf.θ₁₃), paper = 8.5),
    (parameter = "NC normalisation", newtrinos = bf.nc_norm, paper = 1.26),
    (parameter = "Δγ (spectral index)", newtrinos = bf.atm_flux_delta_spectral_index, paper = -0.04),
    (parameter = "effective livetime (yr)", newtrinos = bf.deepcore_lifetime, paper = 2.46),
    (parameter = "optical eff., overall", newtrinos = bf.deepcore_opt_eff_overall, paper = 1.04),
    (parameter = "optical eff., lateral (σ)", newtrinos = bf.deepcore_opt_eff_lateral, paper = -0.27),
    (parameter = "optical eff., head-on", newtrinos = bf.deepcore_opt_eff_headon, paper = -1.22),
    (parameter = "bulk ice scattering", newtrinos = bf.deepcore_ice_scattering, paper = 0.973),
    (parameter = "bulk ice absorption", newtrinos = bf.deepcore_ice_absorption, paper = 1.019)])
cmp.newtrinos = round.(cmp.newtrinos, digits = 3)
CSV.write("results/bestfit_nuisances_vs_paper.csv", cmp)
@printf("ντ CC normalisation: best fit %.2f, 68%% (Wilks) %.2f–%.2f, 90%% %.2f–%.2f; IceCube B: %.2f, %.2f–%.2f, %.2f–%.2f\n",
        ours.best_fit, ours.low68, ours.high68, ours.low90, ours.high90, o.best_fit, o.low68, o.high68, o.low90, o.high90)
@printf("rejection of no ντ CC appearance: %.1f σ (Wilks, √Δχ² at 0); paper (Analysis B, CC): 1.4 σ\n", ours.no_appearance_sigma)
show(stdout, cmp, allrows = true); println()
