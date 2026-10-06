# IceCube Upgrade sensitivity with Newtrinos.jl (module `ic_upgrade`, built on the public IceCube Upgrade neutrino MC
# release, doi:10.21234/qfz1-yh02), compared with A. Ishihara et al. (IceCube), PoS(ICRC2019)1031 — the proceedings
# cited by the release: 3-year 90% C.L. sensitivity in (sin²θ₂₃, Δm²₃₂) and the 1-year ντ normalisation sensitivity.
# The official curves are extracted from the vector figures (data/extract_ishihara.py).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 16 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
GRID = get(ENV, "NEWTRINOS_GRID", "15") |> x -> parse(Int, x)

experiments = (ic_upgrade = Newtrinos.ic_upgrade.configure(),)                       # default: energy scale fixed
experiments_es = (ic_upgrade = Newtrinos.ic_upgrade.configure(; energy_scale_uncertainty = 0.02),)   # optional 2 %
p0 = Newtrinos.get_params(experiments)
# truth: the DeepCore 3-year best fit used in the proceedings (sin²θ₂₃ = 0.51, Δm²₃₂ = 2.31e-3 eV², NO)
p0 = merge(p0, (θ₂₃ = asin(sqrt(0.51)), Δm²₃₁ = 2.31e-3 + p0.Δm²₂₁))

# nuisance model: all systematics of the module (Barr flux, NC and ντ normalisation, energy scale, livetime as free
# normalisation), θ₁₃ with a reactor prior; solar parameters and δCP fixed
function base_priors(p)
    pr = Newtrinos.get_priors(experiments)
    merge(pr, (Δm²₂₁ = p.Δm²₂₁, θ₁₂ = p.θ₁₂, δCP = p.δCP, θ₁₃ = Truncated(Normal(0.149, 0.0025), 0.13, 0.17)))
end
CONFIGS = OrderedDict(
    "default" => pr -> pr,                                                                  # module default (fixed)
    "energy_scale_2pct" => pr -> merge(pr, (ic_upgrade_energy_scale = Newtrinos.get_priors(experiments_es).ic_upgrade_energy_scale,)),
)

# ---- 3-year (sin²θ₂₃, Δm²₃₂) profile on Asimov data
asimov = Newtrinos.generate_asimov_data(experiments, p0)
lik = Newtrinos.generate_likelihood(experiments, asimov)
s2 = (0.37, 0.65); d2 = (2.18e-3, 2.44e-3)
scan_pr(pr) = merge(pr, (θ₂₃ = Uniform(asin(sqrt(s2[1])), asin(sqrt(s2[2]))), Δm²₃₁ = Uniform(d2[1] + p0.Δm²₂₁, d2[2] + p0.Δm²₂₁),
                         nutau_cc_norm = p0.nutau_cc_norm))
res = OrderedDict(name => Newtrinos.profile(lik, scan_pr(f(base_priors(p0))), OrderedDict(:θ₂₃ => GRID, :Δm²₃₁ => GRID), p0,
                                            cache_dir = "cache/th23dm_$name") for (name, f) in CONFIGS)

# ---- 1-year ντ normalisation profile (θ₂₃, Δm²₃₁ free)
p1 = merge(p0, (ic_upgrade_lifetime = 1.0,))
asimov1 = Newtrinos.generate_asimov_data(experiments, p1)
lik1 = Newtrinos.generate_likelihood(experiments, asimov1)
tau_pr(pr) = merge(pr, (ic_upgrade_lifetime = Uniform(0.67, 1.33), nutau_cc_norm = Uniform(0.7, 1.3),
                        θ₂₃ = Uniform(asin(sqrt(0.35)), asin(sqrt(0.67))), Δm²₃₁ = Uniform(2.0e-3, 2.8e-3)))
# additionally statistics only: all flux, normalisation and detector nuisances fixed (oscillation parameters free)
is_nuisance(k) = startswith(string(k), "atm_flux") || k in (:nc_norm, :ic_upgrade_lifetime, :ic_upgrade_energy_scale)
stat_only(pr) = merge(pr, NamedTuple(k => p1[k] for k in keys(pr) if is_nuisance(k)))
TAU_CONFIGS = merge(CONFIGS, OrderedDict("statistics_only" => stat_only))
restau = OrderedDict(name => Newtrinos.profile(lik1, f(tau_pr(base_priors(p1))), OrderedDict(:nutau_cc_norm => 31), p1,
                                               cache_dir = "cache/nutau_$name") for (name, f) in TAU_CONFIGS)
FileIO.save("results/ic_upgrade_sensitivity.jld2", Dict("th23dm32" => res, "nutau" => restau))

# ---- figure: 90% C.L. contours (Δχ² = 4.61, Asimov)
Δχ²(r) = 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior)
off = CSV.read("data/ishihara_contours_90cl.csv", DataFrame)
fig = Figure(size = (720, 540))
ax = Axis(fig[1, 1], xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)", limits = (0.30, 0.70, 2.0, 3.2),
          title = "IceCube Upgrade, 3 yr, 90% C.L. (truth: sin²θ₂₃ = 0.51, Δm²₃₂ = 2.31·10⁻³ eV²)")
for (c, col, w) in (("deepcore_3yr_2018", :orange, 3), ("upgrade_3yr", :red, 5))
    s = off[off.contour .== c, :]
    lines!(ax, s.sin2_theta23, s.dm2_32 .* 1e3, color = (col, 0.6), linewidth = w,
           label = c == "upgrade_3yr" ? "Upgrade 3 yr sensitivity (ICRC2019)" : "DeepCore 3 yr 2018 (ICRC2019 figure)")
end
for (name, col, ls) in (("default", :blue, :solid), ("energy_scale_2pct", :purple, :dash))
    r = res[name]
    contour!(ax, sin.(r.axes.θ₂₃) .^ 2, (r.axes.Δm²₃₁ .- p0.Δm²₂₁) .* 1e3, Δχ²(r), levels = [4.61], color = col,
             linewidth = 2, linestyle = ls)
end
axislegend(ax, [LineElement(color = (:red, 0.6), linewidth = 5), LineElement(color = (:orange, 0.6), linewidth = 3),
                LineElement(color = :blue, linewidth = 2), LineElement(color = :purple, linewidth = 2, linestyle = :dash)],
           ["Upgrade 3 yr sensitivity (ICRC2019)", "DeepCore 3 yr 2018 (ICRC2019 figure)",
            "Newtrinos (default: energy scale fixed)", "Newtrinos, optional 2% energy-scale uncertainty"], position = :lt, framevisible = false, labelsize = 11)
save("ours/numu_disappearance_sensitivity.png", fig)

# ---- figure: ντ normalisation 1σ interval (Δχ² = 1)
function interval(r, x)
    d = Δχ²(r); i0 = argmin(d)
    cross(rng) = (j = findfirst(i -> d[i] > 1, rng); j === nothing ? NaN :
                  (k = rng[j]; kp = k - step(rng); x[kp] + (1 - d[kp]) / (d[k] - d[kp]) * (x[k] - x[kp])))
    (cross(i0:-1:1), cross(i0:1:length(d)))
end
ot = CSV.read("data/ishihara_nutau_1sigma.csv", DataFrame)
up = ot[ot.measurement .== "upgrade_1yr", :]
fig = Figure(size = (760, 340))
ax = Axis(fig[1, 1], xlabel = "N_ντ", yticks = ([1, 2, 3, 4], ["Newtrinos, statistics only", "Newtrinos, optional 2% energy scale",
          "Newtrinos (default)", "ICRC2019 Upgrade 1 yr"]), limits = (0.7, 1.3, 0.5, 4.5), title = "IceCube Upgrade 1 yr: ντ normalisation sensitivity (1σ)")
rangebars!(ax, [4], up.low, up.high, direction = :x, color = :red, linewidth = 6)
rows = [(case = "ICRC2019 Upgrade 1 yr", low = up.low[1], high = up.high[1])]
for (yy, name, col) in ((3, "default", :blue), (2, "energy_scale_2pct", :purple), (1, "statistics_only", :gray40))
    r = restau[name]
    lo, hi = interval(r, collect(r.axes.nutau_cc_norm))
    rangebars!(ax, [yy], [lo], [hi], direction = :x, color = col, linewidth = 6)
    push!(rows, (case = "Newtrinos $name", low = lo, high = hi))
end
vlines!(ax, [1.0], color = :gray, linestyle = :dash)
save("ours/nutau_sensitivity.png", fig)

# ---- numbers: contour extent
ext(x, y, d) = (m = d .< 4.61; (minimum(x[m]), maximum(x[m]), minimum(y[m]), maximum(y[m])))
for (name, r) in res
    xs = [sin(t)^2 for t in r.axes.θ₂₃, _ in r.axes.Δm²₃₁]; ys = [m - p0.Δm²₂₁ for _ in r.axes.θ₂₃, m in r.axes.Δm²₃₁]
    a = ext(xs, ys, Δχ²(r))
    @printf("%-16s 90%% region (grid points): sin²θ₂₃ %.3f–%.3f, Δm²₃₂ %.4e–%.4e\n", name, a...)
end
s = off[off.contour .== "upgrade_3yr", :]
@printf("%-16s 90%% contour:              sin²θ₂₃ %.3f–%.3f, Δm²₃₂ %.4e–%.4e\n", "ICRC2019", extrema(s.sin2_theta23)..., extrema(s.dm2_32)...)
tab = DataFrame(rows)
CSV.write("results/nutau_1sigma.csv", tab)
show(stdout, tab); println()
