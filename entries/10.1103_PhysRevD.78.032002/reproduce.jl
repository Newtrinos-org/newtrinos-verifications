# Super-Kamiokande II solar (J. P. Cravens et al., Phys. Rev. D 78, 032002 (2008)) with Newtrinos.jl (modules
# `sk1_solar`, `sk2_solar`), compared with the SK-II data release (www-sk.icrr.u-tokyo.ac.jp/sk/lowe/sk2_data/, via the
# Internet Archive): the energy-correlated systematic uncertainties (ecorrerr.txt) and the χ² map of the SK-I + SK-II
# combined analysis with free ⁸B and hep fluxes (skc_exclude_dn_flux.txt, columns tan²θ, Δm², χ² kept in data/).
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Distributions, DelimitedFiles, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const SF = Newtrinos.solar_flux

# ---------------- (1) energy-correlated systematics of SK-II: response-based ±1σ vs SK's table ----------------
rows = filter(l -> occursin(r"^\s*\d", l), readlines("data/ecorrerr.txt"))
sk = [(parse.(Float64, split(replace(l, r"^\s*([\d.]+)-\s*([\d.]+)" => s"\1 \2")))) for l in rows]    # E_lo E_hi B+ B- S+ S- R+ R-
mod = Newtrinos.sk2_solar
phys = Newtrinos.solar_common.default_physics()
osc_p = (θ₁₂ = asin(sqrt(0.31)), Δm²₂₁ = 7.5e-5)
# the SK-II module with the response-based treatment (no table) and as configured (SK's table)
function shifts(phase)
    exp = Newtrinos.sk_solar_common.configure(phase, phys)
    p0 = merge(Newtrinos.get_params(exp), osc_p)
    a = exp.assets
    idx = [something(findfirst(i -> a.samples.E_lo[i] == r[1] && a.samples.zenith[i] in ("all", "day"), eachindex(a.samples.E_lo))) for r in sk]
    μ(q) = Newtrinos.sk_solar_common.get_expected(phase, q, exp.physics, a)[idx]
    m0 = μ(p0)
    sh(par, v) = 100 .* (μ(merge(p0, NamedTuple{(par,)}((v,)))) ./ m0 .- 1)
    (S = (sh(:sk2_solar_escale, 1.0), sh(:sk2_solar_escale, -1.0)), R = (sh(:sk2_solar_ereso, 1.0), sh(:sk2_solar_ereso, -1.0)),
     B = (sh(:solar_b8_shape, 1.0), sh(:solar_b8_shape, -1.0)))
end
table = Newtrinos.sk_solar_common.SKPhase(; (k => getfield(mod.PHASE, k) for k in fieldnames(typeof(mod.PHASE)))...)
response = Newtrinos.sk_solar_common.SKPhase(; (k => (k == :correlated_file ? nothing : getfield(mod.PHASE, k)) for k in fieldnames(typeof(mod.PHASE)))...)
st, sr = shifts(table), shifts(response)
Ec = [(r[1] + r[2]) / 2 for r in sk]
fig = Figure(size = (1300, 440))
out = NamedTuple[]
for (col, (key, lab, c)) in enumerate(((:S, "energy scale", 5), (:R, "energy resolution", 7), (:B, "⁸B spectrum shape", 3)))
    ax = Axis(fig[1, col], xlabel = "recoil electron total energy (MeV)", ylabel = col == 1 ? "rate change for ±1σ (%)" : "", title = lab)
    scatter!(ax, Ec, getindex.(sk, c), color = :black, marker = :utriangle, label = "SK-II data release (+1σ)")
    scatter!(ax, Ec, getindex.(sk, c + 1), color = :black, marker = :dtriangle, label = "SK-II data release (−1σ)")
    lines!(ax, Ec, getproperty(sr, key)[1], color = :darkorange, label = "Newtrinos, response-based")
    lines!(ax, Ec, getproperty(sr, key)[2], color = :darkorange)
    if key != :B
        lines!(ax, Ec, getproperty(st, key)[1], color = :steelblue, linestyle = :dash, linewidth = 2, label = "Newtrinos, SK table (default)")
        lines!(ax, Ec, getproperty(st, key)[2], color = :steelblue, linestyle = :dash, linewidth = 2)
    end
    col == 1 && axislegend(ax, position = :lt, framevisible = false, labelsize = 10)
    for (i, e) in enumerate(Ec)
        push!(out, (systematic = lab, E = e, sk_plus = sk[i][c], sk_minus = sk[i][c+1], response_plus = getproperty(sr, key)[1][i],
                    response_minus = getproperty(sr, key)[2][i], newtrinos_plus = getproperty(key == :B ? sr : st, key)[1][i],
                    newtrinos_minus = getproperty(key == :B ? sr : st, key)[2][i]))
    end
end
save("ours/fig_correlated_systematics.png", fig)
CSV.write("results/correlated_systematics.csv", DataFrame(out))

# ---------------- (2) SK-I + SK-II with free ⁸B and hep fluxes: two-flavour χ² over (tan²θ, Δm²) ----------------
physf = merge(phys, (solar_flux = SF.configure(SF.SolarFluxConfig(systematics = SF.SSMPriors())),))
experiments = (sk1_solar = Newtrinos.sk1_solar.configure(physf), sk2_solar = Newtrinos.sk2_solar.configure(physf))
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
fixed = (θ₁₃ = 0.0, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₃₁ = p.Δm²₃₁)
priors = merge(Newtrinos.get_priors(experiments), fixed,
               (solar_norm_b8 = Uniform(0.05, 5.0), solar_norm_hep = Uniform(0.0, 50.0)))   # β, η free as in SK's χ²
t2 = 10 .^ range(-4, 2, length = 25)
dm = 10 .^ range(-9, -3, length = 25)
grid = vec([(a, b) for a in t2, b in dm])
χ2 = fill(NaN, length(grid))
Threads.@threads for k in eachindex(grid)
    a, b = grid[k]
    th = atan(sqrt(a))
    pr = Newtrinos.BAT.distprod(; merge(priors, (θ₁₂ = th, Δm²₂₁ = b))...)
    best = -Inf
    for b8 in (0.5, 1.5)            # two starts of the ⁸B flux scale
        q = merge(p, fixed, (θ₁₂ = th, Δm²₂₁ = b, solar_norm_b8 = b8))
        llh, lp, r = Newtrinos.find_mle_cached(likelihood, pr, q, "cache")
        best = max(best, llh)
    end
    χ2[k] = -2best
end
res = DataFrame(tan2theta = first.(grid), dm2 = last.(grid), chi2 = χ2)
CSV.write("results/sk12_chi2_grid.csv", res)

# SK's map on its own 301 × 301 grid
m = readdlm("data/skc_exclude_dn_flux_chi2.txt", comments = true, comment_char = '*')[:, 1:3]
skt, skd = sort(unique(m[:, 1])), sort(unique(m[:, 2]))
SKχ = fill(NaN, length(skt), length(skd))
for r in eachrow(m)
    SKχ[searchsortedfirst(skt, r[1]), searchsortedfirst(skd, r[2])] = r[3]
end
Z = reshape(χ2, length(t2), length(dm))
fig = Figure(size = (760, 760))
ax = Axis(fig[1, 1], xscale = log10, yscale = log10, xlabel = "tan²θ", ylabel = "Δm² (eV²)",
          title = "SK-I + SK-II, ⁸B and hep free: excluded at 95 % CL (Δχ² > 5.99)", limits = (1e-4, 1e2, 1e-9, 1e-3))
contourf!(ax, skt, skd, SKχ .- minimum(SKχ), levels = [5.99, 1e9], colormap = [(:lightskyblue, 0.8)])
contour!(ax, skt, skd, SKχ .- minimum(SKχ), levels = [5.99], color = :black, linewidth = 1)
contour!(ax, t2, dm, Z .- minimum(Z), levels = [5.99], color = :red, linewidth = 2.5)
vlines!(ax, [1.0], color = :gray, linestyle = :dash)
Legend(fig[2, 1], [PolyElement(color = (:lightskyblue, 0.8), strokecolor = :black, strokewidth = 1), LineElement(color = :red, linewidth = 2.5)],
       ["SK data release χ² map (SK-I + SK-II)", "Newtrinos sk1_solar + sk2_solar"], orientation = :horizontal, framevisible = false)
save("ours/fig_sk12_excluded.png", fig)
# summary: minima, and agreement of the exclusion on our grid points (SK map interpolated in log space)
function skinterp(a, b)
    i = clamp(searchsortedlast(skt, a), 1, length(skt) - 1); j = clamp(searchsortedlast(skd, b), 1, length(skd) - 1)
    u = (log(a) - log(skt[i])) / (log(skt[i+1]) - log(skt[i])); v = (log(b) - log(skd[j])) / (log(skd[j+1]) - log(skd[j]))
    (1 - u) * (1 - v) * SKχ[i, j] + u * (1 - v) * SKχ[i+1, j] + (1 - u) * v * SKχ[i, j+1] + u * v * SKχ[i+1, j+1]
end
skg = [skinterp(a, b) for (a, b) in grid]
exc_nt = (χ2 .- minimum(χ2)) .> 5.99; exc_sk = (skg .- minimum(SKχ)) .> 5.99
i_nt, i_sk = argmin(χ2), argmin(m[:, 3])
summary = DataFrame(quantity = ["best fit tan²θ", "best fit Δm² [eV²]", "fraction of grid points excluded",
                                "grid points with the same exclusion verdict", "Pearson r of Δχ² on the grid (both capped at 25)"],
                    newtrinos = [grid[i_nt][1], grid[i_nt][2], mean(exc_nt), mean(exc_nt .== exc_sk), cor(min.(χ2 .- minimum(χ2), 25), min.(skg .- minimum(SKχ), 25))],
                    sk = [m[i_sk, 1], m[i_sk, 2], mean(exc_sk), NaN, NaN])
CSV.write("results/summary.csv", summary)
show(stdout, summary, allrows = true); println()
