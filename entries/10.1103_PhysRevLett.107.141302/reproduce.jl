# Reproduction of Borexino, Phys. Rev. Lett. 107, 141302 (2011) [arXiv:1104.1816]: the ⁷Be (862 keV) interaction rate from
# the Phase-I spectrum (Fig. 1 top, MC-based fit) with the Newtrinos module borexino_ph1_spectrum.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DensityInterface, Optim, ADTypes, ForwardDiff, CairoMakie, Printf, CSV, DataFrames

cd(@__DIR__); mkpath("ours"); mkpath("results")
const B = Newtrinos.borexino_ph1_spectrum
# independent flux normalisations, so that ⁷Be can be freed on its own
physics = merge(B.default_physics(), (solar_flux = Newtrinos.solar_flux.configure(
    Newtrinos.solar_flux.SolarFluxConfig(systematics = Newtrinos.solar_flux.SSMPriors())),))
bx = B.configure(physics)
exps = (borexino_ph1_spectrum = bx,)
llh = Newtrinos.generate_likelihood(exps); p0 = Newtrinos.get_params(exps); pr = Newtrinos.get_priors(exps)
# Borexino conditions: ⁷Be free; pp, pep, CNO at the (Newtrinos) SSM + LMA prediction as in Borexino's fit (fixed there);
# backgrounds and energy response profiled with the module priors
nuis = collect(keys(bx.params))
toparams(x) = merge(p0, (solar_norm_be7 = x[1],), NamedTuple{Tuple(nuis)}(Tuple(x[2:end])))
nlp(x) = -(logdensityof(llh, toparams(x)) + sum(logpdf(pr[k], x[1+i]) for (i, k) in enumerate(nuis)))
lo = vcat(0.5, [minimum(pr[k]) for k in nuis]); hi = vcat(1.5, [maximum(pr[k]) for k in nuis])
minimise(f, x0, lo, hi) = (r = optimize(f, lo, hi, clamp.(x0, lo .+ 1e-6, hi .- 1e-6), Fminbox(LBFGS()),
                           Optim.Options(iterations=500, outer_iterations=10, g_tol=1e-6); autodiff=ADTypes.AutoForwardDiff());
                           (Optim.minimizer(r), Optim.minimum(r)))
xb, mb = minimise(nlp, vcat(1.0, [p0[k] for k in nuis]), lo, hi)
pb = toparams(xb)
best = B.get_expected(pb, bx.physics, bx.assets)
per = best.solar.be7_862 / xb[1]                     # 862 keV rate per unit normalisation
grid = collect(range(42.0, 54.0, length = 25))
prof = zeros(length(grid))
Threads.@threads for i in eachindex(grid)
    prof[i] = 2 * (minimise(y -> nlp(vcat(grid[i] / per, y)), xb[2:end], lo[2:end], hi[2:end])[2] - mb)
end
CSV.write("results/be7_profile.csv", DataFrame(be7_862_rate_cpd100t = grid, minus2_dlnL = prof))
g = vcat(grid, best.solar.be7_862); pv = vcat(prof, 0.0); o = sortperm(g); g, pv = g[o], pv[o]
ib = findfirst(==(0.0), pv)
cross(i1, i2) = g[i1] + (1 - pv[i1]) * (g[i2] - g[i1]) / (pv[i2] - pv[i1])
il = findlast(>(1), pv[1:ib]); iu = ib - 1 + findfirst(>(1), pv[ib:end])
lo1, hi1 = cross(il, il + 1), cross(iu - 1, iu)
N = bx.assets.observed.le
dev = 2 * sum(best.le .- N .+ N .* log.(max.(N, 1e-300) ./ best.le))
open("results/summary.txt", "w") do io
    @printf(io, "⁷Be (862 keV) = %.2f [%.2f, %.2f] cpd/100 t (−2ΔlnL = 1); low-energy deviance %.1f for %d bins\n",
            best.solar.be7_862, lo1, hi1, dev, length(N))
    println(io, "Borexino (PRL 107, 141302): ⁷Be (862 keV) = 46.0 ± 1.5 (stat) +1.5/−1.6 (sys) cpd/100 t; Fig. 1 MC-based fit 45.5 ± 1.5")
    println(io, "nuisances: ", join([@sprintf("%s=%.3g", replace(string(k), "borexino_ph1s_" => ""), v) for (k, v) in zip(nuis, xb[2:end])], " "))
end
print(read("results/summary.txt", String))
save("ours/fig1_be7_spectrum.png", bx.plot(pb))
fig = Figure(size = (650, 450))
ax = Axis(fig[1, 1], xlabel = "⁷Be (862 keV) rate [cpd/100 t]", ylabel = "−2Δln L", title = "Borexino Phase I: ⁷Be rate")
vspan!(ax, [46.0 - sqrt(1.5^2 + 1.6^2)], [46.0 + sqrt(1.5^2 + 1.5^2)], color = (:gray, 0.2), label = "Borexino stat ⊕ sys")
vspan!(ax, [44.5], [47.5], color = (:gray, 0.4), label = "Borexino stat")
vlines!(ax, [46.0], color = :black, label = "Borexino")
lines!(ax, g, pv, color = :red, linewidth = 2, label = "Newtrinos")
hlines!(ax, [1.0], color = :gray, linestyle = :dot); ylims!(ax, 0, 8)
axislegend(ax, position = :ct, labelsize = 10)
save("ours/be7_rate_profile.png", fig)
