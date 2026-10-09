# Reproduction of KamLAND, Phys. Rev. D 88, 033001 (2013) [arXiv:1303.4667], KamLAND-only three-flavour analysis with
# Newtrinos.jl (module `kamland`: the three data-taking periods, reactor prediction from the IAEA PRIS power histories
# with the common `reactor_flux` model). θ₁₃ free, as in the paper's KamLAND-only fit.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 32 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, Printf
using BAT: distprod

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const K = Newtrinos.kamland

experiments = (kamland = K.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = merge(Newtrinos.get_params(experiments), (θ₁₂ = atan(sqrt(0.48)), θ₁₃ = asin(sqrt(0.02)), Δm²₂₁ = 7.5e-5))
priors = merge(Newtrinos.get_priors(experiments), (Δm²₃₁ = p.Δm²₃₁, θ₂₃ = p.θ₂₃, δCP = p.δCP,
               θ₁₃ = Uniform(0.0, asin(sqrt(0.15))), θ₁₂ = Uniform(atan(sqrt(0.2)), atan(sqrt(1.0))), Δm²₂₁ = Uniform(6.6e-5, 8.6e-5)))

best = Newtrinos.find_mle(likelihood, distprod(; priors...), p)
bf = best[3]
# coarse grids: this entry validates the module, it does not aim at smooth publication-quality contours
th12_dm21 = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 15, :Δm²₂₁ => 17), p, cache_dir = "cache/th12_dm21")
dm21 = Newtrinos.profile(likelihood, priors, OrderedDict(:Δm²₂₁ => 41), p, cache_dir = "cache/dm21")
th12 = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 33), p, cache_dir = "cache/th12")
FileIO.save("results/kamland2013.jld2", Dict("th12_dm21" => th12_dm21, "dm21" => dm21, "th12" => th12, "best_fit" => Dict(pairs(bf))))

Δχ²(r) = 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior)
"best fit and 1σ range from the Δχ² = 1 crossings (linear interpolation between grid points)"
function onesigma(x, y)
    y = y .- minimum(y); i = argmin(y)
    j = findlast(>(1), y[1:i]); lo = j === nothing ? x[1] : x[j] + (1 - y[j]) * (x[j+1] - x[j]) / (y[j+1] - y[j])
    j = findfirst(>(1), y[i:end]); hi = j === nothing ? x[end] : (j += i - 1; x[j-1] + (1 - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1]))
    x[i], hi - x[i], x[i] - lo
end
dm = onesigma(dm21.axes.Δm²₂₁ .* 1e5, Δχ²(dm21)); t12 = onesigma(tan.(th12.axes.θ₁₂) .^ 2, Δχ²(th12))
CSV.write("results/dm21_profile.csv", DataFrame(dm2_21 = dm21.axes.Δm²₂₁, dchi2 = Δχ²(dm21)))
CSV.write("results/tan2_theta12_profile.csv", DataFrame(tan2_theta12 = tan.(th12.axes.θ₁₂) .^ 2, dchi2 = Δχ²(th12)))
CSV.write("results/th12_dm21.csv", DataFrame(tan2_theta12 = vec([tan(a)^2 for a in th12_dm21.axes.θ₁₂, b in th12_dm21.axes.Δm²₂₁]),
                                              dm2_21 = vec([b for a in th12_dm21.axes.θ₁₂, b in th12_dm21.axes.Δm²₂₁]), dchi2 = vec(Δχ²(th12_dm21))))
open("results/summary.txt", "w") do io
    @printf io "KamLAND-only, θ13 free (Newtrinos | KamLAND PRD 88, 033001)\n"
    @printf io "dm2_21 [1e-5 eV^2]   %.3f +%.3f -%.3f | 7.54 +0.19 -0.18\n" dm...
    @printf io "tan2_theta12         %.3f +%.3f -%.3f | 0.481 +0.092 -0.080\n" t12...
    @printf io "sin2_theta13 (best)  %.3f | 0.010 +0.033 -0.034\n" sin(bf.θ₁₃)^2
    @printf io "geo nu (best fit, x reference model 109 U / 27 Th): U %.2f, Th %.2f | 116 U, 8 Th events\n" bf.kamland_geo_u bf.kamland_geo_th
    for k in (:kamland_energy_scale, :kamland_flux_scale, :kamland_flux_scale_late, :kamland_alpha_n, :kamland_li9)
        @printf io "pull %-24s %+.2f sigma\n" k bf[k]
    end
end
print(read("results/summary.txt", String))

# Fig. 4(a): KamLAND-only regions (95 %, 99 %, 99.73 % C.L., 2 dof) and Δχ² profiles
fig = Figure(size = (800, 650))
ax = Axis(fig[2, 1], xlabel = "tan²θ₁₂", ylabel = "Δm²₂₁ (10⁻⁴ eV²)")
x = tan.(th12_dm21.axes.θ₁₂) .^ 2; y = th12_dm21.axes.Δm²₂₁ .* 1e4
for (lv, ls) in zip((0.95, 0.99, 0.9973), (:dot, :dash, :solid))
    contour!(ax, x, y, Δχ²(th12_dm21), levels = [quantile(Chisq(2), lv)], color = :black, linestyle = ls, linewidth = 2)
end
scatter!(ax, [tan(bf.θ₁₂)^2], [bf.Δm²₂₁ * 1e4], color = :black, label = "Newtrinos best fit")
scatter!(ax, [0.481], [0.754], color = :red, marker = :star5, label = "KamLAND best fit")
axislegend(ax, position = :rt)
xlims!(ax, 0.2, 1.0); ylims!(ax, 0.66, 0.86)
axt = Axis(fig[1, 1], ylabel = "Δχ²"); lines!(axt, tan.(th12.axes.θ₁₂) .^ 2, Δχ²(th12), color = :black)
hlines!(axt, [1, 4, 9], color = :gray, linewidth = 0.5); xlims!(axt, 0.2, 1.0); ylims!(axt, 0, 12); hidexdecorations!(axt, grid = false)
axr = Axis(fig[2, 2], xlabel = "Δχ²"); lines!(axr, Δχ²(dm21), dm21.axes.Δm²₂₁ .* 1e4, color = :black)
vlines!(axr, [1, 4, 9], color = :gray, linewidth = 0.5); ylims!(axr, 0.66, 0.86); xlims!(axr, 0, 20); hideydecorations!(axr, grid = false)
colsize!(fig.layout, 2, Relative(0.25)); rowsize!(fig.layout, 1, Relative(0.25))
Label(fig[0, :], "KamLAND only, θ₁₃ free (Newtrinos; zoomed on the allowed region)", fontsize = 14)
save("ours/fig4a_th12_dm21.png", fig)

# Fig. 3: prompt spectra per period with the best-fit contributions (events / 0.425 MeV / day)
a = experiments.kamland.assets; phys = experiments.kamland.physics
μ = K.get_expected(bf, phys, a)
reac = K.reactor_events(bf, phys, a; ε = K.ESCALE_UNC * bf.kamland_energy_scale)
Ec = 0.5 .* (K.EDGES[1:end-1] .+ K.EDGES[2:end])
fig = Figure(size = (650, 950))
for p in 1:3
    pd = a.periods[p]; d = pd.days
    rate = 1 + K.RATE_UNC * bf.kamland_flux_scale + (p > 1 ? K.RATE_UNC_LATE * bf.kamland_flux_scale_late : 0.0)
    acc = pd.acc; an = acc .+ pd.alpha_n .* (1 + K.ALPHA_N_UNC * bf.kamland_alpha_n)
    geo = an .+ pd.geo_u .* bf.kamland_geo_u .+ pd.geo_th .* bf.kamland_geo_th
    ax = Axis(fig[p, 1], ylabel = "events / 0.425 MeV / day", title = "Period $p ($(Int(d)) live days)",
              xlabel = p == 3 ? "E_p (MeV)" : "")
    edges = K.EDGES
    stairs!(ax, edges, vcat(geo, geo[end]) ./ d, step = :post, color = :blue, linewidth = 1)
    barplot!(ax, Ec, geo ./ d, width = 0.425, gap = 0, color = (:blue, 0.25), label = "best-fit geo ν̄e")
    barplot!(ax, Ec, an ./ d, width = 0.425, gap = 0, color = :lightgreen, label = "¹³C(α,n)¹⁶O")
    barplot!(ax, Ec, acc ./ d, width = 0.425, gap = 0, color = :salmon, label = "accidental")
    stairs!(ax, edges, vcat(reac[p] .* rate, 0) ./ d, step = :post, color = :purple, linestyle = :dash, linewidth = 2, label = "best-fit reactor ν̄e")
    stairs!(ax, edges, vcat(μ[p], μ[p][end]) ./ d, step = :post, color = :dodgerblue, linewidth = 2.5, label = "total")
    n = pd.observed
    errorbars!(ax, Ec, n ./ d, sqrt.(n) ./ d, color = :black); scatter!(ax, Ec, n ./ d, color = :black, label = "KamLAND data")
    xlims!(ax, 0, 8.5); ylims!(ax, 0, 0.18)
    p == 3 && axislegend(ax, position = :rt, labelsize = 11)
end
save("ours/fig3_spectra.png", fig)
