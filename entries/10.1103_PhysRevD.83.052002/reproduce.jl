# Reproduction of KamLAND, Phys. Rev. D 83, 052002 (2011) [arXiv:1009.4771], KamLAND-only two-flavour analysis (θ₁₃ = 0)
# with Newtrinos.jl: module `kamland` with `dataset = :kamland2011` (the 2002–2009 spectrum of the paper's Fig. 1, reactor
# prediction from the IAEA PRIS power histories with the common `reactor_flux` model). Compared with KamLAND's official
# Δχ² table (KamLAND data release, θ₁₃ = 0 slice).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 32 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf
using BAT: distprod

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")

experiments = (kamland = Newtrinos.kamland.configure(; dataset = :kamland2011),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = merge(Newtrinos.get_params(experiments), (θ₁₂ = atan(sqrt(0.49)), θ₁₃ = 0.0, Δm²₂₁ = 7.5e-5))
priors = merge(Newtrinos.get_priors(experiments), (Δm²₃₁ = p.Δm²₃₁, θ₁₃ = 0.0, θ₂₃ = p.θ₂₃, δCP = p.δCP,
               θ₁₂ = Uniform(atan(sqrt(0.2)), atan(sqrt(1.0))), Δm²₂₁ = Uniform(6.8e-5, 8.4e-5)))

bf = Newtrinos.find_mle(likelihood, distprod(; priors...), p)[3]
# coarse grids: this entry validates the module, it does not aim at smooth publication-quality contours
th12_dm21 = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 15, :Δm²₂₁ => 17), p, cache_dir = "cache/th12_dm21")
dm21 = Newtrinos.profile(likelihood, priors, OrderedDict(:Δm²₂₁ => 33), p, cache_dir = "cache/dm21")
th12 = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 33), p, cache_dir = "cache/th12")
FileIO.save("results/kamland2011.jld2", Dict("th12_dm21" => th12_dm21, "dm21" => dm21, "th12" => th12, "best_fit" => Dict(pairs(bf))))

Δχ²(r) = 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior)
"best fit and 1σ range from the Δχ² = 1 crossings (linear interpolation between grid points)"
function onesigma(x, y)
    y = y .- minimum(y); i = argmin(y)
    j = findlast(>(1), y[1:i]); lo = j === nothing ? x[1] : x[j] + (1 - y[j]) * (x[j+1] - x[j]) / (y[j+1] - y[j])
    j = findfirst(>(1), y[i:end]); hi = j === nothing ? x[end] : (j += i - 1; x[j-1] + (1 - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1]))
    x[i], hi - x[i], x[i] - lo
end

# official KamLAND-only Δχ²(tan²θ₁₂, sin²θ₁₃, Δm²₂₁) of the data release, slice at θ₁₃ = 0
raw = readdlm(joinpath(pkgdir(Newtrinos), "src/experiments/kamland/kamland_7years/delta_chi2_4th_result.dat"))
sh = (121, 114, 81)
tan2 = reshape(raw[:, 1], sh)[1, :, 1]; dm2 = reshape(raw[:, 3], sh)[1, 1, :]; χ²off = reshape(raw[:, 4], sh)[1, :, :]
χ²off = χ²off .- minimum(χ²off)

dm = onesigma(dm21.axes.Δm²₂₁ .* 1e5, Δχ²(dm21)); t12 = onesigma(tan.(th12.axes.θ₁₂) .^ 2, Δχ²(th12))
dmo = onesigma(dm2 .* 1e5, vec(minimum(χ²off, dims = 1))); t12o = onesigma(tan2, vec(minimum(χ²off, dims = 2)))
CSV.write("results/dm21_profile.csv", DataFrame(dm2_21 = dm21.axes.Δm²₂₁, dchi2 = Δχ²(dm21)))
CSV.write("results/tan2_theta12_profile.csv", DataFrame(tan2_theta12 = tan.(th12.axes.θ₁₂) .^ 2, dchi2 = Δχ²(th12)))
open("results/summary.txt", "w") do io
    @printf io "KamLAND-only, θ13 = 0 (Newtrinos | official Δχ² table | paper)\n"
    @printf io "dm2_21 [1e-5 eV^2]   %.3f +%.3f -%.3f | %.3f +%.3f -%.3f | 7.50 +0.20 -0.20\n" dm... dmo...
    @printf io "tan2_theta12         %.3f +%.3f -%.3f | %.3f +%.3f -%.3f | 0.492 +0.086 -0.067\n" t12... t12o...
    @printf io "geo nu (x reference model 85 U / 21 Th): U %.2f, Th %.2f\n" bf.kamland_geo_u bf.kamland_geo_th
    for k in (:kamland_energy_scale, :kamland_flux_scale, :kamland_alpha_n, :kamland_li9)
        @printf io "pull %-22s %+.2f sigma\n" k bf[k]
    end
end
print(read("results/summary.txt", String))

# Fig. 2(a): 95 %, 99 %, 99.73 % C.L. (2 dof) and Δχ² profiles, Newtrinos (blue) vs official table (black)
fig = Figure(size = (800, 650))
ax = Axis(fig[2, 1], xlabel = "tan²θ₁₂", ylabel = "Δm²₂₁ (10⁻⁴ eV²)")
x = tan.(th12_dm21.axes.θ₁₂) .^ 2; y = th12_dm21.axes.Δm²₂₁ .* 1e4
for (lv, ls) in zip((0.95, 0.99, 0.9973), (:dot, :dash, :solid))
    contour!(ax, tan2, dm2 .* 1e4, χ²off, levels = [quantile(Chisq(2), lv)], color = :black, linestyle = ls, linewidth = 1.5)
    contour!(ax, x, y, Δχ²(th12_dm21), levels = [quantile(Chisq(2), lv)], color = :dodgerblue, linestyle = ls, linewidth = 2.5)
end
lines!(ax, [NaN], [NaN], color = :black, label = "KamLAND (official Δχ² table)")
lines!(ax, [NaN], [NaN], color = :dodgerblue, linewidth = 2.5, label = "Newtrinos")
scatter!(ax, [tan(bf.θ₁₂)^2], [bf.Δm²₂₁ * 1e4], color = :dodgerblue)
axislegend(ax, position = :rt)
xlims!(ax, 0.2, 1.0); ylims!(ax, 0.68, 0.84)
axt = Axis(fig[1, 1], ylabel = "Δχ²")
lines!(axt, tan2, vec(minimum(χ²off, dims = 2)), color = :black, linestyle = :dash, linewidth = 1.5)
lines!(axt, tan.(th12.axes.θ₁₂) .^ 2, Δχ²(th12), color = :dodgerblue, linewidth = 2.5)
hlines!(axt, [1, 4, 9], color = :gray, linewidth = 0.5); xlims!(axt, 0.2, 1.0); ylims!(axt, 0, 12); hidexdecorations!(axt, grid = false)
axr = Axis(fig[2, 2], xlabel = "Δχ²")
lines!(axr, vec(minimum(χ²off, dims = 1)), dm2 .* 1e4, color = :black, linestyle = :dash, linewidth = 1.5)
lines!(axr, Δχ²(dm21), dm21.axes.Δm²₂₁ .* 1e4, color = :dodgerblue, linewidth = 2.5)
vlines!(axr, [1, 4, 9], color = :gray, linewidth = 0.5); ylims!(axr, 0.68, 0.84); xlims!(axr, 0, 20); hideydecorations!(axr, grid = false)
colsize!(fig.layout, 2, Relative(0.25)); rowsize!(fig.layout, 1, Relative(0.25))
Label(fig[0, :], "KamLAND only, θ₁₃ = 0: Newtrinos (blue) vs KamLAND (black); 95 % dotted, 99 % dashed, 99.73 % C.L. solid", fontsize = 13)
save("ours/fig2a_tan2th12_dm21.png", fig)
