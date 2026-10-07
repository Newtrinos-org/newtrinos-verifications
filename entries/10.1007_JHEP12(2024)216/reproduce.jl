# Reproduction of NuFIT 6.0, JHEP 12 (2024) 216 [arXiv:2410.05380], Fig. 11: solar vs KamLAND determination of
# θ₁₂ and Δm²₂₁, with Newtrinos.jl (solar modules `chlorine`, `gallex_gno`, `sage`, `sno`, `sk1_solar`–`sk4_solar`,
# `borexino_ph1`–`borexino_ph3`; `kamland`). Borexino enters through its published interaction rates.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 64 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, LinearAlgebra, Interpolations

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
BLAS.set_num_threads(1)

# CSV export of a profile scan: one row per grid point, ASCII column names, oscillation parameters in physical
# units (sin²θ, Δm² in eV²), dchi2 = -2Δlog L w.r.t. `ref` (default: the scan's best fit)
function save_csv(path, r; ref = maximum(r.values.log_posterior))
    ascii(k) = replace(string(k), "θ" => "theta", "Δm²" => "dm2_", "δ" => "d", "₁" => "1", "₂" => "2", "₃" => "3", "CP" => "cp")
    cols = OrderedDict{String,Any}()
    for (k, v) in pairs(r.values)
        x = vec(v)
        startswith(string(k), "θ") && (cols["sin2_" * ascii(k)] = sin.(x) .^ 2)
        cols[ascii(k)] = x
    end
    cols["dchi2"] = 2 .* (ref .- vec(r.values.log_posterior))
    haskey(cols, "llh") && (cols["log_likelihood"] = pop!(cols, "llh"))
    first_cols = filter(in(keys(cols)), ["sin2_theta12", "dm2_21", "dchi2", "log_likelihood", "log_posterior"])
    CSV.write(path, DataFrame(cols)[:, vcat(first_cols, setdiff(collect(keys(cols)), first_cols))])
end

# NuFIT Fig. 11 conditions: sin²θ₁₃ = 0.0222 fixed; θ₂₃, δCP, Δm²₃₁ irrelevant and fixed; all flux, cross-section and
# detector nuisance parameters profiled with their priors
function setup(experiments)
    p = Newtrinos.get_params(experiments)
    priors = Newtrinos.get_priors(experiments)
    θ13 = asin(sqrt(0.0222))
    p = merge(p, (θ₁₃ = θ13, Δm²₂₁ = 6e-5))
    priors = merge(priors, (θ₁₃ = θ13, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₃₁ = p.Δm²₃₁,
                            θ₁₂ = Uniform(asin(sqrt(0.18)), asin(sqrt(0.42))), Δm²₂₁ = Uniform(1.5e-5, 1.4e-4)))
    Newtrinos.generate_likelihood(experiments), priors, p
end

physics = Newtrinos.solar_common.default_physics()
solar = NamedTuple(name => getproperty(Newtrinos, name).configure(physics)
                   for name in (:chlorine, :gallex_gno, :sage, :sno, :sk1_solar, :sk2_solar, :sk3_solar, :sk4_solar,
                                :borexino_ph1, :borexino_ph2, :borexino_ph3))
kamland = (kamland = Newtrinos.kamland.configure(),)

llh, priors, p = setup(solar)
# coarse grids: this entry validates the setup, it does not aim at smooth publication-quality contours
solar_2d = Newtrinos.profile(llh, priors, OrderedDict(:θ₁₂ => 10, :Δm²₂₁ => 10), p, cache_dir = "cache/solar_2d")
priors_dm = merge(priors, (Δm²₂₁ = Uniform(2e-5, 1e-4),))
solar_dm = Newtrinos.profile(llh, priors_dm, OrderedDict(:Δm²₂₁ => 17), p, cache_dir = "cache/solar_dm")
llh, priors, p = setup(kamland)
kamland_dm = Newtrinos.profile(llh, merge(priors, (Δm²₂₁ = Uniform(2e-5, 1e-4),)), OrderedDict(:Δm²₂₁ => 161), merge(p, (Δm²₂₁ = 7.5e-5,)),
                               cache_dir = "cache/kamland_dm")

FileIO.save("results/nufit6_fig11.jld2", Dict("solar_th12_dm21" => solar_2d, "solar_dm21" => solar_dm, "kamland_dm21" => kamland_dm))
save_csv("results/solar_th12_dm21.csv", solar_2d)
save_csv("results/solar_dm21.csv", solar_dm)
save_csv("results/kamland_dm21.csv", kamland_dm)

Δχ²(r) = 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior)

fig = Figure(size = (900, 520))
# left: solar regions, 1σ–5σ (2 dof), and KamLAND best fit
ax = Axis(fig[1, 1], title = "Newtrinos solar (Cl, Ga, SK I–IV, SNO, Borexino), sin²θ₁₃ = 0.0222",
          xlabel = "sin²θ₁₂", ylabel = "Δm²₂₁ (10⁻⁵ eV²)", titlesize = 13)
x = sin.(solar_2d.axes.θ₁₂) .^ 2; y = solar_2d.axes.Δm²₂₁ .* 1e5
levels = [quantile(Chisq(2), 1 - 2 * ccdf(Normal(), n)) for n in 1:5]
contourf!(ax, x, y, Δχ²(solar_2d), levels = vcat(0, levels), colormap = [:red, :pink, :blue, :magenta, :cyan])
contour!(ax, x, y, Δχ²(solar_2d), levels = levels, color = :black, linewidth = 0.5)
i = argmax(solar_2d.values.log_posterior)
scatter!(ax, [x[i[1]]], [y[i[2]]], color = :white, strokecolor = :black, strokewidth = 1)
kl_best = kamland_dm.axes.Δm²₂₁[argmax(kamland_dm.values.log_posterior)]
hlines!(ax, [kl_best * 1e5], color = :green, linestyle = :dash, label = "KamLAND best fit Δm²₂₁")
xlims!(ax, 0.2, 0.4); ylims!(ax, 0, 14)
axislegend(ax, position = :rt, labelsize = 11)
# right: Δχ²(Δm²₂₁), θ₁₂ profiled
ax2 = Axis(fig[1, 2], xlabel = "Δm²₂₁ (10⁻⁵ eV²)", ylabel = "Δχ²", title = "θ₁₂ and nuisance parameters profiled", titlesize = 13)
lines!(ax2, solar_dm.axes.Δm²₂₁ .* 1e5, Δχ²(solar_dm), color = :red, label = "Solar (B23 MB22-met)")
lines!(ax2, kamland_dm.axes.Δm²₂₁ .* 1e5, Δχ²(kamland_dm), color = :green, label = "KamLAND")
hlines!(ax2, [1, 4, 9], color = :gray, linewidth = 0.5)
xlims!(ax2, 2, 10); ylims!(ax2, 0, 12)
axislegend(ax2, position = :rt, labelsize = 11)
save("ours/fig11_solar_kamland.png", fig)

# key numbers for the summary
# cubic spline through the coarse 1D solar profile
dm_ax = range(first(solar_dm.axes.Δm²₂₁), last(solar_dm.axes.Δm²₂₁), length = length(solar_dm.axes.Δm²₂₁))
@assert collect(dm_ax) ≈ solar_dm.axes.Δm²₂₁
χ²_solar = cubic_spline_interpolation(dm_ax, Δχ²(solar_dm))
dm_fine = range(first(dm_ax), last(dm_ax), length = 2001)
best = dm_fine[argmin(χ²_solar.(dm_fine))]
at_kl = χ²_solar(kl_best) - minimum(χ²_solar.(dm_fine))
open("results/summary.txt", "w") do io
    println(io, "solar best-fit sin2_theta12 = ", round(x[i[1]], digits = 3), ", dm2_21 = ", round(y[i[2]], digits = 2), "e-5 eV^2 (2D grid)")
    println(io, "solar best-fit dm2_21 (1D profile) = ", round(best * 1e5, digits = 2), "e-5 eV^2")
    println(io, "KamLAND best-fit dm2_21 = ", round(kl_best * 1e5, digits = 2), "e-5 eV^2")
    println(io, "solar dchi2 at the KamLAND best fit = ", round(at_kl, digits = 2))
end
print(read("results/summary.txt", String))
