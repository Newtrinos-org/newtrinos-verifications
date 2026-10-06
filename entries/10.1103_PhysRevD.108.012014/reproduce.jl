# Reproduction of IceCube, Phys. Rev. D 108, 012014 (2023) [arXiv:2304.12236] with Newtrinos.jl (module `deepcore`),
# using the public DeepCore verification sample (Harvard Dataverse doi:10.7910/DVN/B4RITM).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")

# CSV export of a profile scan: one row per grid point, ASCII column names, oscillation parameters in physical
# units (sin²θ, Δm²₃₂ = Δm²₃₁ − Δm²₂₁ in eV², δCP in rad), dchi2 = -2Δlog L w.r.t. `ref` (default: the scan's best fit)
function save_csv(path, r; ref = maximum(r.values.log_posterior))
    ascii(k) = replace(string(k), "θ" => "theta", "Δm²" => "dm2_", "δ" => "d", "₁" => "1", "₂" => "2", "₃" => "3", "CP" => "cp")
    cols = OrderedDict{String,Any}()
    for (k, v) in pairs(r.values)
        x = vec(v)
        if startswith(string(k), "θ")
            cols["sin2_" * ascii(k)] = sin.(x) .^ 2
        elseif k == :Δm²₃₁
            cols["dm2_32"] = x .- vec(r.values.Δm²₂₁)
        end
        cols[ascii(k)] = x
    end
    cols["dchi2"] = 2 .* (ref .- vec(r.values.log_posterior))
    haskey(cols, "llh") && (cols["log_likelihood"] = pop!(cols, "llh"))
    first_cols = filter(in(keys(cols)), ["sin2_theta23", "dm2_32", "dcp", "dchi2", "log_likelihood", "log_posterior"])
    CSV.write(path, DataFrame(cols)[:, vcat(first_cols, setdiff(collect(keys(cols)), first_cols))])
end

experiments = (deepcore = Newtrinos.deepcore.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# as in the paper: normal ordering, solar parameters and δCP fixed, θ₁₃ constrained, ντ CC normalisation fixed
priors = merge(priors, (Δm²₂₁ = p.Δm²₂₁, θ₁₂ = p.θ₁₂, δCP = p.δCP, nutau_cc_norm = p.nutau_cc_norm,
                        θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18),
                        Δm²₃₁ = Uniform(0.0022, 0.0027), θ₂₃ = Uniform(0.685, 0.9)))

contour = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₂₃ => 13, :Δm²₃₁ => 13), p, cache_dir = "cache/th23dm31")
th23 = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₂₃ => 21), p, cache_dir = "cache/th23")
dm31 = Newtrinos.profile(likelihood, priors, OrderedDict(:Δm²₃₁ => 21), p, cache_dir = "cache/dm31")
FileIO.save("results/deepcore.jld2", Dict("th23dm31" => contour, "th23" => th23, "dm31" => dm31))
save_csv("results/deepcore_ssth23_dm32.csv", contour); save_csv("results/deepcore_ssth23.csv", th23); save_csv("results/deepcore_dm32.csv", dm31)

# Fig. 27: 90% C.L. contour, with the official contour of the data release
official = CSV.read(joinpath(pkgdir(Newtrinos), "src/experiments/icecube/deepcore_8y_verification_sample",
                             "DeepCore_oscNext_verification_sample__sin2_theta23_dm2_32__90pc_result_bugfix.csv"), DataFrame; header = false)
fig = Figure(size = (700, 520))
ax = Axis(fig[1, 1], title = "IceCube DeepCore, NO: 90% C.L.", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
lines!(ax, official.Column1, official.Column2 .* 1e3, color = :red, label = "IceCube (data release)")
plot!(ax, Newtrinos.NewtrinosResult(axes = (ss23 = sin.(contour.axes.θ₂₃) .^ 2, dm2 = (contour.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3), values = contour.values),
      levels = [0.9], color = :blue, label = "Newtrinos (profile)")
axislegend(ax, position = :rt)
save("ours/contour_90.png", fig)

# Fig. 23: 1D profiles
for (name, r, x, xlabel) in (("mixing", th23, sin.(th23.axes.θ₂₃) .^ 2, "sin²θ₂₃"),
                             ("mass_splitting", dm31, (dm31.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3, "Δm²₃₂ (10⁻³ eV²)"))
    fig = Figure(size = (700, 480))
    ax = Axis(fig[1, 1], title = "IceCube DeepCore, NO", xlabel = xlabel, ylabel = "-2 Δ log L")
    lines!(ax, x, 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior), color = :blue, linewidth = 2, label = "Newtrinos (profile)")
    hlines!(ax, [1.0, 2.71], color = :gray, linestyle = :dot)
    ylims!(ax, 0, 6)
    axislegend(ax, position = :ct)
    save("ours/$(name)_profile.png", fig)
end
