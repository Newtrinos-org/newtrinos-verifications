# Reproduction of MINOS+, Phys. Rev. Lett. 125, 131802 (2020) [arXiv:2006.15208], beam contours with Newtrinos.jl (module `minos`).
# Data: MINOS/MINOS+ data release of arXiv:1710.06488 (two-detector νμ CC spectra, 16e20 POT).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames, DelimitedFiles

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const DATA = joinpath(pkgdir(Newtrinos), "src", "experiments")

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

# draw each confidence level with its own line style (1σ solid, 2σ dashed, 3σ dotted)
function contours!(ax, res, levels; color, label)
    for (lv, ls) in zip(levels, (:solid, :dash, :dot))
        plot!(ax, res, levels = [lv], color = color, linestyle = ls, label = lv == first(levels) ? label : nothing)
    end
end

experiments = (minos = Newtrinos.minos.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
priors = merge(priors, (Δm²₂₁ = p.Δm²₂₁, θ₁₂ = p.θ₁₂, δCP = p.δCP, nc_norm = p.nc_norm, nutau_cc_norm = p.nutau_cc_norm,
                        θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18)))

result = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21), p, cache_dir = "cache/th23dm31")
FileIO.save("results/minos.jld2", Dict("th23dm31" => result))
save_csv("results/minos_ssth23_dm32.csv", result)

# Fig. 3: 68% and 90% C.L., normal ordering; official beam contours digitised from the paper (wpd_datasets.csv)
off = CSV.read(joinpath(DATA, "minos/minos_sterile_16e20_POT/wpd_datasets.csv"), DataFrame, header = 2)
fig = Figure(size = (700, 520))
ax = Axis(fig[1, 1], title = "MINOS/MINOS+, NO: 68%, 90% C.L.", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
closed(x) = vcat(x, x[[1]])
x68, y68 = collect(skipmissing(off.X)), collect(skipmissing(off.Y))
x90, y90 = collect(skipmissing(off.X_1)), collect(skipmissing(off.Y_1))
# same line styles as ours: 68% solid, 90% dashed
lines!(ax, closed(x68), closed(y68) .* 1e3, color = :red, label = "MINOS+ beam (digitised)")
lines!(ax, closed(x90), closed(y90) .* 1e3, color = :red, linestyle = :dash)
contours!(ax, Newtrinos.NewtrinosResult(axes = (ss23 = sin.(result.axes.θ₂₃) .^ 2, dm2 = (result.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3), values = result.values),
          [0.68, 0.9]; color = :blue, label = "Newtrinos (profile)")
xlims!(ax, 0.3, 0.7); ylims!(ax, 1.5, 3.1)
axislegend(ax, position = :rt)
save("ours/beam_contours.png", fig)
