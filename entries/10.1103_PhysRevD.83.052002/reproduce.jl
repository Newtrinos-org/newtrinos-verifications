# Reproduction of KamLAND, Phys. Rev. D 83, 052002 (2011) [arXiv:1009.4771], θ₁₃ = 0 with Newtrinos.jl (module `kamland`).
# Data: spectra digitised from the paper; official Δχ² table of the KamLAND data release.
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

experiments = (kamland = Newtrinos.kamland.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
p = merge(p, (θ₁₃ = 0.0,))
priors = merge(priors, (Δm²₃₁ = p.Δm²₃₁, θ₁₃ = p.θ₁₃, θ₂₃ = p.θ₂₃, δCP = p.δCP))

result = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 21, :Δm²₂₁ => 21), p, cache_dir = "cache/th12dm21")
FileIO.save("results/kamland.jld2", Dict("th12dm21" => result))
save_csv("results/kamland_th12_dm21.csv", result)

# official KamLAND-only Δχ²(tan²θ₁₂, sin²θ₁₃, Δm²₂₁), slice at θ₁₃ = 0
raw = readdlm(joinpath(DATA, "kamland/kamland_7years/delta_chi2_4th_result.dat"))
sh = (121, 114, 81)
tan2 = reshape(raw[:, 1], sh)[1, :, 1]; dm2 = reshape(raw[:, 3], sh)[1, 1, :]; χ² = reshape(raw[:, 4], sh)[1, :, :]

# Fig. 2(a): 95%, 99%, 99.73% C.L.
fig = Figure(size = (650, 600))
ax = Axis(fig[1, 1], title = "KamLAND, θ₁₃ = 0: 95%, 99%, 99.73% C.L.", xlabel = "tan²θ₁₂", ylabel = "Δm²₂₁ (10⁻⁴ eV²)")
for (lv, ls) in zip((0.95, 0.99, 0.9973), (:dot, :dash, :solid))
    contour!(ax, tan2, dm2 .* 1e4, χ² .- minimum(χ²), levels = [quantile(Chisq(2), lv)], color = :red, linestyle = ls, linewidth = 2)
end
lines!(ax, [NaN], [NaN], color = :red, label = "KamLAND (Δχ² table)")
r = result
for (lv, ls) in zip((0.95, 0.99, 0.9973), (:dot, :dash, :solid))
    contour!(ax, tan.(r.axes.θ₁₂) .^ 2, r.axes.Δm²₂₁ .* 1e4, 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior),
             levels = [quantile(Chisq(2), lv)], color = :blue, linestyle = ls, linewidth = 2)
end
lines!(ax, [NaN], [NaN], color = :blue, label = "Newtrinos (profile)")
xlims!(ax, 0.1, 1.0); ylims!(ax, 0.4, 2.0)
axislegend(ax, position = :rt)
save("ours/fig2a_tan2th12_dm21.png", fig)
