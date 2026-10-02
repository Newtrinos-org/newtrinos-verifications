# Reproduction of Daya Bay, Phys. Rev. Lett. 130, 161802 (2023) [arXiv:2211.14988] with Newtrinos.jl (module `dayabay`).
# Data: supplemental material of the paper (3158 days, prompt spectra, backgrounds, Δχ² table).
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

experiments = (dayabay = Newtrinos.dayabay.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
priors = merge(priors, (Δm²₃₁ = Uniform(0.0023, 0.0028), θ₁₃ = Uniform(0.13, 0.16), θ₁₂ = p.θ₁₂, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₂₁ = p.Δm²₂₁))

result = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₃ => 31, :Δm²₃₁ => 31), p, cache_dir = "cache/th13dm31")
FileIO.save("results/dayabay.jld2", Dict("th13dm31" => result))
save_csv("results/dayabay_th13_dm31.csv", result)

# effective mass splitting of the paper: Δm²ee = Δm²₃₁ − sin²θ₁₂ Δm²₂₁ = Δm²₃₂ + cos²θ₁₂ Δm²₂₁
dmee_from31(x) = x - sin(p.θ₁₂)^2 * p.Δm²₂₁
dmee_from32(x) = x + cos(p.θ₁₂)^2 * p.Δm²₂₁
off = CSV.read(joinpath(DATA, "daya_bay/daya_bay_3158days/DayaBay_DeltaChiSq_NO_3158days.txt"), DataFrame,
               skipto = 10, delim = " ", header = ["ss2th13", "dm232", "dchi2"], ignorerepeated = true)
o_s = reshape(off.ss2th13, 100, 100)[1, :]; o_m = reshape(off.dm232, 100, 100)[:, 1]; o_χ² = Array(reshape(off.dchi2, 100, 100))'

# Fig. 1: sin²2θ₁₃ – Δm²ee, 1σ, 2σ, 3σ
fig = Figure(size = (700, 560))
ax = Axis(fig[1, 1], title = "Daya Bay, 3158 days: 1σ, 2σ, 3σ", xlabel = "sin²2θ₁₃", ylabel = "Δm²ee (10⁻³ eV²)")
levels = quantile.(Chisq(2), 1 .- 2 .* ccdf.(Normal(), 1:3))
for (lv, ls) in zip(levels, (:solid, :dash, :dot))
    contour!(ax, o_s, dmee_from32.(o_m) .* 1e3, o_χ², levels = [lv], color = :red, linestyle = ls, linewidth = 2)
    contour!(ax, sin.(2 .* result.axes.θ₁₃) .^ 2, dmee_from31.(result.axes.Δm²₃₁) .* 1e3, 2 .* (maximum(result.values.log_posterior) .- result.values.log_posterior),
             levels = [lv], color = :blue, linestyle = ls, linewidth = 2)
end
lines!(ax, [NaN], [NaN], color = :red, label = "Daya Bay (Δχ² table)")
lines!(ax, [NaN], [NaN], color = :blue, label = "Newtrinos (profile)")
xlims!(ax, 0.072, 0.1); ylims!(ax, 2.2, 2.82)
axislegend(ax, position = :rt)
save("ours/contour_dmee_ss2th13.png", fig)
