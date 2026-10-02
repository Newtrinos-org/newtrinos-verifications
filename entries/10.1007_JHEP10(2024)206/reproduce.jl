# Reproduction of KM3NeT, JHEP 10 (2024) 206 [arXiv:2408.07015] with Newtrinos.jl (module `orca`).
# Data: KM3NeT open data, ORCA6 433 kton-years.
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

experiments = (orca = Newtrinos.orca.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
priors = merge(priors, (Δm²₂₁ = p.Δm²₂₁, θ₁₂ = p.θ₁₂, δCP = p.δCP, atm_flux_updown_sigma = 0.0,
                        θ₁₃ = Truncated(Normal(0.156, 0.008), 0.13, 0.18), θ₂₃ = Uniform(π / 4 - 0.25, π / 4 + 0.25)))

orderings = (NO = (Δm²₃₁ = (0.0015, 0.003), start = 2.2e-3), IO = (Δm²₃₁ = (-0.003, -0.0015), start = -2.2e-3))
results = map(keys(orderings), values(orderings)) do mo, o
    r = Newtrinos.profile(likelihood, @set(priors.Δm²₃₁ = Uniform(o.Δm²₃₁...)), OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21),
                          @set(p.Δm²₃₁ = o.start), cache_dir = "cache/$(mo)")
    mo => r
end |> NamedTuple
global_max = maximum(r -> maximum(r.values.log_posterior), results)
FileIO.save("results/orca6.jld2", Dict(string(k) => v for (k, v) in pairs(results)))
for mo in keys(results)
    save_csv("results/orca6_$(mo)_ssth23_dm31.csv", results[mo]; ref = global_max)
end

# official χ² landscapes of the data release (θ₂₃ in degrees, Δm²₃₁ in eV²), gridded (missing points → NaN)
function official(mo)
    df = CSV.read(joinpath(DATA, "km3net/orca6_433kton/chi2_landscape_$(mo).csv"), DataFrame)
    t = sort(unique(df.theta23)); m = sort(unique(df.dm31))
    χ² = fill(NaN, length(t), length(m))
    for r in eachrow(df)
        χ²[searchsortedfirst(t, r.theta23), searchsortedfirst(m, r.dm31)] = r.chi2
    end
    (ss23 = sind.(t) .^ 2, dm2 = abs.(m) .* 1e3, χ² = χ²)
end
off = (NO = official("NO"), IO = official("IO"))
off_min = minimum(o -> minimum(filter(!isnan, o.χ²)), off)

# Fig. 6 (left): 90% CL, NO and IO
fig = Figure(size = (650, 600))
ax = Axis(fig[1, 1], title = "KM3NeT/ORCA6, 433 kton-years: 90% CL", xlabel = "sin²θ₂₃", ylabel = "|Δm²₃₁| (10⁻³ eV²)")
level = quantile(Chisq(2), 0.9)
for (mo, ls) in ((:NO, :solid), (:IO, :dash))
    o = off[mo]
    contour!(ax, o.ss23, o.dm2, o.χ² .- off_min, levels = [level], color = :red, linestyle = ls, linewidth = 2)
    lines!(ax, [NaN], [NaN], color = :red, linestyle = ls, label = "KM3NeT $(mo) (data release)")
    r = results[mo]
    contour!(ax, sin.(r.axes.θ₂₃) .^ 2, abs.(r.axes.Δm²₃₁) .* 1e3, 2 .* (global_max .- r.values.log_posterior), levels = [level], color = :blue, linestyle = ls, linewidth = 2)
    lines!(ax, [NaN], [NaN], color = :blue, linestyle = ls, label = "Newtrinos $(mo)")
end
xlims!(ax, 0.35, 0.7); ylims!(ax, 1.45, 3.0)
axislegend(ax, position = :rt)
save("ours/contour_NOIO.png", fig)
