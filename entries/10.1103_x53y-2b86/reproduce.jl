# Reproduction of NOvA, Phys. Rev. Lett. 136, 011802 (2026) with Newtrinos.jl (module `nova`).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, HDF5, CSV, DataFrames

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

# draw each confidence level with its own line style (1σ solid, 2σ dashed, 3σ dotted)
function contours!(ax, res, levels; color, label)
    for (lv, ls) in zip(levels, (:solid, :dash, :dot))
        plot!(ax, res, levels = [lv], color = color, linestyle = ls, label = lv == first(levels) ? label : nothing)
    end
end

experiments = (nova = Newtrinos.nova.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# as in the paper: solar parameters fixed, 1D Daya Bay constraint sin²2θ₁₃ = 0.0851 ± 0.0024, flat δCP
ref = Newtrinos.nova.ref_params()
p = merge(p, (θ₁₂ = ref.θ₁₂, Δm²₂₁ = ref.Δm²₂₁, θ₁₃ = ref.θ₁₃))
priors = merge(priors, (θ₁₂ = p.θ₁₂, Δm²₂₁ = p.Δm²₂₁, θ₁₃ = Normal(ref.θ₁₃, 0.0024 / (4 * sqrt(0.0851 * (1 - 0.0851)))),
                        δCP = Uniform(0, 2π), θ₂₃ = Uniform(asin(sqrt(0.40)), asin(sqrt(0.64)))))

orderings = (NO = (Δm²₃₂ = (2.25e-3, 2.65e-3), start = (Δm²₃₂ = 2.441e-3, δCP = 0.87π)),
             IO = (Δm²₃₂ = (-2.65e-3, -2.25e-3), start = (Δm²₃₂ = -2.481e-3, δCP = 1.53π)))

# conditional fits per mass ordering (profile likelihood)
results = map(keys(orderings), values(orderings)) do mo, o
    pr = @set priors.Δm²₃₁ = Uniform((o.Δm²₃₂ .+ p.Δm²₂₁)...)
    p0 = merge(p, (Δm²₃₁ = o.start.Δm²₃₂ + p.Δm²₂₁, δCP = o.start.δCP))
    th23dm = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 21, :Δm²₃₁ => 21), p0, cache_dir = "cache/$(mo)_th23dm31")
    dcpth23 = Newtrinos.profile(likelihood, pr, OrderedDict(:δCP => 25, :θ₂₃ => 21), p0, cache_dir = "cache/$(mo)_dcpth23")
    FileIO.save("results/nova_$(mo).jld2", Dict("th23dm31" => th23dm, "dcpth23" => dcpth23))
    save_csv("results/nova_$(mo)_ssth23_dm32.csv", th23dm); save_csv("results/nova_$(mo)_dcp_ssth23.csv", dcpth23)
    mo => (; th23dm, dcpth23)
end |> NamedTuple

# official Bayesian credible regions shipped with the data release (1, 2, 3σ)
levels = (("06827", :solid), ("09545", :dash), ("09973", :dot))
function official!(ax, pair, mo)
    h5open(Newtrinos.nova.datafile) do f
        g = f["contours/RCDB1D_cond"]
        for (lvl, ls) in levels, k in keys(g)
            occursin("$(pair)_$(mo)_cred_int_$(lvl)_", k) || continue
            lines!(ax, read(g[k]["x"]), read(g[k]["y"]), color = :red, linestyle = ls,
                   label = lvl == "06827" && endswith(k, "_0") ? "NOvA (Bayesian)" : nothing)
        end
    end
end

for mo in keys(orderings)
    r = results[mo].th23dm
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], title = "NOvA 2024, $(mo): 1σ, 2σ, 3σ", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (10⁻³ eV²)")
    official!(ax, "ssth23dm32", mo)
    contours!(ax, Newtrinos.NewtrinosResult(axes = (ss23 = sin.(r.axes.θ₂₃) .^ 2, dm2 = (r.axes.Δm²₃₁ .- p.Δm²₂₁) .* 1e3), values = r.values),
              [0.6827, 0.9545, 0.9973]; color = :blue, label = "Newtrinos (profile)")
    axislegend(ax, position = :rt)
    save("ours/ssth23_dm32_$(mo).png", fig)

    r = results[mo].dcpth23
    fig = Figure(size = (700, 500))
    ax = Axis(fig[1, 1], title = "NOvA 2024, $(mo): 1σ, 2σ, 3σ", xlabel = "δCP / π", ylabel = "sin²θ₂₃")
    official!(ax, "dcpssth23", mo)
    contours!(ax, Newtrinos.NewtrinosResult(axes = (dcp = r.axes.δCP ./ π, ss23 = sin.(r.axes.θ₂₃) .^ 2), values = r.values), [0.6827, 0.9545, 0.9973]; color = :blue, label = "Newtrinos (profile)")
    xlims!(ax, 0, 2); ylims!(ax, 0.35, 0.67)
    axislegend(ax, position = :lb)
    save("ours/dcp_ssth23_$(mo).png", fig)
end
