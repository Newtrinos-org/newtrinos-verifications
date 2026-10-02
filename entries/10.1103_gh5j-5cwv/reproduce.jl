# Reproduction of T2K, Phys. Rev. Lett. 135, 261801 (2025) [arXiv:2506.05889] with Newtrinos.jl (module `t2k`).
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

experiments = (t2k = Newtrinos.t2k.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# as in the paper: solar parameters fixed, reactor constraint sin²θ₁₃ = 0.0220 ± 0.0007, flat δCP
ref = Newtrinos.t2k.ref_params()
p = merge(p, (θ₁₂ = ref.θ₁₂, Δm²₂₁ = ref.Δm²₂₁, θ₁₃ = ref.θ₁₃))
θ₂₃_range = (asin(sqrt(0.35)), asin(sqrt(0.68)))
priors = merge(priors, (θ₁₂ = p.θ₁₂, Δm²₂₁ = p.Δm²₂₁, θ₁₃ = Normal(ref.θ₁₃, 0.0007 / (2 * sin(ref.θ₁₃) * cos(ref.θ₁₃))),
                        δCP = Uniform(0, 2π), θ₂₃ = Uniform(θ₂₃_range...)))

orderings = (NO = (Δm²₃₁ = (2.2e-3, 2.8e-3) .+ p.Δm²₂₁, start = (Δm²₃₁ = 2.506e-3 + p.Δm²₂₁, δCP = mod(-2.18, 2π)),
                   dm2 = x -> x - p.Δm²₂₁, color = :steelblue),
             IO = (Δm²₃₁ = (-2.8e-3, -2.2e-3), start = (Δm²₃₁ = -2.47e-3, δCP = mod(-1.37, 2π)),
                   dm2 = x -> -x, color = :darkorange))

# the θ₂₃ octants are separate local minima: profile each octant, keep the better one per point
function profile_octants(pr, p0, vars, cache_dir)
    oct = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
        Newtrinos.profile(likelihood, @set(pr.θ₂₃ = Uniform(r...)), vars, @set(p0.θ₂₃ = sum(r) / 2), cache_dir = cache_dir)
    end
    better = oct.lower.values.log_posterior .>= oct.upper.values.log_posterior
    Newtrinos.NewtrinosResult(axes = oct.lower.axes, values = map((l, u) -> ifelse.(better, l, u), oct.lower.values, oct.upper.values), meta = oct.lower.meta)
end

results = map(keys(orderings), values(orderings)) do mo, o
    pr = @set priors.Δm²₃₁ = Uniform(o.Δm²₃₁...)
    p0 = merge(p, (Δm²₃₁ = o.start.Δm²₃₁, δCP = o.start.δCP, θ₂₃ = asin(sqrt(0.56))))
    th23dm = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 25, :Δm²₃₁ => 25), p0, cache_dir = "cache/$(mo)_th23dm31")
    dcp = profile_octants(pr, p0, OrderedDict(:δCP => 41), "cache/$(mo)_dcp")
    FileIO.save("results/t2k_$(mo).jld2", Dict("th23dm31" => th23dm, "dcp" => dcp))
    mo => (; th23dm, dcp)
end |> NamedTuple
global_max = maximum(r -> maximum(r.dcp.values.log_posterior), results)
for mo in keys(results)   # dchi2 relative to the global best fit over both orderings
    save_csv("results/t2k_$(mo)_ssth23_dm2.csv", results[mo].th23dm; ref = global_max)
    save_csv("results/t2k_$(mo)_dcp.csv", results[mo].dcp; ref = global_max)
end

official = h5open(Newtrinos.t2k.datafile) do f
    (dcp = Dict(mo => (read(f["official/dcp/$mo/dcp"]), read(f["official/dcp/$mo/dchi2"])) for mo in ("NO", "IO")),
     th23dm2 = Dict(mo => Dict(l => (read(f["official/th23dm2/$mo/$l/ssth23"]), read(f["official/th23dm2/$mo/$l/dm2"])) for l in ("068", "090", "997")) for mo in ("NO", "IO")))
end

# Fig. 3: Δχ²(δCP) relative to the global best fit
wrap(x) = mod(x + π, 2π) - π
fig = Figure(size = (800, 500))
ax = Axis(fig[1, 1], title = "T2K: Δχ² vs δCP", xlabel = "δCP", ylabel = "Δχ²")
for mo in keys(orderings)
    x, y = official.dcp[string(mo)]
    lines!(ax, x, y, color = orderings[mo].color, linestyle = :dash, label = "T2K $(mo)")
    r = results[mo].dcp; d = wrap.(r.axes.δCP); i = sortperm(d)
    lines!(ax, d[i], (2 .* (global_max .- r.values.log_posterior))[i], color = orderings[mo].color, linewidth = 3, label = "Newtrinos $(mo)")
end
xlims!(ax, -π, π); ylims!(ax, 0, 30)
axislegend(ax, position = :lt)
save("ours/fig3_dcp.png", fig)

# Fig. 4: sin²θ₂₃ – Δm²₃₂ (NO) / |Δm²₃₁| (IO), 68%, 90%, 99.7% CL, conditional on each ordering
fig = Figure(size = (800, 550))
ax = Axis(fig[1, 1], title = "T2K: 68%, 90%, 99.7% CL", xlabel = "sin²θ₂₃", ylabel = "Δm²₃₂ (NO) / |Δm²₃₁| (IO) (10⁻³ eV²)")
for mo in keys(orderings)
    o = orderings[mo]
    for (l, ms) in (("068", 4), ("090", 3), ("997", 3))
        x, y = official.th23dm2[string(mo)][l]
        keep = (y .> 2.25e-3) .& (x .< 0.65)          # drop stray digitisation points (legend, labels)
        x, y = x[keep], y[keep]
        scatter!(ax, x, y .* 1e3, color = o.color, markersize = ms, label = l == "090" ? "T2K $(mo) (digitised)" : nothing)
    end
    r = results[mo].th23dm
    contours!(ax, Newtrinos.NewtrinosResult(axes = (ss23 = sin.(r.axes.θ₂₃) .^ 2, dm2 = o.dm2.(r.axes.Δm²₃₁) .* 1e3), values = r.values),
              [0.68, 0.9, 0.997]; color = o.color, label = "Newtrinos $(mo)")
end
xlims!(ax, 0.3, 0.7); ylims!(ax, 2.1, 2.8)
axislegend(ax, position = :lb)
save("ours/fig4_ssth23_dm2.png", fig)
