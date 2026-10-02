# Reproduction of Super-Kamiokande & T2K, Phys. Rev. Lett. 134, 011801 (2025) [arXiv:2405.12488]: joint fit of the
# Newtrinos.jl modules `t2k` and `super_k` as they are (no correlated systematics). Expensive: run with many threads
# (e.g. --threads=100). Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 100 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames, HDF5

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


experiments = (t2k = Newtrinos.t2k.configure(), super_k = Newtrinos.super_k.configure())
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# as in the paper: solar parameters fixed, reactor constraint sin²2θ₁₃ = 0.0853 ± 0.0027 (PDG 2019), flat δCP
θ₁₃ = asin(sqrt(0.0853)) / 2
σθ₁₃ = 0.0027 / (4 * sin(2θ₁₃) * cos(2θ₁₃))
θ₂₃_range = (asin(sqrt(0.35)), asin(sqrt(0.75)))
p = merge(p, (θ₁₃ = θ₁₃,))
priors = merge(priors, (θ₁₂ = p.θ₁₂, Δm²₂₁ = p.Δm²₂₁, θ₁₃ = Normal(θ₁₃, σθ₁₃), δCP = Uniform(0, 2π), θ₂₃ = Uniform(θ₂₃_range...)))
orderings = (NO = (Δm²₃₂ = (2.2e-3, 2.8e-3), start = (Δm²₃₂ = 2.49e-3, δCP = mod(-1.9, 2π))),
             IO = (Δm²₃₂ = (-2.8e-3, -2.2e-3), start = (Δm²₃₂ = -2.53e-3, δCP = mod(-1.5, 2π))))

function profile_octants(pr, p0, vars, cache_dir)   # octants scanned concurrently
    mkpath(cache_dir)
    oct = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
        Threads.@spawn Newtrinos.profile(likelihood, @set(pr.θ₂₃ = Uniform(r...)), vars, @set(p0.θ₂₃ = sum(r) / 2), cache_dir = cache_dir)
    end
    oct = map(fetch, oct)
    better = oct.lower.values.log_posterior .>= oct.upper.values.log_posterior
    Newtrinos.NewtrinosResult(axes = oct.lower.axes, values = map((l, u) -> ifelse.(better, l, u), oct.lower.values, oct.upper.values), meta = oct.lower.meta)
end

results = map(keys(orderings), values(orderings)) do mo, o
    pr = @set priors.Δm²₃₁ = Uniform((o.Δm²₃₂ .+ p.Δm²₂₁)...)
    p0 = merge(p, (Δm²₃₁ = o.start.Δm²₃₂ + p.Δm²₂₁, δCP = o.start.δCP, θ₂₃ = asin(sqrt(0.53))))
    mkpath("cache/$(mo)_th23dcp")
    # all scans of both orderings start at once (Threads.@spawn) to keep many threads busy
    th23dcp = Threads.@spawn Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 21, :δCP => 25), p0, cache_dir = "cache/$(mo)_th23dcp")
    dcp = Threads.@spawn profile_octants(pr, p0, OrderedDict(:δCP => 41), "cache/$(mo)_dcp")
    mo => (; th23dcp, dcp)
end |> NamedTuple
results = map(t -> map(fetch, t), results)

global_max = maximum(r -> maximum(r.dcp.values.log_posterior), results)
FileIO.save("results/t2k_sk.jld2", Dict("$(mo)_$(k)" => results[mo][k] for mo in keys(results) for k in keys(results[mo])))
for mo in keys(results)
    save_csv("results/t2k_sk_$(mo)_ssth23_dcp.csv", results[mo].th23dcp; ref = global_max)
    save_csv("results/t2k_sk_$(mo)_dcp.csv", results[mo].dcp; ref = global_max)
end

# Fig. 1: (sin²θ₂₃, δCP) regions, mass ordering profiled (the paper marginalises it), 1σ and 2σ
wrap(x) = mod(x + π, 2π) - π
rNO, rIO = results.NO.th23dcp, results.IO.th23dcp
lp = max.(rNO.values.log_posterior, rIO.values.log_posterior)
d = wrap.(rNO.axes.δCP); k = sortperm(d)
fig = Figure(size = (750, 520))
ax = Axis(fig[1, 1], title = "SK + T2K, mass ordering profiled: 1σ, 2σ", xlabel = "sin²θ₂₃", ylabel = "δCP")
for (lv, ls) in ((0.6827, :solid), (0.9545, :dot))
    contour!(ax, sin.(rNO.axes.θ₂₃) .^ 2, d[k], 2 .* (maximum(lp) .- lp[:, k]), levels = [quantile(Chisq(2), lv)], color = :crimson, linestyle = ls, linewidth = 3)
end
lines!(ax, [NaN], [NaN], color = :crimson, linewidth = 3, label = "Newtrinos SK + T2K (profile)")
xlims!(ax, 0.35, 0.75); ylims!(ax, -π, π)
axislegend(ax, position = :rt)
save("ours/fig1_ssth23_dcp.png", fig)

# Δχ²(δCP) per ordering vs. the official frequentist analysis ("Frequentist2", data release)
fig = Figure(size = (800, 480))
ax = Axis(fig[1, 1], title = "SK + T2K: Δχ² vs δCP (relative to the global best fit)", xlabel = "δCP", ylabel = "Δχ²")
h5open("data/official_frequentist2.h5") do f
    for (mo, c) in ((:NO, :steelblue), (:IO, :darkorange))
        e = read(f["$(mo)/dCP/edges"]); y = read(f["$(mo)/dCP/dchi2"])
        lines!(ax, (e[1:end-1] .+ e[2:end]) ./ 2, y, color = c, linestyle = :dash, label = "SK + T2K $(mo) (Frequentist2)")
        r = results[mo].dcp; x = wrap.(r.axes.δCP); i = sortperm(x)
        lines!(ax, x[i], 2 .* (global_max .- r.values.log_posterior[i]), color = c, linewidth = 3, label = "Newtrinos $(mo)")
    end
end
xlims!(ax, -π, π); ylims!(ax, 0, 30)
axislegend(ax, position = :lt)
save("ours/dcp_dchi2.png", fig)
