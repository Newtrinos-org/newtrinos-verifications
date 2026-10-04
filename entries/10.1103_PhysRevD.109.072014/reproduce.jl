# Reproduction of Super-Kamiokande, Phys. Rev. D 109, 072014 (2024) [arXiv:2311.05105] with Newtrinos.jl (module `super_k`).
# Data: Super-K I–V atmospheric neutrino data release (zenodo 8401262). Expensive: run with many threads (e.g. --threads=100).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 100 --project=<path to Newtrinos.jl> reproduce.jl
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


experiments = (super_k = Newtrinos.super_k.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# as in the paper's "θ₁₃ free" SK-only analysis (Fig. 14): solar parameters fixed, θ₁₃ free (sin²θ₁₃ ≤ 0.075)
θ₂₃_range = (asin(sqrt(0.30)), asin(sqrt(0.775)))
priors = merge(priors, (θ₁₂ = p.θ₁₂, Δm²₂₁ = p.Δm²₂₁, θ₁₃ = Uniform(0.0, asin(sqrt(0.075))),
                        δCP = Uniform(0, 2π), θ₂₃ = Uniform(θ₂₃_range...)))
orderings = (NO = (Δm²₃₂ = (1.2e-3, 3.6e-3), start = 2.4e-3), IO = (Δm²₃₂ = (-3.6e-3, -1.2e-3), start = -2.4e-3))

function profile_octants(pr, p0, vars, cache_dir)   # θ₂₃ octants are separate minima, scanned concurrently
    mkpath(cache_dir)
    oct = map((lower = (θ₂₃_range[1], π / 4), upper = (π / 4, θ₂₃_range[2]))) do r
        Threads.@spawn Newtrinos.profile(likelihood, @set(pr.θ₂₃ = Uniform(r...)), vars, @set(p0.θ₂₃ = sum(r) / 2), cache_dir = cache_dir)
    end
    oct = map(fetch, oct)
    # the two octant scans use θ₂₃ priors of different widths, which shifts each log_posterior by its own
    # constant −log(width): remove it before comparing the octants (nuisance-parameter priors are kept)
    widths = (lower = π / 4 - θ₂₃_range[1], upper = θ₂₃_range[2] - π / 4)
    oct = map((r, w) -> Newtrinos.NewtrinosResult(axes = r.axes, meta = r.meta,
                                                   values = merge(r.values, (log_posterior = r.values.log_posterior .+ log(w),))), oct, widths)
    better = oct.lower.values.log_posterior .>= oct.upper.values.log_posterior
    Newtrinos.NewtrinosResult(axes = oct.lower.axes, values = map((l, u) -> ifelse.(better, l, u), oct.lower.values, oct.upper.values), meta = oct.lower.meta)
end
function prof(pr, p0, vars, cache_dir)
    mkpath(cache_dir)
    Newtrinos.profile(likelihood, pr, vars, p0, cache_dir = cache_dir)
end

# 1D profiles per ordering (Δm²₃₁ = Δm²₃₂ + Δm²₂₁). All scans are started at once (Threads.@spawn), so their
# ~290 points fill a machine with many threads instead of one 16–25-point scan at a time.
tasks = map(keys(orderings), values(orderings)) do mo, o
    pr = @set priors.Δm²₃₁ = Uniform((o.Δm²₃₂ .+ p.Δm²₂₁)...)
    p0 = merge(p, (Δm²₃₁ = o.start + p.Δm²₂₁, θ₂₃ = asin(sqrt(0.45))))
    mo => (dcp = Threads.@spawn(profile_octants(pr, p0, OrderedDict(:δCP => 21), "cache/$(mo)_dcp")),
           th13 = Threads.@spawn(profile_octants(pr, p0, OrderedDict(:θ₁₃ => 16), "cache/$(mo)_th13")),
           dm2 = Threads.@spawn(profile_octants(pr, p0, OrderedDict(:Δm²₃₁ => 25), "cache/$(mo)_dm2")),
           th23 = Threads.@spawn(prof(pr, p0, OrderedDict(:θ₂₃ => 20), "cache/$(mo)_th23")))
end |> NamedTuple
results = map(t -> map(fetch, t), tasks)

# Each 1D scan carries the constant log density of its own scanned-parameter prior, so Δχ² is taken per
# panel: relative to the best fit over both orderings *of that scan* (NO and IO use equal-width ranges,
# so the IO − NO offset is kept)
panel_max = map(k -> max(maximum(results.NO[k].values.log_posterior), maximum(results.IO[k].values.log_posterior)), keys(results.NO)) |>
            v -> NamedTuple{keys(results.NO)}(v)
FileIO.save("results/superk.jld2", Dict("$(mo)_$(k)" => results[mo][k] for mo in keys(results) for k in keys(results[mo])))
for mo in keys(results), k in keys(results[mo])
    save_csv("results/superk_$(mo)_$(k).csv", results[mo][k]; ref = panel_max[k])
end

# official SK-only Δχ² (θ₁₃ free) on the 4D grid (Δm², sin²θ₂₃, δCP, sin²θ₁₃), profiled to 1D; each table is
# relative to its own ordering, IO is shifted by the quoted Δχ²(IO − NO) = 5.23 (Sec. VII of the paper)
function official(mo)
    a = readdlm(joinpath(DATA, "super_k/sk_atm_2023/chi2/sk_2023_q13-free_$(lowercase(string(mo))).txt"); comments = true)
    a = a[isfinite.(a[:, 5]), :]
    off = mo == :IO ? 5.23 : 0.0
    prof1(col) = (x = sort(unique(a[:, col])); (x, [minimum(a[a[:, col] .== v, 5]) + off for v in x]))
    (dm2 = prof1(1), ss23 = prof1(2), dcp = prof1(3), ss13 = prof1(4))
end
off = (NO = official(:NO), IO = official(:IO))

# Fig. 14: four 1D panels
colors = (NO = :teal, IO = :darkorange)
wrap(x) = mod(x + π, 2π) - π
fig = Figure(size = (1100, 950))
panels = ((1, 1, :dcp, "δCP", r -> wrap.(r.axes.δCP), o -> wrap.(o.dcp[1]), o -> o.dcp[2], (-π, π)),
          (1, 2, :th13, "sin²θ₁₃", r -> sin.(r.axes.θ₁₃) .^ 2, o -> o.ss13[1], o -> o.ss13[2], (0, 0.075)),
          (2, 1, :dm2, "|Δm²₃₂,₃₁| (10⁻³ eV²)", r -> abs.(r.axes.Δm²₃₁ .- (r.axes.Δm²₃₁[1] > 0 ? p.Δm²₂₁ : 0.0)) .* 1e3, o -> o.dm2[1] .* 1e3, o -> o.dm2[2], (1.4, 3.6)),
          (2, 2, :th23, "sin²θ₂₃", r -> sin.(r.axes.θ₂₃) .^ 2, o -> o.ss23[1], o -> o.ss23[2], (0.32, 0.68)))
for (row, col, key, xlabel, xr, xo, yo, lims) in panels
    ax = Axis(fig[row, col], xlabel = xlabel, ylabel = "Δχ²")
    for mo in (:NO, :IO)
        x, y = xo(off[mo]), yo(off[mo]); i = sortperm(x)
        lines!(ax, x[i], y[i], color = colors[mo], linestyle = :dash, label = "SK $(mo) (data release)")
        r = results[mo][key]; x = xr(r); i = sortperm(x)
        lines!(ax, x[i], 2 .* (panel_max[key] .- r.values.log_posterior[i]), color = colors[mo], linewidth = 3, label = "Newtrinos $(mo)")
    end
    hlines!(ax, quantile.(Chisq(1), [0.68, 0.9, 0.95, 0.99]), color = :gray, linestyle = :dot)
    xlims!(ax, lims...); ylims!(ax, 0, 16)
    row == 1 && col == 1 && axislegend(ax, position = :lt)
end
save("ours/fig14_1d_profiles.png", fig)
