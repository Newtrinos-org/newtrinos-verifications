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

# Fig. 4 (top): 1D profiles of |Δm²₃₂| and sin²θ₂₃ for both orderings, with the paper's Table II intervals
# (the paper combines beam and atmospheric samples; Newtrinos has the beam data only)
ord = (NO = (Δm²₃₁ = (1.8e-3, 3.2e-3), start = 2.45e-3), IO = (Δm²₃₁ = (-3.2e-3, -1.8e-3), start = -2.35e-3))
prof1d = map(keys(ord), values(ord)) do mo, o
    pr = merge(priors, (Δm²₃₁ = Uniform(o.Δm²₃₁...),)); p0 = merge(p, (Δm²₃₁ = o.start,))
    mo => (dm2 = Newtrinos.profile(likelihood, pr, OrderedDict(:Δm²₃₁ => 29), p0, cache_dir = "cache/$(mo)_dm2"),
           th23 = Newtrinos.profile(likelihood, pr, OrderedDict(:θ₂₃ => 25), p0, cache_dir = "cache/$(mo)_th23"))
end |> NamedTuple
top = map(k -> maximum(max(maximum(prof1d.NO[k].values.log_posterior), maximum(prof1d.IO[k].values.log_posterior))), (dm2 = :dm2, th23 = :th23))
dm32(r) = abs.(r.axes.Δm²₃₁ .- (r.axes.Δm²₃₁[1] > 0 ? p.Δm²₂₁ : 0.0)) .* 1e3
table2 = (NO = (dm2 = (2.31, 2.48), ss23 = (0.37, 0.65)), IO = (dm2 = (2.37, 2.52), ss23 = (0.37, 0.65)))   # 68 % / 90 % C.L.
fig = Figure(size = (1100, 430))
ax1 = Axis(fig[1, 1], xlabel = "|Δm²₃₂| (10⁻³ eV²)", ylabel = "−2Δln L"); ax2 = Axis(fig[1, 2], xlabel = "sin²θ₂₃")
for (mo, c) in ((:NO, :blue), (:IO, :red))
    r = prof1d[mo].dm2; lines!(ax1, dm32(r), 2 .* (top.dm2 .- r.values.log_posterior), color = c, linewidth = 2.5, label = "Newtrinos $mo (beam)")
    r = prof1d[mo].th23; lines!(ax2, sin.(r.axes.θ₂₃) .^ 2, 2 .* (top.th23 .- r.values.log_posterior), color = c, linewidth = 2.5)
    lines!(ax1, collect(table2[mo].dm2), [1.0, 1.0] .+ (mo == :IO ? 0.15 : 0.0), color = (c, 0.5), linewidth = 6, label = "MINOS+ $mo: 68 % C.L. (Table II)")
    lines!(ax2, collect(table2[mo].ss23), fill(quantile(Chisq(1), 0.9), 2) .+ (mo == :IO ? 0.15 : 0.0), color = (c, 0.5), linewidth = 6)
end
for ax in (ax1, ax2); hlines!(ax, [1.0, quantile(Chisq(1), 0.9)], color = :gray, linestyle = :dot); ylims!(ax, 0, 6); end
xlims!(ax1, 2.0, 2.9); xlims!(ax2, 0.3, 0.7); axislegend(ax1, position = :ct, labelsize = 10)
save("ours/fig4_1d_profiles.png", fig)
open("results/summary.txt", "w") do io
    for mo in (:NO, :IO)
        r = prof1d[mo].dm2; y = 2 .* (top.dm2 .- r.values.log_posterior); x = dm32(r)
        r2 = prof1d[mo].th23; y2 = 2 .* (top.th23 .- r2.values.log_posterior); x2 = sin.(r2.axes.θ₂₃) .^ 2
        inside(x, y, lv) = (s = x[y .<= lv]; (minimum(s), maximum(s)))
        println(io, "$mo: |dm2_32| best $(round(x[argmin(y)], digits = 3)), grid points with Δχ² ≤ 1: $(round.(inside(x, y .- minimum(y), 1.0), digits = 3)); ",
                "sin2_theta23 best $(round(x2[argmin(y2)], digits = 3)), grid points with Δχ² ≤ 2.71: $(round.(inside(x2, y2 .- minimum(y2), 2.71), digits = 3)); min Δχ² $(round(minimum(y), digits = 2))")
    end
end
print(read("results/summary.txt", String))
