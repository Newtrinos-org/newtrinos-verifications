# daemonflux (J. P. Yañez, A. Fedynitch, Phys. Rev. D 107, 123037 (2023)) in Newtrinos.jl (module `atm_flux`:
# Daemonflux() nominal flux and Daemon() systematics) compared with the daemonflux package, and its uncertainties
# compared with the Barr et al. (2006) scheme that the Newtrinos atmospheric modules use by default.
# The daemonflux reference values are produced by data/daemonflux_reference.py.
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, LinearAlgebra, Distributions, ForwardDiff, DataStructures, FileIO

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const AF = Newtrinos.atm_flux
ref = CSV.read("data/daemonflux_reference.csv", DataFrame)
E = sort(unique(ref.E_GeV))
rows = NamedTuple[]

# fluxes averaged over the given cos θz values and their derivatives with respect to the systematic parameters
# x (a flat vector): returns the five quantities of the paper's Fig. 12 for each energy
function quantities(af, cz, x, unflatten)
    nominal = af.nominal_flux(E, cz)
    f = af.sys_flux(nominal, unflatten(x))
    avg(v) = vec(mean(reshape(v, length(E), length(cz)), dims = 2))
    numu, numubar, nue, nuebar = avg(f.numu), avg(f.numubar), avg(f.nue), avg(f.nuebar)
    (numu_sum = numu .+ numubar, nue_sum = nue .+ nuebar, numu_ratio = numu ./ numubar, nue_ratio = nue ./ nuebar,
     flavour_ratio = (numu .+ numubar) ./ (nue .+ nuebar))
end
function relative_errors(af, cz)
    ks = collect(keys(af.params))
    lens = [length(af.params[k]) for k in ks]
    unflatten(x) = begin
        i = 0
        NamedTuple{Tuple(ks)}(Tuple(begin
            v = af.params[k] isa AbstractVector ? x[i+1:i+n] : x[i+1]; i += n; v
        end for (k, n) in zip(ks, lens)))
    end
    x0 = reduce(vcat, [vec(collect(af.params[k])) for k in ks])
    C = cat([(d = af.priors[k]; d isa MvNormal ? cov(d) : fill(var(d isa Truncated ? d.untruncated : d), 1, 1)) for k in ks]...; dims = (1, 2))
    q0 = quantities(af, cz, x0, unflatten)
    map(keys(q0)) do name
        J = ForwardDiff.jacobian(x -> getproperty(quantities(af, cz, x, unflatten), name), x0)
        name => sqrt.(diag(J * C * J')) ./ getproperty(q0, name)
    end |> NamedTuple, q0
end

daemon = AF.configure(AF.AtmFluxConfig(nominal_model = AF.Daemonflux(), systematics_model = AF.Daemon()))
barr = AF.configure(AF.AtmFluxConfig(nominal_model = AF.HKKM("spl-nu-20-01-000.d"), systematics_model = AF.Barr()))
err_d, val_d = relative_errors(daemon, [1.0])          # vertical, as the reference
err_b, _ = relative_errors(barr, [0.95])             # HKKM tables end at cos θz = 0.95

labels = (numu_sum = "νμ + ν̄μ", nue_sum = "νe + ν̄e", numu_ratio = "νμ / ν̄μ", nue_ratio = "νe / ν̄e", flavour_ratio = "νμ / νe")
fig = Figure(size = (1500, 560))
for (k, (q, lab)) in enumerate(pairs(labels))
    g = sort(ref[ref.quantity .== String(q), :], :E_GeV)
    ax = Axis(fig[(k - 1) ÷ 3 + 1, (k - 1) % 3 + 1], xscale = log10, yscale = log10, xlabel = "E_ν (GeV)", ylabel = "relative error",
              title = lab, limits = (5, 1e4, 2e-3, 1.0))
    lines!(ax, g.E_GeV, g.relative_error, color = :black, linewidth = 3, label = "daemonflux (Python package)")
    q in (:numu_ratio, :nue_ratio, :flavour_ratio) && lines!(ax, g.E_GeV, g.relative_error_from_flavours, color = :gray, linewidth = 1.5,
                                                             label = k == 3 ? "daemonflux, from its per-flavour shifts" : nothing)
    lines!(ax, E, getproperty(err_d, q), color = :orange, linestyle = :dash, linewidth = 2, label = "Newtrinos Daemon()")
    lines!(ax, E, getproperty(err_b, q), color = :mediumpurple, linewidth = 2, label = "Newtrinos Barr() (HKKM, South Pole)")
    k in (1, 3) && axislegend(ax, position = :lt, framevisible = false, labelsize = 10)
    push!(rows, (quantity = String(q), comparison = "relative error, Newtrinos Daemon() / daemonflux − 1",
                 max_abs = maximum(abs.(getproperty(err_d, q) ./ g.relative_error .- 1))))
    push!(rows, (quantity = String(q), comparison = "relative error, Newtrinos Daemon() / daemonflux per-flavour shifts − 1",
                 max_abs = maximum(abs.(getproperty(err_d, q) ./ g.relative_error_from_flavours .- 1))))
    # nominal values (daemonflux returns E³-weighted fluxes)
    scale = q in (:numu_sum, :nue_sum) ? E .^ 3 : ones(length(E))
    push!(rows, (quantity = String(q), comparison = "value, Newtrinos Daemonflux() / daemonflux − 1",
                 max_abs = maximum(abs.(getproperty(val_d, q) .* scale ./ g.value .- 1))))
end
save("ours/fig_uncertainties.png", fig)
CSV.write("results/flux_comparison.csv", DataFrame(rows))

# ---------------- flux-systematics budget of the atmospheric analyses (linear propagation of the priors) ----------------
flat(x) = x isa NamedTuple ? reduce(vcat, [flat(v) for v in values(x)]) : vec(collect(x))
osc = (θ₁₂ = asin(sqrt(0.307)), θ₁₃ = asin(sqrt(0.02195)), θ₂₃ = asin(sqrt(0.561)), δCP = deg2rad(177.0), Δm²₂₁ = 7.49e-5, Δm²₃₁ = 2.534e-3)
budget = NamedTuple[]
for (name, mod) in (("IceCube DeepCore 8 y", Newtrinos.deepcore_8y), ("KM3NeT/ORCA6", Newtrinos.orca), ("Super-K I–V atmospheric", Newtrinos.super_k))
    cfg = mod.configure().physics.atm_flux.cfg
    for sysm in (cfg.systematics_model, AF.Daemon())
        af = AF.configure(AF.AtmFluxConfig(nominal_model = cfg.nominal_model, systematics_model = sysm))
        exp = mod.configure(merge(mod.default_physics(), (atm_flux = af,)))
        p0 = merge(Newtrinos.get_params(exp), osc)
        ks = collect(keys(af.params)); lens = [length(af.params[k]) for k in ks]
        x0 = reduce(vcat, [vec(collect(af.params[k])) for k in ks])
        function f(x)
            i = 0; upd = Dict{Symbol, Any}()
            for (k, n) in zip(ks, lens)
                upd[k] = af.params[k] isa AbstractVector ? x[i+1:i+n] : x[i+1]; i += n
            end
            flat(mod.get_expected(merge(p0, NamedTuple(upd)), exp.physics, exp.assets))
        end
        μ = f(x0); J = ForwardDiff.jacobian(f, x0)
        C = cat([(d = af.priors[k]; d isa MvNormal ? cov(d) : fill(var(d isa Truncated ? d.untruncated : d), 1, 1)) for k in ks]...; dims = (1, 2))
        V = J * C * J'
        rel = sqrt.(max.(diag(V), 0)) ./ max.(μ, 1e-12)
        sel = μ .> 1
        push!(budget, (experiment = name, flux_systematics = string(nameof(typeof(sysm))), n_parameters = length(x0),
                       total_rate_rel_error = sqrt(sum(V)) / sum(μ), median_bin_rel_error = median(rel[sel]), max_bin_rel_error = maximum(rel[sel])))
    end
end
CSV.write("results/flux_budget.csv", DataFrame(budget))

# ---------------- DeepCore 8 y: profiles of sin²θ₂₃ and Δm²₃₁ with Barr and with daemonflux systematics ----------------
mod = Newtrinos.deepcore_8y
cfg = mod.configure().physics.atm_flux.cfg
profiles = Dict()
for sysm in (cfg.systematics_model, AF.Daemon())
    tag = string(nameof(typeof(sysm)))
    exp = mod.configure(merge(mod.default_physics(), (atm_flux = AF.configure(AF.AtmFluxConfig(nominal_model = cfg.nominal_model, systematics_model = sysm)),)))
    experiments = (deepcore = exp,)
    likelihood = Newtrinos.generate_likelihood(experiments)
    p = merge(Newtrinos.get_params(experiments), (θ₁₂ = osc.θ₁₂, Δm²₂₁ = osc.Δm²₂₁, θ₁₃ = osc.θ₁₃, δCP = osc.δCP))
    priors = merge(Newtrinos.get_priors(experiments), (θ₁₂ = osc.θ₁₂, Δm²₂₁ = osc.Δm²₂₁, θ₁₃ = osc.θ₁₃, δCP = osc.δCP,
                   θ₂₃ = Uniform(asin(sqrt(0.35)), asin(sqrt(0.65))), Δm²₃₁ = Uniform(2.0e-3, 3.0e-3)))
    p = merge(p, (θ₂₃ = asin(sqrt(0.5)), Δm²₃₁ = 2.5e-3))
    profiles[tag] = (θ₂₃ = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₂₃ => 21), p, cache_dir = "cache/$(tag)_th23"),
                     Δm²₃₁ = Newtrinos.profile(likelihood, priors, OrderedDict(:Δm²₃₁ => 21), p, cache_dir = "cache/$(tag)_dm31"))
end
FileIO.save("results/deepcore_profiles.jld2", Dict(k => Dict("th23" => v.θ₂₃, "dm31" => v.Δm²₃₁) for (k, v) in profiles))
fig = Figure(size = (1100, 460))
prof_rows = NamedTuple[]
for (col, (var, lab, tr)) in enumerate(((:θ₂₃, "sin²θ₂₃", x -> sin(x)^2), (:Δm²₃₁, "Δm²₃₁ [10⁻³ eV²]", x -> 1e3x)))
    ax = Axis(fig[1, col], xlabel = lab, ylabel = col == 1 ? "−2 Δ ln L (profile)" : "", title = "IceCube DeepCore 8 y (data, NO)")
    for (tag, colr) in (("Barr", :mediumpurple), ("Daemon", :orange))
        r = profiles[tag][var]
        x = tr.(collect(r.axes[var])); y = 2 .* (maximum(r.values.log_posterior) .- vec(r.values.log_posterior))
        lines!(ax, x, y, color = colr, linewidth = 2.5, label = tag == "Barr" ? "Barr() (default)" : "Daemon() (daemonflux)")
        i = argmin(y)
        lo, hi = extrema(x[y .<= 1.0])
        push!(prof_rows, (parameter = lab, flux_systematics = tag, best_fit = x[i], interval_68_lo = lo, interval_68_hi = hi))
    end
    hlines!(ax, [1.0], color = :gray, linestyle = :dot)
    col == 1 && axislegend(ax, position = :ct, framevisible = false)
end
save("ours/fig_deepcore_profiles.png", fig)
CSV.write("results/deepcore_profiles.csv", DataFrame(prof_rows))
show(stdout, DataFrame(rows), allrows = true); println()
show(stdout, DataFrame(budget), allrows = true); println()
show(stdout, DataFrame(prof_rows), allrows = true); println()
