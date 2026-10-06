# Combined neutrino-mass-ordering sensitivity of JUNO and the IceCube Upgrade with Newtrinos.jl (modules `juno` with
# its 8-core reactor configuration and `ic_upgrade`, built on the public IceCube Upgrade MC release), compared with
# IceCube-Gen2 & JUNO members, Phys. Rev. D 101, 032006 (2020), Table V and the livetime figure (JUNO 8 cores + Upgrade).
# The paper's curves are extracted from the PGF figure of the arXiv source (data/extract_livetime.py).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 16 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, FileIO, CairoMakie, CSV, DataFrames, Printf, BAT
using BAT: distprod

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")

# true parameters (Table III of the paper, NuFIT 4.0): δCP = 0 and Δm²₂₁ fixed
TRUE = Dict(:NO => (Δm²₃₁ = 2.525e-3, θ₁₂ = deg2rad(33.82), θ₁₃ = deg2rad(8.61), θ₂₃ = deg2rad(49.6), δCP = 0.0, Δm²₂₁ = 7.39e-5),
            :IO => (Δm²₃₁ = -2.438e-3, θ₁₂ = deg2rad(33.82), θ₁₃ = deg2rad(8.65), θ₂₃ = deg2rad(49.8), δCP = 0.0, Δm²₂₁ = 7.39e-5))
LIVETIMES = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

# experiments for a livetime T [years]: JUNO with 300 days per year (6 years = 1800 days, as in the paper),
# the Upgrade with 365.25 days per year (its livetime parameter, in years)
const O = Newtrinos.osc
const UPGRADE = Newtrinos.ic_upgrade.configure()   # module default (point-like oscillation probabilities)
# optional: ray-to-spray averaging of the Upgrade oscillation probabilities with the default 15 % (Gaussian) width
const UPGRADE_SPRAY = Newtrinos.ic_upgrade.configure(merge(Newtrinos.ic_upgrade.default_physics(),
    (osc = O.configure(O.OscillationConfig(interaction = O.SI(), propagation = O.Spray())),)))
function experiments(T, which; spray = false)
    e = OrderedDict{Symbol, Any}()
    which in (:juno, :combined) && (e[:juno] = Newtrinos.juno.configure(; livetime_years = T * 300 / 365))
    which in (:upgrade, :combined) && (e[:ic_upgrade] = spray ? UPGRADE_SPRAY : UPGRADE)
    NamedTuple(e)
end

# nuisance model. Upgrade, as in the paper's Table II: spectral index 5 %, energy scale 10 %, effective-area
# (normalisation) scale 10 %, flux ratios (Barr νe/νμ, ν/ν̄ uncertainties); no zenith-shape, NC or ντ nuisances.
# JUNO: the module's systematics. Oscillation: θ₁₂ and θ₂₃ free within the NuFIT 4.0 3σ ranges, θ₁₃ with a ±0.13°
# prior, Δm²₃₁ within ±3σ of the true value (true ordering) or of its mirror (wrong ordering).
function priors(exps, truth, T; wrong)
    pr = Newtrinos.get_priors(exps)
    s = sign(truth.Δm²₃₁) * (wrong ? -1 : 1)
    dm = s > 0 ? Uniform(2.36e-3, 2.69e-3) : Uniform(-2.60e-3, -2.27e-3)    # equal widths: priors cancel in Δχ²
    osc = (Δm²₃₁ = dm, θ₁₂ = Uniform(deg2rad(31.61), deg2rad(36.27)), θ₁₃ = Truncated(Normal(truth.θ₁₃, deg2rad(0.13)), 0.1, 0.2),
           θ₂₃ = Uniform(deg2rad(40.3), deg2rad(52.4)), δCP = 0.0, Δm²₂₁ = truth.Δm²₂₁)
    pr = merge(pr, osc)
    if haskey(exps, :ic_upgrade)
        pr = merge(pr, (atm_flux_delta_spectral_index = Truncated(Normal(0, 0.05), -0.3, 0.3),
                        ic_upgrade_energy_scale = Truncated(Normal(1, 0.10), 0.5, 1.5),
                        ic_upgrade_lifetime = Truncated(Normal(T, 0.10T), 0.3T, 1.7T),
                        atm_flux_updown_sigma = 0.0, atm_flux_uphorizonzal_sigma = 0.0, nc_norm = 1.0, nutau_cc_norm = 1.0))
    end
    pr
end

function sensitivity(T, which, ord; spray = false, wo_start = nothing)
    exps = experiments(T, which; spray)
    truth = merge(Newtrinos.get_params(exps), TRUE[ord], haskey(exps, :ic_upgrade) ? (ic_upgrade_lifetime = T,) : (;))
    lik = Newtrinos.generate_likelihood(exps, Newtrinos.generate_asimov_data(exps, truth))
    fit(pr, start) = (r = Newtrinos.find_mle(lik, distprod(; pr...), start); (r[2], r[3]))   # (log posterior, best fit)
    lp_true = fit(priors(exps, truth, T; wrong = false), truth)[1]
    # wrong ordering: several starting points (both θ₂₃ octants, |Δm²₃₁| shifted in both directions), or a given start
    m = abs(truth.Δm²₃₁)
    starts = wo_start === nothing ?
        [merge(truth, (Δm²₃₁ = -sign(truth.Δm²₃₁) * dm, θ₂₃ = th)) for dm in (m - 0.13e-3, m - 0.035e-3, m + 0.035e-3, m + 0.13e-3)
         for th in (deg2rad(42.0), truth.θ₂₃)] : [merge(truth, wo_start)]
    fits = [fit(priors(exps, truth, T; wrong = true), s) for s in starts]
    lp_wrong, best = fits[argmax(first.(fits))]
    Δχ² = max(2 * (lp_true - lp_wrong), 0.0)
    (livetime = T, experiment = String(which), true_ordering = String(ord), spray = spray, dchi2 = Δχ², significance = sqrt(Δχ²),
     wo_dm31 = best.Δm²₃₁, wo_theta23_deg = rad2deg(best.θ₂₃), wo_best = best)
end

# each task is checkpointed in cache/ (a rerun at the same Newtrinos commit resumes from there)
function cached(f, name)
    file = "cache/$name.jld2"
    isfile(file) && return FileIO.load(file, "r")
    r = f(); FileIO.save(file, "r", r); r
end
tasks = [(T, w, o) for T in LIVETIMES for w in (:juno, :upgrade, :combined) for o in (:NO, :IO)]
# expensive tasks first, so that the threads stay busy until the end
sort!(tasks, by = t -> (t[2] == :juno, -t[1]))
out = Vector{Any}(undef, length(tasks))
Threads.@threads :dynamic for i in eachindex(tasks)
    T, w, o = tasks[i]
    out[i] = cached(() -> sensitivity(T, w, o), "task_$(w)_$(o)_$(Int(T))y")
    @info "done" T w o out[i].significance
end
# optional Spray (15 %) points at 6 years for the Upgrade and the combination, wrong-ordering fit started from the
# best fit of the default calculation (Spray is ~20x slower)
sp_tasks = [(w, o) for w in (:upgrade, :combined) for o in (:NO, :IO)]
sp = Vector{Any}(undef, length(sp_tasks))
Threads.@threads :dynamic for i in eachindex(sp_tasks)
    w, o = sp_tasks[i]
    b = only(r for r in out if r.livetime == 6.0 && r.experiment == String(w) && r.true_ordering == String(o))
    sp[i] = cached(() -> sensitivity(6.0, w, o; spray = true, wo_start = b.wo_best), "task_spray_$(w)_$(o)_6y")
    @info "done (spray)" w o sp[i].significance
end
tab = select(DataFrame(vcat(out, sp)), Not(:wo_best))
sort!(tab, [:spray, :true_ordering, :experiment, :livetime])
# simple (quadratic) sum of the stand-alone sensitivities, as in the paper
for o in ("NO", "IO"), T in LIVETIMES
    sel(w) = tab[(tab.true_ordering .== o) .& (tab.experiment .== w) .& (tab.livetime .== T) .& .!tab.spray, :dchi2][1]
    push!(tab, (livetime = T, experiment = "simple_sum", true_ordering = o, spray = false, dchi2 = sel("juno") + sel("upgrade"),
                significance = sqrt(sel("juno") + sel("upgrade")), wo_dm31 = NaN, wo_theta23_deg = NaN))
end
CSV.write("results/nmo_sensitivity_vs_livetime.csv", tab)

# ---- figure: significance vs livetime, paper (thick, transparent) and Newtrinos (thin, markers)
paper = CSV.read("data/livetime_nmo_sensitivity.csv", DataFrame)
COL = Dict("upgrade" => :royalblue, "juno" => :black, "combined" => :orangered, "simple_sum" => :darkorange)
PCOL = Dict("upgrade" => "upgrade", "juno" => "juno_8cores", "combined" => "combined", "simple_sum" => "simple_sum")
fig = Figure(size = (1000, 430))
for (k, o) in enumerate(("NO", "IO"))
    ax = Axis(fig[1, k], xlabel = "livetime (years)", ylabel = k == 1 ? "significance (σ)" : "", title = "true $o",
              limits = (0.5, 6.5, 0, 8))
    hlines!(ax, [3, 5], color = (:gray, 0.4), linestyle = :dash)
    for c in ("juno", "upgrade", "simple_sum", "combined")
        p = paper[(paper.true_ordering .== o) .& (paper.curve .== PCOL[c]), :]
        lines!(ax, p.livetime_years, p.significance_sigma, color = (COL[c], 0.35), linewidth = 5,
               linestyle = c == "simple_sum" ? :dash : :solid)
        n = tab[(tab.true_ordering .== o) .& (tab.experiment .== c) .& .!tab.spray, :]
        scatterlines!(ax, n.livetime, n.significance, color = COL[c], linewidth = 1.5, markersize = 7,
                      linestyle = c == "simple_sum" ? :dash : :solid)
        s = tab[(tab.true_ordering .== o) .& (tab.experiment .== c) .& tab.spray, :]
        nrow(s) > 0 && scatter!(ax, s.livetime, s.significance, color = COL[c], marker = :star5, markersize = 16,
                                strokecolor = :white, strokewidth = 0.8)
    end
    if k == 1
        axislegend(ax, [LineElement(color = (:gray, 0.5), linewidth = 5), MarkerElement(color = :gray, marker = :circle),
                        MarkerElement(color = :gray, marker = :star5, markersize = 14),
                        LineElement(color = :orangered), LineElement(color = :darkorange, linestyle = :dash),
                        LineElement(color = :royalblue), LineElement(color = :black)],
                   ["paper (PRD 101, 032006)", "Newtrinos", "Newtrinos, Upgrade with Spray (15 %), optional", "combined", "simple sum", "IceCube Upgrade", "JUNO (8 cores)"],
                   position = :lt, framevisible = false, labelsize = 11)
    end
end
save("ours/fig_livetime.png", fig)
six = tab[tab.livetime .== 6.0, :]
show(stdout, six[:, [:spray, :true_ordering, :experiment, :significance, :wo_dm31, :wo_theta23_deg]], allrows = true); println()
