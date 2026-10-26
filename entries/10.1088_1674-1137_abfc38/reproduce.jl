# Daya Bay unfolded reactor antineutrino spectra (Chin. Phys. C 45, 073001 (2021)) as the data-driven spectrum model of
# Newtrinos.jl (`reactor_flux.DayaBayUnfolded`), compared with the published unfolded spectra and with the
# Huber–Mueller model, and its effect on the KamLAND (2013, three periods) oscillation fit.
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, LinearAlgebra, Distributions, DataStructures, FileIO

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const RF = Newtrinos.reactor_flux
hm, dyb = RF.configure(), RF.configure(RF.DayaBayUnfolded())
t = RF.read_dyb_unfolded()
f_db = RF.DYB_FRACTIONS
# covariance of the IBD-weighted unfolded spectra (²³⁵U, Pu combo, total; 25 bins each)
lines = filter(l -> !startswith(l, "#") && !isempty(strip(l)), readlines(joinpath(pkgdir(Newtrinos), "src", "physics", "reactors", "dayabay_unfolded", "CovMatrix_UnfoldAntineutrinoSpectra.txt")))
C = reduce(vcat, [permutedims(parse.(Float64, split(l))) for l in lines])
σ_unf = sqrt.(diag(C))
bm(f, b) = RF.bin_mean(f, t.edges[b], t.edges[b+1])

spectra = (("²³⁵U", E -> dyb.isotope_spectrum(:U235, E), E -> hm.isotope_spectrum(:U235, E), t.U235, 0),
           ("Pu combo", E -> dyb.isotope_spectrum(:Pu239, E) + 0.183 * dyb.isotope_spectrum(:Pu241, E),
                        E -> hm.isotope_spectrum(:Pu239, E) + 0.183 * hm.isotope_spectrum(:Pu241, E), t.combo, 25),
           ("Total", E -> dyb.spectrum(E, f_db), E -> hm.spectrum(E, f_db), t.total, 50))
rows = NamedTuple[]
Ef = collect(range(1.85, 9.5, length = 600))
fig = Figure(size = (1400, 820))
for (col, (name, sd, sh, data, off)) in enumerate(spectra)
    σx(E) = RF.ibd_xsec(E) * 1e43
    # IBD-weighted bin contents of the data release (data × σ averaged over the bin, as in the release's first column)
    dσ = [bm(E -> sd(E) * σx(E), b) for b in eachindex(data)]
    err = σ_unf[off+1:off+25] .* 1e43
    ax = Axis(fig[1, col], ylabel = col == 1 ? "σ × 10⁻⁴³ [cm²/MeV/fission]" : "", title = name, xticklabelsvisible = false, limits = (1.8, 9.5, 0, 2.05))
    lines!(ax, Ef, sh.(Ef) .* σx.(Ef), color = :gray, linestyle = :dash, label = "Huber–Mueller")
    lines!(ax, Ef, sd.(Ef) .* σx.(Ef), color = :steelblue, linewidth = 2, label = "Newtrinos DayaBayUnfolded()")
    errorbars!(ax, t.centres, dσ, err, color = :black); scatter!(ax, t.centres, dσ, color = :black, markersize = 6, label = "Daya Bay unfolded (Wiener-SVD)")
    col == 1 && axislegend(ax, position = :rt, framevisible = false, labelsize = 10)
    ax2 = Axis(fig[2, col], xlabel = "Antineutrino energy [MeV]", ylabel = col == 1 ? "ratio to Huber–Mueller" : "", limits = (1.8, 9.5, 0.75, 1.25))
    hlines!(ax2, [1.0], color = :gray, linestyle = :dot)
    lines!(ax2, Ef, sd.(Ef) ./ sh.(Ef), color = :steelblue, linewidth = 2)
    rb = data ./ [bm(sh, b) for b in eachindex(data)]
    errorbars!(ax2, t.centres, rb, rb .* err ./ dσ, color = :black); scatter!(ax2, t.centres, rb, color = :black, markersize = 6)
    push!(rows, (spectrum = name, check = "bin means of the model / unfolded bin contents − 1", value = maximum(abs.([bm(sd, b) for b in eachindex(data)] ./ data .- 1))))
    push!(rows, (spectrum = name, check = "Daya Bay / Huber–Mueller, 4–6 MeV bin-weighted mean", value = mean(rb[9:16])))
end
rowsize!(fig.layout, 2, Relative(0.35))
save("ours/fig_unfolded_spectra.png", fig)

# IBD yields (cm²/fission) with the models' normalisations
yield(rf, f) = (E = range(RF.IBD_THRESHOLD, 12.0, length = 4001); step(E) * sum(rf.spectrum(e, f) * rf.ibd_xsec(e) for e in E) * rf.rate_scale(f))
b4 = (U235 = 0.538, U238 = 0.078, Pu239 = 0.328, Pu241 = 0.056)
kl = Newtrinos.kamland.DATASETS.periods2013.fractions
for (lab, f) in (("Daya Bay fractions", f_db), ("Bugey-4 fractions", b4), ("KamLAND fractions", kl))
    push!(rows, (spectrum = "IBD yield, $lab", check = "Huber–Mueller × Bugey-4 anchoring [1e-43 cm²/fission]", value = 1e43 * yield(hm, f)))
    push!(rows, (spectrum = "IBD yield, $lab", check = "Daya Bay unfolded [1e-43 cm²/fission]", value = 1e43 * yield(dyb, f)))
end
CSV.write("results/spectra.csv", DataFrame(rows))

# ---------------- KamLAND (2013, three periods): profiles in Δm²₂₁ and sin²θ₁₂ with both spectrum models ----------------
profiles = Dict()
for (tag, rf) in (("HuberMueller", hm), ("DayaBayUnfolded", dyb))
    exp = Newtrinos.kamland.configure(merge(Newtrinos.kamland.default_physics(), (reactor_flux = rf,)))
    experiments = (kamland = exp,)
    likelihood = Newtrinos.generate_likelihood(experiments)
    p = merge(Newtrinos.get_params(experiments), (θ₁₃ = asin(sqrt(0.0220)), θ₁₂ = asin(sqrt(0.31)), Δm²₂₁ = 7.5e-5))
    priors = merge(Newtrinos.get_priors(experiments), (θ₁₃ = p.θ₁₃, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₃₁ = p.Δm²₃₁,
                   θ₁₂ = Uniform(asin(sqrt(0.2)), asin(sqrt(0.45))), Δm²₂₁ = Uniform(6.8e-5, 8.2e-5)))
    profiles[tag] = (Δm²₂₁ = Newtrinos.profile(likelihood, priors, OrderedDict(:Δm²₂₁ => 41), p, cache_dir = "cache/$(tag)_dm21"),
                     θ₁₂ = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₂ => 41), p, cache_dir = "cache/$(tag)_th12"))
end
FileIO.save("results/kamland_profiles.jld2", Dict(k => Dict("dm21" => v.Δm²₂₁, "th12" => v.θ₁₂) for (k, v) in profiles))
prof_rows = NamedTuple[]
fig = Figure(size = (1100, 460))
for (col, (var, lab, tr)) in enumerate(((:Δm²₂₁, "Δm²₂₁ [10⁻⁵ eV²]", x -> 1e5x), (:θ₁₂, "sin²θ₁₂", x -> sin(x)^2)))
    ax = Axis(fig[1, col], xlabel = lab, ylabel = col == 1 ? "−2 Δ ln L (profile)" : "", title = "KamLAND 2013 (three periods)")
    best = maximum(maximum(profiles[t][var].values.log_posterior) for t in keys(profiles))
    for (tag, colr) in (("HuberMueller", :gray), ("DayaBayUnfolded", :steelblue))
        r = profiles[tag][var]
        x = tr.(collect(r.axes[var])); lp = vec(r.values.log_posterior)
        lines!(ax, x, 2 .* (best .- lp), color = colr, linewidth = 2.5, label = tag == "HuberMueller" ? "Huber–Mueller (Bugey-4 anchored)" : "Daya Bay unfolded")
        y = 2 .* (maximum(lp) .- lp); i = argmin(y); lo, hi = extrema(x[y .<= 1.0])
        push!(prof_rows, (parameter = lab, spectrum = tag, best_fit = x[i], interval_68_lo = lo, interval_68_hi = hi, max_log_posterior = maximum(lp)))
    end
    col == 1 && axislegend(ax, position = :ct, framevisible = false)
end
save("ours/fig_kamland_profiles.png", fig)
CSV.write("results/kamland_profiles.csv", DataFrame(prof_rows))
show(stdout, DataFrame(rows), allrows = true); println()
show(stdout, DataFrame(prof_rows), allrows = true); println()
