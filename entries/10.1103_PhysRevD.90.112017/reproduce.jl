# ANL and BNL deuterium single-pion production, reanalysed with the flux cancelled through the CCQE rate (Wilkinson et
# al., PRD 90, 112017 (2014); values of the NUISANCE data files, in data/), compared with the resonance cross sections of
# Newtrinos.jl (module `xsec`): the GENIE tunes (`generator_sigma`, process RES on free protons and neutrons) and the
# reference model of `CorrelatedXsec` with its prior 1σ band (on free nucleons: the RES normalisation knob, 10 %).
# On a proton the charged-current resonances give only μ⁻ p π⁺, so CC1π⁺p is compared with RES on hydrogen; on a
# neutron both channels, μ⁻ p π⁰ and μ⁻ n π⁺, are summed and compared with RES on a free neutron. The data contain the
# non-resonant single-pion background in addition, which GENIE generates in its DIS process (shown separately, W < 2 GeV
# is not separable here, so DIS is an upper bound).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, Random, Distributions

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

readc(f) = (d = CSV.read(f, DataFrame; comment = "#", header = 1); rename!(d, [:E_lo, :E_hi, :s, :e]); d)
P = (ANL = readc("data/anl_cc1pip_p.csv"), BNL = readc("data/bnl_cc1pip_p.csv"))
# neutron: sum of the two channels in their common bins
function nsum(a, b)
    j = innerjoin(a, b, on = [:E_lo, :E_hi], makeunique = true)
    DataFrame(E_lo = j.E_lo, E_hi = j.E_hi, s = j.s .+ j.s_1, e = sqrt.(j.e .^ 2 .+ j.e_1 .^ 2))
end
N = (ANL = nsum(readc("data/anl_cc1pi0.csv"), readc("data/anl_cc1pip_n.csv")),
     BNL = nsum(readc("data/bnl_cc1pi0.csv"), readc("data/bnl_cc1pip_n.csv")))

tab = X.load_table()
E = collect(range(0.2, 6.0, length = 400))
at(v, e = E) = X.at_energies(tab.E, v, e)
TUNES = Symbol.(sort(tab.tunes))
σ_tune(t, target, p, e = E) = at(getfield(X.generator_sigma(t, target, :numu, false, :CC), p), e) .* e

xs = X.configure(X.CorrelatedXsec())
ref = xs.cfg.reference
sigma(target, e = E) = (f = xs.sigma(e; flavour = :numu, anti = false, current = :CC, target, process = :RES); p -> f(p) .* e)
rng = MersenneTwister(1)
draw() = map(p -> p isa Distribution ? rand(rng, p) : p, xs.priors)
samples = [draw() for _ in 1:500]
p0 = xs.params
q(M, a) = [quantile(r, a) for r in eachrow(M)]

rows = []
for (name, target, D, title) in (("cc1pip_p", :H, P, "νμ p → μ⁻ p π⁺ (D₂)  vs  RES on free protons"),
                                 ("cc1pi_n", :n, N, "νμ n → μ⁻ p π⁰ + μ⁻ n π⁺ (D₂)  vs  RES on free neutrons"))
    fig = Figure(size = (840, 480))
    ax = Axis(fig[1, 1], xlabel = "E_ν (GeV)", ylabel = "σ (10⁻³⁸ cm² per target nucleon)", title = title, limits = (0, 4, 0, 1.4))
    f = sigma(target)
    M = reduce(hcat, [f(p) for p in samples])
    band!(ax, E, q(M, 0.16), q(M, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec RES prior 1σ")
    for (i, t) in enumerate(TUNES)
        lines!(ax, E, σ_tune(t, target, :RES), color = (Makie.wong_colors()[mod1(i, 7)], 0.7), linewidth = 1, label = "GENIE $(t) RES")
    end
    lines!(ax, E, f(p0), color = :dodgerblue, linewidth = 2.5, label = "reference $(ref) RES")
    lines!(ax, E, f(p0) .+ σ_tune(ref, target, :DIS), color = :dodgerblue, linestyle = :dot, linewidth = 1.5, label = "reference RES + DIS (upper bound)")
    for (lab, d, mk) in (("ANL", D.ANL, :rect), ("BNL", D.BNL, :circle))
        x = (d.E_lo .+ d.E_hi) ./ 2
        errorbars!(ax, x, d.s, d.e, color = :black, whiskerwidth = 4)
        errorbars!(ax, x, d.s, x .- d.E_lo, d.E_hi .- x, direction = :x, color = (:black, 0.5))
        scatter!(ax, x, d.s, marker = mk, color = mk == :rect ? :white : :black, strokecolor = :black, strokewidth = 1.2, label = "$(lab) (reanalysed, stat.)")
        # χ² with the bin-averaged model, and the best-fit normalisation of the reference RES
        bins = [collect(range(lo, hi, length = 11)) for (lo, hi) in zip(d.E_lo, d.E_hi)]
        mref = [mean(sigma(target, b)(p0)) for b in bins]
        sd = vec(std(reduce(hcat, [[mean(sigma(target, b)(p)) for b in bins] for p in samples]), dims = 2))
        sel = d.E_lo .>= 0.4                     # threshold bins with ≈ 0 events are left out
        c(m, extra = 0) = sum(((d.s .- m) .^ 2 ./ (d.e .^ 2 .+ extra .^ 2))[sel])
        w = 1 ./ d.e[sel] .^ 2
        a = sum(w .* d.s[sel] .* mref[sel]) / sum(w .* mref[sel] .^ 2)
        push!(rows, merge((channel = name, experiment = lab, ndf = count(sel), reference = c(mref), reference_with_prior = c(mref, sd),
                           best_norm_of_reference_RES = a),
                          NamedTuple(Symbol(t) => c([mean(σ_tune(t, target, :RES, b)) for b in bins]) for t in TUNES)))
    end
    Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
    save("ours/$(name).png", fig)
end
out = DataFrame(rows)
CSV.write("results/chi2.csv", out)
show(stdout, out, allrows = true, allcols = true); println()
