# NOMAD quasi-elastic νμ and ν̄μ cross sections, 3–100 GeV, on its (mainly carbon) target and converted to free nucleons
# (EPJC 63, 355 (2009); tables from the arXiv:0812.4543 source), with the deuterium bubble-chamber data the paper compares
# to (ANL 1977, BNL 1981, FNAL 1983; NUISANCE data files), compared with the quasi-elastic cross sections of Newtrinos.jl
# (module `xsec`): the GENIE tunes (`generator_sigma`, process QE, and QE + MEC) per neutron (ν) or proton (ν̄) of carbon
# and of free nucleons, and the reference model of `CorrelatedXsec` with its prior 1σ band (for free nucleons only the
# axial-mass and normalisation knobs act; the tune-spread modes are defined on nuclei).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics, Random, Distributions

cd(@__DIR__); mkpath("ours"); mkpath("results")
const X = Newtrinos.xsec

readc(f) = CSV.read(f, DataFrame; comment = "#")
Nn, Nb = readc("data/nomad_nu.csv"), readc("data/nomad_nubar.csv")
anl, bnl, fnal = readc("data/anl_d2.csv"), readc("data/bnl_d2.csv"), readc("data/fnal_d2.csv")

tab = X.load_table()
E = 10 .^ range(log10(0.1), log10(120), length = 300)
at(v, e = E) = X.at_energies(tab.E, v, e)
TUNES = Symbol.(sort(tab.tunes))
# σ per struck nucleon: carbon has 6 neutrons and 6 protons out of 12 nucleons → per nucleon × 2; free n / p: per nucleon
pernuc(target) = target === :C ? 2.0 : 1.0
σ_tune(t, target, anti, procs) = at(sum(getfield(X.generator_sigma(t, target, :numu, anti, :CC), p) for p in procs)) .* E .* pernuc(target)

xs = X.configure(X.CorrelatedXsec())
ref = xs.cfg.reference
sigma(target, anti, procs) = (f = xs.sigma(E; flavour = :numu, anti, current = :CC, target, process = procs); p -> f(p) .* E .* pernuc(target))
rng = MersenneTwister(1)
draw() = map(p -> p isa Distribution ? rand(rng, p) : p, xs.priors)
samples = [draw() for _ in 1:500]
p0 = xs.params
q(M, a) = [quantile(r, a) for r in eachrow(M)]
tunecol(i) = (Makie.wong_colors()[mod1(i, 7)], 0.7)

function panel(pos, target, anti, title, ylab, data)
    ax = Axis(pos, xscale = log10, xlabel = "E_ν (GeV)", ylabel = ylab, title = title, limits = (0.1, 120, 0, 1.8),
              xticks = ([0.1, 0.3, 1, 3, 10, 30, 100], ["0.1", "0.3", "1", "3", "10", "30", "100"]))
    fQE = sigma(target, anti, :QE)
    M = reduce(hcat, [fQE(p) for p in samples])
    band!(ax, E, q(M, 0.16), q(M, 0.84), color = (:dodgerblue, 0.25), label = "CorrelatedXsec QE prior 1σ")
    for (i, t) in enumerate(TUNES)
        lines!(ax, E, σ_tune(t, target, anti, (:QE,)), color = tunecol(i), linewidth = 1, label = "GENIE $(t) QE")
    end
    lines!(ax, E, fQE(p0), color = :dodgerblue, linewidth = 2.5, label = "reference $(ref) QE")
    target === :C && lines!(ax, E, sigma(target, anti, (:QE, :MEC))(p0), color = :dodgerblue, linestyle = :dash, linewidth = 2,
                            label = "reference QE + MEC")
    for (d, lab, mk) in data
        errorbars!(ax, d.x, d.y, d.e, color = :black, whiskerwidth = 4)
        scatter!(ax, d.x, d.y, marker = mk, color = mk == :circle ? :black : :white, strokecolor = :black, strokewidth = 1.2, label = lab)
    end
    ax
end
mid(lo, hi) = sqrt.(lo .* hi)
nomad_c(d) = (x = d.E_mean, y = d.sigma_nucl, e = d.err_tot_nucl)
nomad_f(d) = (x = d.E_mean, y = d.sigma_free, e = sqrt.(d.stat_free .^ 2 .+ d.syst_free .^ 2))
specs = (
    ("nu_carbon", :C, false, "NOMAD νμ QE, carbon", "σ_QE per neutron (10⁻³⁸ cm²)", [(nomad_c(Nn), "NOMAD", :circle)]),
    ("nubar_carbon", :C, true, "NOMAD ν̄μ QE, carbon", "σ_QE per proton (10⁻³⁸ cm²)", [(nomad_c(Nb), "NOMAD", :circle)]),
    ("nu_free", :n, false, "νμ n → μ⁻ p: free neutron / deuterium", "σ_QE per neutron (10⁻³⁸ cm²)",
     [(nomad_f(Nn), "NOMAD (converted to free n)", :circle),
      ((x = mid(anl.E_lo, anl.E_hi), y = anl.sigma_fluxtuned, e = anl.err_fluxtuned), "ANL 1977 (D₂)", :rect),
      ((x = mid(bnl.E_lo, bnl.E_hi), y = bnl.sigma, e = bnl.err), "BNL 1981 (D₂)", :utriangle),
      ((x = mid(fnal.E_lo, fnal.E_hi), y = fnal.sigma_nuis, e = fnal.err_nuis), "FNAL 1983 (D₂)", :diamond)]),
)
for (name, target, anti, title, ylab, data) in specs
    fig = Figure(size = (820, 480))
    ax = panel(fig[1, 1], target, anti, title, ylab, data)
    Legend(fig[1, 2], ax, framevisible = false, labelsize = 9)
    save("ours/$(name).png", fig)
end

# χ² (stat ⊕ syst, uncorrelated; NOMAD's normalisation comes from its DIS sample) of the QE prediction at the bin-mean
# energies, for the reference (with and without its prior spread) and the tunes
function chi2s(d, target, anti; y = :sigma_nucl, err = d.err_tot_nucl)
    fx = xs.sigma(d.E_mean; flavour = :numu, anti, current = :CC, target, process = :QE)
    m(p) = fx(p) .* d.E_mean .* pernuc(target)
    sd = vec(std(reduce(hcat, [m(p) for p in samples]), dims = 2))
    c(mm, extra = 0) = sum((d[!, y] .- mm) .^ 2 ./ (err .^ 2 .+ extra .^ 2))
    tunes = [c(at(getfield(X.generator_sigma(t, target, :numu, anti, :CC), :QE), d.E_mean) .* d.E_mean .* pernuc(target)) for t in TUNES]
    (c(m(p0)), c(m(p0), sd), tunes, m(p0))
end
rows = []
for (lab, d, target, anti, y, err) in (("nu carbon", Nn, :C, false, :sigma_nucl, Nn.err_tot_nucl),
                                       ("nubar carbon", Nb, :C, true, :sigma_nucl, Nb.err_tot_nucl),
                                       ("nu free (NOMAD converted)", Nn, :n, false, :sigma_free, sqrt.(Nn.stat_free .^ 2 .+ Nn.syst_free .^ 2)))
    c0, cb, ct, m0 = chi2s(d, target, anti; y, err)
    push!(rows, merge((dataset = lab, ndf = nrow(d), reference = c0, reference_with_prior = cb,
                       mean_data_over_reference = mean(d[!, y] ./ m0)), NamedTuple(Symbol(t) => v for (t, v) in zip(TUNES, ct))))
end
out = DataFrame(rows)
CSV.write("results/chi2.csv", out)
show(stdout, out, allrows = true, allcols = true); println()
