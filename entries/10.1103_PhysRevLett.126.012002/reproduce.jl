# Reproduction of COHERENT, Phys. Rev. Lett. 126, 012002 (2021) [arXiv:2003.10630], Analysis A with Newtrinos.jl (module `coherent_lAr`, SM-scaled cross-section model).
# Data: COHERENT CENNS-10 liquid-argon data release (Analysis A).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames, DensityInterface

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

experiments = (coherent_lAr = Newtrinos.coherent_lAr.configure(xsec_model = :sm_scale),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)

# CEvNS rate scaled as a whole (cevns_xsec_scale = N / N_SM); sin²θ_W fixed, nuclear radius and all
# background / systematic nuisances profiled
priors = merge(priors, (sin2thetaW = p.sin2thetaW, cevns_xsec_scale = Uniform(0.0, 2.5)))
mkpath("cache/scale")
result = Newtrinos.profile(likelihood, priors, OrderedDict(:cevns_xsec_scale => 41), p, cache_dir = "cache/scale")
FileIO.save("results/coherent_lAr.jld2", Dict("scale" => result))
save_csv("results/coherent_lAr_scale.csv", result)

# number of CEvNS events predicted by our model for the SM cross section (scale = 1)
e = experiments.coherent_lAr
N_SM = sum(Newtrinos.coherent_lAr.get_expected(p, e.physics, e.assets))
println("Newtrinos SM prediction: $(round(N_SM, digits = 1)) CEvNS events (paper: 128 ± 17)")

# Fig. 8: -2Δ ln L vs number of CEvNS events (Analysis A curve extracted from the vector figure, data/)
off = CSV.read("data/official_fig8_analysisA.csv", DataFrame)
x = result.axes.cevns_xsec_scale .* N_SM
y = 2 .* (maximum(result.values.log_posterior) .- result.values.log_posterior)
fig = Figure(size = (750, 520))
ax = Axis(fig[1, 1], title = "COHERENT CENNS-10 (LAr), Analysis A", xlabel = "CEvNS counts", ylabel = "-2 Δ ln L")
lines!(ax, off.cevns_counts, off.minus2_dlnL, color = :red, linestyle = :dash, linewidth = 2, label = "COHERENT Analysis A")
vlines!(ax, [128.0], color = :red, linewidth = 1, label = "SM prediction (paper)")
lines!(ax, x, y, color = :blue, linewidth = 3, label = "Newtrinos (profile)")
vlines!(ax, [N_SM], color = :blue, linewidth = 1, linestyle = :dot, label = "SM prediction (Newtrinos)")
xlims!(ax, 0, 300); ylims!(ax, 0, 15.5)
axislegend(ax, position = :ct)
save("ours/fig8_likelihood_nevents.png", fig)
open("results/sm_prediction.txt", "w") do io
    println(io, "Newtrinos SM prediction for Analysis A: $(N_SM) CEvNS events")
end
