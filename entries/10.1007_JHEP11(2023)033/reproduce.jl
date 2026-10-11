# Averaging of fast oscillations in Newtrinos.jl (module `osc`, propagation model `Spray`: the ray-to-spray method of
# M. Maltoni, JHEP 11 (2023) 033), checked against brute-force numerical averaging over the same energy cell, for the
# NOvA baseline (810 km, ρ = 2.84 g/cm³) and Δm²₃₁ from the physical value up to 10 eV² (the range of the paper's
# figure 'Impact of averaging').
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 4 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, StructArrays, ArraysOfArrays

cd(@__DIR__); mkpath("ours"); mkpath("results")
const O = Newtrinos.osc

# the paper's fixed parameters (NuFIT 5.2)
fixed = (θ₁₂ = asin(sqrt(0.303)), θ₁₃ = asin(sqrt(0.022)), θ₂₃ = asin(sqrt(0.572)), δCP = deg2rad(197), Δm²₂₁ = 7.41e-5)
# averaging cell: one step of the NOvA module's logarithmic true-energy grid (400 points over 0.1-10 GeV)
const W = log(10.0 / 0.1) / 399
L, ρYe = 810.0, 2.84 * 0.5
# one constant-density layer, as in the accelerator modules
layers = StructArray{Newtrinos.Layer}(([6371.0], [ρYe], [ρYe]))
paths = VectorOfVectors{Newtrinos.Path}([[Newtrinos.Path(L, 1)]])
ray = O.configure(O.OscillationConfig(interaction = O.SI()))
spray = O.configure(O.OscillationConfig(interaction = O.SI(), propagation = O.Spray(averaging = :uniform, σ_E = W, σ_h = 0.0)))

E = 10 .^ range(log10(0.5), log10(5.0), length = 600)
# brute force: P averaged uniformly over [E(1 − W/2), E(1 + W/2)] (the cell Spray averages over), 801 samples
const U = range(-0.5, 0.5, length = 801)
function brute(p, α, β)
    Ef = vec([e * (1 + u * W) for u in U, e in E])
    P = ray.osc_prob(Ef, paths, layers, p)[:, 1, α, β]
    vec(sum(reshape(P, length(U), length(E)), dims = 1)) ./ length(U)
end

DM = [2.5e-3, 0.02, 0.2, 1.0, 10.0]
CH = (("P(νμ → νe)", 2, 1), ("P(νμ → νμ)", 2, 2))
rows = []
fig = Figure(size = (1150, 1100))
for (r, dm) in enumerate(DM), (c, (lab, α, β)) in enumerate(CH)
    p = merge(fixed, (Δm²₃₁ = dm,))
    Pr = ray.osc_prob(E, paths, layers, p)[:, 1, α, β]
    Ps = spray.osc_prob(E, paths, layers, p)[:, 1, α, β]
    Pb = brute(p, α, β)
    push!(rows, (dm2_31 = dm, channel = lab, max_abs_spray_vs_bruteforce = maximum(abs.(Ps .- Pb)),
                 max_abs_ray_vs_bruteforce = maximum(abs.(Pr .- Pb))))
    ax = Axis(fig[r, c], xscale = log10, title = @sprintf("%s, Δm²₃₁ = %g eV²", lab, dm), xlabel = r == length(DM) ? "E (GeV)" : "",
              xticks = ([0.5, 1, 2, 5], ["0.5", "1", "2", "5"]))
    lines!(ax, E, Pr, color = (:black, 0.5), linewidth = 0.7, label = "ray (no averaging)")
    lines!(ax, E, Pb, color = :orange, linewidth = 3, label = "brute-force cell average")
    lines!(ax, E, Ps, color = :purple, linewidth = 1.5, linestyle = :dash, label = "Spray")
    r == 1 && c == 1 && axislegend(ax, position = :rt, framevisible = false, labelsize = 10)
end
save("ours/spray_vs_bruteforce.png", fig)
tab = DataFrame(rows)
CSV.write("results/spray_vs_bruteforce.csv", tab)
show(stdout, tab, allrows = true); println()

# ---------------- the paper's figure 'Impact of averaging', NOvA and T2K panels: Δχ²(Δm²₃₁) with the Newtrinos modules ----------------
# −2 ln L vs Δm²₃₁ from 10⁻⁵ to 10 eV² with the other oscillation parameters at the paper's values and the nuisance parameters at
# their nominal values, without averaging (ray) and with Spray over one cell of each module's logarithmic true-energy grid
using DensityInterface
cells = (nova = log(10.0 / 0.1) / 399, t2k = log(10.0 / 0.05) / 499)
DMs = 10 .^ range(-5, 1, length = 481)
chi2 = Dict{Tuple{Symbol, Symbol}, Vector{Float64}}()
for name in (:nova, :t2k), (tag, prop) in ((:ray, O.Basic()), (:spray, O.Spray(averaging = :uniform, σ_E = cells[name], σ_h = 0.0)))
    mod = getproperty(Newtrinos, name)
    physics = (osc = O.configure(O.OscillationConfig(interaction = O.SI(), eigen_method = Newtrinos.BargerEigen(), propagation = prop)),)
    exps = NamedTuple{(name,)}((mod.configure(physics),))
    llh = Newtrinos.generate_likelihood(exps); p0 = merge(Newtrinos.get_params(exps), fixed)
    chi2[(name, tag)] = [-2 * logdensityof(llh, merge(p0, (Δm²₃₁ = dm,))) for dm in DMs]
end
CSV.write("results/dchi2_dm31.csv", DataFrame(dm2_31 = DMs, nova_ray = chi2[(:nova, :ray)], nova_spray = chi2[(:nova, :spray)],
                                              t2k_ray = chi2[(:t2k, :ray)], t2k_spray = chi2[(:t2k, :spray)]))
fig = Figure(size = (800, 650))
for (r, (name, c, lab)) in enumerate(((:nova, :magenta, "NOvA"), (:t2k, :red, "T2K")))
    ax = Axis(fig[r, 1], xscale = log10, ylabel = "Δχ²", xlabel = r == 2 ? "Δm²₃₁ (eV²)" : "", title = lab,
              xticks = LogTicks(-5:1))
    m = min(minimum(chi2[(name, :ray)]), minimum(chi2[(name, :spray)]))
    lines!(ax, DMs, chi2[(name, :ray)] .- m, color = :black, linewidth = 0.8, label = "ray (no averaging)")
    lines!(ax, DMs, chi2[(name, :spray)] .- m, color = c, linewidth = 2.5, label = "Spray")
    xlims!(ax, 1e-5, 10); ylims!(ax, 0, nothing); r == 1 && axislegend(ax, position = :rt, framevisible = false)
end
save("ours/dchi2_dm31.png", fig)
for name in (:nova, :t2k)
    hi = DMs .>= 0.2
    @printf("%s: Δm²₃₁ ≥ 0.2 eV²: ray Δχ² spread %.0f, Spray spread %.1f; |Spray − ray| at Δm²₃₁ < 0.02 eV²: %.3f\n", name,
            maximum(chi2[(name, :ray)][hi]) - minimum(chi2[(name, :ray)][hi]), maximum(chi2[(name, :spray)][hi]) - minimum(chi2[(name, :spray)][hi]),
            maximum(abs.(chi2[(name, :spray)][.!hi .& (DMs .< 0.02)] .- chi2[(name, :ray)][.!hi .& (DMs .< 0.02)])))
end
