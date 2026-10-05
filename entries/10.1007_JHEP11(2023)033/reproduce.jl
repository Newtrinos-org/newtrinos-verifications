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
# the Spray code expects an atmosphere segment first: a vanishing one (1 μm, zero density) is prepended
layers = StructArray{Newtrinos.Layer}(([6391.0, 6371.0], [0.0, ρYe], [0.0, ρYe]))
paths = VectorOfVectors{Newtrinos.Path}([StructArray{Newtrinos.Path}(([1e-9, L], [1, 2]))])
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
