# The 3+1 matter-enhanced ν̄μ disappearance through the Earth with Newtrinos.jl (module `osc`, `Sterile` flavour model
# with standard matter interactions, and `earth_layers`), compared with IceCube, Phys. Rev. Lett. 125, 141801 (2020),
# Fig. 1, at the global-fit point Δm²₄₁ = 1.3 eV², sin²2θ₂₄ = 0.07, θ₃₄ = 0.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, StructArrays

cd(@__DIR__); mkpath("ours"); mkpath("results")

osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(flavour = Newtrinos.osc.Sterile(), interaction = Newtrinos.osc.SI()))
earth = Newtrinos.earth_layers.configure()
p = merge(osc.params, (Δm²₄₁ = 1.3, θ₂₄ = asin(sqrt(0.07)) / 2, θ₁₄ = 0.0, θ₃₄ = 0.0, δCP = 0.0))

# the full PREM profile: PREM zoned at its density discontinuities with chord-averaged densities (pieces ≤ 100 km),
# Yₑ 0.466 in the core and 0.494 elsewhere (as the default zones)
z = Newtrinos.earth_layers.PREM_discontinuities().zones
earth_d = Newtrinos.earth_layers.configure(Newtrinos.earth_layers.PREM(zones = z, p_fractions = [z[i] >= 5.6 ? 0.466 : 0.494 for i in 1:length(z)-1]))

E = 10 .^ range(2, 5, length = 300)                     # GeV
CZ = collect(range(-0.999, -0.001, length = 300))
D = Dict()
lay4 = earth.compute_layers()
earths = (default = (earth.compute_paths(CZ, lay4), lay4),
          chord = earth.compute_chord_paths(CZ, lay4),
          fine = earth_d.compute_chord_paths(CZ, earth_d.compute_layers(); max_length = 100.0))
for (name, (paths, layers)) in pairs(earths)
    P = osc.osc_prob(E, paths, layers, p; anti = true)
    D[string(name)] = 100 .* (1 .- permutedims(P[:, :, 2, 2]))  # disappearance [%], [cz, E]
end

cmap = cgrad(:Blues, rev = true)
fig = Figure(size = (1700, 500))
for (k, (name, ttl)) in enumerate((("fine", "full PREM (chord-averaged)"), ("default", "4 zones (default)"), ("chord", "4 zones, chord-averaged")))
    ax = Axis(fig[1, k], yscale = log10, xlabel = "cos θ_z", ylabel = k == 1 ? "E_ν (GeV)" : "",
              title = "ν̄μ disappearance — $ttl", limits = (-1, 0, 1e2, 1e5))
    contourf!(ax, CZ, E, D[name], levels = 0:4:100, colormap = cmap)
    vlines!(ax, [-0.98, -0.83], color = :white, linestyle = :dash, linewidth = 1.5)
    k == 3 && Colorbar(fig[1, 4], colormap = cmap, limits = (0, 100), label = "disappearance (%)")
end
save("ours/fig1_oscillogram.png", fig)

# position and depth of the matter resonance in the mantle and in the core
function resonance(d, czsel)
    sub = d[czsel, :]
    i = argmax(sub); (ci, ei) = Tuple(CartesianIndices(sub)[i])
    (cz = CZ[czsel][ci], E_TeV = E[ei] / 1e3, disappearance_pct = sub[ci, ei])
end
rows = [(earth = k, region = reg, resonance(D[k], sel)...) for k in ("fine", "default", "chord")
        for (reg, sel) in (("mantle", (CZ .> -0.83) .& (CZ .< -0.3)), ("core", CZ .< -0.84))]
tab = DataFrame(rows)
CSV.write("results/resonance.csv", tab)
show(stdout, tab); println()
