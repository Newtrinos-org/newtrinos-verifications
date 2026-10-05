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

# PREM in 100 shells (Yₑ 0.466 core / 0.494 elsewhere) as a finely layered alternative to the default four zones
function fine_layers(n = 100)
    tab = CSV.read(joinpath(pkgdir(Newtrinos), "src/physics/PREM_1s.csv"), DataFrame, header = false)
    r, ρ = reverse(tab[:, 1]), reverse(tab[:, 3])
    ρ_at(x) = (k = clamp(searchsortedlast(r, x), 1, length(r) - 1);
               r[k] == r[k+1] ? ρ[k+1] : ρ[k] + (ρ[k+1] - ρ[k]) * (x - r[k]) / (r[k+1] - r[k]))
    edges = range(0, 6371, length = n + 1)
    radii, pp, nn = [6391.0], [0.0], [0.0]
    for i in n:-1:1
        ρ̄ = sum(ρ_at, range(edges[i], edges[i+1], length = 50)) / 50
        ye = edges[i+1] <= 3480 ? 0.466 : 0.494
        push!(radii, edges[i+1]); push!(pp, ρ̄ * ye); push!(nn, ρ̄ * (1 - ye))
    end
    StructArray{Newtrinos.Layer}((radii, pp, nn))
end

E = 10 .^ range(2, 5, length = 300)                     # GeV
CZ = collect(range(-0.999, -0.001, length = 300))
D = Dict()
for (name, layers) in (("default", earth.compute_layers()), ("fine", fine_layers()))
    P = osc.osc_prob(E, earth.compute_paths(CZ, layers), layers, p; anti = true)
    D[name] = 100 .* (1 .- permutedims(P[:, :, 2, 2]))  # disappearance [%], [cz, E]
end

cmap = cgrad(:Blues, rev = true)
fig = Figure(size = (1180, 500))
for (k, (name, ttl)) in enumerate((("fine", "PREM, 100 shells"), ("default", "default Newtrinos Earth (4 zones)")))
    ax = Axis(fig[1, k], yscale = log10, xlabel = "cos θ_z", ylabel = k == 1 ? "E_ν (GeV)" : "",
              title = "ν̄μ disappearance — $ttl", limits = (-1, 0, 1e2, 1e5))
    contourf!(ax, CZ, E, D[name], levels = 0:4:100, colormap = cmap)
    vlines!(ax, [-0.98, -0.83], color = :white, linestyle = :dash, linewidth = 1.5)
    k == 2 && Colorbar(fig[1, 3], colormap = cmap, limits = (0, 100), label = "disappearance (%)")
end
save("ours/fig1_oscillogram.png", fig)

# position and depth of the matter resonance in the mantle and in the core
function resonance(d, czsel)
    sub = d[czsel, :]
    i = argmax(sub); (ci, ei) = Tuple(CartesianIndices(sub)[i])
    (cz = CZ[czsel][ci], E_TeV = E[ei] / 1e3, disappearance_pct = sub[ci, ei])
end
rows = [(earth = k, region = reg, resonance(D[k], sel)...) for k in ("fine", "default")
        for (reg, sel) in (("mantle", (CZ .> -0.83) .& (CZ .< -0.3)), ("core", CZ .< -0.84))]
tab = DataFrame(rows)
CSV.write("results/resonance.csv", tab)
show(stdout, tab); println()
