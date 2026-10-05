# Atmospheric neutrino oscillograms of the Earth with Newtrinos.jl (modules `osc` with standard matter
# interactions and `earth_layers`), compared with Akhmedov, Razzaque, Smirnov, JHEP 02 (2013) 082, Figs. 1-3.
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, StructArrays, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")

# oscillation parameters of the paper (Sec. 2): "Δm²₃₂" = 2.35e-3 eV², Δm²₂₁ = 7.6e-5 eV², sin²θ₂₃ = 0.42,
# sin²θ₁₂ = 0.312, sin²θ₁₃ = 0.025, δ = 0. The positions of the extrema in the paper's Fig. 2 are reproduced
# when 2.35e-3 eV² is used as Δm²₃₁ (the paper's formulae use Δ = Δm²₃₁ / 2E); IH: Δm²₃₁ → -Δm²₃₁.
const Δm² = 2.35e-3
base = (θ₁₂ = asin(sqrt(0.312)), θ₁₃ = asin(sqrt(0.025)), θ₂₃ = asin(sqrt(0.42)), δCP = 0.0, Δm²₂₁ = 7.6e-5)
params = (NH = (; base..., Δm²₃₁ = Δm²), IH = (; base..., Δm²₃₁ = -Δm²))

osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI()))
earth = Newtrinos.earth_layers.configure()

# (a) the default Newtrinos Earth: PREM averaged into four constant-density zones (as used in the analyses)
layers_default = earth.compute_layers()

# (b) close to the paper's full PREM profile: PREM (the tabulation shipped with Newtrinos) averaged in 100
#     shells of 63.71 km, with electron fraction ye_core / ye_mantle
function fine_layers(ye_core, ye_mantle; n = 100)
    tab = CSV.read(joinpath(pkgdir(Newtrinos), "src/physics/PREM_1s.csv"), DataFrame, header = false)
    r, ρ = reverse(tab[:, 1]), reverse(tab[:, 3])           # ascending radius
    ρ_at(x) = (k = clamp(searchsortedlast(r, x), 1, length(r) - 1);
               r[k] == r[k+1] ? ρ[k+1] : ρ[k] + (ρ[k+1] - ρ[k]) * (x - r[k]) / (r[k+1] - r[k]))
    edges = range(0, 6371, length = n + 1)
    radii, p, nn = [6391.0], [0.0], [0.0]                   # 20 km atmosphere, as in Newtrinos
    for i in n:-1:1                                          # outermost first
        xs = range(edges[i], edges[i+1], length = 50)
        ρ̄ = sum(ρ_at, xs) / length(xs)
        ye = edges[i+1] <= 3480 ? ye_core : ye_mantle
        push!(radii, edges[i+1]); push!(p, ρ̄ * ye); push!(nn, ρ̄ * (1 - ye))
    end
    StructArray{Newtrinos.Layer}((radii, p, nn))
end
layers_paper = fine_layers(0.5, 0.5)        # the paper's amplitudes at cos θz = -1 require Yₑ = 0.5 in the core
layers_fine = fine_layers(0.466, 0.494)     # same Yₑ as the default Newtrinos zones: isolates the layering

const FL = ("e", "μ", "τ")
prob(E, cz, layers, par; anti = false) = osc.osc_prob(E, earth.compute_paths(cz, layers), layers, par; anti)

# ---------------- Figs. 1 and 2: P vs E at four zenith angles ----------------
E1 = collect(range(1, 20, length = 1500))
CZ1 = [-1.0, -0.8, -0.6, -0.4]
P = Dict((h, a) => prob(E1, CZ1, layers_paper, params[h]; anti = a) for h in (:NH, :IH), a in (false, true))

function panel_column(fname, ylabel, curves)
    fig = Figure(size = (520, 820))
    for (j, cz) in enumerate(CZ1)
        ax = Axis(fig[j, 1], ylabel = ylabel, limits = (1, 20, -0.02, 1.02), yticks = 0:0.2:1,
                  xlabel = j == 4 ? "E_ν [GeV]" : "", xticklabelsvisible = j == 4)
        for c in curves
            lines!(ax, E1, c.p[:, j]; color = c.color, linestyle = c.style, linewidth = 1.6)
        end
        text!(ax, 19.5, 0.88, text = @sprintf("cos θz = %.1f", cz), align = (:right, :center), fontsize = 14)
    end
    rowgap!(fig.layout, 4)
    save(fname, fig, px_per_unit = 2)
end
col = (e = :blue, μ = :red, τ = :black)
Pd = P[(:NH, false)]
panel_column("ours/fig1a_numu.png", "P(νμ → νx)", [(p = Pd[:, :, 2, i], color = col[i], style = :solid) for i in 1:3])
panel_column("ours/fig1b_nue.png", "P(νe → νx)", [(p = Pd[:, :, 1, i], color = col[i], style = :solid) for i in 1:3])
for (a, tag, lab) in ((false, "fig2a_nu", "P(νx → νμ)"), (true, "fig2b_nubar", "P(ν̄x → ν̄μ)"))
    curves = [(p = P[(h, a)][:, :, α, 2], color = c, style = s)
              for (h, s) in ((:NH, :solid), (:IH, :dash)) for (α, c) in ((2, :blue), (1, :red))]
    panel_column("ours/$tag.png", lab, curves)
end

# ---------------- Fig. 3: oscillograms (NH, neutrinos), normalised to the maximum in the panel ----------------
E3 = collect(range(1, 17, length = 400))
CZ3 = collect(range(-1, 0, length = 400))
P3paper = prob(E3, CZ3, layers_paper, params.NH)
P3 = prob(E3, CZ3, layers_default, params.NH)
P3fine = prob(E3, CZ3, layers_fine, params.NH)
channels = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3)]
cmap = cgrad([:darkred, :red3, :orangered, :darkorange, :orange, :gold], 10, categorical = true)
function oscillograms(fname, P3, label)
    fig = Figure(size = (1150, 840))
    for (k, (α, β)) in enumerate(channels)
        z = P3[:, :, α, β]'
        zmax = α == β ? 1.0 : maximum(z)
        ttl = α == β ? "P$(FL[α])$(FL[β])" : @sprintf("P%s%s / %.2f", FL[α], FL[β], zmax)
        ax = Axis(fig[(k - 1) ÷ 3 + 1, (k - 1) % 3 + 1], title = ttl, xlabel = "cos θz",
                  ylabel = k in (1, 4) ? "E_ν [GeV]" : "", yticklabelsvisible = k in (1, 4),
                  xticks = ([-1, -0.75, -0.5, -0.25, 0], ["-1", "-0.75", "-0.5", "-0.25", "0"]))
        contourf!(ax, CZ3, E3, z ./ zmax, levels = 0:0.1:1, colormap = cmap)
        contour!(ax, CZ3, E3, z ./ zmax, levels = 0.05:0.05:0.95, color = (:black, 0.35), linewidth = 0.5)
    end
    Colorbar(fig[1:2, 4], colormap = cmap, limits = (0, 1), ticks = 0:0.1:1)
    Label(fig[0, 1:3], label, fontsize = 15)
    save(fname, fig, px_per_unit = 1.5)
end
oscillograms("ours/fig3_oscillograms.png", P3paper, "Earth: PREM in 100 shells, Yₑ = 0.5")
oscillograms("ours/fig3_oscillograms_4zone.png", P3, "Earth: default Newtrinos PREM (4 zones, Yₑ = 0.466-0.496)")

# maxima quoted in the paper's Fig. 3 titles: Peμ 0.42, Peτ 0.61, Pμe 0.42, Pμτ 0.97
paper_max = Dict((1, 2) => 0.42, (1, 3) => 0.61, (2, 1) => 0.42, (2, 3) => 0.97)
rows = map(collect(paper_max)) do ((α, β), pm)
    (channel = "P$(FL[α])$(FL[β])", paper_max = pm,
     newtrinos_max = round(maximum(P3paper[:, :, α, β]), digits = 3),
     newtrinos_max_default_earth = round(maximum(P3[:, :, α, β]), digits = 3))
end
sort!(rows, by = r -> r.channel)
CSV.write("results/fig3_channel_maxima.csv", DataFrame(rows))

# impact of the Earth layering: |P(4 zones) - P(100 shells)| over the Fig. 3 plane
lay = [(channel = "P$(FL[α])$(FL[β])",
        max_abs_diff = round(maximum(abs.(P3[:, :, α, β] .- P3fine[:, :, α, β])), digits = 3),
        mean_abs_diff = round(sum(abs.(P3[:, :, α, β] .- P3fine[:, :, α, β])) / length(P3[:, :, α, β]), digits = 4))
       for (α, β) in channels]
CSV.write("results/layering_4zone_vs_fine.csv", DataFrame(lay))

fig = Figure(size = (1150, 420))
for (k, (α, β)) in enumerate(((2, 2), (2, 1), (1, 1)))
    ax = Axis(fig[1, k], title = "P$(FL[α])$(FL[β]): 4 zones − 100 shells", xlabel = "cos θz",
              ylabel = k == 1 ? "E_ν [GeV]" : "")
    hm = heatmap!(ax, CZ3, E3, (P3[:, :, α, β] .- P3fine[:, :, α, β])', colormap = :balance, colorrange = (-0.3, 0.3))
    k == 3 && Colorbar(fig[1, 4], hm, label = "ΔP")
end
save("ours/layering_difference.png", fig, px_per_unit = 1.5)

println("Fig. 3 channel maxima (paper vs Newtrinos paper-like Earth / default Earth):")
for r in rows
    @printf("  %s  %.2f  %.3f / %.3f\n", r.channel, r.paper_max, r.newtrinos_max, r.newtrinos_max_default_earth)
end
println("4-zone vs 100-shell Earth (same Yₑ), max |ΔP| over 1-17 GeV, cos θz ∈ [-1, 0]:")
for r in lay
    @printf("  %s  max %.3f  mean %.4f\n", r.channel, r.max_abs_diff, r.mean_abs_diff)
end
