# B16 standard solar models (N. Vinyoles et al., ApJ 835, 202 (2017)) in Newtrinos.jl (`solar_flux.B16`) next to B23,
# and the paper's test of the models against the measured ⁸B and ⁷Be fluxes (Fig. 4), with the measured fluxes
# determined here from the Newtrinos solar experiments and KamLAND.
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Distributions, LinearAlgebra, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const SF = Newtrinos.solar_flux
# solar values used in the paper (Bergström et al. 2016, JHEP 03, 132): ⁸B and ⁷Be [cm⁻² s⁻¹]
SUN = (b8 = 5.16e6, be7 = 4.80e9)

# ---------------- (1) the models: fluxes and 1σ (2 dof) ellipses in the (⁷Be, ⁸B) plane ----------------
models = [("B16 GS98", SF.B16(composition = :GS98), :blue), ("B16 AGSS09met", SF.B16(composition = :AGSS09), :red),
          ("B23 GS98", SF.B23(composition = :GS98), :royalblue), ("B23 AGSS09", SF.B23(composition = :AGSS09), :salmon),
          ("B23 MB22m (Newtrinos default)", SF.B23(), :gray30)]
rows = NamedTuple[]
for (name, m, _) in models
    sf = SF.configure(SF.SolarFluxConfig(model = m))
    for c in SF.COMPONENTS
        push!(rows, (model = name, component = String(c), flux = sf.nominal[c], fractional_error = sf.fractional_error[c]))
    end
end
CSV.write("results/model_fluxes.csv", DataFrame(rows))
function ellipse(sf; level = 2.30, n = 200)
    ib, i8 = SF.component_index(:be7), SF.component_index(:b8)
    μ = [sf.nominal.be7 / SUN.be7, sf.nominal.b8 / SUN.b8]
    s = [sf.fractional_error.be7 * μ[1], sf.fractional_error.b8 * μ[2]]
    C = [s[1]^2 sf.correlation[ib, i8]*s[1]*s[2]; sf.correlation[ib, i8]*s[1]*s[2] s[2]^2]
    L = cholesky(Symmetric(C)).L
    pts = [μ .+ sqrt(level) .* (L * [cos(t), sin(t)]) for t in range(0, 2π, length = n)]
    μ, first.(pts), last.(pts)
end

# ---------------- (2) measured ⁸B and ⁷Be fluxes: solar experiments + KamLAND, both fluxes free ----------------
# flux priors of the default B23 (MB22m) model for the other components; ⁸B and ⁷Be free (Uniform)
physics = merge(Newtrinos.solar_common.default_physics(), (solar_flux = SF.configure(SF.SolarFluxConfig(systematics = SF.SSMPriors())),))
sk_spectra(name) = getproperty(Newtrinos, name).configure(physics; daynight = :combined)
experiments = merge(NamedTuple(name => (name in (:sk1_solar, :sk2_solar, :sk3_solar, :sk4_solar) ? sk_spectra(name) :
                                        getproperty(Newtrinos, name).configure(physics))
                               for name in (:chlorine, :gallex_gno, :sage, :sno, :sk1_solar, :sk2_solar, :sk3_solar, :sk4_solar,
                                            :borexino_ph1, :borexino_ph2, :borexino_ph3)),
                    (sk_solar_dn = Newtrinos.sk_solar_dn.configure(physics; dataset = :sk1to4), kamland = Newtrinos.kamland.configure()))
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
Φ0 = physics.solar_flux.nominal
θ13 = asin(sqrt(0.0218))
priors = merge(Newtrinos.get_priors(experiments), (θ₁₃ = θ13, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₃₁ = p.Δm²₃₁,
               θ₁₂ = Uniform(asin(sqrt(0.2)), asin(sqrt(0.45))), Δm²₂₁ = Uniform(5e-5, 1e-4)))
p = merge(p, (θ₁₃ = θ13, θ₁₂ = asin(sqrt(0.31)), Δm²₂₁ = 7.5e-5))
# grid in Φ/Φ_sun (7 × 7; Δχ² is close to quadratic, the 1σ region is taken from a quadratic fit to the grid)
xb = collect(range(0.85, 1.15, length = 13))[1:2:end]       # ⁷Be
y8 = collect(range(0.92, 1.08, length = 13))[1:2:end]       # ⁸B
grid = vec([(a, b) for a in xb, b in y8])
lp = fill(NaN, length(grid))
Threads.@threads for k in eachindex(grid)
    a, b = grid[k]
    nb, n8 = a * SUN.be7 / Φ0.be7, b * SUN.b8 / Φ0.b8
    pr = Newtrinos.BAT.distprod(; merge(priors, (solar_norm_be7 = nb, solar_norm_b8 = n8))...)
    _, lpk, _ = Newtrinos.find_mle_cached(likelihood, pr, merge(p, (solar_norm_be7 = nb, solar_norm_b8 = n8)), "cache")
    lp[k] = lpk
end
χ2 = -2 .* lp
# quadratic surface χ² = c₀ + c₁x + c₂y + c₃x² + c₄xy + c₅y² fitted to the grid
A = hcat(ones(length(grid)), first.(grid), last.(grid), first.(grid) .^ 2, first.(grid) .* last.(grid), last.(grid) .^ 2)
c = A \ χ2
H = [2c[4] c[5]; c[5] 2c[6]]
best = -H \ c[2:3]
χ2min = c[1] + c[2:3]' * best / 2
quad_residual = maximum(abs.(A * c .- χ2))
Z = reshape(χ2 .- χ2min, length(xb), length(y8))
CSV.write("results/b8_be7_profile.csv", DataFrame(be7_over_sun = first.(grid), b8_over_sun = last.(grid), dchi2 = vec(Z),
                                                  dchi2_quadratic = A * c .- χ2min))

fig = Figure(size = (900, 640))
ax = Axis(fig[1, 1], xlabel = "Φ(⁷Be) / Φ(⁷Be)_sun", ylabel = "Φ(⁸B) / Φ(⁸B)_sun", limits = (0.84, 1.16, 0.68, 1.32),
          title = "⁸B and ⁷Be: solar models (1σ, 2 dof) and the Newtrinos solar + KamLAND fit")
for (name, m, col) in models
    μ, ex, ey = ellipse(SF.configure(SF.SolarFluxConfig(model = m)))
    lines!(ax, ex, ey, color = col, linewidth = startswith(name, "B16") ? 2.5 : 1.2, linestyle = startswith(name, "B16") ? :solid : :dash, label = name)
    scatter!(ax, [μ[1]], [μ[2]], color = col, marker = startswith(name, "B16") ? :rect : :circle, markersize = 12)
end
errorbars!(ax, [1.0], [1.0], [0.046], [0.050], direction = :x, color = :black)
errorbars!(ax, [1.0], [1.0], [0.017], [0.025], color = :black)
scatter!(ax, [1.0], [1.0], color = :black, marker = :diamond, markersize = 12, label = "solar values used in the paper (Bergström et al. 2016)")
Lf = cholesky(Symmetric(inv(H / 2))).L      # Δχ² = (v - best)ᵀ (H/2) (v - best)
fit_pts = [best .+ sqrt(2.30) .* (Lf * [cos(t), sin(t)]) for t in range(0, 2π, length = 200)]
lines!(ax, first.(fit_pts), last.(fit_pts), color = :darkgreen, linewidth = 3)
scatter!(ax, [best[1]], [best[2]], color = :darkgreen, marker = :star5, markersize = 18, label = "Newtrinos solar + KamLAND fit (Δχ² = 2.30)")
axislegend(ax, position = :rb, framevisible = false, labelsize = 11)
save("ours/fig_b8_be7.png", fig)

# 1D 1σ intervals of the measured fluxes (other flux profiled): σᵢ = sqrt((2 H⁻¹)ᵢᵢ)
σ = sqrt.(diag(2inv(H)))
ρ = (2inv(H))[1, 2] / (σ[1] * σ[2])
summary = DataFrame(quantity = ["Φ(⁷Be)/Φ_sun", "Φ(⁸B)/Φ_sun"], newtrinos_best = best, newtrinos_sigma = σ,
                    bergstrom = [1.0, 1.0], bergstrom_lo = [1 - 0.046, 1 - 0.017], bergstrom_hi = [1.05, 1.025],
                    correlation = [ρ, ρ], quadratic_fit_max_residual = [quad_residual, quad_residual])
CSV.write("results/summary.csv", summary)
show(stdout, summary, allrows = true); println()
