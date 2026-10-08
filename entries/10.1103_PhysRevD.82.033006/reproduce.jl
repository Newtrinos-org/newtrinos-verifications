# Reproduction of Borexino, Phys. Rev. D 82, 033006 (2010) [arXiv:0808.2868]: the ⁸B ν–e interaction rate above 3 MeV
# from the background-subtracted spectrum (Fig. 7), with the Newtrinos module borexino_ph1_spectrum (high-energy part).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, Optim, CairoMakie, Printf, CSV, DataFrames

cd(@__DIR__); mkpath("ours"); mkpath("results")
const B = Newtrinos.borexino_ph1_spectrum
physics = merge(B.default_physics(), (solar_flux = Newtrinos.solar_flux.configure(
    Newtrinos.solar_flux.SolarFluxConfig(systematics = Newtrinos.solar_flux.SSMPriors())),))
bx = B.configure(physics)
p0 = Newtrinos.get_params((; bx))
he = bx.assets.spectra.he
# χ² of the ⁸B spectrum (Gaussian, published errors) with the energy-scale nuisance (unit Gaussian)
expected(n, s) = B.get_expected(merge(p0, (solar_norm_b8 = n, borexino_ph1s_b8_scale = s)), bx.physics, bx.assets).he
chi2(n, s) = sum(((he.counts .- expected(n, s)) ./ he.sigma) .^ 2) + s^2
prof(n) = optimize(s -> chi2(n, s), -3.0, 3.0).minimum
norms = collect(0.4:0.01:1.6)
χ = prof.(norms)
ib = argmin(χ); nb = norms[ib]
rate(n) = sum(expected(n, 0.0)) / B.EXPOSURE_HE          # cpd/100 t above 3 MeV (recoil energy)
r0 = rate(1.0)
inside = norms[χ .<= χ[ib] + 1]
CSV.write("results/b8_profile.csv", DataFrame(rate_cpd100t = norms .* r0, chi2 = χ .- χ[ib]))
open("results/summary.txt", "w") do io
    @printf(io, "⁸B rate above 3 MeV: %.3f [%.3f, %.3f] cpd/100 t (Δχ² = 1), χ²_min = %.2f for %d bins; B23 MB22-met SSM + LMA (defaults): %.3f\n",
            nb * r0, minimum(inside) * r0, maximum(inside) * r0, χ[ib], length(he.counts), r0)
    println(io, "Borexino (PRD 82, 033006): 0.217 ± 0.038 (stat) ± 0.008 (syst) cpd/100 t")
end
print(read("results/summary.txt", String))
fig = Figure(size = (650, 450))
ax = Axis(fig[1, 1], xlabel = "Energy (MeV)", ylabel = "Counts / 2 MeV / 345.3 days", title = "Borexino Phase I: ⁸B spectrum (background subtracted)")
x = (he.lo .+ he.hi) ./ 2
edges = vcat(he.lo, he.hi[end])
stairs!(ax, edges, vcat(expected(1.0, 0.0), expected(1.0, 0.0)[end]), step = :post, color = :blue, linewidth = 2, label = "Newtrinos: B23 MB22-met + LMA")
stairs!(ax, edges, vcat(expected(nb, 0.0), expected(nb, 0.0)[end]), step = :post, color = :green, linestyle = :dash, linewidth = 2, label = "Newtrinos: best-fit ⁸B normalisation")
errorbars!(ax, x, he.counts, he.sigma, color = :red); scatter!(ax, x, he.counts, color = :red, label = "Data")
xlims!(ax, 3, 15); ylims!(ax, 0, 45)
axislegend(ax, position = :rt)
save("ours/fig7_b8_spectrum.png", fig)
