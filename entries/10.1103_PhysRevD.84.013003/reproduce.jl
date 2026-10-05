# Neutrino oscillations with large extra dimensions in Newtrinos.jl (module `osc`, flavour model `ADD` with a truncated
# Kaluza-Klein tower) compared with Machado, Nunokawa, Zukanovich Funchal, Phys. Rev. D 84, 013003 (2011), Fig. 1, and
# with the semi-analytic solution for the infinite KK tower (the paper's appendix).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, LinearAlgebra

cd(@__DIR__); mkpath("ours"); mkpath("results")
const O = Newtrinos.osc

# parameters of the paper's Fig. 1: a = 0.5 μm, m₀ = 0
s13sq = (1 - sqrt(1 - 0.07)) / 2                            # sin²2θ₁₃ = 0.07
base = (θ₁₂ = asin(sqrt(0.32)), θ₁₃ = asin(sqrt(s13sq)), θ₂₃ = asin(sqrt(0.5)), δCP = 0.0, Δm²₂₁ = 7.59e-5)
par = Dict(:NH => (; base..., Δm²₃₁ = 2.46e-3, m₀ = 0.0, ADD_radius = 0.5),
           :IH => (; base..., Δm²₃₁ = -2.46e-3, m₀ = 0.0, ADD_radius = 0.5))
std3 = O.configure(O.OscillationConfig())
addosc(N, ord) = O.configure(O.OscillationConfig(flavour = O.ADD(N_KK = N, three_flavour = O.ThreeFlavour(ordering = ord))))

# ---- semi-analytic infinite tower (vacuum): eigenvalues λ of a²M†M from λ = (πξ²/2) cot(πλ) in (n, n + 1/2),
#      zero-mode weights W₀² = 1 / (1 + ξ² Σ_A A²/(A² − λ²)²), ξ = √2 m a;  amplitude A_αβ = Σ_j U_αj U*_βj Σ_n W₀² e^{-iλ²L/2Ea²}
const UMEV = 5.067730716156395                              # 1 μm in eV⁻¹ (as osc.jl)
const KM = 1e9 * UMEV                                       # 1 km in eV⁻¹
function kk_modes(ξ; nmax = 300, Asum = 200_000)
    ξ == 0 && return [0.0], [1.0]
    f(λ) = λ * sin(π * λ) - (π * ξ^2 / 2) * cos(π * λ)
    λs = map(0:nmax) do n                                  # bisection on (n, n + 1/2)
        lo, hi = n + 1e-15, n + 0.5
        for _ in 1:200
            mid = (lo + hi) / 2
            (f(lo) * f(mid) <= 0) ? (hi = mid) : (lo = mid)
        end
        (lo + hi) / 2
    end
    A = 1:Asum
    w = [1 / (1 + ξ^2 * (sum(@. A^2 / (A^2 - λ^2)^2) + 1 / Asum)) for λ in λs]   # 1/Asum: tail of Σ 1/A²
    λs, w
end
function P_tower(ord, L_km, E_GeV, α, β; anti = false)
    p = par[ord]
    U = O.get_PMNS(p); U = anti ? conj(U) : U
    m = O.get_abs_masses(p)
    a = p.ADD_radius * UMEV
    modes = [kk_modes(sqrt(2) * mj * a) for mj in m]
    map(E_GeV) do e
        amp = sum(U[α, j] * conj(U[β, j]) * sum(w * cis(-λ^2 * L_km * KM / (2 * e * 1e9 * a^2)) for (λ, w) in zip(modes[j]...)) for j in 1:3)
        abs2(amp)
    end
end

CASES = [(L = 735.0, E = collect(range(1.0, 5.0, length = 1200)), fl = 2, anti = false, xlabel = "E (GeV)", scale = 1.0,
          ylab = "P(νμ → νμ)", ylim = (0, 1)),
         (L = 180.0, E = collect(range(1.5e-3, 8.7e-3, length = 1500)), fl = 1, anti = true, xlabel = "E (MeV)", scale = 1e3,
          ylab = "P(ν̄e → ν̄e)", ylim = (0, 1)),
         (L = 1.0, E = collect(range(1.5e-3, 8.7e-3, length = 3000)), fl = 1, anti = true, xlabel = "E (MeV)", scale = 1e3,
          ylab = "P(ν̄e → ν̄e)", ylim = (0.7, 1))]
NKK = 20
# Gaussian energy smearing σ_E/E = 3 %, evaluated only where ±2σ lies inside the energy range
function smear(E, P; r = 0.03)
    out = fill(NaN, length(E))
    for i in eachindex(E)
        σ = r * E[i]
        (E[i] - 2σ < E[1] || E[i] + 2σ > E[end]) && continue
        w = @. exp(-(E - E[i])^2 / (2σ^2))
        out[i] = sum(w .* P) / sum(w)
    end
    out
end
nanmax(x) = maximum(filter(!isnan, x))
rows = []
fig = Figure(size = (620, 1150))
for (k, c) in enumerate(CASES)
    ax = Axis(fig[k, 1], xlabel = c.xlabel, ylabel = c.ylab, limits = (c.E[1] * c.scale, c.E[end] * c.scale, c.ylim...),
              title = @sprintf("%g km", c.L))
    lines!(ax, c.E .* c.scale, std3.osc_prob(c.E, [c.L], merge(base, (Δm²₃₁ = 2.46e-3,)); anti = c.anti)[:, 1, c.fl, c.fl],
           color = :black, linewidth = 2, label = "standard (Newtrinos)")
    for (ord, col, ls) in ((:NH, :blue, :dash), (:IH, :red, :dot))
        Pn = addosc(NKK, ord == :NH ? :NO : :IO).osc_prob(c.E, [c.L], par[ord]; anti = c.anti)[:, 1, c.fl, c.fl]
        Pt = P_tower(ord, c.L, c.E, c.fl, c.fl; anti = c.anti)
        lines!(ax, c.E .* c.scale, Pn, color = col, linestyle = ls, linewidth = 1.6, label = "LED $(ord), Newtrinos N_KK = $NKK")
        for N in (1, 3, 5, 10, 20)
            PN = N == NKK ? Pn : addosc(N, ord == :NH ? :NO : :IO).osc_prob(c.E, [c.L], par[ord]; anti = c.anti)[:, 1, c.fl, c.fl]
            push!(rows, (baseline_km = c.L, ordering = String(ord), N_KK = N, max_abs_diff_vs_infinite_tower = maximum(abs.(PN .- Pt)),
                         max_abs_diff_smeared_3pct = nanmax(abs.(smear(c.E, PN) .- smear(c.E, Pt)))))
        end
    end
    k == 1 && axislegend(ax, position = :rb, framevisible = false, labelsize = 10)
end
save("ours/fig1_probabilities.png", fig)

tab = DataFrame(rows)
CSV.write("results/kk_truncation.csv", tab)
fig = Figure(size = (1100, 430))
for (k, (col_, ttl)) in enumerate(((:max_abs_diff_vs_infinite_tower, "pointwise"), (:max_abs_diff_smeared_3pct, "after σ_E/E = 3 % smearing")))
    local ax = Axis(fig[1, k], xscale = log10, yscale = log10, xlabel = "number of KK modes N_KK",
                    ylabel = k == 1 ? "max |P(N_KK) − P(∞)|" : "", title = "Newtrinos ADD vs. infinite KK tower: $ttl",
                    xticks = [1, 3, 5, 10, 20], limits = (nothing, nothing, 1e-6, 1))
    for (c, mk) in zip(CASES, (:circle, :rect, :utriangle)), (ord, colr) in ((:NH, :blue), (:IH, :red))
        sub = tab[(tab.baseline_km .== c.L) .& (tab.ordering .== String(ord)), :]
        scatterlines!(ax, sub.N_KK, max.(sub[!, col_], 1e-16), color = colr, marker = mk, label = @sprintf("%s, %g km", ord, c.L))
    end
    k == 1 && axislegend(ax, position = :lb, framevisible = false, labelsize = 10, nbanks = 2)
end
save("ours/kk_truncation.png", fig)
show(stdout, tab, allrows = true); println()
