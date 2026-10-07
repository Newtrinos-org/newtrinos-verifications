# Reproduction of Borexino, "Comprehensive measurement of pp-chain solar neutrinos", Nature 562, 505 (2018), Fig. 3:
# the electron-neutrino survival probability P_ee(E) from the pp, ⁷Be, pep and ⁸B interaction rates, compared with the
# MSW-LMA prediction. Borexino's points and bands are from the Borexino open data (data/Nature2018_Fig3_*.txt).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
const SC = Newtrinos.solar_common
const NE = 3.307e31          # target electrons per 100 t
const DAY = 86400.0

bx = Newtrinos.borexino_ph2.configure()
physics, site = bx.physics, bx.assets.site          # LNGS, 42.45° N
sf = physics.solar_flux
p0 = Newtrinos.get_params((borexino_ph2 = bx,))
# oscillation parameters: NuFIT 3.2 (2018) best fit and 1σ ranges of sin²θ₁₂ and Δm²₂₁
osc(s12, dm) = merge(p0, (θ₁₂ = asin(sqrt(s12)), θ₁₃ = asin(sqrt(0.02206)), Δm²₂₁ = dm))
s12_bf, s12_lo, s12_hi = 0.307, 0.295, 0.320
dm_bf, dm_lo, dm_hi = 7.40e-5, 7.20e-5, 7.61e-5
pbf = osc(s12_bf, dm_bf)

# --- MSW-LMA band: yearly (day + night) averaged P_ee at LNGS, ⁸B production region, envelope of the 1σ corners ---
E = exp.(range(log(0.1), log(20.0), length = 120))
Pee(p) = SC.averaged_prob(physics, site, :b8, E, p)[:, 1]
curves = [Pee(osc(s, d)) for s in (s12_lo, s12_bf, s12_hi), d in (dm_lo, dm_bf, dm_hi)]
P_mid = Pee(pbf); P_min = minimum(reduce(hcat, vec(curves)), dims = 2)[:]; P_max = maximum(reduce(hcat, vec(curves)), dims = 2)[:]
bands = readdlm("data/Nature2018_Fig3_DATAbands.txt", skipstart = 2)
Eb, lmin, lmax, vmin, vmax = bands[:, 2], bands[:, 3], bands[:, 4], bands[:, 5], bands[:, 6]
bx_mid(e) = (i = clamp(searchsortedfirst(Eb, e), 2, length(Eb)); (lmin[i] + lmax[i]) / 2)
dev = maximum(abs.(P_mid .- bx_mid.(E)))
@printf("Newtrinos LMA P_ee (NuFIT 3.2 best fit) vs centre of Borexino's MSW-LMA band: max |Δ| = %.3f over 0.1–20 MeV\n", dev)

# --- "measured" P_ee from the published rates: R = N_e Φ [P ⟨σ_e⟩ + (1 − P) ⟨σ_x⟩] ---
# flux-averaged ν–e cross sections [cm²] above the recoil threshold T_min (continuum: Newtrinos spectral shape)
function σavg(comp; T_min = 0.0, T_max = Inf)
    est = SC.ESTotal(physics, comp; T_min)
    w = est.line ? est.w : est.w .* sf.spectrum(comp, est.E, p0) ./ sf.flux(comp, p0)   # normalised to the total flux
    σe, σx = sum(w .* est.σe), sum(w .* est.σx)
    isfinite(T_max) || return σe, σx
    he, hx = σavg(comp; T_min = T_max)                  # recoil window [T_min, T_max]
    σe - he, σx - hx
end
# total fluxes [cm⁻² s⁻¹]: B16 (GS98) high-metallicity SSM as used by Borexino, and Newtrinos' B23 MB22-met
Φ_HZ = (pp = 5.98e10, be7 = 4.93e9, pep = 1.44e8, b8 = 5.46e6)
Φ_HZ_unc = (pp = 0.006, be7 = 0.06, pep = 0.009, b8 = 0.12)
# measurements [cpd/100 t]: Phase II (PRD 100, 082004; pep with CNO at the HZ prediction) and ⁸B (PRD 101, 062001)
meas = [
    (name = "pp", comp = :pp, E = 0.267, R = 134.0, up = hypot(10, 6), dn = hypot(10, 10), kw = (;)),
    (name = "⁷Be", comp = :be7, E = 0.862, R = 48.3, up = hypot(1.1, 0.4), dn = hypot(1.1, 0.7), kw = (;)),
    (name = "pep", comp = :pep, E = 1.44, R = 2.43, up = hypot(0.36, 0.15), dn = hypot(0.36, 0.22), kw = (;)),
    (name = "⁸B HER-I", comp = :b8, E = 7.4, R = 0.136, up = hypot(0.013, 0.003), dn = hypot(0.013, 0.003), kw = (T_min = 3.2, T_max = 5.7)),
    (name = "⁸B HER", comp = :b8, E = 8.1, R = 0.223, up = hypot(0.015, 0.006), dn = hypot(0.016, 0.006), kw = (T_min = 3.2,)),
    (name = "⁸B HER-II", comp = :b8, E = 9.7, R = 0.087, up = hypot(0.010, 0.005), dn = hypot(0.010, 0.005), kw = (T_min = 5.7,)),
]
pts = readdlm("data/Nature2018_Fig3_DATApoints.txt")
bx_pts = Dict{Float64, Tuple{Float64, Float64}}()
for i in axes(pts, 1)
    pts[i, 1] isa Number && pts[i, 2] isa Number && (bx_pts[pts[i, 1]] = (pts[i, 2], pts[i, 3]))
end
rows = []
for m in meas
    σe, σx = σavg(m.comp; m.kw...)
    Pfrom(R, Φ) = (R / (NE * DAY * Φ) - σx) / (σe - σx)
    Φh = Φ_HZ[m.comp]; Φb = sf.flux(m.comp, p0)
    P_HZ = Pfrom(m.R, Φh)
    err = sqrt(((m.up + m.dn) / 2 / (NE * DAY * Φh * (σe - σx)))^2 + (Φ_HZ_unc[m.comp] * (P_HZ + σx / (σe - σx)))^2)
    # predicted rate at the NuFIT 3.2 best fit with Newtrinos' fluxes, P_ee averaged over the spectrum
    est = SC.ESTotal(physics, m.comp; T_min = get(m.kw, :T_min, 0.0))
    R_pred = SC.es_interaction_rate(physics, site, est, pbf) * NE * DAY
    if haskey(m.kw, :T_max)
        R_pred -= SC.es_interaction_rate(physics, site, SC.ESTotal(physics, m.comp; T_min = m.kw.T_max), pbf) * NE * DAY
    end
    push!(rows, (measurement = m.name, E_MeV = m.E, rate_cpd100t = m.R, rate_newtrinos_cpd100t = R_pred,
                 Pee_borexino = bx_pts[m.E][1], Pee_borexino_err = bx_pts[m.E][2],
                 Pee_newtrinos_B16HZ = P_HZ, Pee_newtrinos_B16HZ_err = err, Pee_newtrinos_B23 = Pfrom(m.R, Φb),
                 sigma_e_cm2 = σe, sigma_x_cm2 = σx))
end
df = DataFrame(rows)
CSV.write("results/pee_rates.csv", df)
show(stdout, df[:, 1:9]; allrows = true); println()

# predicted vs measured rates with the B16 SSM predictions of PRD 100, 082004 (Table V)
open("results/summary.txt", "w") do io
    println(io, "Interaction rates [cpd/100 t]: measured | Newtrinos (B23 MB22-met, NuFIT 3.2) | B16-HZ | B16-LZ (PRD 100, 082004, Table V)")
    ssm = Dict("pp" => (131.1, 132.2), "⁷Be" => (47.9, 43.7), "pep" => (2.74, 2.78))
    for r in eachrow(df)
        h = get(ssm, r.measurement, (NaN, NaN))
        @printf(io, "  %-10s %8.3f | %8.3f | %7.2f | %7.2f\n", r.measurement, r.rate_cpd100t, r.rate_newtrinos_cpd100t, h...)
    end
    println(io, "P_ee: Borexino | Newtrinos from the rates with B16-HZ fluxes | with B23 MB22-met fluxes")
    for r in eachrow(df)
        @printf(io, "  %-10s %.3f ± %.3f | %.3f ± %.3f | %.3f\n", r.measurement, r.Pee_borexino, r.Pee_borexino_err,
                r.Pee_newtrinos_B16HZ, r.Pee_newtrinos_B16HZ_err, r.Pee_newtrinos_B23)
    end
    @printf(io, "LMA P_ee (NuFIT 3.2 best fit, ⁸B production, LNGS) vs centre of Borexino's band: max |Δ| = %.3f\n", dev)
end
print(read("results/summary.txt", String))

# --- figure in the style of Fig. 3 ---
fig = Figure(size = (820, 520))
ax = Axis(fig[1, 1], xscale = log10, xlabel = "Neutrino energy (MeV)", ylabel = "P_ee",
          title = "Borexino survival probability: published (filled) and from the rates with Newtrinos (open)", titlesize = 13,
          xticks = ([0.2, 0.5, 1, 2, 5, 10, 20], ["0.2", "0.5", "1", "2", "5", "10", "20"]))
sel = 0.15 .<= Eb .<= 20
band!(ax, Eb[sel], vmin[sel], vmax[sel], color = (:gray, 0.25), label = "Borexino: vacuum-LMA")
band!(ax, Eb[sel], lmin[sel], lmax[sel], color = (:violet, 0.35), label = "Borexino: MSW-LMA")
band!(ax, E, P_min, P_max, color = (:purple, 0.0), strokecolor = :purple, strokewidth = 1)
lines!(ax, E, P_min, color = :purple, linestyle = :dash, label = "Newtrinos: MSW-LMA (NuFIT 3.2, 1σ)")
lines!(ax, E, P_max, color = :purple, linestyle = :dash)
colors = [:red, :blue, :deepskyblue3, :gray, :green, :gray]
for (i, r) in enumerate(eachrow(df))
    errorbars!(ax, [r.E_MeV], [r.Pee_borexino], [r.Pee_borexino_err], color = colors[i], whiskerwidth = 6)
    scatter!(ax, [r.E_MeV], [r.Pee_borexino], color = colors[i], markersize = 10)
    x = r.E_MeV * 1.06
    errorbars!(ax, [x], [r.Pee_newtrinos_B16HZ], [r.Pee_newtrinos_B16HZ_err], color = colors[i], whiskerwidth = 6, linestyle = :dot)
    scatter!(ax, [x], [r.Pee_newtrinos_B16HZ], color = :white, strokecolor = colors[i], strokewidth = 2, markersize = 10)
    text!(ax, r.E_MeV, r.Pee_borexino + r.Pee_borexino_err + 0.01, text = split(r.measurement)[1] * (occursin("HER", r.measurement) ? "" : ""),
          align = (:center, :bottom), fontsize = 12, color = colors[i], visible = !occursin("HER-", r.measurement))
end
xlims!(ax, 0.15, 20); ylims!(ax, 0.2, 0.8)
axislegend(ax, position = :lb, labelsize = 11)
save("ours/fig3_pee.png", fig)
