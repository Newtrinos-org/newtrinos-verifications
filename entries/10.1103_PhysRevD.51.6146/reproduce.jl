# Verification of the Newtrinos.jl ν–e elastic scattering cross section (module `solar_xsec`, `dσdT_es`) against
# Bahcall, Kamionkowski & Sirlin, "Solar neutrinos: radiative corrections in neutrino-electron scattering experiments",
# Phys. Rev. D 51, 6146 (1995), arXiv:astro-ph/9502003 (BKS).
#
# BKS give the cross section with electroweak (ρ_NC, κ(T)) and QED (f₋, f₊, f₊₋) radiative corrections in their
# appendices (Eqs. A1–A7, B3–B5). They are implemented here as a reference ("BKS formulae"), checked against the
# paper's tables, and compared with Newtrinos, which includes both since commit 8b620af. The cross section of the earlier
# commits (electroweak corrections only, with the νₑ κ(T) also for ν_μ,τ) is shown as "Newtrinos before 8b620af".
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")

const SX = Newtrinos.solar_xsec
const SF = Newtrinos.solar_flux
const xs = SX.configure()
const me = SX.M_E
const α = 1 / 137.035999
const SW2_BKS = 0.2317           # sin²θ̂_W(m_Z) used by BKS
const SW2_OLD = 0.23             # value of the earlier calculations (Bahcall 1987, 1989) BKS compare to
const GF_BKS = 1.16639e-11       # [MeV⁻²]

# ------------------------------------------------------------------------------------------------
# BKS formulae
# ------------------------------------------------------------------------------------------------
# dilogarithm Li₂(x) for 0 ≤ x ≤ 1; BKS's Spence function is L(x) = ∫₀ˣ ln|1 - t| dt/t = -Li₂(x)
function Li2(x)
    x == 0 && return 0.0
    x == 1 && return π^2 / 6
    x > 0.5 && return π^2 / 6 - log(x) * log1p(-x) - Li2(1 - x)
    s = 0.0; p = 1.0
    for k in 1:200
        p *= x; t = p / k^2; s += t
        t < 1e-18 && break
    end
    s
end
L(x) = -Li2(x)

I_T(T) = (x = sqrt(1 + 2me / T); (1 / 3 + (3 - x^2) * (x / 2 * log((x + 1) / (x - 1)) - 1)) / 6)  # Eq. (A5)

# QED functions f₋(z) (B3), (1 - z)² f₊(z) (B4) and f₊₋(z) (B5); q neutrino energy, T recoil kinetic energy
function qed(q, T)
    E = T + me; ℓ = sqrt(T * (T + 2me)); β = ℓ / E; z = T / q
    c = E / ℓ * log1p((T + ℓ) / me) - 1
    a = 1 - z - me / (E + ℓ)
    a <= 0 && return (0.0, 0.0, 0.0)
    la, l1, lz = log(a), log1p(-z), log(z)
    fm = c * (2la - l1 - lz / 2 - 5 / 12) + (L(z) - L(β)) / 2 - l1^2 / 2 - (11 / 12 + z / 2) * l1 +
         z * (lz + log(2q / me) / 2) - (31 / 18 + lz / 12) * β - 11z / 12 + z^2 / 24
    fp = c * ((1 - z)^2 * (2la - l1 - lz / 2 - 2 / 3) - (z^2 * lz + 1 - z) / 2) -
         (1 - z)^2 / 2 * (l1^2 + β * (L(1 - z) - lz * l1)) +
         l1 * (z^2 / 2 * lz + (1 - z) / 3 * (2z - 1 / 2)) -
         z^2 / 2 * L(1 - z) - z * (1 - 2z) / 3 * lz - z * (1 - z) / 6 -
         β / 12 * (lz + (1 - z) * (115 - 109z) / 6)
    fpm = c * 2la
    fm, fp, fpm
end

"""BKS Eq. (A1): dσ/dT [cm²/MeV] for flavour `:e` or `:x`; `ew` electroweak corrections (ρ_NC, κ), `qed` QED corrections."""
function dσdT_bks(fl, q, T; sw2 = SW2_BKS, ew = true, qed_on = true, GF = GF_BKS, κx_as_e = false)
    (T <= 0 || T >= SX.t_max(q)) && return 0.0
    I = I_T(T)
    ρ = ew ? 1.0126 : 1.0
    κ = ew ? (fl == :e || κx_as_e ? 0.9791 + 0.0097I : 0.9970 - 0.00037I) : 1.0
    gL = ρ * (0.5 - κ * sw2) - (fl == :e ? 1 : 0)
    gR = -ρ * κ * sw2
    z = T / q
    fm, fp, fpm = qed_on ? qed(q, T) : (0.0, 0.0, 0.0)
    2GF^2 * me / π * SX.HBARC2 * (gL^2 * (1 + α / π * fm) + gR^2 * ((1 - z)^2 + α / π * fp) - gR * gL * me / q * z * (1 + α / π * fpm))
end
dσdT_nt(fl, q, T) = xs.dσdT_es(fl, q, T)
# variants
bks(; kw...) = (fl, q, T) -> dσdT_bks(fl, q, T; kw...)
const BKS = bks()                                               # as in the paper
const TREE_OLD = bks(sw2 = SW2_OLD, ew = false, qed_on = false)  # the earlier calculations
const TREE_NEW = bks(ew = false, qed_on = false)                 # new sin²θ_W only
const BKS_NT = bks(sw2 = SX.SIN2_W, GF = SX.G_F)                 # BKS with the Newtrinos sin²θ_W and G_F
const OLD_NT = bks(sw2 = SX.SIN2_W, GF = SX.G_F, qed_on = false, κx_as_e = true)  # Newtrinos before 8b620af

# total cross section for T ≥ T_min; T = T_max (1 - w²) resolves the integrable endpoint singularity of the QED terms
function σ_tot(f, fl, q; T_min = 0.0, n = 4000)
    Tmax = SX.t_max(q); Tmax <= T_min && return 0.0
    wmax = sqrt(1 - T_min / Tmax); s = 0.0
    for i in 1:n
        w = (i - 0.5) / n * wmax
        s += f(fl, q, Tmax * (1 - w^2)) * 2Tmax * w
    end
    s * wmax / n
end

# spectrum-averaged dσ/dT; q = q_min + (q_max - q_min) u² for the singularity at q_min(T)
q_min(T) = (T + sqrt(T * (T + 2me))) / 2
function dσdT_avg(f, fl, λ, qmax, T; n = 400)
    qm = q_min(T); qm >= qmax && return 0.0
    s = 0.0
    for i in 1:n
        u = (i - 0.5) / n; q = qm + (qmax - qm) * u^2
        s += λ(q) * f(fl, q, T) * 2u * (qmax - qm)
    end
    s / n
end

# BKS average over Bahcall's ⁸B spectrum; use Bahcall et al. (1996) rather than the Newtrinos default (Winter et al. 2006)
const SHAPES = SF.read_spectra(:bahcall)
const QMAX = (pp = 0.4234, b8 = 16.56, n13 = 1.199, o15 = 1.732)   # Newtrinos spectrum endpoints
const SOURCES = [(:be7_862, "⁷Be 0.862 MeV", 0.8613), (:be7_384, "⁷Be 0.384 MeV", 0.3843),
                 (:pp, "pp", nothing), (:b8, "⁸B", nothing), (:n13, "¹³N", nothing), (:o15, "¹⁵O", nothing)]
λ_of(src) = src == :b8 ? SHAPES.b8[1] : SHAPES[src]

# normalised recoil spectrum P(T) of a source for cross section f, on the grid T
function recoil_P(f, fl, src, qline, T)
    if qline !== nothing
        d = [f(fl, qline, t) for t in T]
        return d ./ σ_tot(f, fl, qline)
    end
    λ, qmax = λ_of(src), QMAX[src]
    d = [dσdT_avg(f, fl, λ, qmax, t) for t in T]
    Tf = range(1e-5, SX.t_max(qmax) * (1 - 1e-9), length = 3000)  # normalisation
    df = [dσdT_avg(f, fl, λ, qmax, t) for t in Tf]
    d ./ (sum(df[1:end-1] .+ df[2:end]) / 2 * step(Tf))
end
# spectrum-averaged total cross section [cm²]
function σ_src(f, fl, src, qline; T_min = 0.0)
    qline !== nothing && return σ_tot(f, fl, qline; T_min)
    λ, qmax = λ_of(src), QMAX[src]
    q = range(q_min(max(T_min, 1e-6)), qmax, length = 600)
    v = [λ(e) * σ_tot(f, fl, e; T_min, n = 1500) for e in q]
    sum(v[1:end-1] .+ v[2:end]) / 2 * step(q)
end

readcsv(name) = readdlm(joinpath("data", name), ',', Float64; skipstart = 2)
const TABFILE = Dict(:be7_862 => "bks_tableI_be7_862.csv", :be7_384 => "bks_tableII_be7_384.csv", :pp => "bks_tableIII_pp.csv",
                     :b8 => "bks_tableIV_b8.csv", :n13 => "bks_tableV_n13.csv", :o15 => "bks_tableVI_o15.csv")
const TABNAME = Dict(:be7_862 => "I", :be7_384 => "II", :pp => "III", :b8 => "IV", :n13 => "V", :o15 => "VI")
const FL = ((:e, "νₑ"), (:x, "ν_μ,τ"))
const CBKS, CNT = :black, :firebrick

# ------------------------------------------------------------------------------------------------
# 1. Total cross sections: Table VII
# ------------------------------------------------------------------------------------------------
t7 = readcsv("bks_tableVII_sigma.csv")
q7 = t7[:, 1]
# the rows 0.38, 0.86 and 1.44 MeV are the ⁷Be and pep lines (Table VIII gives the same values for the ⁷Be lines)
qeval = [get(Dict(0.38 => 0.3843, 0.86 => 0.8613, 1.44 => 1.442), q, q) for q in q7]
io = open("results/sigma_tot.csv", "w")
println(io, "q_MeV,flavour,table_VII_1e-46cm2,bks_formulae_1e-46cm2,newtrinos_1e-46cm2,bks_over_table,newtrinos_over_table,newtrinos_over_bks_nt")
println("Table VII: total cross sections (T_min = 0)")
fig = Figure(size = (1100, 480))
for (k, (fl, lab)) in enumerate(FL)
    tab = t7[:, 1 + k]
    ref = [σ_tot(BKS, fl, q) for q in qeval] ./ 1e-46
    nt = [σ_tot(dσdT_nt, fl, q) for q in qeval] ./ 1e-46
    refnt = [σ_tot(BKS_NT, fl, q) for q in qeval] ./ 1e-46
    for i in eachindex(q7)
        @printf(io, "%.4f,%s,%.4g,%.5g,%.5g,%.5f,%.5f,%.5f\n", qeval[i], fl, tab[i], ref[i], nt[i], ref[i] / tab[i], nt[i] / tab[i], nt[i] / refnt[i])
    end
    rnd = [0.5 * 10.0^(floor(log10(v)) - 2) for v in tab] ./ tab   # rounding to 3 significant digits
    @printf("  %-6s BKS formulae / Table VII: %.4f – %.4f (rounding ±%.2f–%.2f %%); Newtrinos / Table VII: %.4f – %.4f\n",
            lab, extrema(ref ./ tab)..., 100 .* extrema(rnd)..., extrema(nt ./ tab)...)
    local ax = Axis(fig[1, k], xscale = log10, xlabel = "q [MeV]", ylabel = "ratio to Table VII",
              title = "$(lab)–e total cross section, T_min = 0", limits = (nothing, (0.96, 1.04)))
    band!(ax, qeval, 1 .- rnd, 1 .+ rnd, color = (:gray, 0.3), label = "Table VII rounding")
    scatterlines!(ax, qeval, ref ./ tab, color = CBKS, label = "BKS formulae (this entry)")
    scatterlines!(ax, qeval, nt ./ tab, color = CNT, marker = :diamond, label = "Newtrinos")
    hlines!(ax, [1.0], color = :gray, linestyle = :dash)
    k == 1 && axislegend(ax, position = :lb)
end
close(io)
save("ours/sigma_tot.png", fig)

# ------------------------------------------------------------------------------------------------
# 2. Fig. 6: fractional changes of the total cross section
# ------------------------------------------------------------------------------------------------
qg = vcat(range(0.2, 1, length = 17), range(1.25, 20, length = 76))
fig = Figure(size = (900, 650))
ax = Axis(fig[1, 1], xlabel = "q [MeV]", ylabel = "ξ = σ / σ_ref − 1",
          title = "Total cross section: changes w.r.t. sin²θ_W = 0.23 without radiative corrections (as BKS Fig. 6)",
          limits = (0, 20, -0.05, 0.05))
for (fl, lab) in FL
    lw = fl == :e ? 3 : 1.2
    old = [σ_tot(TREE_OLD, fl, q, n = 1500) for q in qg]
    net = [σ_tot(BKS, fl, q, n = 1500) for q in qg] ./ old .- 1
    sw = [σ_tot(TREE_NEW, fl, q, n = 1500) for q in qg] ./ old .- 1
    rc = [σ_tot(BKS, fl, q, n = 1500) / σ_tot(TREE_NEW, fl, q, n = 1500) - 1 for q in qg]
    nt = [σ_tot(dσdT_nt, fl, q, n = 1500) for q in qg] ./ old .- 1
    lines!(ax, qg, net, color = :black, linewidth = lw, label = "$lab: net (BKS)")
    lines!(ax, qg, rc, color = :black, linewidth = lw, linestyle = :dash, label = "$lab: radiative corrections (BKS)")
    lines!(ax, qg, sw, color = :black, linewidth = lw, linestyle = :dot, label = "$lab: sin²θ_W 0.23 → 0.2317")
    lines!(ax, qg, nt, color = CNT, linewidth = lw, label = "$lab: Newtrinos")
    lines!(ax, qg, [σ_tot(OLD_NT, fl, q, n = 1500) for q in qg] ./ old .- 1, color = (CNT, 0.5), linewidth = lw,
           linestyle = :dash, label = "$lab: Newtrinos before 8b620af")
    @printf("Fig. 6 (%s) at q = 10 MeV: net %+.4f, radiative %+.4f, sin²θ_W %+.4f; Newtrinos %+.4f\n", lab,
            net[findfirst(≥(10), qg)], rc[findfirst(≥(10), qg)], sw[findfirst(≥(10), qg)], nt[findfirst(≥(10), qg)])
end
axislegend(ax, position = :rc, nbanks = 2, labelsize = 11)
save("ours/fig6_sigma_tot.png", fig)

# ------------------------------------------------------------------------------------------------
# 3. Figs. 1 and 4: changes of the normalised recoil spectra (⁷Be 0.862 MeV line, ⁸B)
# ------------------------------------------------------------------------------------------------
function xi_plot(src, qline, Tmax, title, file, lims)
    T = collect(range(Tmax * 1e-3, 0.99Tmax, length = qline === nothing ? 60 : 150))
    fig = Figure(size = (900, 650))
    ax = Axis(fig[1, 1], xlabel = "T [MeV]", ylabel = "ξ(T) = P(T) / P₀(T) − 1", title = title, limits = (0, nothing, lims...))
    for (fl, lab) in FL
        lw = fl == :e ? 3 : 1.2
        P0 = recoil_P(TREE_OLD, fl, src, qline, T)
        P0n = recoil_P(TREE_NEW, fl, src, qline, T)
        lines!(ax, T, recoil_P(BKS, fl, src, qline, T) ./ P0 .- 1, color = :black, linewidth = lw, label = "$lab: net (BKS)")
        lines!(ax, T, recoil_P(BKS, fl, src, qline, T) ./ P0n .- 1, color = :black, linewidth = lw, linestyle = :dash, label = "$lab: radiative corrections (BKS)")
        qline === nothing || lines!(ax, T, P0n ./ P0 .- 1, color = :black, linewidth = lw, linestyle = :dot, label = "$lab: sin²θ_W 0.23 → 0.2317")
        lines!(ax, T, recoil_P(dσdT_nt, fl, src, qline, T) ./ P0 .- 1, color = CNT, linewidth = lw, label = "$lab: Newtrinos")
        lines!(ax, T, recoil_P(OLD_NT, fl, src, qline, T) ./ P0 .- 1, color = (CNT, 0.5), linewidth = lw, linestyle = :dash,
               label = "$lab: Newtrinos before 8b620af")
    end
    axislegend(ax, position = :lb, labelsize = 11)
    save(file, fig)
end
xi_plot(:be7_862, 0.8613, SX.t_max(0.8613), "⁷Be 0.862 MeV line: change of the normalised recoil spectrum (as BKS Fig. 1)",
        "ours/fig1_be7_recoil.png", (-0.01, 0.01))
xi_plot(:b8, nothing, 15.0, "⁸B: change of the normalised recoil spectrum (as BKS Fig. 4)", "ours/fig4_b8_recoil.png", (-0.05, 0.02))

# ------------------------------------------------------------------------------------------------
# 4. Tables I–VI: normalised recoil spectra
# ------------------------------------------------------------------------------------------------
fig = Figure(size = (1200, 950))
io = open("results/recoil_spectra.csv", "w")
println(io, "table,source,flavour,T_MeV,P_table,P_bks_formulae,P_newtrinos")
println("Tables I–VI: normalised recoil spectra, ratio to the table (T < 0.95 T_max, P > 1 % of max)")
for (i, (src, lab, qline)) in enumerate(SOURCES)
    tab = readcsv(TABFILE[src])
    local ax = Axis(fig[(i - 1) ÷ 3 + 1, (i - 1) % 3 + 1], xlabel = "T [MeV]", ylabel = "P(T) / Table $(TABNAME[src])",
              title = "$lab (Table $(TABNAME[src]))", limits = (nothing, (0.95, 1.05)))
    Tmax = qline === nothing ? SX.t_max(QMAX[src]) : SX.t_max(qline)
    for (k, (fl, flab)) in enumerate(FL)
        P = tab[:, 1 + k]
        sel = (tab[:, 1] .> 0) .& (P .> 0.01maximum(P))
        T = tab[sel, 1]
        ref = recoil_P(BKS, fl, src, qline, T)
        nt = recoil_P(dσdT_nt, fl, src, qline, T)
        for j in eachindex(T)
            @printf(io, "%s,%s,%s,%.4f,%.4f,%.5f,%.5f\n", TABNAME[src], src, fl, T[j], P[sel][j], ref[j], nt[j])
        end
        core = T .< 0.95 * maximum(tab[:, 1])
        @printf("  %-14s %-6s BKS formulae: %.4f – %.4f   Newtrinos: %.4f – %.4f\n", lab, flab,
                extrema(ref[core] ./ P[sel][core])..., extrema(nt[core] ./ P[sel][core])...)
        lines!(ax, T, ref ./ P[sel], color = CBKS, linestyle = fl == :e ? :solid : :dash, label = "BKS formulae, $flab")
        lines!(ax, T, nt ./ P[sel], color = CNT, linestyle = fl == :e ? :solid : :dash, label = "Newtrinos, $flab")
    end
    hlines!(ax, [1.0], color = :gray, linestyle = :dot)
    i == 1 && axislegend(ax, position = :lb, labelsize = 10)
end
close(io)
save("ours/recoil_tables.png", fig)

# ------------------------------------------------------------------------------------------------
# 5. Table VIII: spectrum-averaged total cross sections
# ------------------------------------------------------------------------------------------------
t8 = readdlm("data/bks_tableVIII_sources.csv", ','; skipstart = 2)
io = open("results/sources.csv", "w")
println(io, "source,flavour,table_VIII_1e-46cm2,bks_formulae_1e-46cm2,newtrinos_1e-46cm2")
println("Table VIII: spectrum-averaged cross sections [1e-46 cm²]  (table / BKS formulae / Newtrinos)")
fig = Figure(size = (1000, 450))
axs = [Axis(fig[1, k], xticks = (1:length(SOURCES), ["⁷Be 862", "⁷Be 384", "pp", "⁸B", "¹³N", "¹⁵O"]), ylabel = "ratio to Table VIII",
            title = "$(flab)–e, spectrum-averaged total cross section", limits = (nothing, (0.95, 1.05))) for (k, (_, flab)) in enumerate(FL)]
for (i, (src, lab, qline)) in enumerate(SOURCES)
    row = findfirst(==(string(src)), string.(t8[:, 1]))
    for (k, (fl, flab)) in enumerate(FL)
        tab = t8[row, 3 + k]
        ref = σ_src(BKS, fl, src, qline) / 1e-46
        nt = σ_src(dσdT_nt, fl, src, qline) / 1e-46
        @printf(io, "%s,%s,%.4g,%.5g,%.5g\n", src, fl, tab, ref, nt)
        @printf("  %-14s %-6s %7.4g  %8.4g  %8.4g\n", lab, flab, tab, ref, nt)
        rnd = 0.5 * (tab < 100 ? 0.1 : 1.0) / tab   # printed precision (one decimal below 100, integers above)
        errorbars!(axs[k], [i], [1.0], [rnd], color = :gray, whiskerwidth = 10)
        scatter!(axs[k], [i - 0.1], [ref / tab], color = CBKS, label = "BKS formulae")
        scatter!(axs[k], [i + 0.1], [nt / tab], color = CNT, marker = :diamond, label = "Newtrinos")
    end
end
hlines!.(axs, Ref([1.0]), color = :gray, linestyle = :dot)
axislegend(axs[1], unique = true, position = :lb)
save("ours/sources.png", fig)
close(io)

# ------------------------------------------------------------------------------------------------
# 6. Newtrinos vs the full BKS cross section at the Newtrinos sin²θ_W, now and before 8b620af
# ------------------------------------------------------------------------------------------------
fig = Figure(size = (1100, 480))
for (k, (fl, flab)) in enumerate(FL)
    local ax = Axis(fig[1, k], xlabel = "T / T_max", ylabel = "dσ/dT ratio", limits = (0, 1, 0.94, 1.06),
              title = "$(flab)–e: Newtrinos / BKS (both with sin²θ_W = $(SX.SIN2_W))")
    for (q, c) in zip((0.8613, 5.0, 10.0, 15.0), (:steelblue, :darkorange, :forestgreen, :purple))
        Tm = SX.t_max(q)
        x = range(0.005, 0.995, length = 300)
        lines!(ax, x, [dσdT_nt(fl, q, Tm * t) / BKS_NT(fl, q, Tm * t) for t in x], color = c, linewidth = 2.5, label = "q = $(q) MeV")
        lines!(ax, x, [OLD_NT(fl, q, Tm * t) / BKS_NT(fl, q, Tm * t) for t in x], color = c, linestyle = :dash)
    end
    hlines!(ax, [1.0], color = :gray, linestyle = :dot)
    text!(ax, 0.02, 0.945, text = "solid: Newtrinos (pinned commit); dashed: Newtrinos before 8b620af", fontsize = 11)
    k == 1 && axislegend(ax, position = :lt, labelsize = 11)
end
save("ours/newtrinos_vs_bks.png", fig)

println("max |Newtrinos / BKS − 1| (dσ/dT, 0.5–99.5 % of T_max, q = 0.2–20 MeV): ",
        maximum(abs(dσdT_nt(fl, q, SX.t_max(q) * t) / BKS_NT(fl, q, SX.t_max(q) * t) - 1)
                for fl in (:e, :x), q in range(0.2, 20, length = 50), t in range(0.005, 0.995, length = 100)))

# cross-section-level comparison of the pinned commit, BKS, and the commits before 8b620af
io = open("results/impact.csv", "w")
println(io, "quantity,flavour,newtrinos_over_bks,before_8b620af_over_bks")
println("Newtrinos / BKS and Newtrinos before 8b620af / BKS (sin²θ_W = $(SX.SIN2_W)):")
for (name, src, qline, Tmin) in (("8B, T > 3.49 MeV (SK-IV, E_tot > 4 MeV)", :b8, nothing, 3.49),
                                 ("8B, T > 4.5 MeV", :b8, nothing, 4.5),
                                 ("7Be 862 keV, all T", :be7_862, 0.8613, 0.0),
                                 ("pp, all T", :pp, nothing, 0.0),
                                 ("10 MeV, all T", :line10, 10.0, 0.0))
    for (fl, flab) in FL
        b = σ_src(BKS_NT, fl, src, qline; T_min = Tmin)
        r, r0 = σ_src(dσdT_nt, fl, src, qline; T_min = Tmin) / b, σ_src(OLD_NT, fl, src, qline; T_min = Tmin) / b
        @printf(io, "\"%s\",%s,%.5f,%.5f\n", name, fl, r, r0)
        @printf("  %-42s %-6s %.4f   before: %.4f\n", name, flab, r, r0)
    end
end
# ⁸B recoil spectrum shape (νe), relative to T = 4 MeV
T = collect(4.0:1.0:14.0)
Pref = recoil_P(BKS_NT, :e, :b8, nothing, T)
r = recoil_P(dσdT_nt, :e, :b8, nothing, T) ./ Pref; r ./= r[1]
r0 = recoil_P(OLD_NT, :e, :b8, nothing, T) ./ Pref; r0 ./= r0[1]
for i in eachindex(T)
    @printf(io, "\"8B nue recoil shape, T = %.1f MeV, relative to T = 4 MeV\",e,%.5f,%.5f\n", T[i], r[i], r0[i])
end
@printf("  ⁸B νₑ recoil-spectrum shape / BKS, T = 14 vs 4 MeV: %+.4f   before: %+.4f\n", r[end] - 1, r0[end] - 1)
close(io)
println("done")
