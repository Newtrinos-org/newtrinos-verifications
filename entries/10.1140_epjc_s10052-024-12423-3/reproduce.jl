# Newtrinos.jl solar oscillations (module `osc`: solar_mass_fractions / solar_prob, Earth regeneration; module
# `solar_flux`: B23 production regions and nadir_exposure) compared with PEANUTS (T. E. Gonzalo, M. Lucente,
# Eur. Phys. J. C 84 (2024) 119; github.com/michelelucente/PEANUTS). The PEANUTS reference values are produced by
# data/peanuts_reference.py (see its header). Run from this directory with the Newtrinos environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results")
const EL = Newtrinos.earth_layers
osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI()))
rows = NamedTuple[]
# production region from a tabulated radial distribution, with the trapezoidal weights PEANUTS integrates with
function production(r, ne, frac)
    dr = diff(r)
    w = frac .* ([dr; 0.0] .+ [0.0; dr]) ./ 2
    (ne = ne, nn = ne ./ 2, w = w)       # the neutron density cancels for three active flavours
end
param(th12, th13, th23, dcp, d21, d31) = (θ₁₂ = th12, θ₁₃ = th13, θ₂₃ = th23, δCP = dcp, Δm²₂₁ = d21, Δm²₃₁ = d31)

# ---------------- (1) the paper's SNO validation (Fig. 8): ⁸B and hep at the solar surface, BS05(AGS,OP) ----------------
sp = CSV.read("data/sno_parameters.csv", DataFrame)[1, :]
psno = param(sp.th12, sp.th13, sp.th23, sp.dcp, sp.d21, sp.d31)
bs = CSV.read("data/bs05_model.csv", DataFrame)
ref1 = CSV.read("data/sno_validation.csv", DataFrame)
fig = Figure(size = (1300, 620))
for (col, (comp, c, title)) in enumerate((("8B", :b8, "⁸B neutrinos"), ("hep", :hep, "hep neutrinos")))
    g = ref1[(ref1.component .== comp), :]
    E = sort(unique(g.E_MeV))
    P = osc.solar_prob(E .* 1e-3, production(bs.r, bs.ne, bs[!, c]), psno)
    ax = Axis(fig[1, col], title = title, ylabel = "P(νe → να)", xticklabelsvisible = false, limits = (0, 20, 0.15, 0.6))
    for (k, lab, colr) in ((1, "νe", :dodgerblue), (2, "νμ", :darkorange), (3, "ντ", :forestgreen))
        lines!(ax, E, P[:, k], color = colr, linewidth = 2.5, label = lab)
    end
    for consts in ("published", "matched")
        gc = g[g.constants .== consts, :]
        lines!(ax, gc.E_MeV, gc.Pe, color = :black, linestyle = consts == "matched" ? :dash : :dot, linewidth = 1,
               label = "PEANUTS νe ($consts constants)")
        push!(rows, (case = "SNO validation (BS05), $comp, PEANUTS constants $consts", quantity = "P(νe → νe)",
                     max_abs_diff = maximum(abs.(P[:, 1] .- gc.Pe))))
    end
    col == 1 && axislegend(ax, position = :rt, framevisible = false, labelsize = 11)
    ax2 = Axis(fig[2, col], xlabel = "Energy (MeV)", ylabel = "Newtrinos − PEANUTS", limits = (0, 20, nothing, nothing))
    for (consts, ls) in (("published", :dot), ("matched", :solid))
        gc = g[g.constants .== consts, :]
        lines!(ax2, gc.E_MeV, P[:, 1] .- gc.Pe, color = :dodgerblue, linestyle = ls, label = "P_ee, $consts")
    end
    col == 1 && axislegend(ax2, position = :rb, framevisible = false, labelsize = 11)
    rowsize!(fig.layout, 2, Relative(0.3))
end
save("ours/fig_sno_8B_hep.png", fig)

# ---------------- (2) yearly nadir exposure (Fig. 5) ----------------
ex = CSV.read("data/nadir_exposure.csv", DataFrame)
fig = Figure(size = (800, 520))
ax = Axis(fig[1, 1], xlabel = "nadir angle η / π", ylabel = "weight (∫ W dη = 1)", title = "Annual nadir exposure",
          limits = (0, 1, -0.2, 5))
for (lat, colr) in ((0.0, :dodgerblue), (45.0, :darkorange), (89.0, :forestgreen))
    g = ex[ex.lat_deg .== lat, :]
    lines!(ax, g.eta ./ π, g.W, color = colr, linewidth = 2.5, label = "λ = $(Int(lat))° (PEANUTS)")
    η_edges = range(0, π, length = 401)
    e = Newtrinos.solar_flux.nadir_exposure(lat; cz_edges = -cos.(η_edges))   # cos θz of the Sun = −cos η
    W = e.w ./ diff(η_edges)
    ηc = (η_edges[1:end-1] .+ η_edges[2:end]) ./ 2
    lines!(ax, ηc ./ π, W, color = :black, linestyle = :dash, linewidth = 1, label = lat == 0 ? "Newtrinos (flux-weighted)" : nothing)
    # compare the night fraction and the mean nadir angle
    Wp = g.W ./ sum(g.W)
    push!(rows, (case = "exposure, latitude $(Int(lat))°: night fraction", quantity = "PEANUTS − Newtrinos",
                 max_abs_diff = sum(Wp[g.eta .< π / 2]) - sum(e.w[ηc .< π / 2])))
end
axislegend(ax, position = :rt, framevisible = false)
save("ours/fig_nadir_exposure.png", fig)

# ---------------- (3) B23 solar model, all components, daytime ----------------
pj = CSV.read("data/ref_parameters.csv", DataFrame)[1, :]
p = param(asin(sqrt(pj.s12)), asin(sqrt(pj.s13)), asin(sqrt(pj.s23)), pj.dcp, pj.d21, pj.d31)
b23 = CSV.read(joinpath(pkgdir(Newtrinos), "src", "physics", "solar", "b23_MB22m_structure.csv"), DataFrame; comment = "#")
sf = Newtrinos.solar_flux.configure()
ref3 = CSV.read("data/day_pee.csv", DataFrame)
fig = Figure(size = (900, 640))
ax = Axis(fig[1, 1], xscale = log10, xlabel = "E (MeV)", ylabel = "P_ee(Newtrinos) − P_ee(PEANUTS)",
          title = "B23 (MB22m), daytime, all components", limits = (0.1, 18.8, nothing, nothing))
# spectral endpoints [MeV] (the reference grid runs to 18.8 MeV for every component)
endpoint = Dict("pp" => 0.42, "pep" => 1.45, "hep" => 18.8, "be7" => 0.87, "b8" => 16.5, "n13" => 1.2, "o15" => 1.73, "f17" => 1.74)
for (k, comp) in enumerate(("pp", "pep", "hep", "be7", "b8", "n13", "o15", "f17"))
    for consts in ("published", "matched")
        g = ref3[(ref3.component .== comp) .& (ref3.constants .== consts) .& (ref3.E_MeV .<= endpoint[comp]), :]
        Pt = osc.solar_prob(g.E_MeV .* 1e-3, production(b23.r, b23.ne, b23[!, comp]), p)[:, 1]   # same quadrature
        Pn = osc.solar_prob(g.E_MeV .* 1e-3, sf.production[Symbol(comp)], p)[:, 1]                # Newtrinos default
        push!(rows, (case = "B23 day, $comp, constants $consts, full radial table", quantity = "P(νe → νe)", max_abs_diff = maximum(abs.(Pt .- g.Pe))))
        push!(rows, (case = "B23 day, $comp, constants $consts, Newtrinos default (40 production points)", quantity = "P(νe → νe)", max_abs_diff = maximum(abs.(Pn .- g.Pe))))
        if length(g.E_MeV) > 2
            lines!(ax, g.E_MeV, Pn .- g.Pe, color = Cycled(k), linestyle = consts == "matched" ? :solid : :dot,
                   label = consts == "matched" ? comp : nothing)
        else
            scatter!(ax, g.E_MeV, Pn .- g.Pe, color = Cycled(k), marker = consts == "matched" ? :circle : :xcross,
                     label = consts == "matched" ? comp : nothing)
        end
    end
end
axislegend(ax, position = :lt, framevisible = false, nbanks = 2, labelsize = 11)
text!(ax, 0.98, 0.04, space = :relative, align = (:right, :bottom), fontsize = 11,
      text = "solid: matter potential matched; dotted: PEANUTS as published (V/k with 1/(2ħc) rounded to 2.533)")
save("ours/fig_b23_day.png", fig)

# ---------------- (4) Earth regeneration, Super-K depth (1 km) ----------------
ref4 = CSV.read("data/earth_mass_to_e.csv", DataFrame)
U, h = Newtrinos.osc._solar_matrices(osc.cfg, p)
e = EL.configure(EL.PREM_discontinuities(continental = true))
L = e.compute_layers()
rdet = 6371.0 - 1.0
fig = Figure(size = (1300, 520))
for (col, E0) in enumerate((5.0, 10.0, 16.0))
    ax = Axis(fig[1, col], xlabel = "cos θz of the Sun", ylabel = col == 1 ? "P(ν₂ → νe) − |U_e2|²" : "", title = "E = $(E0) MeV",
              limits = (-1, 0, nothing, nothing))
    for (model, mode, consts, lab, style) in (("lisi-montanino", "analytical", "matched", "PEANUTS, Lisi–Montanino Earth", (:black, :dash)),
                                              ("newtrinos-zones", "analytical", "matched", "PEANUTS, Newtrinos zones", (:darkorange, :solid)))
        g = sort(ref4[(ref4.model .== model) .& (ref4.mode .== mode) .& (ref4.constants .== consts) .& (ref4.E_MeV .== E0), :], :cz)
        lines!(ax, g.cz, g.P2e .- abs2(U[1, 2]), color = style[1], linestyle = style[2], linewidth = model == "newtrinos-zones" ? 3 : 1.5, label = lab)
    end
    g = sort(ref4[(ref4.model .== "newtrinos-zones") .& (ref4.constants .== "matched") .& (ref4.mode .== "analytical") .& (ref4.E_MeV .== E0), :], :cz)
    Rz = Newtrinos.osc.earth_mass_to_flavour(U, h, [E0 * 1e-3], e.compute_paths(g.cz, L; r_detector = rdet), L, osc.cfg.interaction)[1, :, 2, 1]
    paths, chord = e.compute_chord_paths(g.cz, L; r_detector = rdet)
    Rc = Newtrinos.osc.earth_mass_to_flavour(U, h, [E0 * 1e-3], paths, chord, osc.cfg.interaction)[1, :, 2, 1]
    lines!(ax, g.cz, Rz .- abs2(U[1, 2]), color = :dodgerblue, linewidth = 1.2, label = "Newtrinos, zones")
    lines!(ax, g.cz, Rc .- abs2(U[1, 2]), color = :purple, linewidth = 1.5, label = "Newtrinos, chord-averaged (solar default)")
    col == 1 && axislegend(ax, position = :lt, framevisible = false, labelsize = 10)
end
for E0 in sort(unique(ref4.E_MeV))
    g = sort(ref4[(ref4.model .== "newtrinos-zones") .& (ref4.constants .== "matched") .& (ref4.mode .== "analytical") .& (ref4.E_MeV .== E0), :], :cz)
    R = Newtrinos.osc.earth_mass_to_flavour(U, h, [E0 * 1e-3], e.compute_paths(g.cz, L; r_detector = rdet), L, osc.cfg.interaction)[1, :, :, 1]
    push!(rows, (case = "Earth, Newtrinos zones, E = $E0 MeV (PEANUTS analytical)", quantity = "P(νi → νe)",
                 max_abs_diff = maximum(abs.(R .- Matrix(g[:, [:P1e, :P2e, :P3e]])))))
    gl = sort(ref4[(ref4.model .== "lisi-montanino") .& (ref4.constants .== "matched") .& (ref4.mode .== "analytical") .& (ref4.E_MeV .== E0), :], :cz)
    paths, chord = e.compute_chord_paths(gl.cz, L; r_detector = rdet)
    Rc = Newtrinos.osc.earth_mass_to_flavour(U, h, [E0 * 1e-3], paths, chord, osc.cfg.interaction)[1, :, :, 1]
    push!(rows, (case = "Earth, Newtrinos solar default vs Lisi–Montanino (PEANUTS analytical), E = $E0 MeV", quantity = "P(νi → νe)",
                 max_abs_diff = maximum(abs.(Rc .- Matrix(gl[:, [:P1e, :P2e, :P3e]])))))
end
# PEANUTS' own analytical vs numerical evolutor at the spot points
num = ref4[ref4.mode .== "numerical", :]
for r in eachrow(num)
    a = ref4[(ref4.mode .== "analytical") .& (ref4.model .== r.model) .& (ref4.constants .== r.constants) .& (ref4.E_MeV .== r.E_MeV) .& (abs.(ref4.eta .- r.eta) .< 1e-9), :]
    nrow(a) == 1 && push!(rows, (case = "PEANUTS analytical − numerical, $(r.model), E = $(r.E_MeV) MeV, η = $(round(r.eta, digits = 3))",
                                 quantity = "P(ν₂ → νe)", max_abs_diff = a.P2e[1] - r.P2e))
end
save("ours/fig_earth_regeneration.png", fig)

# ---------------- (5) ⁸B day and night at Kamioka ----------------
ref5 = CSV.read("data/exposure_night.csv", DataFrame)
g = ref5[ref5.constants .== "matched", :]
edges = range(-1, 1, length = 2001)   # 1000 night bins: regeneration oscillates fast in cos θz
exk = Newtrinos.solar_flux.nadir_exposure(36.43; cz_edges = edges)
night = exk.cz .< 0
wn = exk.w[night] ./ sum(exk.w[night])
paths, chord = e.compute_chord_paths(exk.cz[night], L; r_detector = rdet)
Pd = osc.solar_prob(g.E_MeV .* 1e-3, sf.production.b8, p)[:, 1]
Pn = osc.solar_prob(g.E_MeV .* 1e-3, sf.production.b8, paths, chord, p)[:, :, 1] * wn
adn(d, n) = 2 .* (n .- d) ./ (n .+ d)
push!(rows, (case = "⁸B Kamioka day (B23)", quantity = "P_ee", max_abs_diff = maximum(abs.(Pd .- g.Pee_day))))
push!(rows, (case = "⁸B Kamioka night: Newtrinos (PREM, continental crust) − PEANUTS (Lisi–Montanino)", quantity = "P_ee",
             max_abs_diff = maximum(abs.(Pn .- g.Pee_night))))
push!(rows, (case = "⁸B Kamioka A_DN: Newtrinos − PEANUTS", quantity = "A_DN", max_abs_diff = maximum(abs.(adn(Pd, Pn) .- adn(g.Pee_day, g.Pee_night)))))
fig = Figure(size = (760, 520))
ax = Axis(fig[1, 1], xlabel = "E_ν (MeV)", ylabel = "A_DN = 2(N − D)/(N + D)", title = "⁸B day/night asymmetry at Kamioka (36.43° N, 1 km deep)")
lines!(ax, g.E_MeV, 100 .* adn(Pd, Pn), color = :dodgerblue, linewidth = 2.5, label = "Newtrinos (PREM zones, continental crust)")
lines!(ax, g.E_MeV, 100 .* adn(g.Pee_day, g.Pee_night), color = :black, linestyle = :dash, label = "PEANUTS (Lisi–Montanino Earth)")
ax.ytickformat = v -> [@sprintf("%.1f %%", x) for x in v]
axislegend(ax, position = :lt, framevisible = false)
save("ours/fig_daynight_kamioka.png", fig)

tab = DataFrame(rows)
CSV.write("results/max_abs_differences.csv", tab)
show(stdout, tab, allrows = true, truncate = 120); println()
