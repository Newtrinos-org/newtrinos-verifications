# Newtrinos.jl Earth crossing (modules `osc`, `earth_layers`) compared with NuFast-Earth (P. B. Denton, S. J. Parke,
# Phys. Rev. D 113, 053002 (2026); github.com/PeterDenton/NuFast-Earth), for atmospheric and solar neutrinos.
# The NuFast-Earth reference values are produced by data/make_reference.sh (after data/make_fine_shells.jl).
# Run from this directory with the Newtrinos environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf, StructArrays, ArraysOfArrays, Statistics

cd(@__DIR__); mkpath("ours"); mkpath("results")
const EL = Newtrinos.earth_layers
# NuFIT 6.0 normal ordering, as in the paper (Table I)
p = (θ₁₂ = asin(sqrt(0.307)), θ₁₃ = asin(sqrt(0.02195)), θ₂₃ = asin(sqrt(0.561)), δCP = deg2rad(177.0), Δm²₂₁ = 7.49e-5, Δm²₃₁ = 2.534e-3)
osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(interaction = Newtrinos.osc.SI()))
const CH = [("Pee", 1, 1), ("Pem", 1, 2), ("Pet", 1, 3), ("Pme", 2, 1), ("Pmm", 2, 2), ("Pmt", 2, 3), ("Pte", 3, 1), ("Ptm", 3, 2), ("Ptt", 3, 3)]

# NuFast atmospheric output → P[anti + 1, E, cz, α, β] on its (E, cz) grid
function read_atm(file)
    df = CSV.read(file, DataFrame)
    E, cz = sort(unique(df.E_GeV)), sort(unique(df.cosz))
    P = zeros(2, length(E), length(cz), 3, 3)
    for r in eachrow(df), (n, a, b) in CH
        P[r.anti + 1, searchsortedfirst(E, r.E_GeV), searchsortedfirst(cz, r.cosz), a, b] = r[n]
    end
    E, cz, P
end
newtrinos_atm(E, paths, layers) = cat((reshape(osc.osc_prob(E, paths, layers, p; anti), 1, length(E), length(paths), 3, 3) for anti in (false, true))...; dims = 1)
rows = NamedTuple[]

# ---------------- (1) the paper's oscillogram on NuFast's Earth model PREM_NDiscontinuityLayer(2,10,10,5) ----------------
# the same constant-density shells as Newtrinos layers (outermost first), with the 10 km production height as a
# zero-density atmosphere and the detector 2 km deep
ndl = CSV.read("data/shells_ndl_2_10_10_5.csv", DataFrame)
ρ = vcat(0.0, reverse(ndl.rhoYe))
ndl_layers = StructArray{Newtrinos.Layer}((vcat(6381.0, reverse(ndl.r_outer_km)), ρ, ρ))
E1, cz1, R1 = read_atm("data/ref_oscillogram_ndl.csv")
P1 = newtrinos_atm(E1, EL.compute_paths(cz1, ndl_layers; r_detector = 6369.0), ndl_layers)
for (k, anti) in enumerate(("ν", "ν̄")), (n, a, b) in CH
    push!(rows, (case = "oscillogram, PREM_NDiscontinuityLayer(2,10,10,5)", channel = "$n ($anti)",
                 max_abs_diff = maximum(abs.(P1[k, :, :, a, b] .- R1[k, :, :, a, b]))))
end

fig = Figure(size = (1400, 900))
for (col, (lab, k, a, b, cmax)) in enumerate((("P(νμ → νμ)", 1, 2, 2, 1.0), ("P(ν̄μ → ν̄μ)", 2, 2, 2, 1.0),
                                               ("P(νμ → νe)", 1, 2, 1, 0.6), ("P(ν̄μ → ν̄e)", 2, 2, 1, 0.12)))
    ax = Axis(fig[1, 2col - 1], xlabel = "E_ν (GeV)", ylabel = col == 1 ? "cos θ_z" : "", title = "Newtrinos: $lab, NO")
    hm = heatmap!(ax, E1, cz1, P1[k, :, :, a, b], colormap = :Oranges, colorrange = (0, cmax))
    Colorbar(fig[1, 2col], hm)
    ax2 = Axis(fig[2, 2col - 1], xlabel = "E_ν (GeV)", ylabel = col == 1 ? "cos θ_z" : "", title = "|Newtrinos − NuFast-Earth|")
    hm2 = heatmap!(ax2, E1, cz1, log10.(max.(abs.(P1[k, :, :, a, b] .- R1[k, :, :, a, b]), 1e-12)), colormap = :magma, colorrange = (-12, -5))
    Colorbar(fig[2, 2col], hm2, label = "log₁₀ |ΔP|")
end
save("ours/fig_oscillogram_NO.png", fig)

# ---------------- (2) Earth discretisation: Newtrinos Earth options vs 1 km shells of the same PREM table ----------------
disc = NamedTuple[]
fig = Figure(size = (1400, 760))
for (row, (cname, cfg)) in enumerate((("default PREM() (4 zones)", EL.PREM()), ("PREM_discontinuities() (14 zones)", EL.PREM_discontinuities())))
    E2, cz2, R2 = read_atm("data/ref_fine_$(row == 1 ? "default" : "discontinuities").csv")
    e = EL.configure(cfg)
    L = e.compute_layers()
    variants = (("zones", (e.compute_paths(cz2, L), L)),
                ("zones, chord-averaged", e.compute_chord_paths(cz2, L)),
                ("chord-averaged, ≤ 200 km pieces", e.compute_chord_paths(cz2, L; max_length = 200.0)))
    for (col, (vname, (paths, layers))) in enumerate(variants)
        P2 = newtrinos_atm(E2, paths, layers)
        d(a, b) = abs.(P2[:, :, :, a, b] .- R2[:, :, :, a, b])
        push!(disc, (earth = cname, variant = vname, n_layers = length(layers),
                     max_dPmumu = maximum(d(2, 2)), rms_dPmumu = sqrt(mean(abs2, d(2, 2))),
                     max_dPmue = maximum(d(2, 1)), max_dPee = maximum(d(1, 1))))
        ax = Axis(fig[row, 2col - 1], xscale = log10, xlabel = "E_ν (GeV)", ylabel = col == 1 ? "cos θ_z" : "",
                  title = "$cname\n$vname", titlesize = 13)
        hm = heatmap!(ax, E2, cz2, log10.(max.(d(2, 2)[1, :, :], 1e-6)), colormap = :magma, colorrange = (-5, -0.5))
        Colorbar(fig[row, 2col], hm, label = "log₁₀ |ΔP(νμ → νμ)|")
    end
end
save("ours/fig_earth_discretisation.png", fig)
CSV.write("results/earth_discretisation.csv", DataFrame(disc))

# effect on binned predictions (expected events per analysis bin) at the parameter point above
flat(x) = x isa NamedTuple ? reduce(vcat, [flat(v) for v in values(x)]) : vec(collect(x))
asimov(μ, ref) = sum((μ .- ref) .^ 2 ./ max.(ref, 1e-9))
point(exp) = merge(Newtrinos.get_params(exp), p)
impact = NamedTuple[]
for (name, mod) in (("IceCube DeepCore 8 y", Newtrinos.deepcore_8y), ("KM3NeT/ORCA6", Newtrinos.orca))
    exp = mod.configure()
    cz = exp.assets.binning.cz_fine
    e, ed = exp.physics.earth_layers, EL.configure(EL.PREM_discontinuities())
    L, Ld = e.compute_layers(), ed.compute_layers()
    μ(paths, layers; q = point(exp)) = flat(mod.get_expected(q, exp.physics, merge(exp.assets, (layers = layers, paths = paths))))
    ref = μ(ed.compute_chord_paths(cz, Ld; max_length = 200.0)...)
    io = μ(ed.compute_chord_paths(cz, Ld; max_length = 200.0)...; q = merge(point(exp), (Δm²₃₁ = -p.Δm²₃₁ + p.Δm²₂₁,)))
    for (vname, (paths, layers)) in (("default 4 zones", (e.compute_paths(cz, L), L)), ("PREM_discontinuities zones", (ed.compute_paths(cz, Ld), Ld)))
        m = μ(paths, layers)
        push!(impact, (experiment = name, earth = vname, reference = "PREM_discontinuities, chord-averaged ≤ 200 km",
                       asimov_chi2 = asimov(m, ref), max_rel_bin_shift = maximum(abs.(m .- ref) ./ max.(ref, 1e-9)),
                       asimov_chi2_NO_vs_IO = asimov(io, ref)))
    end
end
# Super-K computes its paths inside the forward model (with the floated electron density), so only zonings are compared
let mods = Dict(k => Newtrinos.super_k.configure(merge(Newtrinos.super_k.default_physics(),
                     (earth_layers = EL.configure(EL.VariableDensity(prem = prem)),))) for (k, prem) in (:d => EL.PREM(), :z => EL.PREM_discontinuities()))
    μ(exp; q = point(exp)) = flat(Newtrinos.super_k.get_expected(q, exp.physics, exp.assets))
    ref = μ(mods[:z])
    io = μ(mods[:z]; q = merge(point(mods[:z]), (Δm²₃₁ = -p.Δm²₃₁ + p.Δm²₂₁,)))
    m = μ(mods[:d])
    push!(impact, (experiment = "Super-K I–V atmospheric", earth = "default 4 zones", reference = "PREM_discontinuities zones",
                   asimov_chi2 = asimov(m, ref), max_rel_bin_shift = maximum(abs.(m .- ref) ./ max.(ref, 1e-9)), asimov_chi2_NO_vs_IO = asimov(io, ref)))
end
CSV.write("results/earth_model_impact.csv", DataFrame(impact))

# ---------------- (3) solar day and night (the paper's Fig. 6) on PREM_NDiscontinuityLayer(2,10,10,5) ----------------
# single production point ρYₑ = 100·2/3 g/cm³ (no solar profile attached, i.e. adiabatic, as in NuFast-Earth)
sol = CSV.read("data/ref_solar_ndl.csv", DataFrame)
w = CSV.read("data/ref_solar_weights.csv", DataFrame)
day = filter(r -> r.cosz == "day", sol)
night = filter(r -> r.cosz != "day", sol)
night.cz = parse.(Float64, night.cosz)
E3 = sort(unique(day.E_GeV)); czn = w.cosz
Rn = zeros(length(E3), length(czn))
for r in eachrow(night)
    Rn[searchsortedfirst(E3, r.E_GeV), searchsortedfirst(czn, r.cz)] = r.Pee
end
production = (ne = [100 * 2 / 3], nn = [100 / 3], w = [1.0])
Pday = osc.solar_prob(E3, production, p)[:, 1]
Pn = osc.solar_prob(E3, production, EL.compute_paths(czn, ndl_layers; r_detector = 6369.0), ndl_layers, p)[:, :, 1]
push!(rows, (case = "solar day", channel = "Pee", max_abs_diff = maximum(abs.(Pday .- day.Pee))))
push!(rows, (case = "solar night, per cos θz", channel = "Pee", max_abs_diff = maximum(abs.(Pn .- Rn))))
wd = w[!, "w_44.356"] ./ sum(w[!, "w_44.356"])
night_nf, night_nt = Rn * wd, Pn * wd
push!(rows, (case = "solar night, DUNE annual average (NuFast weights)", channel = "Pee", max_abs_diff = maximum(abs.(night_nt .- night_nf))))
# Newtrinos' own exposure (solar_flux.nadir_exposure: ephemeris, 1/d² flux weighting) on the same cos θz grid
edges = vcat(-1.0, (czn[1:end-1] .+ czn[2:end]) ./ 2, 0.0, 1.0)
ex = Newtrinos.solar_flux.nadir_exposure(44.356; cz_edges = edges)
wn = ex.w[1:end-1] ./ sum(ex.w[1:end-1])
night_ex = Pn * wn
push!(rows, (case = "solar night, DUNE: Newtrinos exposure − NuFast weights", channel = "Pee", max_abs_diff = maximum(abs.(night_ex .- night_nf))))

fig = Figure(size = (760, 720))
ax = Axis(fig[1, 1], ylabel = "P_ee", xticklabelsvisible = false, limits = (5, 20, 0.30, 0.45))
lines!(ax, E3 .* 1e3, Pday, color = :dodgerblue, linewidth = 2.5, label = "day (Newtrinos)")
lines!(ax, E3 .* 1e3, night_nt, color = :darkorange, linewidth = 2.5, label = "night DUNE (Newtrinos)")
lines!(ax, E3 .* 1e3, day.Pee, color = :black, linestyle = :dash, linewidth = 1, label = "NuFast-Earth")
lines!(ax, E3 .* 1e3, night_nf, color = :black, linestyle = :dash, linewidth = 1)
axislegend(ax, position = :rt, framevisible = false)
ax2 = Axis(fig[2, 1], xlabel = "E (MeV)", ylabel = "ΔP_ee", limits = (5, 20, nothing, nothing))
lines!(ax2, E3 .* 1e3, Pday .- day.Pee, color = :dodgerblue, label = "day")
lines!(ax2, E3 .* 1e3, night_nt .- night_nf, color = :darkorange, label = "night (same exposure weights)")
lines!(ax2, E3 .* 1e3, night_ex .- night_nf, color = :purple, linestyle = :dot, label = "night: Newtrinos exposure − NuFast weights")
axislegend(ax2, position = :lt, framevisible = false, labelsize = 11)
rowsize!(fig.layout, 2, Relative(0.35))
save("ours/fig_solar_day_night.png", fig)

tab = DataFrame(rows)
CSV.write("results/max_abs_differences.csv", tab)
show(stdout, tab, allrows = true); println()
show(stdout, DataFrame(disc), allrows = true); println()
show(stdout, DataFrame(impact), allrows = true); println()
