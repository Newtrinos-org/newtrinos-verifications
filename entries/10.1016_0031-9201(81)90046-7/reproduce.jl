# Verification of the Newtrinos.jl Earth model (module `earth_layers`) against the Preliminary Reference
# Earth Model, Dziewonski & Anderson, Phys. Earth Planet. Inter. 25, 297 (1981).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")

# PREM density from the paper's Table 1: piecewise polynomials in x = r / 6371 km  [g/cm³]
const R_E = 6371.0
const PREM_TABLE1 = [   # (r_min, r_max) km => coefficients of 1, x, x², x³
    (0.0, 1221.5) => (13.0885, 0.0, -8.8381, 0.0),         # inner core
    (1221.5, 3480.0) => (12.5815, -1.2638, -3.6426, -5.5281), # outer core
    (3480.0, 5701.0) => (7.9565, -6.4761, 5.5283, -3.0807),   # lower mantle (incl. D'')
    (5701.0, 5771.0) => (5.3197, -1.4836, 0.0, 0.0),          # transition zone
    (5771.0, 5971.0) => (11.2494, -8.0298, 0.0, 0.0),
    (5971.0, 6151.0) => (7.1089, -3.8045, 0.0, 0.0),
    (6151.0, 6346.6) => (2.6910, 0.6924, 0.0, 0.0),           # LVZ and LID
    (6346.6, 6356.0) => (2.9, 0.0, 0.0, 0.0),                 # crust
    (6356.0, 6368.0) => (2.6, 0.0, 0.0, 0.0),
    (6368.0, 6371.0) => (1.02, 0.0, 0.0, 0.0),                # ocean
]
function ρ_prem(r)
    for ((a, b), c) in PREM_TABLE1
        if a <= r <= b
            x = r / R_E
            return c[1] + c[2] * x + c[3] * x^2 + c[4] * x^3
        end
    end
    0.0
end

# Newtrinos layers: constant-density shells (outermost first) with radius, proton and neutron density
earth = Newtrinos.earth_layers.configure()
layers = earth.compute_layers()
ρ_layers = layers.p_density .+ layers.n_density
ρ_newtrinos(r) = (i = findlast(R -> R >= r, layers.radius); i === nothing ? 0.0 : ρ_layers[i])

# the tabulation shipped with Newtrinos (PREM_1s.csv) against the Table 1 polynomials
tab = CSV.read(joinpath(pkgdir(Newtrinos), "src/physics/PREM_1s.csv"), DataFrame, header = false)
r_tab, ρ_tab = tab[:, 1], tab[:, 3]
# discontinuities are listed twice in the table; evaluate the polynomial just inside each tabulated side
side(i) = i > 1 && r_tab[i - 1] == r_tab[i] ? -1e-6 : 1e-6
dev = [abs(ρ_tab[i] - ρ_prem(clamp(r_tab[i] + side(i), 0.0, R_E))) for i in eachindex(r_tab)]
@printf("max |PREM_1s.csv − Table 1| = %.4f g/cm³\n", maximum(dev))

# --- Fig. 1: radial density profile ---
r = range(0, R_E, 4000)
fig = Figure(size = (800, 520))
ax = Axis(fig[1, 1], xlabel = "radius (km)", ylabel = "density (g/cm³)", title = "Earth density: PREM vs. Newtrinos layers")
lines!(ax, r, ρ_prem.(r), color = :red, linewidth = 2, label = "PREM (Table 1 polynomials)")
scatter!(ax, r_tab, ρ_tab, color = :black, markersize = 4, label = "PREM_1s.csv (input to Newtrinos)")
lines!(ax, r, ρ_newtrinos.(r), color = :blue, linewidth = 3, label = "Newtrinos layers")
for (R, name) in ((1221.5, "ICB"), (3480.0, "CMB"))
    vlines!(ax, [R], color = :gray, linestyle = :dot)
end
xlims!(ax, 0, R_E); ylims!(ax, 0, 14)
axislegend(ax, position = :rt)
save("ours/density_profile.png", fig)

# --- Fig. 2: average density along atmospheric-neutrino paths vs. cos θ_z ---
# from a detector at r_d (the Newtrinos default, 2 km below the surface) back to where the neutrino entered
# the Earth: r(s)² = r_d² + s² + 2 s r_d cos θ_z, ⟨ρ⟩ = (1/L) ∫ ρ(r(s)) ds
const r_d = 6369.0
function path_mean(ρ, cz; n = 20000)
    L = -r_d * cz + sqrt(r_d^2 * cz^2 + R_E^2 - r_d^2)
    s = range(0, L, n)
    rs = sqrt.(max.(r_d^2 .+ s .^ 2 .+ 2 .* s .* r_d .* cz, 0.0))
    sum(ρ.(rs)) / n
end
cz = collect(range(-1.0, -0.02, 99))
mean_prem = path_mean.(ρ_prem, cz)
# Newtrinos: length-weighted density from its own path segmentation (detector at the surface)
paths = earth.compute_paths(cz, layers; r_detector = r_d)
mean_newtrinos = [sum(seg.length * ρ_layers[seg.layer_idx] for seg in p) / sum(seg.length for seg in p) for p in paths]
mean_newtrinos_earth = [(Lt = sum(seg.length for seg in p if ρ_layers[seg.layer_idx] > 0);
                         sum(seg.length * ρ_layers[seg.layer_idx] for seg in p) / Lt) for p in paths]
CSV.write("results/path_mean_density.csv", DataFrame(coszen = cz, prem = mean_prem, newtrinos = mean_newtrinos_earth))
fig = Figure(size = (800, 600))
ax1 = Axis(fig[1, 1], ylabel = "⟨ρ⟩ along the path (g/cm³)", title = "Path-averaged density through the Earth")
lines!(ax1, cz, mean_prem, color = :red, linewidth = 2, label = "PREM (Table 1)")
lines!(ax1, cz, mean_newtrinos_earth, color = :blue, linewidth = 3, linestyle = :dash, label = "Newtrinos layers")
vlines!(ax1, [-sqrt(1 - (3480 / R_E)^2), -sqrt(1 - (1221.5 / R_E)^2)], color = :gray, linestyle = :dot)
axislegend(ax1, position = :rt)
ax2 = Axis(fig[2, 1], xlabel = "cos θ_z", ylabel = "Newtrinos / PREM − 1 (%)")
lines!(ax2, cz, 100 .* (mean_newtrinos_earth ./ mean_prem .- 1), color = :blue, linewidth = 2)
hlines!(ax2, [0.0], color = :gray)
rowsize!(fig.layout, 2, Relative(0.3)); linkxaxes!(ax1, ax2); hidexdecorations!(ax1, grid = false)
save("ours/path_mean_density.png", fig)
@printf("path-averaged density: max deviation %.2f %%\n", 100 * maximum(abs.(mean_newtrinos_earth ./ mean_prem .- 1)))
