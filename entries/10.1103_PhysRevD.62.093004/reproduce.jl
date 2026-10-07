# Verification of the Newtrinos.jl solar MSW conversion and Earth regeneration against the LMA survival probability of
# Bahcall, Krastev & Smirnov, Phys. Rev. D 62, 093004 (2000), Fig. 1, tabulated by Bahcall in data/Psurv_LMA.dat
# (day, day + night and night, averaged over one year and over the ⁸B production region).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
raw = readdlm("data/Psurv_LMA.dat", skipstart = 12)
ref = Float64.(raw[[x isa Number for x in raw[:, 1]], 1:4])     # E [MeV], day, day + night, night
E = ref[:, 1]

# two-flavour LMA point of the paper (values at the end of the data file)
physics = Newtrinos.solar_common.default_physics()
p = merge(Newtrinos.get_params((chlorine = Newtrinos.chlorine.configure(physics),)),
          (θ₁₂ = asin(sqrt(0.7943)) / 2, θ₁₃ = 0.0, Δm²₂₁ = 2.712e-5))
# Sudbury (46.3° N, SNO at 2 km depth); regeneration computed at all energies;
# 160 night bins in cos(zenith): with coarse bins (the experiment modules use 20) the night probability shows
# binning wiggles of up to ~0.01 at individual energies
site = Newtrinos.solar_common.Site(physics, 46.3; depth_km = 2.0, n_night = 160, E_night_min = 0.0)
P = Newtrinos.solar_common.survival(physics, site, :b8, E, p)
day = P.day[:, 1]
night = vec(sum(P.night[:, :, 1] .* site.w_night', dims = 2)) ./ sum(site.w_night)
avg = site.w_day .* day .+ sum(site.w_night) .* night
df = DataFrame(E_MeV = E, day_newtrinos = day, day_bahcall = ref[:, 2], avg_newtrinos = avg, avg_bahcall = ref[:, 3],
               night_newtrinos = night, night_bahcall = ref[:, 4])
CSV.write("results/survival_lma.csv", df)
@printf("max |Δ|: day %.4f, day + night %.4f, night %.4f\n", maximum(abs.(day .- ref[:, 2])), maximum(abs.(avg .- ref[:, 3])),
        maximum(abs.(night .- ref[:, 4])))

fig = Figure(size = (700, 620))
ax = Axis(fig[1, 1], ylabel = "P(νe → νe)", title = "LMA: Δm² = 2.71×10⁻⁵ eV², sin²2θ = 0.794, ⁸B production, Sudbury",
          titlesize = 13)
for (lab, col, c) in (("day", 2, :orange), ("day + night", 3, :black), ("night", 4, :dodgerblue))
    lines!(ax, E, ref[:, col], color = c, linewidth = 3, alpha = 0.5, label = "Bahcall et al.: $lab")
end
lines!(ax, E, day, color = :orange, linestyle = :dash, label = "Newtrinos: day")
lines!(ax, E, avg, color = :black, linestyle = :dash, label = "Newtrinos: day + night")
lines!(ax, E, night, color = :dodgerblue, linestyle = :dash, label = "Newtrinos: night")
xlims!(ax, 0, 20); ylims!(ax, 0.2, 0.65); hidexdecorations!(ax, grid = false)
axislegend(ax, position = :rt, labelsize = 10, nbanks = 2)
axr = Axis(fig[2, 1], xlabel = "Eν (MeV)", ylabel = "Newtrinos − Bahcall")
lines!(axr, E, day .- ref[:, 2], color = :orange); lines!(axr, E, avg .- ref[:, 3], color = :black)
lines!(axr, E, night .- ref[:, 4], color = :dodgerblue)
hlines!(axr, [0], color = :gray); xlims!(axr, 0, 20); ylims!(axr, -0.015, 0.015)
rowsize!(fig.layout, 2, Relative(0.28))
save("ours/fig_survival_lma.png", fig)
