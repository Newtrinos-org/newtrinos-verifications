# Verification of the Newtrinos.jl solar MSW conversion (adiabatic and non-adiabatic) and Earth regeneration against the
# survival probabilities of Bahcall, Krastev & Smirnov, Phys. Rev. D 62, 093004 (2000), Fig. 1 (LMA, SMA and LOW panels),
# tabulated by Bahcall in data/Psurv_*.dat (day, day + night and night, averaged over one year and over the ⁸B production
# region; the two-flavour parameters are given at the end of each file).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
physics = Newtrinos.solar_common.default_physics()
p0 = Newtrinos.get_params((chlorine = Newtrinos.chlorine.configure(physics),))
# Sudbury (46.3° N, SNO at 2 km depth); regeneration computed at all energies; 160 night bins in cos(zenith): with
# coarse bins (the experiment modules use 20) the night probability shows binning wiggles of up to ~0.01
site = Newtrinos.solar_common.Site(physics, 46.3; depth_km = 2.0, n_night = 160, E_night_min = 0.0)

function read_bahcall(sol)
    lines = readlines("data/Psurv_$sol.dat")
    rows = [parse.(Float64, split(l)) for l in lines if occursin(r"^\s*[0-9]\.[0-9]+E[+-][0-9]+\s", l)]
    par = match(r"sin2theta\s*=\s*(\S+)\s+dm2\s*=\s*(\S+)", join(lines, " "))
    reduce(vcat, permutedims.(rows)), parse(Float64, par[1]), parse(Float64, par[2])
end

fig = Figure(size = (1350, 560))
for (k, sol) in enumerate(("LMA", "SMA", "LOW"))
    ref, s22, dm2 = read_bahcall(sol)
    E = ref[:, 1]
    p = merge(p0, (θ₁₂ = asin(sqrt(s22)) / 2, θ₁₃ = 0.0, Δm²₂₁ = dm2))
    P = Newtrinos.solar_common.survival(physics, site, :b8, E, p)
    day = P.day[:, 1]
    night = vec(sum(P.night[:, :, 1] .* site.w_night', dims = 2)) ./ sum(site.w_night)
    avg = site.w_day .* day .+ sum(site.w_night) .* night
    CSV.write("results/survival_$(lowercase(sol)).csv",
              DataFrame(E_MeV = E, day_newtrinos = day, day_bahcall = ref[:, 2], avg_newtrinos = avg, avg_bahcall = ref[:, 3],
                        night_newtrinos = night, night_bahcall = ref[:, 4]))
    above = E .>= 1.0
    @printf("%s (sin²2θ = %.4g, Δm² = %.4g eV²): max |Δ| day %.4f, night %.4f (all E); above 1 MeV day %.4f, night %.4f\n",
            sol, s22, dm2, maximum(abs.(day .- ref[:, 2])), maximum(abs.(night .- ref[:, 4])),
            maximum(abs.(day[above] .- ref[above, 2])), maximum(abs.(night[above] .- ref[above, 4])))
    ax = Axis(fig[1, k], ylabel = k == 1 ? "P(νe → νe)" : "", titlesize = 13,
              title = @sprintf("%s: Δm² = %.3g eV², sin²2θ = %.3g", sol, dm2, s22))
    for (lab, col, c) in (("day", 2, :orange), ("day + night", 3, :black), ("night", 4, :dodgerblue))
        lines!(ax, E, ref[:, col], color = c, linewidth = 3, alpha = 0.5, label = "Bahcall et al.: $lab")
    end
    lines!(ax, E, day, color = :orange, linestyle = :dash, label = "Newtrinos: day")
    lines!(ax, E, avg, color = :black, linestyle = :dash, label = "Newtrinos: day + night")
    lines!(ax, E, night, color = :dodgerblue, linestyle = :dash, label = "Newtrinos: night")
    xlims!(ax, 0, 20); ylims!(ax, 0, 1); hidexdecorations!(ax, grid = false)
    k == 1 && axislegend(ax, position = :rt, labelsize = 9)
    axr = Axis(fig[2, k], xlabel = "Eν (MeV)", ylabel = k == 1 ? "Newtrinos − Bahcall" : "")
    lines!(axr, E, day .- ref[:, 2], color = :orange); lines!(axr, E, avg .- ref[:, 3], color = :black)
    lines!(axr, E, night .- ref[:, 4], color = :dodgerblue)
    hlines!(axr, [0], color = :gray); xlims!(axr, 0, 20); ylims!(axr, -0.06, 0.06)
end
rowsize!(fig.layout, 2, Relative(0.3))
save("ours/fig_survival_msw.png", fig)
