# Verification of the Newtrinos.jl solar zenith-angle exposure (`solar_flux.nadir_exposure`) against the exposure
# functions of Bahcall & Krastev, Phys. Rev. C 56, 2839 (1997), Figs. 5 and 6, tabulated by Bahcall in
# data/exposurefunctions (fraction of the year in 0.5° bins of the solar zenith angle, weighted with 1/d²).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
tab = readdlm("data/exposurefunctions", skipstart = 3)
α = tab[:, 1]                                   # bin centres [deg]
edges = cosd.(reverse(0:0.5:180))               # cos(zenith) bin edges, ascending
sites = (("Kamioka (Super-Kamiokande)", 36.4), ("Sudbury (SNO)", 46.3), ("Gran Sasso", 42.4))
df = DataFrame(zenith_deg = α)
for (k, (name, lat)) in enumerate(sites)
    ours = reverse(Newtrinos.solar_flux.nadir_exposure(lat; cz_edges = edges, n_days = 2000, n_hours = 2880).w)
    ref = tab[:, k + 1] ./ sum(tab[:, k + 1])
    col = split(name)[1]
    df[!, "$(col)_newtrinos"] = ours; df[!, "$(col)_bahcall"] = ref
    @printf("%-28s max |Δ| = %.1e (peak bin %.1e); night fraction %.4f vs %.4f; behind the core (α > 146.8°) %.5f vs %.5f\n",
            name, maximum(abs.(ours .- ref)), maximum(ref), sum(ours[α .> 90]), sum(ref[α .> 90]),
            sum(ours[α .> 146.8]), sum(ref[α .> 146.8]))
end
CSV.write("results/exposure_functions.csv", df)

# f(α) per degree, as plotted in the paper; residuals below
function panel!(fig, j, name)
    col = split(name)[1]
    ax = Axis(fig[1, j], ylabel = "f(α) (deg⁻¹)", title = name, titlesize = 13)
    lines!(ax, α, 2 .* df[!, "$(col)_bahcall"], color = :black, linewidth = 3, label = "Bahcall & Krastev")
    lines!(ax, α, 2 .* df[!, "$(col)_newtrinos"], color = :orange, linewidth = 1.5, linestyle = :dash, label = "Newtrinos")
    xlims!(ax, 0, 180); ylims!(ax, 0, 0.02); hidexdecorations!(ax, grid = false)
    axislegend(ax, position = :cb, labelsize = 10)
    axr = Axis(fig[2, j], xlabel = "α (deg)", ylabel = "Δf (deg⁻¹)")
    lines!(axr, α, 2 .* (df[!, "$(col)_newtrinos"] .- df[!, "$(col)_bahcall"]), color = :orange)
    xlims!(axr, 0, 180); ylims!(axr, -5e-4, 5e-4)
end
fig = Figure(size = (900, 560))
panel!(fig, 1, sites[1][1]); panel!(fig, 2, sites[2][1])
rowsize!(fig.layout, 2, Relative(0.25))
save("ours/fig5_exposure_sk_sno.png", fig)
fig = Figure(size = (480, 560))
panel!(fig, 1, sites[3][1])
rowsize!(fig.layout, 2, Relative(0.25))
save("ours/fig6_exposure_gran_sasso.png", fig)
