# Verification of the Newtrinos.jl solar spectra and absorption cross sections (modules `solar_flux`, `solar_xsec`)
# against the chlorine and gallium capture rates of Bahcall & Pinsonneault, Phys. Rev. Lett. 92, 121301 (2004),
# Table 1, per neutrino source as given by Bahcall's rate code (data/exportrates.output).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, CairoMakie, CSV, DataFrames, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results")
const SF = Newtrinos.solar_flux
sf = SF.configure()
xs = Newtrinos.solar_xsec.configure()
p = merge(sf.params, xs.params)

# BP04 fluxes [cm⁻² s⁻¹] of the code input (data/exportrates.dat; Table 1 has ⁸B = 5.79e6)
bp04 = (pp = 5.938e10, pep = 1.401e8, hep = 7.88e3, be7 = 4.857e9, b8 = 5.822e6, n13 = 5.712e8, o15 = 5.031e8, f17 = 5.912e6)
p = merge(p, (solar_norms = [bp04[c] / sf.nominal[c] for c in SF.COMPONENTS],))

# Bahcall's per-source capture rates [SNU] (data/exportrates.output, "Case number 1")
lines = readlines("data/exportrates.output")
i0 = findfirst(l -> occursin("Source    Fluxes", l), lines)
key = Dict("pp" => :pp, "pep" => :pep, "7Be" => :be7, "8B" => :b8, "13N" => :n13, "15O" => :o15, "17F" => :f17, "hep" => :hep)
ref = Dict{Symbol, Tuple{Float64, Float64}}()
for l in lines[i0+1:i0+10]
    w = split(l)
    length(w) == 5 && haskey(key, w[1]) && (ref[key[w[1]]] = (parse(Float64, w[3]), parse(Float64, w[4])))
end

# capture rates without oscillations: flux × cross section, integrated above threshold
const THR = (cl37 = Newtrinos.solar_xsec.CL_THRESHOLD, ga71 = Newtrinos.solar_xsec.GA_THRESHOLD)
const ENDPOINT = (pp = 0.4234, hep = 18.79, b8 = 16.56, n13 = 1.199, o15 = 1.732, f17 = 1.740)
function rate(comp, target)
    if haskey(sf.lines, comp)
        l = sf.lines[comp]
        return sf.flux(comp, p) * sum(l.br .* [xs.capture(target, e, p) for e in l.E]) / 1e-36
    end
    THR[target] >= ENDPOINT[comp] && return 0.0
    E = range(THR[target], ENDPOINT[comp], length = 4001)
    f = sf.spectrum(comp, collect(E), p) .* [xs.capture(target, e, p) for e in E]
    step(E) * (sum(f) - (f[1] + f[end]) / 2) / 1e-36
end
comps = collect(SF.COMPONENTS)
df = DataFrame(source = string.(comps),
               cl_newtrinos = [rate(c, :cl37) for c in comps], cl_bahcall = [ref[c][1] for c in comps],
               ga_newtrinos = [rate(c, :ga71) for c in comps], ga_bahcall = [ref[c][2] for c in comps])
push!(df, ("total", sum(df.cl_newtrinos), 8.50, sum(df.ga_newtrinos), 130.54))   # totals of exportrates.output
CSV.write("results/capture_rates.csv", df)
show(stdout, df; allrows = true); println()

labels = ["pp", "pep", "hep", "⁷Be", "⁸B", "¹³N", "¹⁵O", "¹⁷F"]
fig = Figure(size = (900, 420))
for (k, (col, name, unit)) in enumerate(((:cl, "Chlorine (³⁷Cl)", "rate (SNU)"), (:ga, "Gallium (⁷¹Ga)", "rate (SNU)")))
    ours = df[1:8, Symbol(col, :_newtrinos)]; theirs = df[1:8, Symbol(col, :_bahcall)]
    tot_o, tot_b = df[9, Symbol(col, :_newtrinos)], df[9, Symbol(col, :_bahcall)]
    ax = Axis(fig[1, k], xticks = (1:8, labels), ylabel = unit, yscale = log10,
              title = @sprintf("%s: Newtrinos %.2f SNU, Bahcall %.2f SNU", name, tot_o, tot_b), titlesize = 13)
    sel = theirs .> 0
    barplot!(ax, (1:8)[sel] .- 0.18, theirs[sel], width = 0.36, fillto = 0.01, color = :gray, label = "Bahcall (BP04)")
    barplot!(ax, (1:8)[sel] .+ 0.18, ours[sel], width = 0.36, fillto = 0.01, color = :dodgerblue, label = "Newtrinos")
    ylims!(ax, 0.01, 200)
    k == 1 && axislegend(ax, position = :rt)
end
Label(fig[0, :], "Capture rates without oscillations, BP04 fluxes", fontsize = 15, font = :bold)
save("ours/capture_rates.png", fig)
