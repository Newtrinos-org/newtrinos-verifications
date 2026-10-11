# Reproduction of Daya Bay, Phys. Rev. Lett. 130, 161802 (2023) [arXiv:2211.14988] with Newtrinos.jl (module `dayabay`).
# Data: supplemental material of the paper (3158 days, prompt spectra, backgrounds, Δχ² table).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DataStructures, Accessors, FileIO, CairoMakie, CSV, DataFrames, DelimitedFiles, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const DATA = joinpath(pkgdir(Newtrinos), "src", "experiments")

# CSV export of a profile scan: one row per grid point, ASCII column names, oscillation parameters in physical
# units (sin²θ, Δm²₃₂ = Δm²₃₁ − Δm²₂₁ in eV², δCP in rad), dchi2 = -2Δlog L w.r.t. `ref` (default: the scan's best fit)
function save_csv(path, r; ref = maximum(r.values.log_posterior))
    ascii(k) = replace(string(k), "θ" => "theta", "Δm²" => "dm2_", "δ" => "d", "₁" => "1", "₂" => "2", "₃" => "3", "CP" => "cp")
    cols = OrderedDict{String,Any}()
    for (k, v) in pairs(r.values)
        x = vec(v)
        if startswith(string(k), "θ")
            cols["sin2_" * ascii(k)] = sin.(x) .^ 2
        elseif k == :Δm²₃₁
            cols["dm2_32"] = x .- vec(r.values.Δm²₂₁)
        end
        cols[ascii(k)] = x
    end
    cols["dchi2"] = 2 .* (ref .- vec(r.values.log_posterior))
    haskey(cols, "llh") && (cols["log_likelihood"] = pop!(cols, "llh"))
    first_cols = filter(in(keys(cols)), ["sin2_theta23", "dm2_32", "dcp", "dchi2", "log_likelihood", "log_posterior"])
    CSV.write(path, DataFrame(cols)[:, vcat(first_cols, setdiff(collect(keys(cols)), first_cols))])
end

# draw each confidence level with its own line style (1σ solid, 2σ dashed, 3σ dotted)
function contours!(ax, res, levels; color, label)
    for (lv, ls) in zip(levels, (:solid, :dash, :dot))
        plot!(ax, res, levels = [lv], color = color, linestyle = ls, label = lv == first(levels) ? label : nothing)
    end
end

experiments = (dayabay = Newtrinos.dayabay.configure(),)
likelihood = Newtrinos.generate_likelihood(experiments)
p = Newtrinos.get_params(experiments)
priors = Newtrinos.get_priors(experiments)
priors = merge(priors, (Δm²₃₁ = Uniform(0.0023, 0.0028), θ₁₃ = Uniform(0.13, 0.16), θ₁₂ = p.θ₁₂, θ₂₃ = p.θ₂₃, δCP = p.δCP, Δm²₂₁ = p.Δm²₂₁))

result = Newtrinos.profile(likelihood, priors, OrderedDict(:θ₁₃ => 31, :Δm²₃₁ => 31), p, cache_dir = "cache/th13dm31")
FileIO.save("results/dayabay.jld2", Dict("th13dm31" => result))
save_csv("results/dayabay_th13_dm31.csv", result)

# effective mass splitting of the paper: Δm²ee = Δm²₃₁ − sin²θ₁₂ Δm²₂₁ = Δm²₃₂ + cos²θ₁₂ Δm²₂₁
dmee_from31(x) = x - sin(p.θ₁₂)^2 * p.Δm²₂₁
dmee_from32(x) = x + cos(p.θ₁₂)^2 * p.Δm²₂₁
off = CSV.read(joinpath(DATA, "daya_bay/daya_bay_3158days/DayaBay_DeltaChiSq_NO_3158days.txt"), DataFrame,
               skipto = 10, delim = " ", header = ["ss2th13", "dm232", "dchi2"], ignorerepeated = true)
o_s = reshape(off.ss2th13, 100, 100)[1, :]; o_m = reshape(off.dm232, 100, 100)[:, 1]; o_χ² = Array(reshape(off.dchi2, 100, 100))'

# Fig. 1: sin²2θ₁₃ – Δm²ee, 1σ, 2σ, 3σ
fig = Figure(size = (700, 560))
ax = Axis(fig[1, 1], title = "Daya Bay, 3158 days: 1σ, 2σ, 3σ", xlabel = "sin²2θ₁₃", ylabel = "Δm²ee (10⁻³ eV²)")
levels = quantile.(Chisq(2), 1 .- 2 .* ccdf.(Normal(), 1:3))
for (lv, ls) in zip(levels, (:solid, :dash, :dot))
    contour!(ax, o_s, dmee_from32.(o_m) .* 1e3, o_χ², levels = [lv], color = :red, linestyle = ls, linewidth = 2)
    contour!(ax, sin.(2 .* result.axes.θ₁₃) .^ 2, dmee_from31.(result.axes.Δm²₃₁) .* 1e3, 2 .* (maximum(result.values.log_posterior) .- result.values.log_posterior),
             levels = [lv], color = :blue, linestyle = ls, linewidth = 2)
end
lines!(ax, [NaN], [NaN], color = :red, label = "Daya Bay (Δχ² table)")
lines!(ax, [NaN], [NaN], color = :blue, label = "Newtrinos (profile)")
xlims!(ax, 0.072, 0.1); ylims!(ax, 2.2, 2.82)
axislegend(ax, position = :rt)
save("ours/contour_dmee_ss2th13.png", fig)

# 1D results: sin²2θ₁₃ and Δm²ee from the 2D profile (minimum over the other parameter), Δχ² = 1 crossings
Δχ²2d = 2 .* (maximum(result.values.log_posterior) .- result.values.log_posterior)
function onesigma(x, y)
    y = y .- minimum(y); i = argmin(y)
    j = findlast(>(1), y[1:i]); lo = j === nothing ? x[1] : x[j] + (1 - y[j]) * (x[j+1] - x[j]) / (y[j+1] - y[j])
    j = findfirst(>(1), y[i:end]); hi = j === nothing ? x[end] : (j += i - 1; x[j-1] + (1 - y[j-1]) * (x[j] - x[j-1]) / (y[j] - y[j-1]))
    x[i], hi - x[i], x[i] - lo
end
s2 = onesigma(sin.(2 .* result.axes.θ₁₃) .^ 2, vec(minimum(Δχ²2d, dims = 2)))
me = onesigma(dmee_from31.(result.axes.Δm²₃₁) .* 1e3, vec(minimum(Δχ²2d, dims = 1)))
o_s2 = onesigma(o_s, vec(minimum(o_χ², dims = 2))); o_me = onesigma(dmee_from32.(o_m) .* 1e3, vec(minimum(o_χ², dims = 1)))
open("results/summary.txt", "w") do io
    println(io, "Newtrinos | official Δχ² table | paper (NO)")
    @printf(io, "sin2_2theta13       %.4f +%.4f -%.4f | %.4f +%.4f -%.4f | 0.0851 ± 0.0024\n", s2..., o_s2...)
    @printf(io, "dm2_ee [1e-3 eV^2]  %.3f +%.3f -%.3f | %.3f +%.3f -%.3f | 2.519 ± 0.060\n", me..., o_me...)
end
print(read("results/summary.txt", String))

# Fig. 2 (EH3 panel): far-hall prompt spectrum (backgrounds subtracted), best fit, no oscillation, and ratio
a = experiments.dayabay.assets; phys = experiments.dayabay.physics
ib = argmax(result.values.log_posterior)
bf = merge(p, (θ₁₃ = result.axes.θ₁₃[ib[1]], Δm²₃₁ = result.axes.Δm²₃₁[ib[2]]))
μ = Newtrinos.dayabay.get_expected(bf, phys, a)
μ0 = Newtrinos.dayabay.get_expected(merge(bf, (θ₁₃ = 0.0,)), phys, a)
eh3 = CSV.read(joinpath(DATA, "daya_bay/daya_bay_3158days/DayaBay_IBDPromptSpectrum_EH3_3158days.txt"), DataFrame,
               comment = "#", delim = " ", ignorerepeated = true, header = [:Emin, :Emax, :Ec, :o6, :o8, :o7, :p6, :p8, :p7])
μ_db = eh3.p6 .+ eh3.p8 .+ eh3.p7
w = diff(a.energy_bins); N = a.observed
CSV.write("results/eh3_spectrum.csv", DataFrame(E_lo = a.energy_bins[1:end-1], E_hi = a.energy_bins[2:end], observed_minus_bkg = N,
                                                newtrinos_best_fit = μ, newtrinos_no_osc = μ0, dayabay_best_fit = μ_db))
fig = Figure(size = (750, 650))
ax = Axis(fig[1, 1], ylabel = "Events / MeV", title = "Daya Bay EH3 (far hall), backgrounds subtracted")
stairs!(ax, a.energy_bins, vcat(μ0 ./ w, 0), step = :post, color = :gray50, linewidth = 2, label = "Newtrinos: no oscillation")
stairs!(ax, a.energy_bins, vcat(μ_db ./ w, 0), step = :post, color = :red, linestyle = :dash, linewidth = 2, label = "Daya Bay: best fit")
stairs!(ax, a.energy_bins, vcat(μ ./ w, 0), step = :post, color = :blue, linewidth = 1.5, label = "Newtrinos: best fit")
scatter!(ax, a.energy, N ./ w, color = :black, markersize = 6, label = "data")
xlims!(ax, 0.7, 12); axislegend(ax, position = :rt); hidexdecorations!(ax, grid = false)
ax2 = Axis(fig[2, 1], xlabel = "Prompt energy (MeV)", ylabel = "Ratio to no oscillation")
errorbars!(ax2, a.energy, N ./ μ0, sqrt.(N) ./ μ0, color = :black); scatter!(ax2, a.energy, N ./ μ0, color = :black, markersize = 6)
stairs!(ax2, a.energy_bins, vcat(μ ./ μ0, 1), step = :post, color = :blue, linewidth = 1.5)
stairs!(ax2, a.energy_bins, vcat(μ_db ./ μ0, 1), step = :post, color = :red, linestyle = :dash, linewidth = 1.5)
hlines!(ax2, [1.0], color = :gray50); xlims!(ax2, 0.7, 12); ylims!(ax2, 0.85, 1.05)
rowsize!(fig.layout, 2, Relative(0.35)); linkxaxes!(ax, ax2)
save("ours/fig2_eh3_spectrum.png", fig)
