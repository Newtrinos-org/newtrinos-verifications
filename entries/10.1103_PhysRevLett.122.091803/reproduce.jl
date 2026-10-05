# Reproduction of the MINOS/MINOS+ sterile-neutrino search, Phys. Rev. Lett. 122, 091803 (2019) [arXiv:1710.06488],
# with Newtrinos.jl (module `minos`, which is built on this paper's data release, and `osc` with the 3+1 `Sterile` model).
# Run from this directory with the Newtrinos.jl environment at the pinned commit:
#   julia -t 8 --project=<path to Newtrinos.jl> reproduce.jl
using Newtrinos, Distributions, DensityInterface, DataStructures, LinearAlgebra, FileIO, CairoMakie, CSV, DataFrames, HDF5, Printf

cd(@__DIR__); mkpath("ours"); mkpath("results"); mkpath("cache")
const RELEASE = joinpath(pkgdir(Newtrinos), "src/experiments/minos/minos_sterile_16e20_POT/dataRelease.h5")

osc = Newtrinos.osc.configure(Newtrinos.osc.OscillationConfig(flavour = Newtrinos.osc.Sterile()))
physics = (osc = osc, xsec = Newtrinos.xsec.configure())
experiments = (minos = Newtrinos.minos.configure(physics),)
A = experiments.minos.assets

# (a) the Newtrinos module likelihood: FD spectra conditioned on the ND (Gaussian), CC and NC
lik_module = Newtrinos.generate_likelihood(experiments)

# (b) the paper's χ² (dataRelease_chi2Calc_compile.C of the release): one Gaussian χ² over the joint [FD; ND]
#     spectrum with covariance V = μμᵀ ∘ V_rel + diag(μ), built from the same Newtrinos predictions μ
function chi2_paper(p)
    sum(("CC", "NC")) do ch
        μ = vcat(Newtrinos.minos.get_expected_per_channel(p, physics, A.ch_data["FD"*ch]),
                 Newtrinos.minos.get_expected_per_channel(p, physics, A.ch_data["ND"*ch]))
        d = vcat(A.ch_data["FD"*ch].observed, A.ch_data["ND"*ch].observed)
        V = (μ * μ') .* (ch == "CC" ? A.TotalCCCovar : A.TotalNCCovar) + Diagonal(μ)
        r = d - μ
        dot(r, V \ r)
    end
end
lik_paper = logfuncdensity(p -> -chi2_paper(p) / 2)

# official χ² surface of the release: bins in θ₂₄ (rad, 50 log bins) × Δm²₄₁ (eV², 100 log bins)
h5 = h5open(RELEASE)
θedges, medges = read(h5["dm241vsth24_bins1"]), read(h5["dm241vsth24_bins2"])
χ²official = permutedims(read(h5["dm241vsth24_hist"]))           # [θ₂₄ bin, Δm²₄₁ bin]
fcθ, fcm = read(h5["dm241vsth24_UpValuesFC90CL_bins1"]), read(h5["dm241vsth24_UpValuesFC90CL_bins2"])
fc_up = permutedims(read(h5["dm241vsth24_UpValuesFC90CL_hist"]))  # FC 90% critical Δχ² values
contour_official = [(read(h5["minosPlus90CL_$(i)_x"]), read(h5["minosPlus90CL_$(i)_y"])) for i in 0:1]
close(h5)
logc(e) = sqrt.(e[1:end-1] .* e[2:end])
θc, mc = logc(θedges), logc(medges)

# fit setup of the paper: θ₂₃, θ₃₄, Δm²₃₁ free (penalty Δm²₃₂ = 2.5 ± 0.5 × 10⁻³ eV²); θ₁₄ = 0, phases 0;
# θ₁₂, θ₁₃, Δm²₂₁ at the values of the release's χ² macro
p = (θ₁₂ = 0.5540758073, θ₁₃ = 0.149116, θ₂₃ = 0.9286, δCP = 0.0, Δm²₂₁ = 7.54e-5, Δm²₃₁ = 2.5e-3 + 7.54e-5,
     Δm²₄₁ = 1.0, θ₁₄ = 0.0, θ₂₄ = 0.05, θ₃₄ = 0.05, nc_norm = 1.0, nutau_cc_norm = 1.0)
priors = (θ₁₂ = p.θ₁₂, θ₁₃ = p.θ₁₃, θ₂₃ = Uniform(0, π / 2), δCP = 0.0, Δm²₂₁ = p.Δm²₂₁,
          Δm²₃₁ = Truncated(Normal(2.5e-3 + 7.54e-5, 5e-4), 1e-4, 6e-3), Δm²₄₁ = LogUniform(mc[1], mc[99]),
          θ₁₄ = 0.0, θ₂₄ = LogUniform(θc[1], θc[49]), θ₃₄ = Uniform(0, π / 2), nc_norm = 1.0, nutau_cc_norm = 1.0)
# grid: every second bin centre of the official surface (25 × 50 points); LogUniform quantiles land on them
grid = OrderedDict(:θ₂₄ => 25, :Δm²₄₁ => 50)

# the profile has several local minima (θ₂₃ octant, θ₃₄, degeneracies with Δm²₃₁ at small Δm²₄₁): the paper χ²
# is profiled from four starting points and the best fit is kept at each grid point
starts = [(θ₂₃ = a, θ₃₄ = b) for a in (0.93, 0.64) for b in (0.05, 0.7)]
function best_of(rs)
    k = argmax(cat((r.values.log_posterior for r in rs)..., dims = 3), dims = 3)
    vals = NamedTuple{keys(rs[1].values)}(Tuple(cat((getproperty(r.values, f) for r in rs)..., dims = 3)[k][:, :, 1]
                                                for f in keys(rs[1].values)))
    Newtrinos.NewtrinosResult(axes = rs[1].axes, values = vals, meta = rs[1].meta)
end
res = Dict(
    "paper_chi2" => best_of([Newtrinos.profile(lik_paper, priors, grid, merge(p, s), cache_dir = "cache/paper_chi2_$i")
                             for (i, s) in enumerate(starts)]),
    "module" => Newtrinos.profile(lik_module, priors, grid, p, cache_dir = "cache/module"))
FileIO.save("results/minos_sterile.jld2", Dict(k => v for (k, v) in res))

ss24 = sin.(res["paper_chi2"].axes.θ₂₄) .^ 2
dm41 = collect(res["paper_chi2"].axes.Δm²₄₁)
Δχ²(r) = 2 .* (maximum(r.values.log_posterior) .- r.values.log_posterior)
ours = Dict(k => Δχ²(v) for (k, v) in res)
iθ = [argmin(abs.(log.(θc) .- log(t))) for t in res["paper_chi2"].axes.θ₂₄]
im = [argmin(abs.(log.(mc) .- log(m))) for m in dm41]
off = χ²official[iθ, im] .- minimum(χ²official)                   # official Δχ² at our grid points
fcv = [fc_up[argmin(abs.(log.(logc(fcθ)) .- log(θc[i]))), argmin(abs.(log.(logc(fcm)) .- log(mc[j])))] for i in iθ, j in im]

CSV.write("results/minos_sterile_dchi2.csv",
          DataFrame([(sin2_theta24 = ss24[i], dm2_41 = dm41[j], dchi2_official = off[i, j],
                      dchi2_newtrinos_paper_chi2 = ours["paper_chi2"][i, j], dchi2_newtrinos_module = ours["module"][i, j],
                      fc90_critical = fcv[i, j], theta23 = res["paper_chi2"].values.θ₂₃[i, j],
                      theta34 = res["paper_chi2"].values.θ₃₄[i, j], dm2_31 = res["paper_chi2"].values.Δm²₃₁[i, j])
                     for j in eachindex(dm41) for i in eachindex(ss24)]))

function limit_axis(fig, title)
    Axis(fig[1, 1], xscale = log10, yscale = log10, title = title, xlabel = "sin²θ₂₄", ylabel = "Δm²₄₁ (eV²)",
         limits = (5e-5, 1, 1e-4, 1e3), xminorticksvisible = true, yminorticksvisible = true)
end
official_line!(ax) = for (k, (x, y)) in enumerate(contour_official)
    lines!(ax, x, y, color = :black, linewidth = 2.5, label = k == 1 ? "MINOS/MINOS+ 90% C.L. (FC, release)" : nothing)
end

limit_legend!(ax, first_label) = axislegend(ax,
    [LineElement(color = :black, linewidth = 2.5), LineElement(color = :dodgerblue, linewidth = 2),
     LineElement(color = :orangered, linewidth = 2, linestyle = :dash)],
    [first_label, "Newtrinos, paper χ²", "Newtrinos, minos module likelihood"], position = :lb, framevisible = false, labelsize = 12)

# Fig. 3: 90% C.L. exclusion with the Feldman-Cousins critical values published in the release
fig = Figure(size = (660, 660))
ax = limit_axis(fig, "MINOS/MINOS+ 3+1: 90% C.L. (FC critical values)")
official_line!(ax)
contour!(ax, ss24, dm41, ours["paper_chi2"] .- fcv, levels = [0], color = :dodgerblue, linewidth = 2,
         label = "Newtrinos, paper χ²")
contour!(ax, ss24, dm41, ours["module"] .- fcv, levels = [0], color = :orangered, linewidth = 2, linestyle = :dash,
         label = "Newtrinos, minos module likelihood")
limit_legend!(ax, "MINOS/MINOS+ 90% C.L. (FC, release)")
save("ours/fig3_limit.png", fig)

# the profile itself: official Δχ² surface vs ours at the same points, both cut at the Wilks 90% value (2 dof)
fig = Figure(size = (660, 660))
ax = limit_axis(fig, "Δχ² = 4.61 (Wilks, 2 dof): official surface vs Newtrinos")
contour!(ax, ss24, dm41, off, levels = [4.61], color = :black, linewidth = 2.5, label = "official χ² surface (release)")
contour!(ax, ss24, dm41, ours["paper_chi2"], levels = [4.61], color = :dodgerblue, linewidth = 2, label = "Newtrinos, paper χ²")
contour!(ax, ss24, dm41, ours["module"], levels = [4.61], color = :orangered, linewidth = 2, linestyle = :dash,
         label = "Newtrinos, minos module likelihood")
limit_legend!(ax, "official χ² surface (release)")
save("ours/wilks_comparison.png", fig)

# Δχ² vs sin²θ₂₄ at fixed Δm²₄₁
fig = Figure(size = (1100, 720))
for (k, m) in enumerate((2e-3, 2e-2, 0.5, 5.0, 50.0, 500.0))
    j = argmin(abs.(log.(dm41) .- log(m)))
    local ax = Axis(fig[(k - 1) ÷ 3 + 1, (k - 1) % 3 + 1], xscale = log10, title = @sprintf("Δm²₄₁ = %.3g eV²", dm41[j]),
              xlabel = "sin²θ₂₄", ylabel = "Δχ²", limits = (1e-4, 1, 0, 25))
    lines!(ax, ss24, off[:, j], color = :black, linewidth = 2.5, label = "official (release)")
    lines!(ax, ss24, ours["paper_chi2"][:, j], color = :dodgerblue, linewidth = 2, label = "Newtrinos, paper χ²")
    lines!(ax, ss24, ours["module"][:, j], color = :orangered, linewidth = 2, linestyle = :dash, label = "Newtrinos, module")
    lines!(ax, ss24, fcv[:, j], color = :gray, linestyle = :dot, label = "FC 90% critical value")
    k == 1 && axislegend(ax, position = :lt, framevisible = false, labelsize = 11)
end
save("ours/dchi2_slices.png", fig)

# the paper's quoted limit at Δm²₄₁ = 0.5 eV²: sin²θ₂₄ < 0.006 (90% C.L.)
function crossing(x, y, level)
    i = findfirst(>(level), y); (isnothing(i) || i == 1) && return NaN
    exp(log(x[i-1]) + (level - y[i-1]) / (y[i] - y[i-1]) * (log(x[i]) - log(x[i-1])))
end
j = argmin(abs.(log.(dm41) .- log(0.5)))
println(@sprintf("Δm²₄₁ = %.3g eV², 90%% C.L. (FC) upper limit on sin²θ₂₄: official surface %.4f, Newtrinos paper χ² %.4f, module %.4f (paper: 0.006)",
                 dm41[j], crossing(ss24, off[:, j] .- fcv[:, j], 0), crossing(ss24, ours["paper_chi2"][:, j] .- fcv[:, j], 0),
                 crossing(ss24, ours["module"][:, j] .- fcv[:, j], 0)))
for (k, v) in ours
    println(@sprintf("%s: max |Δχ² − official| = %.2f, median %.3f (Δχ²_official < 25)", k,
                     maximum(abs.(v .- off)[off .< 25]), median(abs.(v .- off)[off .< 25])))
end
