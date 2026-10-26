# Converged Earth references for the discretisation study: 1 km shells (constant density each) built from the
# PREM table and the per-zone Yₑ of a Newtrinos Earth configuration, written as "r_outer[km] ρYₑ[g/cm³]" lines for
# nufast_driver ("earth shells <file>"). Run with the Newtrinos environment before make_reference.sh:
#   julia --project=<Newtrinos.jl> make_fine_shells.jl
using Newtrinos, Printf
const EL = Newtrinos.earth_layers
cd(@__DIR__)

function fine_shells(cfg, file; dr = 1.0)
    zones = EL._nonempty_zones(cfg, EL._prem_table(cfg))
    profiles = EL._zone_profiles(cfg)                    # (radius, density) per zone, zones in Newtrinos order
    rmax = [maximum(p[1]) for p in profiles]
    open(file, "w") do io
        for r_out in dr:dr:6371.0
            r = r_out - dr / 2
            k = findlast(>=(r), rmax)                     # innermost zone that reaches out to r
            @printf(io, "%.1f %.17g\n", r_out, EL._interp_flat(profiles[k]..., r) * cfg.p_fractions[zones[k]])
        end
    end
end

fine_shells(EL.PREM(), "shells_fine_default.txt")
fine_shells(EL.PREM_discontinuities(), "shells_fine_discontinuities.txt")
