#!/usr/bin/env julia
# Run one entry's reproduce.jl against a clean checkout of Newtrinos.jl at the commit pinned in its
# entry.toml, and record the provenance (commit, Julia, machine, date, wall time, sha256 of all outputs)
# in run.toml next to it. Fit caches (cache/) are reused only if they come from the same commit. The resolved Newtrinos environment is stored as newtrinos_Manifest.toml.
#
#   julia tools/run_entry.jl entries/<slug> --newtrinos=/path/to/Newtrinos.jl [--threads=8]
#
# --newtrinos is a local clone of Newtrinos.jl that contains the pinned commit; a detached git worktree
# of that commit is created under .worktrees/ (so local modifications of the clone are never used).
using TOML, SHA, Dates

function parse_args(args)
    entry = nothing; opts = Dict("threads" => "8")
    for a in args
        if startswith(a, "--")
            k, v = split(a[3:end], "=", limit = 2); opts[k] = v
        else
            entry = a
        end
    end
    entry === nothing && error("usage: julia tools/run_entry.jl entries/<slug> --newtrinos=<clone> [--threads=N]")
    haskey(opts, "newtrinos") || error("--newtrinos=<path to a Newtrinos.jl clone> is required")
    abspath(entry), opts
end

entry, opts = parse_args(ARGS)
meta = TOML.parsefile(joinpath(entry, "entry.toml"))
rev = meta["newtrinos"]["commit"]
clone = abspath(opts["newtrinos"])
root = dirname(dirname(entry))

commit = strip(read(`git -C $clone rev-parse $rev^\{commit\}`, String))
worktree = joinpath(root, ".worktrees", "newtrinos-$(commit[1:10])")
if !isdir(worktree)
    mkpath(dirname(worktree))
    run(`git -C $clone worktree add --detach $worktree $commit`)
end
@assert strip(read(`git -C $worktree rev-parse HEAD`, String)) == commit
@assert isempty(strip(read(`git -C $worktree status --porcelain --untracked-files=no`, String))) "worktree is not clean"

julia = Base.julia_cmd()
run(`$julia --project=$worktree -e "using Pkg; Pkg.instantiate(); Pkg.precompile()"`)

# fresh outputs; fit caches are reused only if they were produced with the same Newtrinos commit
for d in ("ours", "results")
    rm(joinpath(entry, d), recursive = true, force = true)
end
cache = joinpath(entry, "cache"); stamp = joinpath(cache, "NEWTRINOS_COMMIT")
if isdir(cache) && !(isfile(stamp) && strip(read(stamp, String)) == commit)
    rm(cache, recursive = true)
end
mkpath(cache); write(stamp, commit)
count_cached() = sum(((_, _, fs),) -> count(endswith(".jld2"), fs), walkdir(cache); init = 0)
n_cached = count_cached()
started = now(UTC)
t0 = time()
cd(entry) do
    run(`$julia -t $(opts["threads"]) --project=$worktree reproduce.jl`)
end
wall = time() - t0

files = Dict{String,String}()
for d in ("ours", "results"), (dir, _, fs) in walkdir(joinpath(entry, d)), f in fs
    path = joinpath(dir, f)
    files[relpath(path, entry)] = bytes2hex(open(sha256, path))
end
cp(joinpath(worktree, "Manifest.toml"), joinpath(entry, "newtrinos_Manifest.toml"), force = true)

jl_version = strip(read(`$julia -e "print(VERSION)"`, String))
run_info = Dict(
    "newtrinos" => Dict("repository" => meta["newtrinos"]["repository"], "commit" => commit),
    "run" => Dict("date" => Dates.format(started, "yyyy-mm-ddTHH:MM:SS") * "Z", "wall_time_s" => round(wall, digits = 1),
                  "julia" => jl_version, "threads" => parse(Int, opts["threads"]),
                  "cpu" => Sys.cpu_info()[1].model, "cpu_threads" => Sys.CPU_THREADS, "os" => string(Sys.KERNEL),
                  "command" => "julia -t $(opts["threads"]) --project=<Newtrinos.jl @ $(commit[1:10])> reproduce.jl"),
    "files" => files,
    "fit_cache" => Dict("reused_points" => n_cached, "total_points" => count_cached(),
                        "note" => "cached fit points are reused only from runs at the same Newtrinos commit"),
)
open(joinpath(entry, "run.toml"), "w") do io
    println(io, "# written by tools/run_entry.jl — do not edit")
    TOML.print(io, run_info; sorted = true)
end
println("done: $(relpath(entry, root)) in $(round(wall / 60, digits = 1)) min, $(length(files)) output files")
