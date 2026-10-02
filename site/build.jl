#!/usr/bin/env julia
# Build the static site: entries/*/entry.toml (+ run.toml) → public/
#   julia site/build.jl [--out=public]
using TOML, Dates

const ROOT = dirname(@__DIR__)
const OUT = let o = filter(a -> startswith(a, "--out="), ARGS)
    isempty(o) ? joinpath(ROOT, "public") : abspath(o[1][7:end])
end
# --artifact: preview build for a claude.ai artifact (home page as a fragment, explicit index.html links)
const ARTIFACT = "--artifact" in ARGS
# directory links: "entries/" → "entries/index.html" in the artifact build (no directory indexes there)
d(path) = ARTIFACT && (isempty(path) || endswith(path, "/")) ? path * "index.html" : path
const SITE = (name = "Newtrinos verifications",
              tagline = "Reproductions of published neutrino results with Newtrinos.jl",
              repo = "https://github.com/Newtrinos-org/newtrinos-verifications")

# ------------------------------------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------------------------------------
esc(s) = replace(string(s), "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
slugify(s) = lowercase(replace(string(s), r"[^A-Za-z0-9]+" => "-"))
doi_slug(doi) = replace(doi, "/" => "_")

function tojson(x)
    x isa AbstractDict && return "{" * join(["$(tojson(string(k))):$(tojson(v))" for (k, v) in x], ",") * "}"
    x isa AbstractVector && return "[" * join(tojson.(x), ",") * "]"
    x isa AbstractString && return "\"" * replace(x, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n") * "\""
    x isa Bool && return string(x)
    x isa Number && return string(x)
    x === nothing && return "null"
    tojson(string(x))
end

function filesize_str(path)
    n = filesize(path)
    n < 1024 ? "$n B" : n < 1024^2 ? "$(round(n / 1024, digits = 1)) kB" : "$(round(n / 1024^2, digits = 1)) MB"
end

const STATUS_LABEL = Dict("reproduced" => "Reproduced", "partial" => "Partially reproduced", "in-progress" => "In progress")

tag(text; cls = "") = """<span class="tag $cls">$(esc(text))</span>"""

function page(title, body; depth = 0, description = SITE.tagline, active = "")
    html = full_page(title, body; depth, description, active)
    ARTIFACT && depth == 0 || return html
    # the artifact host wraps the main page in its own document skeleton: keep head links, drop the wrappers
    html = replace(html, r"<!doctype html>\n<html lang=\"en\">\n<head>\n" => "", "</head>\n" => "", r"<body([^>]*)>" => s"<div class=\"page-root\"\1>", "</body>\n</html>\n" => "</div>\n")
    replace(html, r"<meta charset=\"utf-8\">\n<meta name=\"viewport\"[^>]*>\n" => "")
end

function full_page(title, body; depth = 0, description = SITE.tagline, active = "")
    r = depth == 0 ? "./" : "../"^depth
    nav(href, label, key) = """<a href="$r$(d(href))"$(active == key ? " class=\"active\"" : "")>$label</a>"""
    """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$(esc(title))</title>
<meta name="description" content="$(esc(description))">
<link rel="icon" type="image/svg+xml" href="$(r)assets/logo.svg">
<link rel="stylesheet" href="$(r)assets/style.css">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">
</head>
<body data-root="$r" data-index="$(ARTIFACT ? "index.html" : "")">
<header class="site-header">
  <div class="wrap header-inner">
    <a class="brand" href="$(d(r))"><picture><source srcset="$(r)assets/logo-dark.svg" media="(prefers-color-scheme: dark)"><img class="brand-logo" src="$(r)assets/logo.svg" alt="Newtrinos.jl logo" width="32" height="32"></picture> $(SITE.name)</a>
    <nav class="main-nav">
      $(nav("entries/", "Entries", "entries"))
      $(nav("experiments/", "Experiments", "experiments"))
      $(nav("about/", "About", "about"))
    </nav>
  </div>
</header>
<main class="wrap">
$body
</main>
<footer class="site-footer">
  <div class="wrap">
    Built with <a href="https://github.com/Newtrinos-org/Newtrinos.jl">Newtrinos.jl</a> ·
    <a href="https://newtrinos-org.github.io/Newtrinos.jl/stable/">docs</a> ·
    <a href="https://doi.org/10.21105/joss.09644">JOSS</a> ·
    <a href="$(SITE.repo)">source</a> · generated $(Dates.format(now(UTC), "yyyy-mm-dd"))
  </div>
</footer>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/julia.min.js"></script>
<script src="$(r)assets/app.js"></script>
</body>
</html>
"""
end

# ------------------------------------------------------------------------------------------------
# load entries
# ------------------------------------------------------------------------------------------------
function load_entries()
    entries = []
    for slug in sort(readdir(joinpath(ROOT, "entries")))
        dir = joinpath(ROOT, "entries", slug)
        isfile(joinpath(dir, "entry.toml")) || continue
        e = TOML.parsefile(joinpath(dir, "entry.toml"))
        e["slug"] = slug
        e["dir"] = dir
        e["run"] = isfile(joinpath(dir, "run.toml")) ? TOML.parsefile(joinpath(dir, "run.toml")) : nothing
        e["run"] === nothing && (e["status"] = "in-progress")
        @assert doi_slug(e["doi"]) == slug "entry folder $slug does not match its DOI $(e["doi"])"
        push!(entries, e)
    end
    sort!(entries, by = e -> (e["added"], e["year"]), rev = true)
end

function index_record(e)
    Dict("slug" => e["slug"], "title" => e["title"], "short" => e["short"], "doi" => e["doi"], "journal" => e["journal"],
         "year" => e["year"], "arxiv" => get(e, "arxiv", ""), "inspire" => get(e, "inspire", ""),
         "experiments" => e["experiments"], "parameters" => get(e, "parameters", String[]), "status" => e["status"],
         "added" => e["added"], "figures" => length(get(e, "figures", [])),
         "thumb" => isempty(get(e, "figures", [])) ? "" : "entries/$(e["slug"])/" * e["figures"][1]["ours"])
end

# ------------------------------------------------------------------------------------------------
# pages
# ------------------------------------------------------------------------------------------------
links_html(e; r = "") = join(filter(!isempty, [
    """<a class="btn" href="https://doi.org/$(esc(e["doi"]))">DOI</a>""",
    haskey(e, "arxiv") ? """<a class="btn" href="https://arxiv.org/abs/$(esc(e["arxiv"]))">arXiv</a>""" : "",
    haskey(e, "inspire") ? """<a class="btn" href="https://inspirehep.net/literature/$(e["inspire"])">INSPIRE</a>""" : "",
]), " ")

function entry_card(e; r = "")
    tags = join(vcat([tag(x; cls = "exp") for x in e["experiments"]], [tag(e["year"])],
                     [tag(STATUS_LABEL[e["status"]]; cls = "status-$(e["status"])")]), " ")
    """
<article class="card">
  <a class="card-link" href="$(r)$(d("entries/$(e["slug"])/"))">
    <h3>$(esc(e["short"]))</h3>
    <p class="muted">$(esc(e["title"]))</p>
  </a>
  <div class="tags">$tags</div>
  <div class="meta muted">$(esc(e["journal"])) ($(e["year"])) · $(length(get(e, "figures", []))) figure(s) · added $(e["added"])</div>
</article>"""
end

function experiment_counts(entries)
    c = Dict{String,Int}()
    for e in entries, x in e["experiments"]
        c[x] = get(c, x, 0) + 1
    end
    sort(collect(c), by = x -> (-x[2], x[1]))
end

function home_page(entries)
    nfig = sum(e -> length(get(e, "figures", [])), entries; init = 0)
    exps = experiment_counts(entries)
    grid = join(["""<a class="exp-tile" href="$(d("entries/"))?experiment=$(esc(x))"><span>$(esc(x))</span><b>$n</b></a>""" for (x, n) in exps], "\n")
    recent = join(entry_card.(entries[1:min(end, 6)]), "\n")
    body = """
<section class="hero">
  <h1>$(SITE.name)</h1>
  <p>$(SITE.tagline). Every entry shows the original figure from the paper next to our reproduction, the code
  that produced it, the exact Newtrinos.jl commit and the downloadable fit results.</p>
  <div class="stats">
    <div><b>$(length(entries))</b><span>papers</span></div>
    <div><b>$(length(exps))</b><span>experiments</span></div>
    <div><b>$nfig</b><span>figures</span></div>
  </div>
</section>
<section class="tool">
  <h2>Newtrinos.jl</h2>
  <p>All reproductions use <b>Newtrinos.jl</b>, an open-source Julia package for global analyses of neutrino
  data: modular physics models (oscillations, fluxes, cross sections, Earth model), experiment likelihoods
  built from public data releases, and inference tools for fits, profiles and scans.</p>
  <div class="links">
    <a class="btn" href="https://github.com/Newtrinos-org/Newtrinos.jl">Code on GitHub</a>
    <a class="btn" href="https://newtrinos-org.github.io/Newtrinos.jl/stable/">Documentation</a>
    <a class="btn" href="https://doi.org/10.21105/joss.09644">JOSS paper</a>
    <a class="btn" href="https://doi.org/10.5281/zenodo.23036152">Zenodo archive</a>
  </div>
  <p class="muted small">If you use Newtrinos.jl, please cite: <i>Newtrinos.jl: A Julia Package for Global Analysis of
  Neutrino Data</i>, Journal of Open Source Software 11(125), 9644, <a href="https://doi.org/10.21105/joss.09644">doi:10.21105/joss.09644</a>.</p>
</section>
<section>
  <h2>Browse by experiment</h2>
  <div class="exp-grid">$grid</div>
</section>
<section>
  <h2>Recently added</h2>
  <div class="cards">$recent</div>
  <p><a href="$(d("entries/"))">All entries →</a></p>
</section>"""
    page(SITE.name, body)
end

function entries_page(entries)
    body = """
<h1>Entries</h1>
<div class="browse">
  <aside class="facets" id="facets"><p class="muted">Loading…</p></aside>
  <section>
    <div class="toolbar">
      <input id="search" type="search" placeholder="Search title, DOI, arXiv…" aria-label="Search">
      <select id="sort" aria-label="Sort">
        <option value="newest">Newest first</option>
        <option value="oldest">Oldest first</option>
        <option value="added">Recently added</option>
        <option value="title">Title A–Z</option>
      </select>
    </div>
    <p class="muted" id="count"></p>
    <div class="cards" id="list">
      $(join(entry_card.(entries; r = "../"), "\n"))
    </div>
  </section>
</div>"""
    page("Entries · $(SITE.name)", body; depth = 1, active = "entries")
end

function experiments_page(entries)
    rows = join(["""<a class="exp-tile" href="../$(d("entries/"))?experiment=$(esc(x))"><span>$(esc(x))</span><b>$n</b></a>""" for (x, n) in experiment_counts(entries)], "\n")
    page("Experiments · $(SITE.name)", """<h1>Experiments</h1><div class="exp-grid">$rows</div>"""; depth = 1, active = "experiments")
end

function about_page()
    body = """
<h1>About</h1>
<div class="prose">
<p>This site collects reproductions of published neutrino-oscillation results with
<a href="https://github.com/philippeller/Newtrinos.jl">Newtrinos.jl</a>, using public material only
(data releases and published figures). Entries are organised by the DOI of the reproduced paper.</p>
<p>Each entry pins the exact Newtrinos.jl commit. The fits are run with <code>tools/run_entry.jl</code>, which
checks out that commit into a clean worktree, runs the entry's <code>reproduce.jl</code> and records the commit,
Julia version, machine, run time and SHA-256 checksums of all outputs in <code>run.toml</code>.</p>
<p>Original figures are reproduced from the cited publications for the purpose of scientific comparison;
all rights remain with the respective collaborations and publishers.</p>
</div>"""
    page("About · $(SITE.name)", body; depth = 1, active = "about")
end

function entry_page(e)
    r = "../../"
    run = e["run"]
    figs = join(map(get(e, "figures", [])) do f
        orig = haskey(f, "original") ?
            """<figure><figcaption>Paper · $(esc(get(f, "paper_ref", "")))</figcaption><a href="$(esc(f["original"]))"><img src="$(esc(f["original"]))" alt="Original: $(esc(f["title"]))" loading="lazy"></a></figure>""" :
            """<figure class="missing"><figcaption>Paper</figcaption><p class="muted">$(esc(get(f, "paper_ref", "no figure")))</p></figure>"""
        ours = isfile(joinpath(e["dir"], f["ours"])) ?
            """<figure><figcaption>Newtrinos.jl</figcaption><a href="$(esc(f["ours"]))"><img src="$(esc(f["ours"]))" alt="Reproduction: $(esc(f["title"]))" loading="lazy"></a></figure>""" :
            """<figure class="missing"><figcaption>Newtrinos.jl</figcaption><p class="muted">Reproduction pending</p></figure>"""
        """
<section class="figure-pair" id="$(esc(f["id"]))">
  <h3>$(esc(f["title"])) <span class="muted">· $(esc(get(f, "paper_ref", "")))</span></h3>
  <div class="pair">$orig$ours</div>
</section>"""
    end, "\n")

    caveats = isempty(get(e, "caveats", [])) ? "" :
        "<h2>Known limitations</h2><ul>" * join(["<li>$(esc(c))</li>" for c in e["caveats"]]) * "</ul>"
    extlinks = join(["""<a class="btn" href="$(esc(l["url"]))">$(esc(l["label"]))</a>""" for l in get(e, "links", [])], " ")

    code = read(joinpath(e["dir"], "reproduce.jl"), String)
    commit = run === nothing ? e["newtrinos"]["commit"] : run["newtrinos"]["commit"]
    repo = e["newtrinos"]["repository"]

    prov = if run === nothing
        "<p class=\"muted\">This entry has not been run yet.</p>"
    else
        rr = run["run"]
        """
<table class="kv">
  <tr><th>Newtrinos.jl commit</th><td><a href="$(esc(repo))/tree/$(commit)"><code>$(commit[1:10])</code></a> ($(esc(join(e["newtrinos"]["modules"], ", "))))</td></tr>
  <tr><th>Run date</th><td>$(esc(rr["date"]))</td></tr>
  <tr><th>Wall time</th><td>$(round(rr["wall_time_s"] / 60, digits = 1)) min ($(rr["threads"]) threads)$(haskey(run, "fit_cache") && run["fit_cache"]["reused_points"] > 0 ? ", $(run["fit_cache"]["reused_points"]) of $(run["fit_cache"]["total_points"]) fit points reused from an earlier run at the same commit" : "")</td></tr>
  <tr><th>Julia</th><td>$(esc(rr["julia"]))</td></tr>
  <tr><th>Machine</th><td>$(esc(rr["cpu"])), $(rr["cpu_threads"]) hardware threads, $(esc(rr["os"]))</td></tr>
  <tr><th>Environment</th><td><a href="newtrinos_Manifest.toml" download>newtrinos_Manifest.toml</a> · <a href="run.toml" download>run.toml</a></td></tr>
</table>"""
    end

    downloads = if run === nothing
        ""
    else
        rows = join(["""<tr><td><a href="$(esc(p))" download>$(esc(p))</a></td><td>$(filesize_str(joinpath(e["dir"], p)))</td><td><code title="$(h)">$(h[1:12])…</code></td></tr>"""
                     for (p, h) in sort(collect(run["files"]))], "\n")
        """<p class="muted small">CSV files hold one row per scan point: the scanned and profiled parameters
(<code>sin2_theta23</code>, <code>dm2_32</code> in eV², <code>dcp</code> in rad, nuisance parameters), <code>dchi2</code> = −2Δ log L
and the log-likelihood. JLD2 files contain the full <code>NewtrinosResult</code> objects.</p>
<div class="table-scroll"><table class="files"><tr><th>File</th><th>Size</th><th>SHA-256</th></tr>$rows</table></div>"""
    end

    tags = join(vcat([tag(x; cls = "exp") for x in e["experiments"]], [tag(e["year"])], [tag(p; cls = "param") for p in get(e, "parameters", [])],
                     [tag(STATUS_LABEL[e["status"]]; cls = "status-$(e["status"])")]), " ")
    body = """
<nav class="crumbs"><a href="$(r)$(d("entries/"))">Entries</a> / <span>$(esc(e["doi"]))</span></nav>
<header class="entry-head">
  <p class="muted">$(esc(e["collaboration"])) · $(esc(e["journal"])) ($(e["year"]))</p>
  <h1>$(esc(e["title"]))</h1>
  <div class="tags">$tags</div>
  <div class="links">$(links_html(e)) $extlinks</div>
</header>
<p class="summary">$(esc(e["summary"]))</p>

<h2>Figures</h2>
$figs
$caveats

<h2>Reproduce</h2>
<ol class="steps">
  <li><code>git clone $(esc(repo)).git && cd Newtrinos.jl && git checkout $(commit[1:10])</code></li>
  <li>Download <a href="reproduce.jl" download>reproduce.jl</a>$(isdir(joinpath(e["dir"], "data")) ? " and the <code>data/</code> folder of this entry" : "") into an empty directory.</li>
  <li><code>julia -t 8 --project=/path/to/Newtrinos.jl reproduce.jl</code></li>
</ol>
<div class="code">
  <button class="copy" data-copy="code-reproduce">Copy</button>
  <pre><code id="code-reproduce" class="language-julia">$(esc(code))</code></pre>
</div>

<h2>Provenance</h2>
$prov

<h2>Downloads</h2>
$downloads

<p class="muted small">Original figures: $(esc(e["collaboration"])), $(esc(e["journal"])) ($(e["year"])),
<a href="https://doi.org/$(esc(e["doi"]))">doi:$(esc(e["doi"]))</a>; reproduced for scientific comparison, all rights with the authors and publisher.</p>
"""
    page("$(e["short"]) · $(SITE.name)", body; depth = 2, description = e["title"], active = "entries")
end

# ------------------------------------------------------------------------------------------------
# build
# ------------------------------------------------------------------------------------------------
function build()
    entries = load_entries()
    rm(OUT, recursive = true, force = true)
    mkpath(OUT)
    cp(joinpath(@__DIR__, "assets"), joinpath(OUT, "assets"))
    write(joinpath(OUT, "index.html"), home_page(entries))
    for (d, html) in (("entries", entries_page(entries)), ("experiments", experiments_page(entries)), ("about", about_page()))
        mkpath(joinpath(OUT, d)); write(joinpath(OUT, d, "index.html"), html)
    end
    write(joinpath(OUT, "index.json"), tojson(index_record.(entries)))
    for e in entries
        dst = joinpath(OUT, "entries", e["slug"])
        mkpath(dst)
        for item in ("original", "ours", "results", "data", "reproduce.jl", "run.toml", "newtrinos_Manifest.toml")
            src = joinpath(e["dir"], item)
            ispath(src) && cp(src, joinpath(dst, item), force = true)
        end
        write(joinpath(dst, "index.html"), entry_page(e))
    end
    write(joinpath(OUT, ".nojekyll"), "")
    println("built $(length(entries)) entries → $OUT")
end

build()
