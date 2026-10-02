# Newtrinos verifications

Reproductions of published neutrino results with [Newtrinos.jl](https://github.com/philippeller/Newtrinos.jl),
organised by the DOI of the reproduced paper. Each entry shows the original paper figure next to the
reproduction, the code that produced it, the exact Newtrinos.jl commit and the fit results.

> The Newtrinos modules and analyses shown here are largely written by an AI coding assistant (Claude) with
> human steering ("vibe coded"); the side-by-side comparisons are the validation.

## Layout

```
entries/<doi with / → _>/
  entry.toml                 paper metadata, figures (original ↔ ours), caveats, pinned Newtrinos commit
  reproduce.jl               minimal Newtrinos code: writes ours/ and results/
  original/                  figures cropped from the paper
  ours/, results/            outputs of reproduce.jl (committed)
  data/                      optional extra inputs (e.g. extracted official regions)
  run.toml                   provenance, written by tools/run_entry.jl
  newtrinos_Manifest.toml    resolved Newtrinos environment of the run
tools/run_entry.jl           run an entry against a clean worktree of the pinned commit
site/build.jl                build the static site into public/
```

## Adding or updating an entry

1. Create `entries/<slug>/` with `entry.toml` (copy an existing one), `reproduce.jl` and `original/*.png`.
2. Run it against the pinned commit (needs a local Newtrinos.jl clone that contains the commit):
   ```bash
   julia tools/run_entry.jl entries/<slug> --newtrinos=../Newtrinos.jl --threads=8
   ```
   This creates a detached worktree under `.worktrees/`, instantiates it, removes old outputs, runs
   `reproduce.jl` and writes `run.toml` (commit, Julia version, machine, date, wall time, SHA-256 of outputs).
3. Build and preview the site:
   ```bash
   julia site/build.jl && python3 -m http.server -d public
   ```

Fits are never run in CI; the GitHub Pages workflow only builds the site from the committed outputs.

## License

- **Code** (`tools/`, `site/`, every `entries/*/reproduce.jl`): [MIT](LICENSE).
- **Our results** (`entries/*/ours/`, `entries/*/results/`, `run.toml`): [CC BY 4.0](LICENSE-CC-BY-4.0.txt).
  Please cite the reproduced paper and [Newtrinos.jl](https://doi.org/10.21105/joss.09644) when reusing them.
- **Not covered:** the original figures in `entries/*/original/` and inputs derived from them (`entries/*/data/`)
  remain the property of the respective collaborations and publishers. They are reproduced here, with citation,
  for scientific comparison only.
