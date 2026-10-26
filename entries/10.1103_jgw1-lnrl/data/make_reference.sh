#!/usr/bin/env bash
# NuFast-Earth reference values for this entry. Needs git, g++ (C++17) and the fine-shell files from
# make_fine_shells.jl. Writes the ref_*.csv files next to this script.
#   bash make_reference.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
commit=327586dddcb8c01282feb7560d029b63ea1257f4   # NuFast-Earth, github.com/PeterDenton/NuFast-Earth (2026-09-22)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
git clone -q https://github.com/PeterDenton/NuFast-Earth.git "$work/NuFast-Earth"
git -C "$work/NuFast-Earth" checkout -q $commit
N="$work/NuFast-Earth"
g++ -O2 -std=c++17 -I"$N/include" "$here/nufast_driver.cpp" "$N"/src/{Earth,Geometry,NuFastEarth,Oscillation,Solar}.cpp -o "$work/driver"
cd "$work"

# paper setup (Sec. V): NuFIT 6.0 NO, PREM_NDiscontinuityLayer(2,10,10,5), detector depth 2 km, production height 10 km
osc="s12sq 0.307
s13sq 0.02195
s23sq 0.561
delta $(python3 -c 'import math; print(repr(177*math.pi/180))')
Dmsq21 7.49e-5
Dmsq31 2.534e-3"
seq_list() { python3 -c "import numpy as np, sys; print(' '.join(repr(float(x)) for x in $1))"; }

printf 'earth prem_ndl4 2 10 10 5\n' > in.txt
./driver shells in.txt "$here/shells_ndl_2_10_10_5.csv"

# (1) oscillogram of the paper: E = 2–40 GeV (linear), cos θz = −1 … 0.1
cat > in.txt <<EOF
$osc
earth prem_ndl4 2 10 10 5
depth 2
production_height 10
E $(seq_list 'np.linspace(2, 40, 77)')
cosz $(seq_list 'np.round(np.linspace(-1, 0.1, 111), 10)')
EOF
./driver atm in.txt "$here/ref_oscillogram_ndl.csv"

# (2) Earth discretisation: 1 km shells of the PREM table of two Newtrinos configurations, Newtrinos geometry
#     (detector at r = 6369 km, 20 km atmosphere)
for cfg in default discontinuities; do
cat > in.txt <<EOF
$osc
earth shells $here/shells_fine_$cfg.txt
depth 2
production_height 20
E $(seq_list 'np.geomspace(2, 50, 41)')
cosz $(seq_list 'np.round(np.linspace(-1, 0, 81), 10)')
EOF
./driver atm in.txt "$here/ref_fine_$cfg.csv"
done

# (3) solar day and night (paper Fig. 6): ρYₑ(Sun) = 100·2/3 g/cm³, E = 5–20 MeV, night cos θz grid,
#     and NuFast's annual exposure weights at the DUNE and Super-K latitudes
cat > in.txt <<EOF
$osc
earth prem_ndl4 2 10 10 5
depth 2
production_height 0
rhoYe_sun $(python3 -c 'print(repr(100*2/3))')
E $(seq_list 'np.linspace(5e-3, 20e-3, 151)')
cosz $(seq_list 'np.round(np.linspace(-1, 0, 401), 10)')
lat 44.356 36.48
EOF
./driver solar in.txt "$here/ref_solar_ndl.csv"
./driver weights in.txt "$here/ref_solar_weights.csv"
echo "references written to $here"
