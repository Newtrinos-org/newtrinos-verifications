# Provenance of data/official_*.csv: predictions of the official Daya Bay model
# (https://github.com/dagflow-team/dayabay-analysis, dayabay_model 1.8.0, dayabay-data-official 1.0.1, hdf5 analysis dataset).
#   official_nominal.csv: at the nominal parameters of the model
#   official_bestfit.csv: at the best fit of the real-data pull fit with the Poisson statistic, i.e. the output of
#     fits/fit_dayabay_dgm.py --data real --statistic full.pull.chi2poisson
#       --free-parameters survival_probability neutrino_per_fission_factor
#       --constrained-parameters survival_probability detector reactor background
#       --profile-parameters survival_probability.DeltaMSq32 survival_probability.SinSq2Theta13 --output fit.yaml
#   official_fits.csv: best fit, MINOS errors (errorsdict_profiled) and χ²min ("fun") from the fit.yaml outputs of the
#     command above with --statistic full.pull.chi2poisson and full.pull.chi2cnp
# Usage (inside a dayabay-analysis checkout with its requirements): python official_extract.py fit.yaml
import csv, sys
import yaml
from dayabay_data_official import get_path_data
from dayabay_model import model_dayabay

model = model_dayabay(path_data=get_path_data("hdf5"), concatenation_mode="detector_period")
st = model.storage
labels = [(k[1], k[2]) for k in st["outputs.eventscount.stages"].walkkeys() if k[0] == "erec"]


def write(path):
    pred = st["outputs.eventscount.final.concatenated.selected"].data
    data = st["outputs.data.real.concatenated.selected"].data
    assert len(labels) * 26 == len(pred)
    with open(path, "w") as f:
        w = csv.writer(f)
        w.writerow(["detector", "period", "bin", "pred", "data"])
        for i, (d, p) in enumerate(labels):
            for b in range(26):
                w.writerow([d, p, b, pred[i * 26 + b], data[i * 26 + b]])


write("official_nominal.csv")
fit = yaml.safe_load(open(sys.argv[1]))
pars = {".".join(map(str, k)): v for k, v in st["parameters.all"].walkitems()}
for name, value in fit["xdict"].items():
    pars[name].value = value
write("official_bestfit.csv")
