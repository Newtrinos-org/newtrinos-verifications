// NuFast-Earth driver for the Newtrinos.jl cross-check.
// usage: driver <mode> <input.txt> <output.csv>
//   mode = atm   : P(alpha -> beta) on an (E, cos z) grid, neutrinos and antineutrinos
//   mode = solar : solar day (one row per E) and night (E x cos z) P(nu_e -> beta)
//   mode = shells: the constant-density shells of the Earth model (outer radius [km], rhoYe [g/cc])
//   mode = weights: NuFast's annual solar exposure weight (Solar_Weight / sin(eta)) per cos z, for latitudes "lat" [deg]
// input.txt: key value lines
//   s12sq s13sq s23sq delta Dmsq21 Dmsq31   oscillation parameters (delta in rad)
//   earth  shells <file> | prem_full | prem_ndl <n> | prem_ndl4 <n_ic> <n_oc> <n_im> <n_om>   Earth model
//          (shells file: one line "r_outer rhoYe" per shell, increasing radius, constant density)
//   depth  <km>  production_height <km>  rhoYe_sun <g/cc>
//   E <list in GeV>   cosz <list>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include "NuFastEarth.h"
#include "Earth.h"
#include "Solar.h"

using namespace NuFast;

class Shells : public Earth_Density
{
	public:
		Shells(const std::vector<double> &r, const std::vector<double> &rho) : rs(r), rhos(rho)
		{
			n_discontinuities = r.size();
			discontinuities = r;
			r_E = r.back();
			constant_shells = true;
		}
		double rhoYe(double r)
		{
			for (size_t i = 0; i < rs.size(); i++) if (r < rs[i]) return rhos[i];
			return 0;
		}
	private:
		std::vector<double> rs, rhos;
};

int main(int argc, char **argv)
{
	if (argc != 4) { fprintf(stderr, "usage: driver atm|solar input output\n"); return 1; }
	std::string mode = argv[1];
	std::ifstream in(argv[2]);
	std::map<std::string, std::vector<std::string>> kv;
	std::string line;
	while (std::getline(in, line))
	{
		std::istringstream ss(line);
		std::string key, tok;
		if (!(ss >> key) || key[0] == '#') continue;
		while (ss >> tok) kv[key].push_back(tok);
	}
	auto num = [&](const char *k) { return std::stod(kv.at(k).at(0)); };
	auto list = [&](const char *k) { std::vector<double> v; for (auto &s : kv.at(k)) v.push_back(std::stod(s)); return v; };

	Earth_Density *earth;
	std::string em = kv.at("earth").at(0);
	if (em == "shells")
	{
		std::ifstream sf(kv.at("earth").at(1));
		std::vector<double> r, rho;
		double a, b;
		while (sf >> a >> b) { r.push_back(a); rho.push_back(b); }
		earth = new Shells(r, rho);
	}
	else if (em == "prem_full") earth = new PREM_Full();
	else if (em == "prem_ndl") earth = new PREM_NDiscontinuityLayer(std::stoi(kv.at("earth").at(1)));
	else if (em == "prem_ndl4") earth = new PREM_NDiscontinuityLayer(std::stoi(kv.at("earth").at(1)), std::stoi(kv.at("earth").at(2)),
	                                                                 std::stoi(kv.at("earth").at(3)), std::stoi(kv.at("earth").at(4)));
	else { fprintf(stderr, "unknown earth model %s\n", em.c_str()); return 1; }

	FILE *out = fopen(argv[3], "w");
	if (mode == "shells")
	{
		// same evaluation as NuFast uses for constant shells (rhoYe just inside each outer boundary)
		fprintf(out, "r_outer_km,rhoYe\n");
		for (int j = 0; j < earth->n_discontinuities; j++)
			fprintf(out, "%.12g,%.12g\n", earth->discontinuities[j], earth->rhoYe(earth->discontinuities[j] - 1e-8));
		fclose(out);
		return 0;
	}
	std::vector<double> coszs = list("cosz");
	if (mode == "weights")
	{
		std::vector<double> lats = list("lat");
		fprintf(out, "cosz");
		for (double l : lats) fprintf(out, ",w_%g", l);
		fprintf(out, "\n");
		for (double cz : coszs)
		{
			double eta = M_PI - acos(cz) + 1e-8; // as in NuFast's Solar_Day_Night_Validation
			fprintf(out, "%.12g", cz);
			for (double l : lats) fprintf(out, ",%.12g", Solar_Weight(eta, l * M_PI / 180) / fabs(sin(eta)));
			fprintf(out, "\n");
		}
		fclose(out);
		return 0;
	}
	std::vector<double> Es = list("E");
	if (mode == "atm")
	{
		fprintf(out, "anti,E_GeV,cosz,Pee,Pem,Pet,Pme,Pmm,Pmt,Pte,Ptm,Ptt\n");
		for (int anti = 0; anti < 2; anti++)
		{
			Probability_Engine pe;
			pe.Set_Oscillation_Parameters(num("s12sq"), num("s13sq"), num("s23sq"), num("delta"), num("Dmsq21"), num("Dmsq31"), anti == 0);
			pe.Set_Eigenvalue_Precision(-1); // exact eigenvalues
			pe.Set_Spectra(Es, coszs);
			pe.Set_Earth(num("depth"), earth);
			pe.Set_Production_Height(num("production_height"));
			auto P = pe.Get_Probabilities();
			for (size_t i = 0; i < Es.size(); i++)
				for (size_t j = 0; j < coszs.size(); j++)
				{
					fprintf(out, "%d,%.12g,%.12g", anti, Es[i], coszs[j]);
					for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) fprintf(out, ",%.12g", P[i][j].arr[a][b]);
					fprintf(out, "\n");
				}
		}
	}
	else if (mode == "solar")
	{
		Probability_Engine pe;
		pe.Set_Oscillation_Parameters(num("s12sq"), num("s13sq"), num("s23sq"), num("delta"), num("Dmsq21"), num("Dmsq31"), true);
		pe.Set_Eigenvalue_Precision(-1);
		pe.Set_Spectra(Es, coszs);
		pe.Set_Earth(num("depth"), earth);
		pe.Set_rhoYe_Sun(num("rhoYe_sun"));
		auto D = pe.Get_Solar_Day_Probabilities();
		auto N = pe.Get_Solar_Night_Probabilities();
		fprintf(out, "E_GeV,cosz,Pee,Pem,Pet\n");
		for (size_t i = 0; i < Es.size(); i++)
		{
			fprintf(out, "%.12g,day,%.12g,%.12g,%.12g\n", Es[i], D[i].arr[0][0], D[i].arr[0][1], D[i].arr[0][2]);
			for (size_t j = 0; j < coszs.size(); j++)
				fprintf(out, "%.12g,%.12g,%.12g,%.12g,%.12g\n", Es[i], coszs[j], N[i][j].arr[0][0], N[i][j].arr[0][1], N[i][j].arr[0][2]);
		}
	}
	fclose(out);
	return 0;
}
