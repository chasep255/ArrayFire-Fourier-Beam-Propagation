#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"
#include <unistd.h>
#include <cstring>
#include <vector>
#include <fstream>
#include "modes.hpp"
#include <memory>

const int CELLS = 256;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA * 4.24234;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 30;

void displayModalSpectrum(af::array p)
{
	p = p.as(f32);
	float avg = af::mean<float>(p);
	float* ph = p.host<float>();
	int lbound = 0, ubound = p.dims(0) - 1;
	for(; lbound < p.dims(0) && ph[lbound] < avg; lbound++);
	for(; ubound > 0 && ph[ubound] < avg; ubound--);
	af::freeHost(ph);
	
	int steps = 2 * p.dims(0);
	p = p(af::seq((double)lbound, (double)ubound));
	af::array x = 2 * af::Pi * af::seq(lbound, ubound + 1) / (steps * DZ);
	af::Window w(2000, 1000);
	while(!w.close())
	{
		w.plot(x, p);
	}
}

void displayTiledModes(const ModesAnalysisConfig& cfg, af::array modes)
{
	std::unique_ptr<BeamPropagator<float>[]> bps(new BeamPropagator<float>[modes.dims(2)]);
	
	for(int i = 0; i < modes.dims(2); i++)
	{
		bps[i] = BeamPropagator<float>(cfg.cell_count, cfg.cell_dim, cfg.lambda);
		bps[i].setMask(cfg.mask);
		bps[i].setElectricField(modes.slice(i));
	}
	
	int cols = std::ceil(std::sqrt(modes.dims(2)));
	int rows = modes.dims(2) / cols;
	rows += rows * cols != modes.dims(2);
	af::Window wnd(cols * cfg.cell_count, rows * cfg.cell_count);
	wnd.grid(rows, cols);
	while(!wnd.close())
	{
		for(int i = 0; i < modes.dims(2); i++)
		{
			bps[i].step(cfg.dz * 0.89645);
		}
		for(int i = 0; i < modes.dims(2); i++)
		{
			af::array tmp = af::abs(bps[i].getElectricField()).as(f32);
			wnd(i / cols, i % cols).image(tmp / af::max<float>(tmp));
		}
		wnd.show();
	}
}

int main()
{
	af::setBackend(AF_BACKEND_CUDA);
	af::info();
	
	af::array efld = af::range(CELLS) - CELLS / 2;
	efld *= efld;
	efld = af::tile(efld, 1, CELLS) + af::tile(efld.T(), CELLS);
	efld = af::exp(-efld / 1000.0);
	
//	af::array efld = af::constant(0.0, af::dim4(CELLS, CELLS));
//	efld(af::seq(90, 166), af::seq(90, 166)) = 1.0;
//	efld(af::seq(100, 156), af::seq(100, 156)) = 0.0;
	
//	af::array efld = af::constant(0.0, af::dim4(CELLS, CELLS));
//	efld(af::seq(90, 166), af::seq(90, 166)) = 1.0;
//	efld(af::seq(100, 156), af::seq(100, 156)) = 0.0;
	
	af::array mask = af::constant(1.45, af::dim4(CELLS, CELLS));
	af::seq in_guide(49, CELLS - 50);
	mask(in_guide, in_guide) = 1.46;
	
	//efld(in_guide, in_guide) = af::randu(af::dim4(in_guide.size, in_guide.size));
	
	af::Window wnd(CELLS, CELLS);
	while(!wnd.close())
	{
		wnd.image(efld);
	}
	
	ModesAnalysisConfig cfg;
	cfg.dz = DZ;
	cfg.steps = std::ceil(0.1 / DZ);
	cfg.cell_dim = CELL_DIM;
	cfg.cell_count = CELLS;
	cfg.initial_field = efld;
	cfg.lambda = LAMBDA;
	cfg.mask = mask;
	cfg.in_guide = af::constant(0, af::dim4(CELLS, CELLS));
	cfg.in_guide(in_guide, in_guide) = 1;
	
	af::array spectrum = modesComputeSpectrum<double>(cfg);
	displayModalSpectrum(spectrum);
	
	spectrum *= spectrum > af::shift(spectrum, -1) && spectrum > af::shift(spectrum, 1);
	af::array betas;
	af::sort(spectrum, betas, spectrum, 0, false);
	
	int nmodes = 20;
	nmodes = std::min(nmodes, af::count<int>(spectrum));
	spectrum = af::array();
	betas = betas(af::seq(nmodes));
	betas = modesSpectralIndexToBeta(betas, cfg);
	
	af::array modes = modesComputeFunctions<double>(cfg, betas);
	displayTiledModes(cfg, modes);
	
	for(int i = 0; i < nmodes; i++)
	{
		af::array img = af::abs(modes.slice(i)).as(f32);
		img /= af::max<float>(img);
		char fn[100];
		sprintf(fn, "images/mode_%f.jpg", betas(i).scalar<float>());
		af::saveImage(fn, img);
	}
	return 0;
}
