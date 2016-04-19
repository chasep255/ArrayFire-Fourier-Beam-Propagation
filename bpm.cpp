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
const float LAMBDA = 1000.0e-9;
const float DZ = LAMBDA;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 30;

void displayModalSpectrum(af::array p)
{
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
			bps[i].step(cfg.dz);
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
	
//	af::array efld = af::range(CELLS) - CELLS / 2;
//	
//	efld *= efld;
//	efld = af::tile(efld, 1, CELLS) + af::tile(efld.T(), CELLS);
//	efld = af::exp(-efld / 1000.0);
	
//	//double square
//	af::array efld = af::constant(0.0, af::dim4(CELLS, CELLS));
//	efld(af::seq(90, 166), af::seq(90, 166)) = 1.0;
//	efld(af::seq(100, 156), af::seq(100, 156)) = 0.0;

  //CROSS
	af::array efld = af::constant(0.0, af::dim4(CELLS, CELLS));
	efld(af::seq(115, 141), af::seq(90, 166)) = 1.0;
	efld(af::seq(90, 166), af::seq(115, 141)) = 1.0;
	
	af::array mask = af::constant(1.45, af::dim4(CELLS, CELLS));
	af::seq in_guide(49, CELLS - 50);
	mask(in_guide, in_guide) = 1.46;
	
//	af::array mask = af::range(CELLS) - CELLS / 2;
//	mask *= mask;
//	mask = af::tile(mask, 1, CELLS) + af::tile(mask.T(), CELLS);
//	mask = af::select(mask < 100.0 * 100.0, af::constant(1.456, mask.dims()), af::constant(1.455, mask.dims()));
//	af::print("", mask);
	
//	efld = af::constant(0, af::dim4(CELLS, CELLS));
//	af::array shark = af::resize(af::loadImage("shark.jpg", false), in_guide.size, in_guide.size);
//	efld(in_guide, in_guide) = shark;
		
	BeamPropagator<float> bp(CELLS, LAMBDA, CELL_DIM);
	bp.setElectricField(efld);
	bp.enableAbosrbingBoundaries();
	bp.setMask(mask);
	
	af::Window wnd(CELLS, CELLS);
	
	while(!wnd.close() && bp.getZ() < 0.5)
	{
		af::array ef = af::abs(bp.getElectricField()).as(f32);
		ef /= af::max<float>(ef);
		wnd.image(ef);
		char fn[100];
		std::sprintf(fn, "images/img%lf.png", bp.getZ());
		af::saveImage(fn, ef);
		for(int i = 0; i < 20; i++)
			bp.step(DZ);
		
	}
	
	return 0;
}
