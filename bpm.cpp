#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"
#include <unistd.h>

const int CELLS = 256;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 15;

struct MaskFunctor
{
	af::array operator()(double z)
	{
		return af::constant(1.0, af::dim4(CELLS, CELLS));
	}
};

int main()
{
	af::setBackend(AF_BACKEND_CUDA);
	af::info();
	
	af::array efld = af::constant(0.0, af::dim4(CELLS, CELLS));
	af::seq s(CELLS / 2 - 50, CELLS / 2 + 50);
	efld(s, s) = 1.0;
	
	BeamPropagator<float> bp(CELLS, CELL_DIM, LAMBDA);
	bp.enableAbosrbingBoundaries();
	bp.setElectricField(efld);
	
	int nframes = 1000;
	af::array data(CELLS, CELLS, nframes, c32);
	for(int i = 0; i < nframes; i++)
	{
		bp.step(DZ);
		data(af::span, af::span, i) = bp.getElectricField();
	}
				
	data = af::reorder(data, 2, 0, 1); //make z the first dim
	af::fftInPlace(data); //fft along the first dimension only which is z
	data = af::reorder(data, 1, 2, 0); //make z the last dim
	
	af::Window wnd(CELLS, CELLS);
	for(int i = 0; i < nframes && !wnd.close(); i++)
	{
		af::array img = af::abs(data(af::span, af::span, i));
		img /= af::max<float>(img);
		wnd.image(img);
		usleep(100000);
	}
	
//	MaskFunctor m;
//
//	af::Window wnd(bp.getGridDim(), bp.getGridDim());
//	while(!wnd.close())
//	{
//		af::timer tm = af::timer::start();
//		for(int i = 0; i < RENDER_EVERY; i++)
//			bp.stepDynamic(DZ, m);
//		af::sync();
//		double t = af::timer::stop(tm);
//		double steps_per_minute = RENDER_EVERY * 60 / t;
//		
//		af::array efld_mag = af::abs(bp.getElectricField());
//		af::array img(af::dim4(bp.getGridDim(), bp.getGridDim(), 3));
//		img(af::span, af::span, 0) = efld_mag;
//		img(af::span, af::span, 1) = efld_mag;
//		img(af::span, af::span, 2) = af::abs(bp.getLastMask()) - 1.0;
//		wnd.image(img);
//		
//		std::cout << "z = " << bp.getZ() << "m \t Steps / Min = " << (int)steps_per_minute << std::endl;
//	}
	return 0;
}
