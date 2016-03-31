#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"

const int CELLS = 512;
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
	af::seq s(CELLS / 2 - 100, CELLS / 2 + 100);
	efld(s, s) = 1.0;
	
	BeamPropagator<double> bp(CELLS, CELL_DIM, LAMBDA);
	bp.enableAbosrbingBoundaries();
	bp.setElectricField(efld);
	
//	af::array c(1000);
//	for(int i = 0; i < c.dims(0); i++)
//	{
//		bp.step(DZ);
//		c(i) = af::sum<double>(efld * af::conjg(bp.getElectricField()));
//	}
//	
//	af::Window wnd(2000, 1000);
//	while(!wnd.close())
//	{
//		wnd.plot(af::range(c.dims(0)), af::abs(af::fft(c)));
//	}
	
	MaskFunctor m;
	
	af::Window wnd(bp.getGridDim(), bp.getGridDim());
	while(!wnd.close())
	{
		af::timer tm = af::timer::start();
		for(int i = 0; i < RENDER_EVERY; i++)
			bp.stepDynamic(DZ, m);
		af::sync();
		double t = af::timer::stop(tm);
		double steps_per_minute = RENDER_EVERY * 60 / t;
		
		af::array efld_mag = af::abs(bp.getElectricField());
		af::array img(af::dim4(bp.getGridDim(), bp.getGridDim(), 3));
		img(af::span, af::span, 0) = efld_mag;
		img(af::span, af::span, 1) = efld_mag;
		img(af::span, af::span, 2) = af::abs(bp.getLastMask()) - 1.0;
		
		wnd.image(img);
		
		std::cout << "z = " << bp.getZ() << "m \t Steps / Min = " << (int)steps_per_minute << std::endl;
	}
	return 0;
}
