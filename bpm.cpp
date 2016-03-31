#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include <unistd.h>
#include <sys/time.h>
#include "BeamPropagator.hpp"

double prof_time()
{
	struct timeval now;
	gettimeofday(&now, NULL);
	return now.tv_sec + 1.0e-6 * now.tv_usec;
}

const int CELLS = 512;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 10;

struct MaskFunctor
{
	af::array operator()(double z)
	{
//		if(z < 0.02)
//		{
//			af::array wave_guide = af::constant(1.0, af::dim4(CELLS, CELLS));
//			af::array r = af::range(CELLS / 2);
//			r = af::join(0, af::flip(r, 0), r);
//			r *= r;
//			r = af::tile(r, 1, CELLS) + af::tile(r.T(), CELLS);
//			
//			wave_guide = 1.0 + 0.2 * (r < (200 * 200) * (0.02 - z) / 0.02);
//			return wave_guide;
//		}
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
	
	af::Window wnd(bp.getGridDim(), bp.getGridDim());
	while(!wnd.close())
	{
		for(int i = 0; i < RENDER_EVERY; i++)
			bp.stepDynamic(DZ, MaskFunctor());
		af::array efld_mag = af::abs(bp.getElectricField());
		af::array img(af::dim4(bp.getGridDim(), bp.getGridDim(), 3));
		img(af::span, af::span, 0) = efld_mag;
		img(af::span, af::span, 1) = efld_mag;
		img(af::span, af::span, 2) = af::abs(bp.getLastMask()) - 1.0;
		
		wnd.image(img);
		
		std::cout << bp.getZ() << std::endl;
	}
	return 0;
}
