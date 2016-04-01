#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"
#include <unistd.h>
#include <cstring>

const int CELLS = 256;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA * 2;
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
	efld(s, s) = 0.1;
	
	af::array mask = af::constant(1.0, af::dim4(CELLS, CELLS));
	int wg_border = 50;
	
	af::seq in_guide(wg_border - 1, CELLS - wg_border - 1);
	mask(in_guide, in_guide) = 1.1;
	
	BeamPropagator<float> bp(CELLS, CELL_DIM, LAMBDA);
	bp.enableAbosrbingBoundaries();
	bp.setElectricField(efld);
	bp.setMask(mask);
	
	double distance = 0.02;
	int steps = distance / DZ;
	std::cout << steps << std::endl;
	af::array p(steps, c32);
	int percent = steps / 100;
	efld = af::conjg(efld);
	int pcounter = 0;
	for(int i = 0; i < steps; i++)
	{
		bp.step(DZ);
		p(i) = af::sum<double>(efld(in_guide, in_guide) * bp.getElectricField()(in_guide, in_guide));
		if(i % percent == percent - 1)
			std::cout << pcounter++ << std::endl;
	}
	
	af::fftInPlace(p);
	p = af::abs(p);
	p *= p > af::shift(p, -1) && p > af::shift(p, 1);
	
	
	af::Window w(2000, 1000);
	while(!w.close())
	{
		w.plot(af::range(steps), p.as(f32));
	}
	

//	
//	af::array max_values, max_idx;
//	af::max(max_values, max_idx, p(af::seq(p.dims(0) / 2)));
//	max_idx = 2.0 * M_PI * max_idx.as(f32) / (DZ * p.dims(0));
//	af::print("", af::join(1, max_values.as(f32), max_idx));
//	
//	double beta = max_idx.scalar<float>();
//	bp.setZ(0);
//	bp.setElectricField(af::conjg(efld));
//	
//	af::array mode = af::constant(0.0, af::dim4(in_guide.size, in_guide.size), c64);
//	
//	pcounter = 0;
//	for(int i = 0; i < steps; i++)
//	{
//		bp.step(DZ);
//		std::complex<double> f = std::exp(beta * std::complex<double>(0, 1) * bp.getZ());
//		mode += bp.getElectricField()(in_guide, in_guide) * af::cdouble(f.real(), f.imag());
//		if(i % percent == percent - 1)
//			std::cout << pcounter++ << std::endl;
//	}
//	
//	af::Window w(in_guide.size, in_guide.size);
//	mode = af::abs(mode);
//	while(!w.close())
//	{
//		w.image(mode.as(f32) / af::max<double>(mode));
//	}
	//w.plot(2.0 * M_PI * af::range(p.dims(0)) / (DZ * p.dims(0)), af::log(p.as(f32)));
	
//	int nframes = 200;
//	af::array data(CELLS, CELLS, nframes, c32);
//	for(int i = 0; i < nframes; i++)
//	{
//		bp.step(DZ);
//		data(af::span, af::span, i) = bp.getElectricField();
//	}
//				
//	data = af::reorder(data, 2, 0, 1); //make z the first dim
//	af::fftInPlace(data); //fft along the first dimension only which is z
//	data = af::reorder(data, 1, 2, 0); //make z the last dim
//	
//	af::Window wnd(CELLS, CELLS);
//	for(int i = 0; i < nframes && !wnd.close(); i++)
//	{
//		af::array img = af::arg(data(af::span, af::span, i));
//		img -= af::min<float>(img);
//		img /= af::max<float>(img);
//		wnd.image(img);
//		char fn[100];
//		sprintf(fn, "images/%d.jpg", i);
//		af::saveImage(fn, img);
//		usleep(100000);
//	}

//	af::Window wnd(bp.getGridDim(), bp.getGridDim());
//	while(!wnd.close())
//	{
//		af::timer tm = af::timer::start();
//		
//		for(int i = 0; i < RENDER_EVERY; i++)
//			bp.step(DZ);
//		
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
