#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"
#include <unistd.h>
#include <cstring>
#include <vector>
#include <fstream>
#include "modes.hpp"

const int CELLS = 256;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA * 5;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 30;

void display_modal_spectrum(af::array p)
{
	float avg = af::mean<float>(p);
	float* ph = p.host<float>();
	int lbound = 0, ubound = p.dims(0) - 1;
	for(; lbound < p.dims(0) && ph[lbound] < avg; lbound++);
	for(; ubound > 0 && ph[ubound] < avg; ubound--);
	af::freeHost(ph);
	
	int steps = 2 * p.dims(0);
	std::cout << lbound << "\t" << ubound << std::endl;
	p = p(af::seq((double)lbound, (double)ubound));
	af::array x = 2 * af::Pi * af::seq(lbound, ubound + 1) / (steps * DZ);
	af::Window w(2000, 1000);
	while(!w.close())
	{
		w.plot(x, p);
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
	
	af::array mask = af::constant(1.0, af::dim4(CELLS, CELLS));
	af::seq in_guide(49, CELLS - 50);
	mask(in_guide, in_guide) = 1.452;
	
	BeamPropagator<float> bp(CELLS, CELL_DIM, LAMBDA);
	bp.enableAbosrbingBoundaries();
	bp.setElectricField(efld);
	bp.setMask(mask);
	
	double distance = 1.0;
	int steps = distance / DZ;
	std::cout << steps << std::endl;
	af::array p(steps, c32);
	int percent = steps / 100;
	efld = af::conjg(efld);
	
	int pcounter = 0;
	for(int i = 0; i < steps; i++)
	{
		bp.step(DZ);
		
		p(i) = af::sum<double>(efld * bp.getElectricField()) * 
				(1.0 - cos(2.0 * M_PI * (double)i / steps));
		
		if(i % percent == percent - 1)
			std::cout << pcounter++ << std::endl;
	}
	
	af::fftInPlace(p);
	p = af::abs(p(af::seq(p.dims(0) / 2)));
	
	display_modal_spectrum(p);
	std::cout << p.dims(0) << std::endl;
	af::array bk_one = af::shift(p, -1);
	af::array fw_one = af::shift(p, 1);
	p *= p > bk_one && p > fw_one;
	
	af::array beta;
	af::sort(p, beta, p, 0, false);
	beta = beta.as(f64);
	beta *= 2.0 * af::Pi / (DZ * steps);
	
	int nmodes = std::min(20, af::count<int>(p));
	
	beta = beta(af::seq(nmodes));

	
	bp.setElectricField(af::conjg(efld));
	bp.setZ(0);
	
	af::array modes = af::constant(0.0, af::dim4(CELLS, CELLS, nmodes), c64);
	pcounter = 0;
	
	for(int i = 0; i < steps; i++)
	{
		bp.step(DZ);
		const af::array& e = bp.getElectricField();
		af::array phi = af::exp(beta.as(c64) * af::cdouble(0, 1) * bp.getZ());
		
		gfor(af::seq j, nmodes)
			modes(af::span, af::span, j) += e * (1.0 - cos(2.0 * M_PI * (double)i / steps)) * 
				af::tile(phi(j), e.dims());
		
		if(i % percent == percent - 1)
			std::cout << pcounter++ << std::endl;
	}
	
	for(int i = 0; i < nmodes; i++)
	{
		af::array img = af::abs(modes.slice(i)).as(f32);
		img /= af::max<float>(img);
		char file_name[200];
		sprintf(file_name, "images/%03d.jpg", i);
		af::saveImage(file_name, img);
	}
	
	for(int m = 0; m < nmodes; m++)
	{
		bp.setElectricField(modes.slice(m) / af::max<float>(modes.slice(m)));
		bp.setZ(0);
		
		af::Window wnd(bp.getGridDim(), bp.getGridDim());
		while(!wnd.close())
		{
			af::timer tm = af::timer::start();
			
			for(int i = 0; i < RENDER_EVERY; i++)
				bp.step(DZ);
			
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
	}
	return 0;
}
