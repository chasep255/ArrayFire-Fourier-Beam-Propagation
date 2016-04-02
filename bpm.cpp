#include <arrayfire.h>
#include <iostream>
#include <cmath>
#include "BeamPropagator.hpp"
#include <unistd.h>
#include <cstring>

const int CELLS = 512;
const float LAMBDA = 500.0e-9;
const float DZ = LAMBDA * 20;
const float CELL_DIM = LAMBDA;
const int RENDER_EVERY = 15;

int main()
{
	af::setBackend(AF_BACKEND_CUDA);
	af::info();
	
	//create gaussian beam initial condition for electric field
	af::array efld = af::range(CELLS) - CELLS / 2;
	efld *= efld;
	efld = af::tile(efld, 1, CELLS) + af::tile(efld.T(), CELLS);
	efld = af::exp(-efld / 600.0);
	
	
	//create square cross section wave guide
	af::array mask = af::constant(1.0, af::dim4(CELLS, CELLS));
	int wg_border = 50;
	af::seq in_guide(wg_border - 1, CELLS - wg_border - 1);
	mask(in_guide, in_guide) = 1.1;
	
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
	
	//propagate the first time to compute P(z)
	int pcounter = 0;
	for(int i = 0; i < steps; i++)
	{
		bp.step(DZ);
		
		//compute the correlation multiplied with the window function
		p(i) = af::sum<double>(efld(in_guide, in_guide) * bp.getElectricField()(in_guide, in_guide)) * 
				(1.0 - cos(2.0 * M_PI * (double)i / steps));
		
		if(i % percent == percent - 1)
			std::cout << pcounter++ << std::endl;
	}
	
	//FFT to go from P(z) -> P(beta)
	af::fftInPlace(p);
	//rotate it so zero is at the center
	p = af::shift(p, steps / 2);
	
	af::Window w(2000, 1000);
	while(!w.close())
		w.plot(af::range(p.dims(0)) - p.dims(0) / 2, af::abs(p));
	
	//compute the magnitued of the cplx numbers
	p = af::abs(p);
	
	//find local maxima (zero everything which is not a local max)
	af::array bk_two = af::shift(p, -2);
	af::array bk_one = af::shift(p, -1);
	af::array fw_one = af::shift(p, 1);
	af::array fw_two = af::shift(p, 2);
	p *= p > bk_one && p > fw_one && fw_one >= fw_two && bk_one >= bk_two;
	
	//sort P(beta) storing the sorted indices in beta
	af::array beta;
	af::sort(p, beta, p, 0, false);
	
	beta = beta.as(f64);
	
	//shift beta to zero in the center
	beta -= steps / 2;
	//convert index numbers to frequency
	beta *= 2.0 * af::Pi / (DZ * steps);
	
	
	int nmodes = std::min(100, af::count<int>(p));
	std::cout << "Modes Found: " << af::count<int>(p) << std::endl;
	
	//clip beta array since we only need the modes we will be computing eigenfunctions for
	beta = beta(af::seq(nmodes));
	
	//reset bp
	bp.setElectricField(af::conjg(efld));
	bp.setZ(0);
	
	//array of eigenfunctions
	af::array modes = af::constant(0.0, af::dim4(CELLS, CELLS, nmodes), c64);
	pcounter = 0;
	
	//compute the eignenfunctions
	for(int i = 0; i < steps; i++)
	{
		bp.step(DZ);
		const af::array& e = bp.getElectricField();
		af::array phi = af::exp(beta.as(c64) * af::cdouble(0, 1) * bp.getZ());
		
		//equation (22) for each of the desired modes
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
