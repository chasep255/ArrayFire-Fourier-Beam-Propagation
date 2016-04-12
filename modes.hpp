#ifndef _MODES_HPP_
#define _MODES_HPP_

#include <arrayfire.h>
#include "BeamPropagator.hpp"
#include "FpTypeTraits.hpp"
#include <cmath>

struct ModesAnalysisConfig
{
	double dz = 0.0;
	long steps;
	double lambda = 0.0;
	double cell_dim = 0.0;
	size_t cell_count = 0;
	af::array mask;
	af::array initial_field;
	af::array in_guide;
};

template<typename real>
af::array modesComputeSpectrum(const ModesAnalysisConfig& cfg)
{
	typedef FpTypeTraits<real> tt;
	//set up a BeamPropagator
	BeamPropagator<real> bp(cfg.cell_count, cfg.cell_dim, cfg.lambda);
	bp.setMask(cfg.mask);
	bp.setElectricField(cfg.initial_field);
	//array to store P(z)
	af::array p(cfg.steps, tt::af_ctype);
	//Complex conjugate of E(x,y,0)
	af::array e_conj = af::conjg(cfg.initial_field).as(tt::af_ctype);
	
	for(long s = 0; s < cfg.steps; s++)
	{
		//perform a step
		bp.step(cfg.dz);
		//compute the correlation function by numerically integrating
		p(s) = af::sum<double>(e_conj * bp.getElectricField() * cfg.in_guide) *
					(1.0 - cos(2.0 * M_PI * (double)s / cfg.steps));
	}
	//return the magnitude of the FFT, also truncate the negative betas
	return af::abs(af::fft(p)(af::seq(cfg.steps / 2)));
}

af::array modesSpectralIndexToBeta(af::array i, const ModesAnalysisConfig& cfg)
{
	return 2.0 * af::Pi / (cfg.steps * cfg.dz) * i;
}

double modesSpectralIndexToBeta(long i, const ModesAnalysisConfig& cfg)
{
	return 2.0 * af::Pi / (cfg.steps * cfg.dz) * i;
}

template<typename real>
af::array modesComputeFunctions(const ModesAnalysisConfig& cfg, af::array betas)
{
	BeamPropagator<real> bp(cfg.cell_count, cfg.cell_dim, cfg.lambda);
	bp.setMask(cfg.mask);
	bp.setElectricField(cfg.initial_field);
	//3D matrix where each 2D slice stores an eigenfunction being calculated
	af::array modes = af::constant(0.0, af::dim4(cfg.cell_count, cfg.cell_count, betas.elements()), c64);
	for(long s = 0; s < cfg.steps; s++)
	{
		bp.step(cfg.dz);
		const af::array& e = bp.getElectricField();
		//phi = exp(iBz)
		af::array phi = af::exp(betas.as(c64) * af::cdouble(0, 1) * bp.getZ());
		//ArrayFire parallel for loop
		gfor(af::seq j, betas.elements())
			modes(af::span, af::span, j) += e * (1.0 - cos(2.0 * M_PI * (double)s / cfg.steps)) * 
				af::tile(phi(j), e.dims()) * cfg.in_guide;
	}
	return modes;
}

#endif
