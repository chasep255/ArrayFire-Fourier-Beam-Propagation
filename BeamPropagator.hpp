#ifndef _BEAM_PROPAGATOR_HPP_
#define _BEAM_PROPAGATOR_HPP_

#include <arrayfire.h>
#include <stddef.h>
#include <type_traits>
#include <cmath>
#include <limits>

template<typename real>
struct FpTypeTraits;

template<>
struct FpTypeTraits<double>
{
	static const af_dtype af_rtype = f64;
	static const af_dtype af_ctype = c64;
	typedef af::cdouble af_complex_t;
};

template<>
struct FpTypeTraits<float>
{
	static const af_dtype af_rtype = f32;
	static const af_dtype af_ctype = c32;
	typedef af::cfloat af_complex_t;
};

template<typename real>
class BeamPropagator
{	
	private:
	
	const static af_dtype ctype = FpTypeTraits<real>::af_ctype;
	const static af_dtype rtype = FpTypeTraits<real>::af_rtype;
	typedef typename FpTypeTraits<real>::af_complex_t af_complex_t;
		
	public:
	
	typedef typename FpTypeTraits<real>::af_complex_t complex_t;
	
	BeamPropagator(size_t _n, real _cell_size, real _lambda) :
		n(_n), cell_size(_cell_size), lambda(_lambda), z(0)
	{
		electric_field = af::constant(0.0, af::dim4(n, n), ctype);
		mask = af::constant(1.0, af::dim4(n, n), ctype);
		
		real k0 = 2 * real(M_PI) / lambda;
		af::array rng = af::range(n);
		af::array k = 2.0 * af::Pi * af::select(rng > n / 2, (rng - n) , rng) / (n * cell_size);
		kz = af::sqrt(af::constant(k0 * k0, af::dim4(n, n)) - af::tile((k * k).T(), n) - af::tile(k * k, 1, n));
	}
	
	void step(real dz)
	{
		af_complex_t i(0, 1);
		electric_field *= af::exp(-dz * i * (2 * M_PI / lambda) * mask);
		af::fft2InPlace(electric_field);
		electric_field *= af::exp(-dz * i * kz);
		af::ifft2InPlace(electric_field);
		
		if(clamp_boundaries)
			electric_field *= gain;
		
		z += dz;
	}
	
	template<typename MaskCallback>
	void stepDynamic(real dz, MaskCallback n, real dn_limit = -1.0)
	{
		af::array next_mask = n(z + dz);
		int steps = 1;
		if(dn_limit > 0)
		{
			af::array dn = af::abs(next_mask - mask);
			real max_dn = af::max<real>(dn);
			steps = std::max((real)1, std::ceil(max_dn / dn_limit));
		}
		
		real step_size = dz / steps;
		for(int s = 0; s < steps; s++)
		{
			if(s < steps - 1)
				setMask(n(z + step_size));
			else
				setMask(next_mask);
			
			step(step_size);
		}
	}
	
	af::array& getElectricField()
	{
		return electric_field;
	}
	
	void setElectricField(af::array e)
	{
		this->electric_field = e.as(ctype);
	}
	
	void setMask(af::array mask)
	{
		this->mask = mask.as(ctype);
	}
	
	af::array& getLastMask()
	{
		return mask;
	}
	
	const af::array& getLastMask() const
	{
		return mask;
	}
	
	size_t getGridDim() const
	{
		return n;
	}
	
	double getZ() const
	{
		return z;
	}
	
	void enableAbosrbingBoundaries(double boundary_percent = 0.05, double sigmas = 0.3)
	{
		clamp_boundaries = true;
		gain = af::constant(1.0, af::dim4(n, n));
		int border_size = std::max(1.0, std::ceil(boundary_percent * n));
		af::array border_gain = af::range(border_size);
		
		double sigma = border_size / sigmas;
		border_gain = af::exp(-border_gain * border_gain / (2.0 * sigma * sigma));
		af::print("", border_gain);
		border_gain = af::tile(border_gain, 1, n);
		gain(af::span, af::seq(border_size)) = af::flip(border_gain.T(), 1);
		gain(af::span, af::seq(n - border_size, n - 1, 1.0)) = border_gain.T();
		
		gain(af::seq(border_size), af::span) = af::min(af::flip(border_gain, 0), 
				gain(af::seq(border_size), af::span));
		
		gain(af::seq(n - border_size, n - 1, 1.0), af::span) = af::min(border_gain, 
				gain(af::seq(n - border_size, n - 1, 1.0), af::span));
	}
	
	void disableAbosrbingBoundaries()
	{
		clamp_boundaries = false;
		gain = af::array();
	}
	
	private:
	
	size_t n;
	real cell_size;
	real lambda;
	double z;
	af::array mask;
	af::array electric_field;
	af::array kz;
	
	bool clamp_boundaries = false;
	af::array gain;
};

#endif
