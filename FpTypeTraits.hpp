#ifndef _FP_TYPE_TRAITS_HPP_
#define _FP_TYPE_TRAITS_HPP_

#include <arrayfire.h>

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

#endif
