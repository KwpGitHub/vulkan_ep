#pragma once
#ifndef CONV_H
#define CONV_H

#include "layer.hpp"

enum padding_mode {
	zero,
	circle,
	mirror,
};

//kn2col

class Convolution :
	public layer
{
public:
	Convolution(int in_channels, int out_channels, int kernel_size[3], int stride[3], int padding[3], padding_mode padding_mode = zero);

protected:

	int _in_channels;
	int _out_channels;
	int _kernel_size[3] = { 0,0,0 };
	int _stride[3] = { 0,0,0 };
	int _padding[3] = { 0,0,0 };

	padding_mode _padding_mode;

	std::vector<float> x = std::vector<float>(16, 0.0);;
	std::vector<float> y = std::vector<float>(16, 0.0);
	std::vector<float> weight = std::vector<float>(16, 0.0);
	std::vector<float> bias = std::vector<float>(16, 0.0);

	kernel::Array<float> X = kernel::Array<float>(device, x);
	kernel::Array<float> Y = kernel::Array<float>(device, y);;
	kernel::Array<float> WEIGHT = kernel::Array<float>(device, weight);;
	kernel::Array<float> BIAS = kernel::Array<float>(device, bias);;

	Shape output_shape;
	Shape input_shape;
	Shape weight_shape;
	Shape bias_shape;

	struct Params { uint32_t x; uint32_t y; uint32_t z; };

	using Spec = kernel::typelist<uint32_t, uint32_t, uint32_t>;
	kernel::Program<Spec, Params> program = kernel::Program<Spec, Params>(device, LINEAR_LAYER);

	void operator () (std::vector<float>& inpt);
	void operator () (kernel::Array<float>& inpt);

	void init();
	void forward(); 

};

#endif //!CONV_H