#pragma once
#include "layer.hpp"

enum padding_mode {
	zero,
	circle,
	mirror,
};

class Convolution :
	public layer
{
public:
	Convolution(int in_channels, int out_channels, int kernel_size[3], int stride[3], int padding[3], padding_mode padding_mode = zero) :
		_in_channels(in_channels),
		_out_channels(out_channels),
		_padding_mode(padding_mode)
	{
		for (int i = 0; i < 3; ++i) {
			_kernel_size[i] = kernel_size[i];
			_stride[i] = stride[i];
			_padding[i] = padding[i];
		}
	}

protected:

	int _in_channels;
	int _out_channels;
	int _kernel_size[3] = { 0,0,0 };
	int _stride[3] = { 0,0,0 };
	int _padding[3] = { 0,0,0 };
	padding_mode _padding_mode;

	std::vector<float> x;
	std::vector<float> y;
	std::vector<float> weight;
	std::vector<float> bias;
	std::vector<float> bias;

	Shape output_shape;
	Shape input_shape;
	Shape weight_shape;
	Shape bias_shape;

	struct Params { uint32_t size; float a; };

	using Spec = pipeline::typelist<uint32_t>;
	pipeline::Program<Spec, Params> program = pipeline::Program<Spec, Params>(device, LINEAR_LAYER);

};

