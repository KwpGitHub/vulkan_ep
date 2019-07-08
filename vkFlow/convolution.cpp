#include "convolution.h"

Convolution::Convolution(int in_channels, int out_channels, int kernel_size[3], int stride[3], int padding[3], padding_mode padding_mode ) :
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

void Convolution::operator ()(std::vector<float>& inpt) {
	x = inpt;
	X = kernel::Array<float>(device, x);

}

void Convolution::operator ()(kernel::Array<float> &inpt) {
	//X = inpt;
}

void Convolution::init() {

	weight_shape.C = input_shape.C;
	weight_shape.H = _kernel_size[0];
	weight_shape.W = _kernel_size[1];
	weight_shape.D = _kernel_size[2];
	weight_shape.N = _out_channels;

	size_t kernel_x = weight_shape.C;
	size_t kernel_y = weight_shape.D * weight_shape.H * weight_shape.W * weight_shape.N;

	size_t input_x = input_shape.D * input_shape.H * input_shape.W;
	size_t input_y = input_shape.C;

	output_shape.C = _out_channels;
	output_shape.H = (input_shape.H - _kernel_size[0] + 2 * _padding[0]) / _stride[0] + 1;
	output_shape.W = (input_shape.W - _kernel_size[0] + 2 * _padding[0]) / _stride[0] + 1;
	output_shape.D = (input_shape.D - _kernel_size[0] + 2 * _padding[0]) / _stride[0] + 1;
}

void Convolution::forward() {

}