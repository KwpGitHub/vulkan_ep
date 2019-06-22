#pragma once
#include "kernel/kernel.h"


#define LINEAR_LAYER "../assets/shaders/linear_layer.bin"
#define CONVOLUTION_LAYER "../assets/shaders/convolution_layer.bin"
#define RNN_LAYER "../assets/shaders/rnn_layer.bin"


namespace runtime_info {
	static kernel::Instance instance;
	static uint32_t deviceID;
};

struct Shape {
	int N;
	int C;
	int D;
	int H;
	int W;
};