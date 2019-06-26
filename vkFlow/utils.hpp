#pragma once
#ifndef UTILS_H
#define UTILS_H

#include "kernel/kernel.h"
#include <vector>

#define TILE_SIZE 64
#define GRID_SIZE 8

#define LINEAR_LAYER "../assets/shaders/linear_layer.bin"
#define CONVOLUTION_LAYER "../assets/shaders/convolution_layer.bin"
#define RNN_LAYER "../assets/shaders/rnn_layer.bin"	

struct Shape {
	size_t N;
	size_t C;
	size_t D;
	size_t H;
	size_t W;

	size_t size() {
		return N * C * D * H * W;
	}
};



#endif //!UTILS_Hs