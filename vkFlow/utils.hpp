#pragma once
#include "pipeline/pipeline.h"
#define LINEAR_LAYER "../assets/shaders/linear_layer.bin"

namespace runtime_info {
	static pipeline::Instance instance;
	static uint32_t deviceID;
};

struct Shape {
	int N;
	int C;
	int D;
	int H;
	int W;
	;
};