#pragma once

#include "pipeline/pipeline.h"

class linear_layer
{
	pipeline::Device device;
	pipeline::Array<float, pipeline::mem::Device> input;
	pipeline::Array<float, pipeline::mem::Device> output;

	using Specs = pipeline::typelist<uint32_t, uint32_t, uint32_t, uint32_t>;
	struct Params { uint32_t b; uint32_t w; uint32_t h; uint32_t c; };
	pipeline::Program<Specs, Params> program;

public:
	linear_layer();
	~linear_layer();

}

