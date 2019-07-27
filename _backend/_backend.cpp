#include <pybind11/pybind11.h>
#include <vector>
#include "kernel/vuh.h"


void test() {
	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);

	auto instance = vuh::Instance();
	auto device = instance.devices().at(0);    // just get the first available device

	auto d_y = vuh::Array<float>(device, y);   // create device arrays and copy data
	auto d_x = vuh::Array<float>(device, x);

	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	struct Params { uint32_t size; float a; };    // shader push-constants interface

	auto program = vuh::Program<Specs, Params>(device, "saxpy.spv");
	program.grid(128/64, 1, 1).spec(64, 1, 1)({ 128, 0.1 }, d_y, d_x); 
	d_y.toHost(begin(y));	

	return;
}

void create_instance() {
	
}


PYBIND11_MODULE(_backend, m) {
	m.def("run", &create_instance);
	m.def("test", &test);
}