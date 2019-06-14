#include <pybind11/pybind11.h>
#include <vector>
#include "documentaion.h"
#include "pipeline/pipeline.h"


namespace py = pybind11;
/*
 * Implements an example function.
 */


void Run() {
	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);

	pipeline::Instance instance = pipeline::Instance();
	pipeline::Device device = instance.devices().at(0);

	pipeline::Array<float> device_x = pipeline::Array<float>(device, y);
	pipeline::Array<float> device_y = pipeline::Array<float>(device, x);

	using Spec = pipeline::typelist<uint32_t>;
	struct Params { uint32_t size; float a; };

	auto program = pipeline::Program<Spec, Params>(device, "shader");
	program.grid(128 / 64).spec(64)({ 128, 0.1 }, device_y, device_x);

	device_y.toHost(begin(y));
}

PYBIND11_MODULE(vkFlow, m) {
	m.doc() = "Run Pipeline";
	m.def("Run()", &Run, "Run Model",)
}