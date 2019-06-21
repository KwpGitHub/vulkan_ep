#include <pybind11/pybind11.h>
#include <vector>
#include "pipeline/pipeline.h"
#include "utils.hpp"

namespace py = pybind11;

void Run() {
	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);
	

	auto device = runtime_info::instance.devices().at(runtime_info::deviceID);

	pipeline::Array<float> device_x = pipeline::Array<float>(device, x);
	pipeline::Array<float> device_y = pipeline::Array<float>(device, y);

	using Spec = pipeline::typelist<uint32_t>;
	struct Params { uint32_t size; float a; };
	auto program = pipeline::Program<Spec, Params>(device, LINEAR_LAYER);
	program.grid(128 / 64).spec(64)({ 128, 0.1 }, device_y, device_x);
	program.grid(128 / 64).spec(64)({ 128, 0.4 }, device_x, device_y);
	device_y.toHost(begin(y));
	device_x.toHost(begin(x));
	int t = 0;

}

void set_device(int t) {
	runtime_info::deviceID = t;
	runtime_info::instance = pipeline::Instance();
}

PYBIND11_MODULE(vkFlow, m) {
	m.def("Run", &Run);
	m.def("set_device", &set_device);


#ifdef VERSION_INFO
	m.attr("__version__") = VERSION_INFO;
#else
	m.attr("__version__") = "dev";
#endif
}