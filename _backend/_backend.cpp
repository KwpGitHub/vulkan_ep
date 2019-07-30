#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include "kernel/vuh.h"
#include "tensor.h"

namespace py = pybind11;

void test() {
	auto y = std::vector<float>(128, 1.0f);
	auto x = std::vector<float>(128, 2.0f);

	auto instance = vuh::Instance();
	auto device = instance.devices().at(0);    // just get the first available device

	auto d_y = vuh::Array<float>(device, y);   // create device arrays and copy data
	auto d_x = vuh::Array<float>(device, x);

	using Specs = vuh::typelist<uint32_t, uint32_t, uint32_t>;     // shader specialization constants interface
	struct Params { uint32_t size; float a; };    // shader push-constants interface

	auto program = vuh::Program<Specs, Params>(device, "C:\\Users\\monish\\source\\repos\\vulkan_ep\\_backend/saxpy.spv");
	program.grid(128/64, 1, 1).spec(64, 1, 1)({ 128, 0.1 }, d_y, d_x); 
	d_y.toHost(begin(y));	

	return;
}

void create_instance() {
	backend::instance = new vuh::Instance();
}

void build_input_tensor(py::array_t<float> input){
	py::buffer_info buf = input.request();
	auto s = buf.shape;
	float* p = (float*)buf.ptr;

	std::vector<float> data;
	std::vector<uint32_t> shape;

	for (auto _s : s) 
		shape.push_back((uint32_t)_s);

	for (int i = 0; i < std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>()); ++i)
		data.push_back(p[i]);

	backend::Tensor x = backend::Tensor(data, shape);

}

void create_tensor(py::str name, py::list data, py::list shape) {
	std::vector<float> d;
	std::vector<uint32_t> s;

	for (auto x : shape)
		s.push_back(x.cast<uint32_t>());
	for (auto x : data)
		d.push_back(x.cast<float>());
	backend::Tensor* x = new backend::Tensor(d, s);
	backend::tensor_dict.insert(std::pair<std::string, backend::Tensor*>(std::string(name), x));
}

PYBIND11_MODULE(_backend, m) {
	m.def("create_instance", &create_instance);
	m.def("create_tensor", &create_tensor);
	m.def("input", &build_input_tensor);
	m.def("test", &test);
}