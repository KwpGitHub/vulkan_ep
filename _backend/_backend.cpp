#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <numeric>
#include "kernel/vuh.h"
#include "tensor.h"
#include "layer.h"
#include "layers_map.h"

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

	//auto program = vuh::Program<Specs, Params>(device, "C:\\Users\\monish\\source\\repos\\vulkan_ep\\_backend/saxpy.spv");
	auto program = vuh::Program<Specs, Params>(device, "saxpy.spv");

	program.grid(128/64, 1, 1).spec(64, 1, 1).bind({ 128, 0.1f }, d_y, d_x); 
	program.run();
	d_y.toHost(begin(y));
	int error_count = 0;
	for (int i = 0; i < 128; ++i) {
		if (abs(y[i] - (1.0 + 0.1 * x[i])) > 1e-7) 
			error_count++;
	}

	if (error_count != 0)
		std::cout << ":::PIPELINE VALIDATION SUCCESS:::" << std::endl;
	else
		std::cout << ":::PIPELINE VALIDATION FAILURE:::" << std::endl;
	
	return;
}

void create_instance(py::str file_path) {
	std::cout << file_path << std::endl;
	backend::file_path = std::string(file_path) + std::string("..\\_backend\\");
	backend::instance = new vuh::Instance();
	backend::device = new vuh::Device(backend::instance->devices().at(0));
}

void create_tensor_from_numpy(py::str name, py::array_t<float> input){
	py::buffer_info buf = input.request();
	auto s = buf.shape;
	float* p = (float*)buf.ptr;

	std::vector<float> data;
	std::vector<uint32_t> s;

	for (auto _s : s) 
		s.push_back((uint32_t)_s);

	for (int i = 0; i < std::accumulate(s.begin(), s.end(), 1, std::multiplies<uint32_t>()); ++i)
		data.push_back(p[i]);

	backend::Shape_t _shape = { 1,1,1,1,1 };

	switch (s.size()) {
	case 1: _shape = { s[0], 1,1,1,1 };
			break;
	case 2: _shape = { s[0], 1, 1, 1, s[1] };
			break;
	case 3: _shape = { s[0], 1, 1, s[1], s[2] };
			break;
	case 4: _shape = { s[0], s[1], 1, s[2], s[3] };
			break;
	case 5: _shape = { s[0], s[1], s[2], s[3], s[4] };
	}

	std::cout << "UNINIT_TENSOR ::: " << name << std::endl;
	backend::Tensor* x = new backend::Tensor(data, _shape);
	backend::tensor_dict.insert(std::pair<std::string, backend::Tensor*>(std::string(name), x));

}

void create_tensor(py::str name, py::list data, py::list shape) {
	std::vector<float> d;
	std::vector<uint32_t> s;

	for (auto x : shape)
		s.push_back(x.cast<uint32_t>());
	for (auto x : data)
		d.push_back(x.cast<float>());
	

	backend::Shape_t _shape = { 1,1,1,1,1 };

	switch (s.size()) {
	case 1: _shape = { s[0], 1,1,1,1 };
			break;
	case 2: _shape = { s[0], 1, 1, 1, s[1] };
			break;
	case 3: _shape = { s[0], 1, 1, s[1], s[2] };
			break;
	case 4: _shape = { s[0], s[1], 1, s[2], s[3] };
			break;
	case 5: _shape = { s[0], s[1], s[2], s[3], s[4] };
	}

	std::cout << "TENSOR ::: "<< name << std::endl;
	backend::Tensor* x = new backend::Tensor(d, _shape);	
	backend::tensor_dict.insert(std::pair<std::string, backend::Tensor*>(std::string(name), x));
}

void create_layer(py::str name, py::str opType, py::list inputs, py::list outputs, py::dict attribute) {
	std::vector<std::string> i;
	std::vector<std::string> o;
	std::string n = std::string(name);
	std::string oT = std::string(opType);
	std::map<std::string, std::vector<std::string>> a;

	for (auto attr : attribute) {
		auto param = std::string(py::str(attr.first));
		std::vector<std::string> tmp;
		for (auto x : attr.second) {
			tmp.push_back(std::string(py::str(x)));
		}
		a.insert(std::pair<std::string, std::vector<std::string>>(param, tmp));
	}

	for (auto x : inputs)
		i.push_back(x.cast<std::string>());

	for (auto x : outputs)
		o.push_back(x.cast<std::string>());
	
	std::cout << "LAYERS ::: " << name << "\n\t input:[ ";
	for (auto x : i)
		std::cout << x << " ";
	std::cout << "] \n\t output:[";
	for (auto x : o)
		std::cout << x << " ";
	std::cout << "]" << std::endl;

	auto layer_create_func = backend::layer_map[oT];
	auto layer = layer_create_func(n, i, o, a);
	backend::layer_dict[n] = layer;

}

PYBIND11_MODULE(_backend, m) {
	m.def("create_instance", &create_instance);
	m.def("create_tensor", &create_tensor);
	m.def("create_layer", &create_layer);
	m.def("create_tensor_from_numpy", &create_tensor_from_numpy);
	m.def("test", &test);
}