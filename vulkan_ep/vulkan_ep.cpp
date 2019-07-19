#include <vector>
#include <pybind11/pybind11.h>
#include <iostream>


namespace py = pybind11;

void create_layer(py::str layer_type, py::dict parameter) {
	for (auto item : parameter) {
		std::cout << "key=" << std::string(py::str(item.first));
	}
}

void run() {
	
}

PYBIND11_MODULE(vkFlow, m) {
	m.def("run", &run);
	m.def("create_layer", &create_layer);
}