#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>

namespace py = pybind11;

void create_layer(py::dict node, py::dict data) {
	
}

void run() {

}



PYBIND11_MODULE(onnx_ep, m) {
	m.def("run", &run);
	m.def("create_layer", &create_layer);
}