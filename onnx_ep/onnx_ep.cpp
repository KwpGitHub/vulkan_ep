#include <vector>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <iostream>
#include "paramdict.h"
#include "mat.h"

namespace py = pybind11;

enum {

};

void create_layer(py::dict node, py::dict data) {
	std::string name = std::string(py::str(node["name"]));
	std::string op_type = std::string(py::str(node["op_type"]));
	std::vector<std::string> input;
	std::vector<std::string> output;
	
	
	std::string data_name = std::string(py::str(node["name"]));
	std::vector<int> dims;
	std::vector<int> type;
	node["data"];
	 
}

void run() {

}



PYBIND11_MODULE(onnx_ep, m) {
	m.def("run", &run);
	m.def("create_layer", &create_layer);
}