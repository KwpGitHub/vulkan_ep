#include <vector>
#include <pybind11/pybind11.h>
#include <iostream>
#include "layer_repo.h"

namespace py = pybind11;

void run() {
	std::cout << "HELLO" << std::endl;

	auto l = new backend::AbsVal_t();

	auto t = backend::model_layers;
	bool x = t[0]->support_inplace;
	std::cout << l << std::endl;
}

PYBIND11_MODULE(vkFlow, m) {
	m.def("run", &run);
}