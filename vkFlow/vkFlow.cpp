#include <vector>
#include <pybind11/pybind11.h>
#include <iostream>
#include "layer_repo.h"

namespace py = pybind11;

static backend::VulkanDevice* g_vkdev = 0;

void run() {
	std::cout << "HELLO" << std::endl;

	auto l = new layer::AbsVal_t();
	auto in = backend::Mat(227, 227, 3);
	in.fill<float>(-1.0);
	auto t = layer::model_layers;
	bool x = t[0]->support_inplace;

	if (x) {
		int out = t[0]->forward_inplace(in);
	}

	std::cout << in.total() << std::endl;

	for (int i = 0; i < 227 * 227 * 3; ++i) {
		if (in[i] != 1) {
			std::cerr << i << " " << in[i] << std::endl;
		}
	}
	std::cout << "RAN ABS" << std::endl;

}

PYBIND11_MODULE(vkFlow, m) {
	m.def("run", &run);
}